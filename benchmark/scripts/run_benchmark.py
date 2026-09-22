"""
run_benchmark.py — Orchestrated two-phase benchmark runner (v2).

What this replaces:
  The v1 nested-loop driver (experiments × models × languages × samples)
  had no model lifecycle management, no telemetry, and no single-process
  guarantee. Running several of those drivers concurrently on the same
  RTX 4060 corrupted the throughput numbers by 0.35×–4.4× (see
  REPORT.md §data-reliability for the evidence).

What this adds (md §17 audit requirements):
  * Two-phase orchestration: every model finishes A_fixed_tokens before
    any model starts B_fixed_semantic_length (and vice versa). Eliminates
    the cross-experiment taint that comes from running A and B of the
    same model with different GPU memory pressure.
  * Models are sorted ascending by parameter_count_b (ties broken by
    name) — smallest first.
  * Strict single-process lock: refuse to start if `.run.pid` already
    points at a live PID. The orchestrator refuses to run if Ollama has
    a different model loaded (it's unloaded first).
  * Per-model lifecycle hook via `ModelOrchestrator`:
      preflight → ensure_loaded → loop (lang × sample × repeat)
        → unload → wait_for_unloaded → summary.json.
  * Every raw JSON now carries a `host` block: nvidia-smi / psutil /
    /proc snapshots before and after each call, plus client-side
    request_sent / first_byte / response_received timestamps and HTTP
    response headers. This lets post-hoc analysis distinguish genuine
    inference time from queueing time and from GPU contention.
  * Retry-with-backoff per cell (3 attempts by default).
  * Auto-resume: cells whose raw JSON exists are skipped. Resume mode
    is on by default; `--no-resume` re-runs everything.
  * Continues on per-model failure: a model that 500's through all its
    retries is logged but does not abort the whole run.
  * Per-model summary JSON under `results/raw/<out-dir>/model_summary/`
    plus a top-level RUN_REPORT.md.

Output layout:
  results/raw/<out-dir>/
    ├── A_fixed_tokens/
    │     <model>__<lang>__<sample>__r<repeat>.json  (raw per cell)
    ├── B_fixed_semantic_length/
    │     <model>__<lang>__<sample>__r<repeat>.json
    ├── model_summary/
    │     <model>__<experiment>.json                (per-model rollup)
    ├── run_meta.json                                 (top-level meta)
    └── RUN_REPORT.md                                 (human summary)

Usage examples:
  python -m scripts.run_benchmark
      # full matrix, both phases, default out-dir = new-YYYYMMDD-HHMMSS

  python -m scripts.run_benchmark --phase A
      # only A_fixed_tokens across all models

  python -m scripts.run_benchmark --phase both --out-dir new-20260906-1
      # explicit out-dir (so two runs land in different folders)

  python -m scripts.run_benchmark --models qwen3:0.6b --samples 3 --repeats 2
      # tiny smoke matrix
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from scripts.model_orchestrator import ModelOrchestrator


# --------------------------------------------------------------------------- #
def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def sort_models(models_cfg: list[dict]) -> list[dict]:
    """Sort by (parameter_count_b asc, name case-insensitive asc).

    Missing numeric falls back to +inf so unconfigured models run last.
    """
    def key(m: dict):
        b = m.get("parameter_count_b")
        if b is None:
            try:
                b = float(str(m.get("parameter_size", "0")).rstrip("BMm").replace("B", "") or "0")
                if "M" in str(m.get("parameter_size", "")):
                    b = b / 1000.0
            except Exception:
                b = float("inf")
        return (b, m["name"].lower())

    return sorted(models_cfg, key=key)


def load_prompts(experiment_key: str, root: Path) -> dict:
    cfg_path = root / "prompts" / (
        "experiment_a.json" if experiment_key == "A_fixed_tokens" else "experiment_b.json"
    )
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def load_samples(root: Path) -> list[dict]:
    samples = []
    with (root / "datasets" / "samples.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def render_user_prompt(
    template: dict, language: str, source_text: str, target_chars: Optional[int]
) -> str:
    tmpl = template["templates"][language]
    return tmpl.format(source=source_text, n=target_chars or 0)


def raw_path_for(
    out_dir: Path, experiment: str, model: str, language: str,
    sample_id: str, repeat: int,
) -> Path:
    safe_model = model.replace("/", "_").replace(":", "_")
    return (
        out_dir
        / experiment
        / f"{safe_model}__{language}__{sample_id}__r{repeat:02d}.json"
    )


def save_raw(
    out_dir: Path, experiment: str, model: str, language: str,
    sample_id: str, repeat: int, payload: dict,
) -> Path:
    p = raw_path_for(out_dir, experiment, model, language, sample_id, repeat)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def raw_exists(
    out_dir: Path, experiment: str, model: str, language: str,
    sample_id: str, repeat: int,
) -> bool:
    return raw_path_for(out_dir, experiment, model, language, sample_id, repeat).exists()


def save_model_summary(out_dir: Path, summary: dict) -> Path:
    safe_model = summary["model"].replace("/", "_").replace(":", "_")
    p = out_dir / "model_summary" / f"{safe_model}__{summary['experiment']}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
THINKING_MODELS = {
    "qwen3:0.6b", "qwen3:0.6b-q8", "qwen3.5:0.8b", "qwen3-vl:2b",
    "deepseek-r1:1.5b",
    "huihui_ai/deepseek-r1-abliterated:1.5b",
    "lfm2.5-thinking:latest",
}


def num_predict_for(model: str, base: int, inf: dict) -> int:
    override = (
        inf.get("num_predict_thinking", 2048)
        if model in THINKING_MODELS
        else inf.get("num_predict_default", 1024)
    )
    return max(base, override)


def run_one_model(
    *,
    out_dir: Path,
    experiment: str,
    experiment_cfg: dict,
    model_name: str,
    languages: list[dict],
    samples: list[dict],
    inf: dict,
    prompt_template: dict,
    log,
    resume: bool,
    overwrite: bool,
    retries: int,
) -> dict:
    """Drive one model through every (lang × sample × repeat) cell.

    Catches per-cell and per-model exceptions so a single bad cell does
    not abort the whole benchmark. The orchestrator tracks aggregate
    counters; this function returns its summary dict.
    """
    base_num_predict = (
        experiment_cfg["num_predict"]
        if experiment == "A_fixed_tokens"
        else experiment_cfg.get("max_tokens_safety", 1024)
    )
    num_predict = num_predict_for(model_name, base_num_predict, inf)
    target_chars = experiment_cfg.get("target_char_count")
    think = None  # ollama_client default: False for thinking models, True otherwise

    orch = ModelOrchestrator(model=model_name, log=log, retry_attempts=retries)
    log(f"  >>> {model_name} × {experiment} START")
    orch.start_timer()

    try:
        orch.preflight()
        orch.ensure_loaded()
    except Exception as exc:
        orch.errors.append({
            "phase": "lifecycle",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "wall_clock_unix": time.time(),
        })
        log(f"  >>> {model_name} lifecycle FAILED: {exc}")
        summary = orch.summary()
        summary.experiment = experiment  # type: ignore[attr-defined]
        save_model_summary(out_dir, {**summary.to_dict(), "experiment": experiment})
        return summary.to_dict()

    cells_target = len(languages) * len(samples) * inf["repeat"]
    log(f"  >>> {model_name} cells_target = {cells_target}")

    for lang in languages:
        for sample in samples:
            source_text = sample["texts"].get(lang["code"], "")
            user_prompt = render_user_prompt(
                prompt_template, lang["code"], source_text, target_chars
            )
            for r in range(inf["repeat"]):
                if resume and not overwrite and raw_exists(
                    out_dir, experiment, model_name, lang["code"], sample["sample_id"], r
                ):
                    orch.record_skipped(1)
                    continue
                payload = orch.invoke(
                    system=prompt_template["system"],
                    user=user_prompt,
                    temperature=inf["temperature"],
                    top_p=inf["top_p"],
                    top_k=inf["top_k"],
                    seed=inf["seed"],
                    num_ctx=inf["num_ctx"],
                    num_predict=num_predict,
                    keep_alive=inf["keep_alive"],
                    timeout_s=inf["request_timeout_s"],
                    think=think,
                    prompt_user_text=user_prompt,
                    prompt_source_text=source_text,
                    sample_id=sample["sample_id"],
                    repeat=r,
                    language=lang["code"],
                    experiment=experiment,
                )
                if payload is None:
                    continue
                save_raw(
                    out_dir, experiment, model_name, lang["code"],
                    sample["sample_id"], r, payload,
                )

    log(
        f"  >>> {model_name} cells={orch.cells_attempted} "
        f"saved={orch.cells_saved} skipped={orch.cells_skipped} "
        f"failed={orch.cells_failed} retries={orch.retry_count}"
    )

    try:
        orch.unload()
    except Exception as exc:
        orch.errors.append({
            "phase": "unload", "error": str(exc),
            "traceback": traceback.format_exc(),
            "wall_clock_unix": time.time(),
        })
        log(f"  >>> {model_name} unload EXC: {exc}")

    summary = orch.summary()
    summary_dict = {**summary.to_dict(), "experiment": experiment}
    save_model_summary(out_dir, summary_dict)
    log(f"  >>> {model_name} × {experiment} DONE")
    return summary_dict


# --------------------------------------------------------------------------- #
def write_run_report(out_dir: Path, run_meta: dict, all_summaries: list[dict]) -> Path:
    path = out_dir / "RUN_REPORT.md"
    lines = [
        f"# Run report — {run_meta['out_dir']}",
        "",
        f"- started:   {run_meta['started_iso']}",
        f"- ended:     {datetime.fromtimestamp(run_meta['ended_unix'], tz=timezone.utc).isoformat()}",
        f"- elapsed_s: {run_meta['ended_unix'] - run_meta['started_unix']:.1f}",
        f"- phase:     {run_meta['phase']}",
        f"- models:    {len(run_meta['models'])} (sorted by parameter_count_b asc)",
        f"- languages: {len(run_meta['languages'])} (incl. rus_Cyrl)",
        f"- samples:   {run_meta['n_samples']}",
        f"- repeats:   {run_meta['repeat']}",
        "",
        "## Per-model rollup",
        "",
        "| experiment | model | cells | skipped | failed | retries | peak VRAM (MiB) | unload ok | unload (s) | initial load (ms) | errors |",
        "|---|---|---:|---:|---:|---:|---:|:---:|---:|---:|---:|",
    ]
    for s in all_summaries:
        lines.append(
            f"| {s.get('experiment','')} | {s['model']} | "
            f"{s['cells_attempted']} | {s['cells_skipped']} | "
            f"{s['cells_failed']} | {s['retries']} | "
            f"{s['peak_vram_mib']:.0f} | "
            f"{'✓' if s['unload_succeeded'] else '✗'} | "
            f"{s['unload_elapsed_s']:.1f} | "
            f"{(s['initial_load_duration_ns'] or 0)/1e6:.0f} | "
            f"{len(s['errors'])} |"
        )
    lines += [
        "",
        "## Output layout",
        "",
        f"- Raw per-cell JSON: `{out_dir}/<experiment>/<model>__<lang>__<sample>__rNN.json`",
        f"- Per-model summaries: `{out_dir}/model_summary/<model>__<experiment>.json`",
        f"- This report: `{out_dir}/RUN_REPORT.md`",
        "",
        "## How to analyse",
        "",
        "```bash",
        f"python3 -m scripts.calculate_metrics --runs-dir {out_dir}",
        f"python3 -m scripts.analyze_results  --runs-dir {out_dir}",
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def acquire_single_process_lock(root: Path, log) -> bool:
    """Return True if lock acquired, False if another run is already live.

    The returned value is sticky: callers MUST only call
    `release_single_process_lock` when they got `True` back here. Otherwise
    the lock file would be wiped on the failed attempt and a second
    concurrent run could start.

    Edge case — `os.kill(pid, 0)` returns success for the *current* PID
    too, so a stale lock file that happens to share our PID (PID reuse
    after crash, or interrupted cleanup) would otherwise look like a
    conflict with ourselves. We treat that case as stale and overwrite.
    """
    import os
    lock = root / ".run.pid"
    if lock.exists():
        try:
            old = int(lock.read_text().strip())
            if old == os.getpid():
                log(f"  stale .run.pid ({old}) — same as current PID, removing")
                lock.unlink(missing_ok=True)
            else:
                os.kill(old, 0)
                log(f"ERROR: benchmark already running, PID={old}")
                return False
        except (ProcessLookupError, ValueError, OSError):
            log(f"  stale .run.pid ({lock.read_text().strip()}) — removing")
            lock.unlink(missing_ok=True)
    lock.write_text(str(os.getpid()))
    log(f"  acquired single-process lock (PID={os.getpid()})")
    return True


def release_single_process_lock(root: Path) -> None:
    (root / ".run.pid").unlink(missing_ok=True)


def install_signal_handlers(root: Path, log) -> None:
    """Install SIGTERM/SIGINT handlers so the lock is always released.

    Without this, a SIGTERM that lands mid Ollama HTTP call can kill the
    process before the `finally` block runs, leaving `.run.pid` orphaned
    and blocking future runs.
    """
    import atexit
    import signal

    def _handler(signum, frame):
        log(f"  received signal {signum} — cleaning up lock and exiting")
        release_single_process_lock(root)
        import sys
        sys.exit(128 + signum)

    atexit.register(lambda: release_single_process_lock(root))
    try:
        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGINT, _handler)
    except (ValueError, OSError):
        pass  # main thread only — fine in single-process mode


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml", type=Path)
    parser.add_argument("--models", default="",
                        help="comma-separated subset of model names")
    parser.add_argument("--languages", default="",
                        help="comma-separated subset of language codes")
    parser.add_argument("--experiment", default="",
                        help="legacy single experiment (A_fixed_tokens / B_fixed_semantic_length)")
    parser.add_argument("--phase", default="both", choices=["A", "B", "both"],
                        help="which experiment to run this invocation")
    parser.add_argument("--out-dir", default="",
                        help="raw output directory under results/raw/. "
                             "Defaults to new-YYYYMMDD-HHMMSS.")
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="tiny matrix: 1 model × 1 lang × 1 sample × 1 repeat")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="(default) skip cells whose raw JSON exists")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-run every cell even if a raw JSON exists")
    parser.add_argument("--retries", type=int, default=3,
                        help="per-cell retry count on transient errors")
    parser.add_argument("--log", default="logs/run.log", type=Path)
    parser.add_argument("--no-lock", action="store_true",
                        help="skip the single-process PID lock (debug only)")
    args = parser.parse_args()

    root = _project_root()
    log_fh = args.log.parent.mkdir(parents=True, exist_ok=True) or args.log.open("a", encoding="utf-8")
    log_lock = lambda msg: _emit(log_fh, msg)
    log_lock(f"=== run_benchmark start (pid={__import__('os').getpid()}) ===")

    lock_held = False
    if not args.no_lock:
        if not acquire_single_process_lock(root, log_lock):
            log_fh.close()
            return 1
        lock_held = True
        install_signal_handlers(root, log_lock)

    try:
        return _run(args, root, log_lock)
    finally:
        if lock_held:
            release_single_process_lock(root)
        log_fh.close()


def _emit(log_fh, msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line)
    log_fh.write(line + "\n")
    log_fh.flush()


def _run(args, root: Path, log) -> int:
    cfg = load_config(args.config)
    inf = dict(cfg["inference"])
    if args.repeats is not None:
        inf["repeat"] = args.repeats
    if args.smoke:
        inf["repeat"] = 1
        inf["warmup_runs"] = 0

    samples = load_samples(root)
    if args.samples is not None:
        samples = samples[: args.samples]
    if args.smoke:
        samples = samples[:1]

    models_all = sort_models(cfg["models"])
    if args.models:
        wanted = set(args.models.split(","))
        models = [m for m in models_all if m["name"] in wanted]
    elif args.smoke:
        models = models_all[:1]
    else:
        models = models_all

    languages = cfg["languages"]
    if args.languages:
        wanted = set(args.languages.split(","))
        languages = [l for l in languages if l["code"] in wanted]
    if args.smoke:
        languages = languages[:1]

    experiments = cfg["experiments"]
    phase_order = []
    if args.experiment:
        phase_order = [args.experiment]
    elif args.phase == "A":
        phase_order = ["A_fixed_tokens"]
    elif args.phase == "B":
        phase_order = ["B_fixed_semantic_length"]
    else:
        phase_order = ["A_fixed_tokens", "B_fixed_semantic_length"]

    if args.out_dir:
        out_dir_name = args.out_dir
    else:
        out_dir_name = f"new-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out_dir = root / "results" / "raw" / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    started_unix = time.time()
    run_meta = {
        "out_dir": str(out_dir.relative_to(root)),
        "started_unix": started_unix,
        "started_iso": datetime.fromtimestamp(started_unix, tz=timezone.utc).isoformat(),
        "phase": args.phase,
        "experiment": args.experiment or args.phase,
        "models": [m["name"] for m in models],
        "languages": [l["code"] for l in languages],
        "n_samples": len(samples),
        "repeat": inf["repeat"],
        "resume": args.resume,
        "overwrite": args.overwrite,
        "retries": args.retries,
        "config_path": str(args.config),
    }

    log(
        f"=== run_benchmark config: "
        f"models={len(models)} langs={len(languages)} "
        f"samples={len(samples)} repeats={inf['repeat']} "
        f"phase={args.phase} resume={args.resume} out_dir={out_dir.name}"
    )
    log(f"  models in order: {[m['name'] for m in models]}")
    log(f"  languages: {[l['code'] for l in languages]}")

    all_summaries: list[dict] = []
    for exp_key in phase_order:
        exp_cfg = experiments[exp_key]
        prompt_template = load_prompts(exp_key, root)
        log(f"-- PHASE {exp_key} --")
        for m in models:
            try:
                summary = run_one_model(
                    out_dir=out_dir,
                    experiment=exp_key,
                    experiment_cfg=exp_cfg,
                    model_name=m["name"],
                    languages=languages,
                    samples=samples,
                    inf=inf,
                    prompt_template=prompt_template,
                    log=log,
                    resume=args.resume,
                    overwrite=args.overwrite,
                    retries=args.retries,
                )
                all_summaries.append(summary)
            except Exception as exc:
                tb = traceback.format_exc()
                log(f"!!! {m['name']} × {exp_key} FATAL: {exc}\n{tb}")
                all_summaries.append({
                    "experiment": exp_key,
                    "model": m["name"],
                    "cells_attempted": 0,
                    "cells_saved": 0,
                    "cells_failed": 0,
                    "cells_skipped": 0,
                    "retries": 0,
                    "warmups_attempted": 0,
                    "warmups_failed": 0,
                    "peak_vram_mib": 0.0,
                    "initial_load_duration_ns": None,
                    "unload_succeeded": False,
                    "unload_elapsed_s": 0.0,
                    "errors": [{
                        "phase": "run_one_model",
                        "error": str(exc),
                        "traceback": tb,
                        "wall_clock_unix": time.time(),
                    }],
                })

    run_meta["ended_unix"] = time.time()
    (out_dir / "run_meta.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    report_path = write_run_report(out_dir, run_meta, all_summaries)
    log(f"=== run_benchmark done — report at {report_path.relative_to(root)} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())