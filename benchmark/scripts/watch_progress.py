#!/usr/bin/env python3
"""
watch_progress.py — Live dashboard for the orchestrated v2 benchmark.

Two modes:

  (a) DEFAULT: refresh every 2 s.  Prints a one-screen dashboard with:
        - which (model × language × experiment) cell is currently running
        - how many cells of the matrix are done
        - ETA based on rolling average per-cell time
        - last 5 log lines
        - currently-loaded Ollama model + peak VRAM observed so far

  (b) --once:  print the dashboard once and exit (for piping to files)

  (c) --plain: print the same info as plain key=value lines (for grep/jq)

  (d) --runs-dir PATH: watch a specific results/raw/new-XXX folder
                       instead of auto-detecting the latest one.

Reads:
  .run.pid                        ← PID of the running benchmark
  configs/config.yaml             ← to know the full matrix dimensions
  logs/run.log                    ← run_benchmark writes here
  results/raw/<run>/<exp>/*.json  ← one file per (model, lang, sample, repeat)
  results/raw/<run>/run_meta.json ← current phase / model order
  results/raw/<run>/model_summary/*.json  ← per-model rollup

Usage:
  python3 -m scripts.watch_progress
  python3 -m scripts.watch_progress --once
  python3 -m scripts.watch_progress --plain
  python3 -m scripts.watch_progress --interval 5
  python3 -m scripts.watch_progress --runs-dir results/raw/new-20260906-120000
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from scripts.ollama_client import list_loaded_models


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "results" / "raw"


def _safe_load_yaml(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def _running(pidfile: Path) -> int | None:
    if not pidfile.exists():
        return None
    try:
        pid = int(pidfile.read_text().strip())
    except ValueError:
        return None
    try:
        os.kill(pid, 0)
    except OSError:
        return None
    return pid


def _log_tail(path: Path, n: int = 5) -> list[str]:
    if not path.exists():
        return []
    try:
        with path.open(encoding="utf-8") as fh:
            lines = fh.readlines()
        return [l.rstrip("\n") for l in lines[-n:]]
    except Exception:
        return []


def _current_cell(log_lines: list[str]) -> str:
    """Return the most recent '>>> model START' or 'unload' line."""
    for line in reversed(log_lines):
        if ">>>" in line or "PHASE" in line or "unload" in line:
            return line.split("]", 1)[-1].strip()
    return "(idle)"


def _latest_run_dir() -> Path | None:
    if not RAW_ROOT.exists():
        return None
    candidates = sorted(
        (p for p in RAW_ROOT.iterdir() if p.is_dir() and p.name.startswith("new-")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return RAW_ROOT
    if any(candidates[0].rglob("*.json")):
        return candidates[0]
    return RAW_ROOT


def _scan_done(run_dir: Path, n_repeats: int) -> dict:
    cells_partial: Counter = Counter()
    cells_full: set = set()
    files: list[Path] = []
    if not run_dir.exists():
        return {"n_files": 0, "cells_partial": {}, "cells_full": set()}
    for p in run_dir.rglob("*.json"):
        if p.name == "run_meta.json":
            continue
        if "model_summary" in p.parts:
            continue
        files.append(p)
        stem = p.stem
        bits = stem.split("__")
        if len(bits) >= 4:
            exp = p.relative_to(run_dir).parts[0]
            model, lang = bits[0], bits[1]
            cells_partial[(exp, model, lang)] += 1
            if cells_partial[(exp, model, lang)] >= n_repeats:
                cells_full.add((exp, model, lang))
    return {
        "n_files": len(files),
        "cells_partial": dict(cells_partial),
        "cells_full": cells_full,
    }


def _eta_seconds(done_cells: int, total_cells: int,
                 start_time: float | None, running: bool) -> float | None:
    if not running:
        return None
    if not start_time or done_cells <= 0:
        return None
    elapsed = time.time() - start_time
    rate = done_cells / elapsed
    if rate <= 0:
        return None
    return max(0.0, (total_cells - done_cells) / rate)


def _fmt_eta(s: float | None) -> str:
    if s is None:
        return "—"
    if s < 60:
        return f"{s:.0f}s"
    m, sec = divmod(s, 60)
    if m < 60:
        return f"{m:.0f}m{sec:.0f}s"
    h, m = divmod(m, 60)
    return f"{h:.0f}h{m:.0f}m"


def _bar(done: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "[" + " " * width + "]"
    pct = done / total
    fill = int(pct * width)
    return "[" + "#" * fill + "-" * (width - fill) + f"] {pct*100:5.1f}%"


def _start_time_from_log(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    try:
        first = log_path.read_text(encoding="utf-8").split("\n", 1)[0]
        if "===" in first:
            today = time.strftime("%Y-%m-%d", time.localtime())
            stamp = first.split("]", 1)[0].lstrip("[")
            return time.mktime(time.strptime(f"{today} {stamp}", "%Y-%m-%d %H:%M:%S"))
    except Exception:
        pass
    try:
        return log_path.stat().st_mtime
    except Exception:
        return None


def _load_run_meta(run_dir: Path) -> dict:
    p = run_dir / "run_meta.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _per_model_summary(run_dir: Path) -> dict:
    out: dict = {}
    p = run_dir / "model_summary"
    if not p.exists():
        return out
    for f in p.glob("*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            out[d["model"]] = d
        except Exception:
            continue
    return out


# --------------------------------------------------------------------------- #
def render_dashboard(plain: bool = False, runs_dir: Path | None = None) -> dict:
    cfg = _safe_load_yaml(ROOT / "configs" / "config.yaml")
    models_cfg = sorted(
        cfg.get("models", []),
        key=lambda m: (m.get("parameter_count_b", float("inf")), m["name"].lower()),
    )
    models = [m["name"] for m in models_cfg]
    langs = [l["code"] for l in cfg.get("languages", [])]
    exps = list(cfg.get("experiments", {}).keys())
    n_samples = cfg.get("dataset", {}).get("num_samples", 200)
    n_repeats = cfg.get("inference", {}).get("repeat", 5)

    run_dir = runs_dir or _latest_run_dir() or RAW_ROOT
    if not run_dir.is_absolute():
        run_dir = (ROOT / run_dir).resolve()
    run_meta = _load_run_meta(run_dir)
    scan = _scan_done(run_dir, n_repeats=n_repeats)
    model_summaries = _per_model_summary(run_dir)

    pid = _running(ROOT / ".run.pid")
    log_lines = _log_tail(ROOT / "logs" / "run.log", n=8)
    cur = _current_cell(log_lines)
    start_ts = _start_time_from_log(ROOT / "logs" / "run.log")

    total_cells = len(models) * len(langs) * len(exps)
    full_cells = len(scan["cells_full"])
    partial_cells = len(scan["cells_partial"])
    running = pid is not None
    eta = _eta_seconds(full_cells, total_cells, start_ts, running)

    loaded = list_loaded_models()
    loaded_names = [m.get("name") for m in loaded]

    breakdown: dict[str, Counter] = defaultdict(Counter)
    full_breakdown: dict[str, set] = defaultdict(set)
    for (exp, model, lang), n in scan["cells_partial"].items():
        breakdown[model][(exp, lang)] = n
    for (exp, model, lang) in scan["cells_full"]:
        full_breakdown[model].add((exp, lang))

    if plain:
        lines = [
            f"running_pid={pid if pid else 'none'}",
            f"runs_dir={run_dir.relative_to(ROOT)}",
            f"n_files={scan['n_files']}",
            f"cells_full={full_cells}/{total_cells}",
            f"cells_partial={partial_cells}/{total_cells}",
            f"current={cur}",
            f"loaded_models={','.join(n for n in loaded_names if n)}",
            f"phase={run_meta.get('phase', '?')}",
            f"models={len(models)}",
            f"languages={len(langs)}",
            f"experiments={len(exps)}",
            f"eta={_fmt_eta(eta)}",
        ]
        print("\n".join(lines))
        return {
            "running": pid is not None,
            "n_files": scan["n_files"],
            "cells_full": full_cells,
            "cells_total": total_cells,
            "current": cur,
            "eta": eta,
            "loaded_models": loaded_names,
            "phase": run_meta.get("phase"),
        }

    W = 80
    lines = []
    lines.append("=" * W)
    lines.append(
        f"  Token/s vs Bit/s — orchestrated v2 dashboard"
        f"     {'RUNNING' if pid else 'STOPPED'}"
    )
    lines.append("=" * W)
    status_pid = f"PID {pid}" if pid else "not running"
    lines.append(f"  {status_pid}    refresh with --interval N    Ctrl-C to quit")
    lines.append(f"  Run dir: {run_dir.relative_to(ROOT)}")
    lines.append(
        f"  Phase:   {run_meta.get('phase', '?')}    "
        f"Ollama loaded: {','.join(n for n in loaded_names if n) or '(none)'}"
    )
    lines.append("")
    lines.append(
        f"  Matrix: {len(models)} models × {len(langs)} languages × "
        f"{len(exps)} experiments × {n_samples} samples × {n_repeats} repeats"
    )
    lines.append(f"  Total cells: {total_cells}")
    lines.append("")
    lines.append("  Progress (full cells): " + _bar(full_cells, total_cells, width=40))
    lines.append(f"  Files on disk:        {scan['n_files']}")
    lines.append(f"  Cells with ≥1 file:   {partial_cells} / {total_cells}")
    lines.append(f"  Cells fully done:     {full_cells} / {total_cells}")
    lines.append(f"  ETA:                  {_fmt_eta(eta)}")
    lines.append("")
    lines.append(f"  Current: {cur}")
    lines.append("")
    lines.append("  Per-model status (full=lang×exp cells with all repeats):")
    lines.append("    " + "-" * (W - 4))
    lines.append(f"    {'model':36s}  full  partial  files  peak_VRAM  unload")
    for m in models:
        cells_m = breakdown.get(m, Counter())
        n_partial = len(cells_m)
        n_files_m = sum(cells_m.values())
        n_full = len(full_breakdown.get(m, set()))
        ms = model_summaries.get(m, {})
        peak = ms.get("peak_vram_mib", 0.0)
        unload_ok = ms.get("unload_succeeded")
        unload_marker = (
            "✓" if unload_ok is True
            else ("✗" if unload_ok is False else "—")
        )
        lines.append(
            f"    {m:36s}  {n_full:4d}  {n_partial:7d}  {n_files_m:5d}  "
            f"{peak:8.0f}MiB  {unload_marker}"
        )
    lines.append("")
    lines.append("  Recent log (last 5 lines):")
    lines.append("    " + "-" * (W - 4))
    for line in log_lines[-5:]:
        if len(line) > W - 4:
            line = line[: W - 7] + "..."
        lines.append(f"  {line}")
    lines.append("=" * W)
    print("\n".join(lines))

    return {
        "running": pid is not None,
        "n_files": scan["n_files"],
        "cells_full": full_cells,
        "cells_total": total_cells,
        "current": cur,
        "eta": eta,
        "loaded_models": loaded_names,
    }


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--interval", type=float, default=2.0,
                    help="refresh interval in seconds (default 2)")
    ap.add_argument("--once", action="store_true",
                    help="print dashboard once and exit")
    ap.add_argument("--plain", action="store_true",
                    help="print key=value format (for grep / scripts)")
    ap.add_argument("--runs-dir", type=Path, default=None,
                    help="watch a specific results/raw/new-* directory "
                         "instead of auto-detecting the latest one")
    args = ap.parse_args()

    if args.once:
        render_dashboard(plain=args.plain, runs_dir=args.runs_dir)
        return 0

    try:
        while True:
            sys.stdout.write("\x1b[H\x1b[2J")
            sys.stdout.flush()
            render_dashboard(plain=args.plain, runs_dir=args.runs_dir)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[watch] exiting")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())