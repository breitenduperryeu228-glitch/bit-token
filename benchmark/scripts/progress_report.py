#!/usr/bin/env python3
"""progress_report.py — detailed markdown progress report for the v2 benchmark.

Outputs a human-readable report with:
  * overall progress (cells done / total)
  * per-model × per-experiment breakdown
  * session elapsed time, live generation rate
  * ETA for the current model and the whole run
  * host telemetry (GPU / VRAM / temp) and process status

Usage:
  python3 -m scripts.progress_report
  python3 -m scripts.progress_report --runs-dir results/raw/new-20260906-202656
  python3 -m scripts.progress_report --json     # machine-readable
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = "results/raw/new-20260906-202656"
LOG = ROOT / "logs" / "run.log"
PIDFILE = ROOT / ".run.pid"

EXP_A = "A_fixed_tokens"
EXP_B = "B_fixed_semantic_length"
EXPS = [EXP_A, EXP_B]


def sanitize(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")


def fmt_dur(sec: float | None) -> str:
    if sec is None or sec < 0:
        return "—"
    sec = int(sec)
    d, r = divmod(sec, 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    parts = []
    if d:
        parts.append(f"{d}d")
    if h or d:
        parts.append(f"{h}h")
    if m or h or d:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)


def fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def bar(done: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + " " * width + "]"
    pct = min(1.0, done / total)
    fill = int(pct * width)
    return "`[" + "#" * fill + "-" * (width - fill) + f"] {pct * 100:5.1f}%`"


def load_cfg() -> dict:
    with (ROOT / "configs" / "config.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def parse_models(cfg: dict) -> list[str]:
    ms = sorted(
        cfg.get("models", []),
        key=lambda m: (m.get("parameter_count_b", 9e9), m["name"].lower()),
    )
    return [m["name"] for m in ms]


def scan(run_dir: Path) -> dict:
    """Count raw files per (exp, sanitized-model) and collect mtimes."""
    out: dict[str, dict[str, int]] = {e: {} for e in EXPS}
    mtimes: list[float] = []
    for exp in EXPS:
        d = run_dir / exp
        if not d.is_dir():
            continue
        counts: dict[str, int] = {}
        with os.scandir(d) as it:
            for entry in it:
                if not entry.name.endswith(".json"):
                    continue
                if entry.name == "run_meta.json":
                    continue
                prefix = entry.name.split("__", 1)[0]
                counts[prefix] = counts.get(prefix, 0) + 1
                try:
                    mtimes.append(entry.stat().st_mtime)
                except OSError:
                    pass
        out[exp] = counts
    return {"counts": out, "mtimes": mtimes}


START_RE = re.compile(r"^\[(\d\d):(\d\d):(\d\d)\]\s+>>>\s+(.+?)\s+×\s+(\S+)\s+START")
DONE_RE = re.compile(r"^\[(\d\d):(\d\d):(\d\d)\]\s+>>>\s+(.+?)\s+×\s+(\S+)\s+DONE")
SESSION_RE = re.compile(r"^\[(\d\d):(\d\d):(\d\d)\]\s+===\s+run_benchmark start")


def _tod_seconds(h: str, m: str, s: str) -> int:
    return int(h) * 3600 + int(m) * 60 + int(s)


def parse_log(log_path: Path) -> dict:
    """Return session start (unix) + historical (model,exp) durations."""
    if not log_path.exists():
        return {"session_start": None, "durations": {}}
    base_date = datetime.fromtimestamp(log_path.stat().st_mtime).date()
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()

    session_tod = None
    open_start: dict[tuple[str, str], int] = {}
    durations: dict[tuple[str, str], float] = {}
    prev_tod = None

    for ln in lines:
        m = SESSION_RE.match(ln)
        if m:
            session_tod = _tod_seconds(*m.groups())
            continue
        ms = START_RE.match(ln)
        if ms:
            h, mi, s, model, exp = ms.groups()
            tod = _tod_seconds(h, mi, s)
            open_start[(model, exp)] = tod
            prev_tod = tod
            continue
        md = DONE_RE.match(ln)
        if md:
            h, mi, s, model, exp = md.groups()
            tod = _tod_seconds(h, mi, s)
            st = open_start.pop((model, exp), None)
            if st is not None:
                dur = tod - st
                if dur < 0:  # crossed midnight
                    dur += 86400
                # Keep the FIRST completed duration: later sessions only
                # re-visit already-done models and finish in seconds (skips),
                # which would clobber the real historical timing.
                key = (model, exp)
                if key not in durations:
                    durations[key] = float(dur)
            prev_tod = tod

    session_start = None
    if session_tod is not None:
        session_start = datetime.combine(base_date, datetime.min.time()).timestamp() + session_tod
        # if the log's last mtime is on the next day, adjust
        if session_start > log_path.stat().st_mtime + 60:
            session_start -= 86400
    return {"session_start": session_start, "durations": durations}


def host_telemetry() -> dict:
    info = {"gpu_util": None, "vram_used": None, "vram_total": None, "temp": None}
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()
        if out:
            p = [x.strip() for x in out[0].split(",")]
            info.update(gpu_util=p[0], vram_used=p[1], vram_total=p[2], temp=p[3])
    except Exception:
        pass
    return info


def pid_running() -> int | None:
    if not PIDFILE.exists():
        return None
    try:
        pid = int(PIDFILE.read_text().strip())
    except ValueError:
        return None
    try:
        os.kill(pid, 0)
    except OSError:
        return None
    return pid


def build_report(run_dir: Path) -> dict:
    cfg = load_cfg()
    models = parse_models(cfg)
    n_repeats = cfg.get("inference", {}).get("repeat", 5)
    n_samples = cfg.get("dataset", {}).get("num_samples", 200)
    n_langs = len(cfg.get("languages", []))
    per_model_exp = n_samples * n_langs * n_repeats
    total = len(models) * len(EXPS) * per_model_exp

    sc = scan(run_dir)
    log = parse_log(LOG)
    now = time.time()
    session_start = log["session_start"]

    # session-produced cells = files written after session start
    if session_start:
        session_cells = sum(1 for t in sc["mtimes"] if t >= session_start)
        session_elapsed = now - session_start
    else:
        session_cells = 0
        session_elapsed = 0.0

    # live rate from files written in the last 15 minutes
    window = 15 * 60
    recent = sum(1 for t in sc["mtimes"] if t >= now - window)
    live_rate_per_min = recent / (window / 60)

    rows = []
    done_total = 0
    for model in models:
        pref = sanitize(model)
        a = sc["counts"][EXP_A].get(pref, 0)
        b = sc["counts"][EXP_B].get(pref, 0)
        done_total += a + b
        rows.append({"model": model, "a": a, "b": b, "target": per_model_exp})

    # current position: newest file for the current model tells us the language
    current_lang = None
    current_lang_done = 0
    current_lang_target = n_samples * n_repeats

    # current model/exp = last START without DONE
    current = None
    if LOG.exists():
        lines = LOG.read_text(encoding="utf-8", errors="replace").splitlines()
        started = None
        done_pairs = set()
        for ln in lines:
            ms = START_RE.match(ln)
            if ms:
                started = (ms.group(4), ms.group(5))
                continue
            md = DONE_RE.match(ln)
            if md:
                done_pairs.add((md.group(4), md.group(5)))
        if started and started not in done_pairs:
            current = started
    if current is None:
        for model in models:
            if sc["counts"][EXP_B].get(sanitize(model), 0) < per_model_exp:
                current = (model, EXP_B)
                break

    # remaining work
    remaining = total - done_total
    current_remaining = 0
    if current:
        pref = sanitize(current[0])
        cur_done = sc["counts"][current[1]].get(pref, 0)
        current_remaining = max(0, per_model_exp - cur_done)
        # find newest file for current model -> language + per-language count
        d = run_dir / current[1]
        if d.is_dir():
            newest_mtime = -1.0
            newest_name = None
            lang_counts: dict[str, int] = {}
            with os.scandir(d) as it:
                for entry in it:
                    if not entry.name.endswith(".json"):
                        continue
                    if not entry.name.startswith(pref + "__"):
                        continue
                    bits = entry.name.split("__")
                    if len(bits) >= 4:
                        lang_counts[bits[1]] = lang_counts.get(bits[1], 0) + 1
                    try:
                        mt = entry.stat().st_mtime
                    except OSError:
                        continue
                    if mt > newest_mtime:
                        newest_mtime = mt
                        newest_name = entry.name
            if newest_name:
                bits = newest_name.split("__")
                if len(bits) >= 4:
                    current_lang = bits[1]
                    current_lang_done = lang_counts.get(current_lang, 0)

    eta_current = None
    if current and live_rate_per_min > 0:
        eta_current = current_remaining / live_rate_per_min * 60

    # crude whole-run ETA: current model at live rate + future models at
    # their historical A-phase per-cell time (B often faster; estimate only)
    future_secs = 0.0
    for model in models:
        pref = sanitize(model)
        b_done = sc["counts"][EXP_B].get(pref, 0)
        if b_done >= per_model_exp:
            continue
        if current and model == current[0]:
            continue
        dur = log["durations"].get((model, EXP_A))
        if dur:
            future_secs += dur * (per_model_exp - b_done) / per_model_exp
    eta_total = None
    if eta_current is not None:
        eta_total = eta_current + future_secs

    return {
        "run_dir": str(run_dir),
        "now": now,
        "session_start": session_start,
        "session_elapsed": session_elapsed,
        "session_cells": session_cells,
        "live_rate_per_min": live_rate_per_min,
        "rows": rows,
        "done_total": done_total,
        "total": total,
        "remaining": remaining,
        "current": current,
        "current_lang": current_lang,
        "current_lang_done": current_lang_done,
        "current_lang_target": current_lang_target,
        "current_remaining": current_remaining,
        "eta_current": eta_current,
        "eta_total": eta_total,
        "per_model_exp": per_model_exp,
        "durations": {f"{k[0]}__{k[1]}": v for k, v in log["durations"].items()},
        "pid": pid_running(),
        "host": host_telemetry(),
    }


def render_markdown(r: dict) -> str:
    L = []
    pct = r["done_total"] / r["total"] * 100 if r["total"] else 0
    L.append(f"# 基准测试进度报告")
    L.append("")
    L.append(f"- 报告时间：`{fmt_ts(r['now'])}`")
    status = f"运行中 (PID {r['pid']})" if r["pid"] else "**未运行**"
    L.append(f"- 运行状态：{status}")
    L.append(f"- 数据目录：`{r['run_dir']}`")
    L.append("")
    L.append("## 总进度")
    L.append("")
    L.append(f"**{r['done_total']:,} / {r['total']:,} cell（{pct:.1f}%）**")
    L.append("")
    L.append(bar(r["done_total"], r["total"]))
    L.append("")
    L.append("## 分模型 × 实验明细")
    L.append("")
    L.append("| 模型 | A_fixed_tokens | B_fixed_semantic | 小计 / 目标 | 状态 |")
    L.append("|---|---:|---:|---:|---|")
    order = [row["model"] for row in r["rows"]]
    for row in r["rows"]:
        a, b, tgt = row["a"], row["b"], row["target"]
        sub = a + b
        total_tgt = 2 * tgt
        if a >= tgt and b >= tgt:
            st = "✅ 完成"
        elif b >= tgt:
            st = "✅ B完成"
        elif a >= tgt and b > 0:
            st = "🔄 B进行中"
        elif a >= tgt:
            st = "⏳ 待B"
        else:
            st = "⏳ 待A"
        if r["current"] and row["model"] == r["current"][0]:
            st = "🔄 **当前**"
        L.append(f"| {row['model']} | {a:,} / {tgt:,} | {b:,} / {tgt:,} | {sub:,} / {total_tgt:,} | {st} |")
    L.append("")
    L.append("## 时间与速率")
    L.append("")
    L.append("| 指标 | 数值 |")
    L.append("|---|---|")
    L.append(f"| 本次会话开始 | {fmt_ts(r['session_start']) if r['session_start'] else '—'} |")
    L.append(f"| 本次会话已运行 | {fmt_dur(r['session_elapsed'])} |")
    L.append(f"| 本次会话新增 cell | {r['session_cells']:,} |")
    L.append(f"| 近 15 分钟生成速率 | {r['live_rate_per_min']:.2f} cell/分钟 |")
    if r["current"]:
        m, e = r["current"]
        L.append(f"| 当前模型 | {m} × {e} |")
        if r.get("current_lang"):
            L.append(
                f"| 当前语言 | {r['current_lang']} "
                f"（{r['current_lang_done']:,} / {r['current_lang_target']:,}） |"
            )
        L.append(f"| 当前模型剩余 | {r['current_remaining']:,} cell |")
        L.append(f"| 当前模型预计剩余 | {fmt_dur(r['eta_current'])} |")
    L.append(f"| 全部剩余 cell | {r['remaining']:,} |")
    L.append(f"| 全部预计剩余 | {fmt_dur(r['eta_total'])} |")
    L.append("")
    if r.get("durations"):
        L.append("## 历史阶段耗时（已完成模型，实测）")
        L.append("")
        L.append("| 模型 | A_fixed_tokens | B_fixed_semantic |")
        L.append("|---|---:|---:|")
        for row in r["rows"]:
            m = row["model"]
            a = r["durations"].get(f"{m}__{EXP_A}")
            b = r["durations"].get(f"{m}__{EXP_B}")
            L.append(f"| {m} | {fmt_dur(a)} | {fmt_dur(b)} |")
        L.append("")
    host = r["host"]
    if host.get("gpu_util") is not None:
        L.append("## 主机状态")
        L.append("")
        L.append("| GPU 利用率 | 显存 | 温度 |")
        L.append("|---|---|---|")
        L.append(
            f"| {host['gpu_util']}% | {host['vram_used']} / {host['vram_total']} MiB | {host['temp']}°C |"
        )
        L.append("")
    return "\n".join(L)


def latest_run_dir() -> Path:
    """Newest results/raw/new-* directory that actually contains data."""
    raw = ROOT / "results" / "raw"
    if not raw.is_dir():
        return ROOT / DEFAULT_RUN
    cands = sorted(
        (p for p in raw.iterdir() if p.is_dir() and p.name.startswith("new-")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for p in cands:
        if any(p.rglob("*.json")):
            return p
    return (ROOT / DEFAULT_RUN) if (ROOT / DEFAULT_RUN).exists() else raw


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", default=None,
                    help="results/raw/new-* dir (default: auto-detect newest)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    if args.runs_dir:
        run_dir = Path(args.runs_dir)
        if not run_dir.is_absolute():
            run_dir = (ROOT / run_dir).resolve()
    else:
        run_dir = latest_run_dir()
    r = build_report(run_dir)
    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=2, default=str))
    else:
        print(render_markdown(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
