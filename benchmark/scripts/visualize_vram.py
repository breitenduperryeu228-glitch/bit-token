"""
visualize_vram.py — Plot GPU VRAM occupancy over the lifetime of a run.

Reads the `host.pre_call` / `host.post_call` nvidia-smi snapshots embedded
in every raw JSON produced by the v2 orchestrator, and draws:

  * VRAM occupancy (used / total, %) over wall-clock time.
  * Each model's segment is shaded / colored so the load→unload sawtooth
    is visible (one model occupies the GPU at a time; a clean run shows
    the baseline dip between models).

Usage:
  python3 -m scripts.visualize_vram --runs-dir results/raw/new-20260906-202656
  python3 -m scripts.visualize_vram --runs-dir results/raw/new-XXXX --out figures/vram.png
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def extract_vram_series(runs_dir: Path) -> dict:
    """Return a dict keyed by model -> list of (timestamp, vram_used_mib,
    vram_total_mib). One point per pre_call and per post_call snapshot."""
    series: dict = defaultdict(list)
    for path in runs_dir.rglob("*.json"):
        if path.name == "run_meta.json" or "model_summary" in path.parts:
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        host = raw.get("host") or {}
        model = raw.get("model") or "unknown"
        for phase in ("pre_call", "post_call"):
            block = (host.get(phase) or {}).get("nvidia_smi") or {}
            gpus = block.get("gpus") or []
            ts = block.get("timestamp_unix") or raw.get("wall_clock_unix")
            if not gpus or ts is None:
                continue
            g = gpus[0]
            used = g.get("vram_used_mib")
            total = g.get("vram_total_mib") or 8192.0
            if used is None:
                continue
            series[model].append((ts, used, total))
    for model in series:
        series[model].sort(key=lambda p: p[0])
    return series


def plot(series: dict, out_path: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # baseline = first observed total (all models share one GPU)
    total_mib = None
    for pts in series.values():
        for _, _, t in pts:
            total_mib = t
            break
        if total_mib:
            break
    total_mib = total_mib or 8192.0

    # normalise timestamps to minutes since run start
    t0 = None
    for pts in series.values():
        if pts:
            t0 = pts[0][0]
            break
    if t0 is None:
        raise RuntimeError("no VRAM snapshots found")

    colors = plt.cm.tab10.colors
    for i, (model, pts) in enumerate(sorted(series.items())):
        xs = [(p[0] - t0) / 60.0 for p in pts]
        ys = [p[1] / total_mib * 100.0 for p in pts]
        color = colors[i % len(colors)]
        ax1.plot(xs, ys, lw=0.6, color=color, label=model, alpha=0.9)

        # shade the time window this model was active
        if len(pts) >= 2:
            x_start = (pts[0][0] - t0) / 60.0
            x_end = (pts[-1][0] - t0) / 60.0
            ax1.axvspan(
                x_start, x_end, color=color, alpha=0.08,
            )

    ax1.set_ylabel("VRAM occupancy (%)")
    ax1.set_title(
        f"GPU VRAM occupancy over time — {len(series)} models, "
        f"total VRAM {total_mib:.0f} MiB"
    )
    ax1.legend(loc="upper right", fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)

    # second axis: absolute MiB
    ax2.set_ylabel("VRAM used (MiB)")
    ax2.set_xlabel("Time since run start (minutes)")
    for i, (model, pts) in enumerate(sorted(series.items())):
        xs = [(p[0] - t0) / 60.0 for p in pts]
        ys = [p[1] for p in pts]
        ax2.plot(xs, ys, lw=0.5, color=colors[i % len(colors)], alpha=0.8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="",
                    help="results/raw/new-XXX folder (default: latest new-*)")
    ap.add_argument("--out", default="",
                    help="output PNG path (default: figures/vram_curve.png)")
    args = ap.parse_args()

    root = _project_root()
    if args.runs_dir:
        runs_dir = Path(args.runs_dir)
        if not runs_dir.is_absolute():
            runs_dir = root / runs_dir
    else:
        candidates = sorted(
            (root / "results" / "raw").glob("new-*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        runs_dir = candidates[0] if candidates else (root / "results" / "raw")

    series = extract_vram_series(runs_dir)
    if not series:
        print("[vram] no VRAM snapshots found")
        return 1

    out_path = Path(args.out) if args.out else (root / "figures" / "vram_curve.png")
    if not out_path.is_absolute():
        out_path = root / out_path
    plot(series, out_path)

    n_points = sum(len(v) for v in series.values())
    print(f"[vram] {n_points} points across {len(series)} models")
    print(f"[vram] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())