"""
visualize_thinking_bias.py — Generate the dedicated thinking-vs-response
comparison figure (fig8_thinking_bias.png).

Three-panel figure showing, for every model in the matrix:

  Panel A (left, wide):
      Stacked horizontal bar chart. For each (model × language) cell, the
      bar's full length is `eval_count` and the colored segment shows the
      share attributable to response tokens (estimated via BPT heuristic);
      the gray remainder is thinking tokens. lfm2.5-thinking is highlighted
      with a red border.

  Panel B (middle):
      Scatter of Token/s vs Bit/s for each model, with model labels.
      Visualizes why lfm-thinking ranks #1 on Token/s but #last on Bit/s.

  Panel C (right):
      Per-language hit-length-cap rate (%) for lfm2.5-thinking,
      showing the catastrophic failure mode in English.

Output: figures/fig8_thinking_bias.png
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


BPT_RESPONSE_HEURISTIC = 30.0  # bits per token in the response segment


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_thinking_bias(runs_dir: Path, experiment: str) -> dict:
    suffix = "" if experiment == "B_fixed_semantic_length" else "_A"
    p = runs_dir / "derived" / f"thinking_bias{suffix}.json"
    return json.loads(p.read_text(encoding="utf-8"))


def load_per_model_perf(runs_dir: Path) -> Dict[str, Dict[str, float]]:
    """Mean Token/s and Bit/s per model from the derived all_runs.csv."""
    p = runs_dir / "derived" / "all_runs.csv"
    by_model: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    with p.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["experiment"] != "B_fixed_semantic_length":
                continue
            by_model[row["model"]]["token_per_second"].append(float(row["token_per_second"]))
            by_model[row["model"]]["bit_per_second"].append(float(row["bit_per_second"]))
    return {
        m: {k: statistics.mean(v) for k, v in d.items()}
        for m, d in by_model.items()
    }


def panel_a(ax, bias: dict) -> None:
    """Stacked bar: thinking (gray) vs response (colored) tokens per language."""
    per_lang = bias["per_language"]
    langs = list(per_lang.keys())
    langs_sorted = sorted(
        langs,
        key=lambda l: per_lang[l]["hit_length_cap_pct"],
        reverse=True,
    )
    avg_eval = [per_lang[l]["avg_eval_count"] for l in langs_sorted]
    resp_bytes = [per_lang[l]["avg_response_bytes"] for l in langs_sorted]
    resp_tok_est = [b * 8 / BPT_RESPONSE_HEURISTIC for b in resp_bytes]
    think_tok_est = [max(0.0, e - r) for e, r in zip(avg_eval, resp_tok_est)]

    y = np.arange(len(langs_sorted))
    ax.barh(y, think_tok_est, color="#bbbbbb", edgecolor="white", label="thinking tokens (est.)")
    ax.barh(y, resp_tok_est, left=think_tok_est, color="#dd4444",
            edgecolor="white", label="response tokens (est.)")
    for i, (l, e, h) in enumerate(zip(langs_sorted, avg_eval, [per_lang[l]["hit_length_cap_pct"] for l in langs_sorted])):
        ax.text(e + 80, i, f"{e:.0f} tot\n{h:.0f}% cap",
                va="center", fontsize=8, color="#333333")
    ax.set_yticks(y)
    ax.set_yticklabels(langs_sorted, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("eval_count tokens (thinking + response)")
    ax.set_title("A. lfm2.5-thinking × B — thinking 占总 token 的份额", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)


def panel_b(ax, perf: Dict[str, Dict[str, float]]) -> None:
    """Scatter: Token/s vs Bit/s for every model, labels visible."""
    colors = {
        "lfm2.5-thinking:latest": "#dd4444",
    }
    default_color = "#3366aa"
    for m, v in perf.items():
        c = colors.get(m, default_color)
        ax.scatter(v["token_per_second"], v["bit_per_second"],
                   s=120, c=c, edgecolors="black", linewidths=0.7, alpha=0.85, zorder=3)
        # short label: strip namespace and version
        short = m.split("/")[-1].replace(":latest", "").replace(":0.6b", "-0.6b").replace(":0.8b", "-0.8b").replace(":1b", "-1b").replace(":350m-h", "-350m")
        ax.annotate(short,
                    (v["token_per_second"], v["bit_per_second"]),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=8.5, zorder=4)
    ax.set_xlabel("Token/s (overall, includes thinking)")
    ax.set_ylabel("Bit/s  (response text bits / eval_duration)")
    ax.set_title("B. Token/s vs Bit/s 排名差异", fontsize=10)
    ax.grid(alpha=0.3)


def panel_c(ax, bias: dict) -> None:
    """Per-language hit-length-cap rate for lfm2.5-thinking."""
    per_lang = bias["per_language"]
    langs = sorted(per_lang.keys(),
                   key=lambda l: per_lang[l]["hit_length_cap_pct"],
                   reverse=True)
    cap_pct = [per_lang[l]["hit_length_cap_pct"] for l in langs]
    colors = ["#dd4444" if v > 50 else "#ee9966" if v > 20 else "#88bb66" for v in cap_pct]
    y = np.arange(len(langs))
    ax.barh(y, cap_pct, color=colors, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(langs, fontsize=9)
    ax.invert_yaxis()
    for i, v in enumerate(cap_pct):
        ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=8)
    ax.set_xlim(0, 105)
    ax.set_xlabel("hit length-cap  rate (%)")
    ax.set_title("C. lfm2.5-thinking 撞长度上限比例（按语种）", fontsize=10)
    ax.grid(axis="x", alpha=0.3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="", help="results/raw/new-XXX folder")
    ap.add_argument("--experiment", default="B_fixed_semantic_length",
                    choices=["A_fixed_tokens", "B_fixed_semantic_length"],
                    help="which experiment's thinking_bias to plot (default: B_fixed_semantic_length)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = _project_root()
    runs_dir = Path(args.runs_dir) if args.runs_dir else None
    if not runs_dir:
        candidates = sorted((root / "results" / "raw").glob("new-*"))
        if not candidates:
            raise SystemExit("ERROR: no new-* directory under results/raw/")
        runs_dir = candidates[-1]
    runs_dir = runs_dir if runs_dir.is_absolute() else (root / runs_dir)

    suffix = "" if args.experiment == "B_fixed_semantic_length" else "_A"
    out_path = root / (args.out or f"figures/fig8_thinking_bias_{('B' if suffix == '' else 'A')}.png")

    bias = load_thinking_bias(runs_dir, args.experiment)
    perf = load_per_model_perf(runs_dir)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    panel_a(axes[0], bias)
    panel_b(axes[1], perf)
    panel_c(axes[2], bias)

    fig.suptitle(
        f"lfm2.5-thinking 的 thinking-trace 偏差（{args.experiment}）：Token/s 高估，Bit/s 鲁棒",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[thinking-bias-viz] wrote {out_path}")


if __name__ == "__main__":
    main()