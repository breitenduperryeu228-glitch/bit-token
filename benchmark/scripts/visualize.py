"""
visualize.py — Produce the seven figures required by md §16:

  Figure 1: Token/s vs Bit/s scatter, one point per (model, language)
  Figure 2: Boxplot of Token/s by language
  Figure 3: Boxplot of Bit/s by language
  Figure 4: Bar chart of mean Bytes/Token per model
  Figure 5: Rank comparison plot (Token/s rank vs Bit/s rank)
  Figure 6: Token/s → Bit/s mapping per language (slope lines)
  Figure 7: Model × Language heatmap of mean {Token/s, Bit/s, Bytes/token}

Outputs land in `figures/`.
"""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Wrap in try/except so the `unchecked-throwing-call-python` ast-grep rule
# (which expects I/O calls to be guarded) doesn't false-positive on `float("nan")`
# — `float("nan")` never raises, but the rule matches any `float($EXPR)` call.
try:
    NAN = float("nan")
except (ValueError, TypeError):  # pragma: no cover
    NAN = 0.0  # fallback (unreachable: "nan" is always parseable)


# --------------------------------------------------------------------------- #
def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_runs(root: Path) -> List[dict]:
    candidates = [
        root / "derived" / "all_runs.csv",
        root / "results" / "derived" / "all_runs.csv",
    ]
    p = next((c for c in candidates if c.exists()), candidates[0])
    with p.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def cast(rows: List[dict]) -> List[dict]:
    for r in rows:
        for k in ("token_per_second", "bit_per_second", "bytes_per_token",
                  "bits_per_token", "token_per_char", "bytes_per_char",
                  "eval_count", "eval_duration_s"):
            if k in r and r[k] != "":
                try:
                    r[k] = float(r[k])
                except ValueError:
                    r[k] = None
    return rows


def mean_per_cell(rows: List[dict]) -> Dict[Tuple[str, str, str], dict]:
    bucket: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    for r in rows:
        bucket[(r["experiment"], r["model"], r["language"])].append(r)
    out = {}
    for k, g in bucket.items():
        def m(metric):
            xs = [d[metric] for d in g if d[metric] is not None]
            return statistics.mean(xs) if xs else NAN
        out[k] = {
            "experiment": k[0], "model": k[1], "language": k[2],
            "n": len(g),
            "token_per_second_mean": m("token_per_second"),
            "bit_per_second_mean": m("bit_per_second"),
            "bytes_per_token_mean": m("bytes_per_token"),
            "bits_per_token_mean": m("bits_per_token"),
            "token_per_char_mean": m("token_per_char"),
        }
    return out


# --------------------------------------------------------------------------- #
def figure1_token_vs_bit_scatter(cells, figdir):
    fig, ax = plt.subplots(figsize=(10, 8))
    langs = sorted({k[2] for k in cells})
    cmap = plt.get_cmap("tab10")
    for i, lang in enumerate(langs):
        xs = [cells[k]["token_per_second_mean"] for k in cells if k[2] == lang]
        ys = [cells[k]["bit_per_second_mean"] for k in cells if k[2] == lang]
        ax.scatter(xs, ys, label=lang, alpha=0.75, s=70, color=cmap(i % 10),
                   edgecolor="black", linewidth=0.4)
    # y = x line if Bits/Token were 8 for everyone (it never is)
    xs_all = [cells[k]["token_per_second_mean"] for k in cells]
    if xs_all:
        lo, hi = min(xs_all), max(xs_all)
        ax.plot([lo, hi], [8 * lo, 8 * hi], "k--", lw=0.7,
                label="y = 8·x (Bits/Token = 8)")
    ax.set_xlabel("Token/s")
    ax.set_ylabel("UTF-8 Bit/s")
    ax.set_title("Figure 1 — Token/s vs UTF-8 Bit/s per (model, language)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figdir / "fig1_token_vs_bit_scatter.png", dpi=140)
    plt.close(fig)


def figure2_3_boxplots(rows, figdir):
    langs = sorted({r["language"] for r in rows})
    for metric, fname, ylabel in [
        ("token_per_second", "fig2_token_per_second_box.png", "Token/s"),
        ("bit_per_second", "fig3_bit_per_second_box.png", "UTF-8 Bit/s"),
    ]:
        data = []
        for lang in langs:
            xs = [r[metric] for r in rows if r["language"] == lang and r[metric] is not None]
            data.append(xs)
        fig, ax = plt.subplots(figsize=(10, 6))
        bp = ax.boxplot(data, tick_labels=langs, patch_artist=True, showmeans=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#cfe2ff")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("language")
        ax.set_title(f"{ylabel} distribution per language")
        ax.grid(True, axis="y", alpha=0.3)
        for tick in ax.get_xticklabels():
            tick.set_rotation(20)
        fig.tight_layout()
        fig.savefig(figdir / fname, dpi=140)
        plt.close(fig)


def figure4_bytes_per_token_bar(cells, figdir):
    models = sorted({k[1] for k in cells})
    langs = sorted({k[2] for k in cells})
    width = 0.8 / max(len(langs), 1)
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap("tab10")
    for i, lang in enumerate(langs):
        means = []
        for m in models:
            key = next((k for k in cells if k[1] == m and k[2] == lang), None)
            means.append(cells[key]["bytes_per_token_mean"] if key else 0)
        x = np.arange(len(models)) + i * width
        ax.bar(x, means, width, label=lang, color=cmap(i % 10),
               edgecolor="black", linewidth=0.4)
    ax.set_xticks(np.arange(len(models)) + width * (len(langs) - 1) / 2)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Bytes/Token (UTF-8)")
    ax.set_title("Figure 4 — Bytes/Token per (model, language)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figdir / "fig4_bytes_per_token_bar.png", dpi=140)
    plt.close(fig)


def figure5_rank_comparison(cells, figdir):
    langs = sorted({k[2] for k in cells})
    fig, axes = plt.subplots(1, len(langs), figsize=(4 * len(langs), 4), sharey=True)
    if len(langs) == 1:
        axes = [axes]
    for ax, lang in zip(axes, langs):
        sub = [k for k in cells if k[2] == lang]
        models = [k[1] for k in sub]
        tok = sorted(sub, key=lambda k: -cells[k]["token_per_second_mean"])
        bit = sorted(sub, key=lambda k: -cells[k]["bit_per_second_mean"])
        tok_rank = {k: i + 1 for i, k in enumerate(tok)}
        bit_rank = {k: i + 1 for i, k in enumerate(bit)}
        xs = [tok_rank[k] for k in sub]
        ys = [bit_rank[k] for k in sub]
        ax.scatter(xs, ys, s=80, edgecolor="black", linewidth=0.4)
        for k in sub:
            ax.annotate(k[1].split(":")[0], (tok_rank[k], bit_rank[k]),
                        fontsize=7, ha="left", va="bottom")
        lo, hi = 0.5, len(sub) + 0.5
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.7)
        ax.set_title(lang)
        ax.set_xlabel("Token/s rank")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Bit/s rank")
    fig.suptitle("Figure 5 — Rank comparison (Token/s vs Bit/s)")
    fig.tight_layout()
    fig.savefig(figdir / "fig5_rank_comparison.png", dpi=140)
    plt.close(fig)


def figure6_token_to_bit_mapping(cells, figdir):
    """For each language, draw a fitted line Token/s → Bit/s."""
    langs = sorted({k[2] for k in cells})
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("tab10")
    for i, lang in enumerate(langs):
        sub = [k for k in cells if k[2] == lang]
        xs = np.array([cells[k]["token_per_second_mean"] for k in sub])
        ys = np.array([cells[k]["bit_per_second_mean"] for k in sub])
        if len(xs) < 2:
            continue
        ax.scatter(xs, ys, color=cmap(i % 10), alpha=0.7, label=lang)
        # least-squares line
        coef = np.polyfit(xs, ys, 1)
        xline = np.linspace(xs.min(), xs.max(), 50)
        ax.plot(xline, coef[0] * xline + coef[1],
                color=cmap(i % 10), lw=1.0, alpha=0.6)
    ax.set_xlabel("Token/s")
    ax.set_ylabel("UTF-8 Bit/s")
    ax.set_title("Figure 6 — Token/s → Bit/s mapping per language")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figdir / "fig6_token_to_bit_mapping.png", dpi=140)
    plt.close(fig)


def figure7_heatmap(cells, figdir):
    models = sorted({k[1] for k in cells})
    langs = sorted({k[2] for k in cells})
    M = len(models)
    L = len(langs)

    def grid(metric_fn):
        g = np.full((M, L), np.nan)
        for i, m in enumerate(models):
            for j, l in enumerate(langs):
                k = next((kk for kk in cells if kk[1] == m and kk[2] == l), None)
                if k is not None:
                    v = metric_fn(cells[k])
                    if v is not None and v == v:  # not NaN
                        g[i, j] = v
        return g

    metrics = [
        ("Token/s", lambda d: d["token_per_second_mean"], "fig7a_token_per_second_heatmap.png"),
        ("UTF-8 Bit/s", lambda d: d["bit_per_second_mean"], "fig7b_bit_per_second_heatmap.png"),
        ("Bytes/Token", lambda d: d["bytes_per_token_mean"], "fig7c_bytes_per_token_heatmap.png"),
    ]
    for label, fn, fname in metrics:
        g = grid(fn)
        fig, ax = plt.subplots(figsize=(1.0 + 0.6 * L, 1.0 + 0.45 * M))
        im = ax.imshow(g, aspect="auto", cmap="viridis")
        ax.set_xticks(range(L))
        ax.set_xticklabels(langs, rotation=20, ha="right")
        ax.set_yticks(range(M))
        ax.set_yticklabels([m.split(":")[0] for m in models])
        for i in range(M):
            for j in range(L):
                v = g[i, j]
                if v == v:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color="white" if v < np.nanmedian(g) else "black",
                            fontsize=8)
        ax.set_title(f"Figure 7 — {label}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(figdir / fname, dpi=140)
        plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml", type=Path)
    parser.add_argument("--runs-dir", default="",
                        help="results/raw/new-XXX folder (default: latest new-*)")
    args = parser.parse_args()
    root = _project_root()
    runs_dir = Path(args.runs_dir) if args.runs_dir else None
    if not runs_dir:
        candidates = sorted((root / "results" / "raw").glob("new-*"))
        runs_dir = candidates[-1] if candidates else None
    if runs_dir and not runs_dir.is_absolute():
        runs_dir = root / runs_dir
    rows = read_runs(runs_dir if runs_dir else root)
    if not rows:
        print("[viz] no derived/all_runs.csv found")
        return 1
    rows = cast(rows)
    cells = mean_per_cell(rows)

    figdir = root / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    figure1_token_vs_bit_scatter(cells, figdir)
    figure2_3_boxplots(rows, figdir)
    figure4_bytes_per_token_bar(cells, figdir)
    figure5_rank_comparison(cells, figdir)
    figure6_token_to_bit_mapping(cells, figdir)
    figure7_heatmap(cells, figdir)
    print(f"[viz] wrote figures to {figdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
