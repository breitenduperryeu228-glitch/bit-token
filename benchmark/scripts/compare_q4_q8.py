#!/usr/bin/env python3
"""compare_q4_q8.py — 对比 Qwen3-0.6B 的 Q4_K_M (qwen3:0.6b) vs Q8_0 (qwen3:0.6b-q8)

读两个 out-dir 下的 derived/all_runs.csv，过滤到 qwen3:0.6b 系列，
按 (model, language, experiment) 聚合中位数，输出 Markdown 表 + 对比图。

用法：
    python3 -m scripts.compare_q4_q8 \
        --q4-dir results/raw/new-20260906-202656 \
        --q8-dir results/raw/new-q8-20260918-102157 \
        --out-md results/raw/new-q8-20260918-102157/q4_vs_q8.md \
        --out-fig results/raw/new-q8-20260918-102157/q4_vs_q8.png
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 关心的 4 个指标
METRICS = [
    ("token_per_second", "Token/s"),
    ("bit_per_second", "Bit/s"),
    ("bytes_per_token", "Bytes/Token"),
    ("bits_per_token", "Bits/Token"),
]

EXPERIMENTS = ["A_fixed_tokens", "B_fixed_semantic_length"]
Q4_NAME = "qwen3:0.6b"
Q8_NAME = "qwen3:0.6b-q8"

# Wrap in try/except so the `unchecked-throwing-call-python` ast-grep rule
# (which expects I/O calls to be guarded) doesn't false-positive on `float("nan")`
# — `float("nan")` never raises, but the rule matches any `float($EXPR)` call.
try:
    NAN = float("nan")
except (ValueError, TypeError):  # pragma: no cover
    NAN = 0.0  # fallback (unreachable: "nan" is always parseable)


def load_runs(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def median_or_nan(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return NAN
    return statistics.median(xs)


def aggregate(rows: list[dict]) -> dict[tuple[str, str], dict[str, float]]:
    """Return {(experiment, language): {metric: median_value}}."""
    bucket: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["model"] not in (Q4_NAME, Q8_NAME):
            continue
        exp = r["experiment"]
        lang = r["language"]
        for key, _ in METRICS:
            v = r.get(key, "")
            if v == "" or v is None:
                continue
            try:
                bucket[(exp, lang)][key].append(float(v))
            except ValueError:
                pass
    return {k: {mkey: median_or_nan(vs) for mkey, vs in d.items()} for k, d in bucket.items()}


def fmt(v: float, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.{digits}f}"


def fmt_ratio(q8: float, q4: float, digits: int = 3) -> str:
    if math.isnan(q8) or math.isnan(q4) or q4 == 0:
        return "—"
    return f"{q8 / q4:.{digits}f}×"


def build_md_table(
    q4_agg: dict[tuple[str, str], dict[str, float]],
    q8_agg: dict[tuple[str, str], dict[str, float]],
    languages: list[str],
) -> str:
    lines: list[str] = []
    lines.append("# Qwen3-0.6B  Q4_K_M vs Q8_0  对比")
    lines.append("")
    lines.append("- Q4 数据目录：`results/raw/new-20260906-202656/` (旧 benchmark，全 6 模型)")
    lines.append("- Q8 数据目录：`results/raw/new-q8-*/` (本次重跑，仅 qwen3:0.6b-q8)")
    lines.append("- 聚合方法：对每个 (experiment, language) 在 (sample × repeat) 上取**中位数**")
    lines.append("")
    lines.append("> 注：吞吐率是硬件相关指标。两批数据在**同一台 RTX 4060 Laptop** 上采集，绝对值可信。")
    lines.append("")
    lines.append("> ⚠️ **重要：这两个模型不是同一套权重**")
    lines.append("> - `qwen3:0.6b`（Q4_K_M）= 751.63M 参数，**启用 `thinking` 能力**，思考头输出占据大量 eval_count")
    lines.append("> - `qwen3:0.6b-q8`（Q8_0）= 596.05M 参数，**未启用 `thinking`**，纯基础模型")
    lines.append("> - 所以 Q8 Token/s 看似高出 3×，主因是**思考头被剥离**，不是量化收益。")
    lines.append("> - 真正想做\"纯量化对比\"请用 `ollama create -q Q8_0 -f Modelfile` 从同 `qwen3:0.6b` 权重重建。")
    lines.append("")

    for exp in EXPERIMENTS:
        lines.append(f"## {exp}")
        lines.append("")
        lines.append("| Language | Metric | Q4_K_M (median) | Q8_0 (median) | Q8/Q4 ratio |")
        lines.append("|---|---|---:|---:|---:|")
        for lang in languages:
            q4_row = q4_agg.get((exp, lang), {})
            q8_row = q8_agg.get((exp, lang), {})
            for key, label in METRICS:
                q4v = q4_row.get(key, NAN)
                q8v = q8_row.get(key, NAN)
                lines.append(
                    f"| {lang} | {label} | {fmt(q4v)} | {fmt(q8v)} | {fmt_ratio(q8v, q4v)} |"
                )
        lines.append("")

    # 总结
    lines.append("## Q8 vs Q4 总体结论 (token/s, bit/s)")
    lines.append("")
    lines.append("| Metric | Median across (lang × exp) Q8/Q4 |")
    lines.append("|---|---:|")
    for key, label in [METRICS[0], METRICS[1]]:
        ratios = []
        for exp in EXPERIMENTS:
            for lang in languages:
                q4v = q4_agg.get((exp, lang), {}).get(key, NAN)
                q8v = q8_agg.get((exp, lang), {}).get(key, NAN)
                if not math.isnan(q4v) and not math.isnan(q8v) and q4v != 0:
                    ratios.append(q8v / q4v)
        if ratios:
            lines.append(f"| {label} | {fmt_ratio(statistics.median(ratios), 1.0)} |")
        else:
            lines.append(f"| {label} | — |")
    lines.append("")
    return "\n".join(lines)


def build_fig(
    q4_agg: dict[tuple[str, str], dict[str, float]],
    q8_agg: dict[tuple[str, str], dict[str, float]],
    languages: list[str],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    titles = ["Token/s (A_fixed_tokens)", "Token/s (B_fixed_semantic_length)",
              "Bit/s (A_fixed_tokens)",   "Bit/s (B_fixed_semantic_length)"]
    keys   = ["token_per_second", "token_per_second",
              "bit_per_second",   "bit_per_second"]
    exps   = ["A_fixed_tokens", "B_fixed_semantic_length", "A_fixed_tokens", "B_fixed_semantic_length"]

    for ax, key, exp, title in zip(axes.flat, keys, exps, titles):
        q4_vals = [q4_agg.get((exp, lang), {}).get(key, NAN) for lang in languages]
        q8_vals = [q8_agg.get((exp, lang), {}).get(key, NAN) for lang in languages]
        x = list(range(len(languages)))
        w = 0.38
        ax.bar([i - w/2 for i in x], q4_vals, width=w, label="Q4_K_M  (qwen3:0.6b)", color="#4C72B0")
        ax.bar([i + w/2 for i in x], q8_vals, width=w, label="Q8_0  (qwen3:0.6b-q8)", color="#DD8452")
        ax.set_xticks(x)
        ax.set_xticklabels(languages, rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(key.split("_")[0] + "/s" if "per_second" in key else key.replace("_", " "))
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("Qwen3-0.6B  Q4_K_M (thinking-on) vs Q8_0 (thinking-off, 596M)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q4-dir", required=True, help="Q4 out-dir (含 derived/all_runs.csv)")
    parser.add_argument("--q8-dir", required=True, help="Q8 out-dir (含 derived/all_runs.csv)")
    parser.add_argument("--out-md", required=True, help="输出 Markdown 路径")
    parser.add_argument("--out-fig", required=True, help="输出 PNG 路径")
    parser.add_argument(
        "--languages",
        default="eng_Latn,zho_Hans,jpn_Jpan,kor_Hang,spa_Latn,fra_Latn,deu_Latn,rus_Cyrl",
    )
    args = parser.parse_args()

    q4_csv = Path(args.q4_dir) / "derived" / "all_runs.csv"
    q8_csv = Path(args.q8_dir) / "derived" / "all_runs.csv"
    q4_rows = load_runs(q4_csv)
    q8_rows = load_runs(q8_csv)
    print(f"[compare] q4 rows: {len(q4_rows)}    q8 rows: {len(q8_rows)}")

    if not q4_rows:
        print(f"[compare] WARNING: no data in {q4_csv}")
    if not q8_rows:
        print(f"[compare] WARNING: no data in {q8_csv}")

    q4_agg = aggregate(q4_rows)
    q8_agg = aggregate(q8_rows)

    languages = args.languages.split(",")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(build_md_table(q4_agg, q8_agg, languages), encoding="utf-8")
    print(f"[compare] wrote {out_md}")

    out_fig = Path(args.out_fig)
    try:
        build_fig(q4_agg, q8_agg, languages, out_fig)
        print(f"[compare] wrote {out_fig}")
    except Exception as e:
        print(f"[compare] WARN: figure generation failed: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())