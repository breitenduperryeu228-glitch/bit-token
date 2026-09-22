"""
report.py — Synthesise a Markdown report that answers the six research
questions (md §14) and explicitly references the analysis JSON, the
per-cell summary CSV, and the figures in `figures/`.

This is what the user reads last.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    import yaml
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def fmt(v, default="—"):
    if v is None:
        return default
    if isinstance(v, float):
        if v != v:
            return default
        return f"{v:.4g}"
    return str(v)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml", type=Path)
    args = parser.parse_args()

    root = _project_root()
    cfg = load_config(args.config)
    derived = root / "results" / "derived"
    analysis = json.loads((derived / "analysis.json").read_text(encoding="utf-8"))
    summary_rows = list(csv.DictReader((derived / "per_cell_summary.csv").open(encoding="utf-8")))

    now = datetime.now(timezone.utc).isoformat()

    md: List[str] = []
    md.append(f"# Token/s vs UTF-8 Bit/s — Benchmark Report")
    md.append(f"_Generated: {now}_\n")

    md.append("## 0. Setup")
    md.append(f"- Hardware: {cfg['hardware']['gpu']} ({cfg['hardware']['vram_gb']} GB VRAM), "
              f"{cfg['hardware']['ram_gb']} GB RAM")
    md.append(f"- Software: Ollama {cfg['software']['ollama_version']}, Python "
              f"{cfg['software']['python_version']}, CUDA {cfg['software']['cuda']}")
    md.append(f"- Models benchmarked: {', '.join(m['name'] for m in cfg['models'])}")
    md.append(f"- Languages: {', '.join(l['code'] for l in cfg['languages'])}")
    md.append(f"- Dataset: {cfg['dataset']['name']} "
              f"({cfg['dataset']['num_samples']} aligned samples per language)")
    md.append(f"- Repeats per cell: {cfg['inference']['repeat']} (plus "
              f"{cfg['inference']['warmup_runs']} warm-up not counted)")
    md.append("")

    # ----------------------------- RQ1 ------------------------------
    rq1 = analysis["RQ1_ranking_consistency"]
    md.append("## RQ1 — Do Token/s rankings agree with Bit/s rankings?")
    md.append(f"- **Spearman ρ** (overall, all cells): "
              f"{fmt(rq1['overall']['spearman_r'])} (p = {fmt(rq1['overall']['spearman_p'])})")
    md.append(f"- **Kendall τ** (overall): "
              f"{fmt(rq1['overall']['kendall_tau'])} (p = {fmt(rq1['overall']['kendall_p'])})")
    md.append("")
    md.append("Per-language correlations:")
    md.append("")
    md.append("| Language | Spearman ρ | p | Kendall τ | p |")
    md.append("|---|---|---|---|---|")
    for lang, c in rq1["by_language"].items():
        md.append(f"| {lang} | {fmt(c['spearman_r'])} | {fmt(c['spearman_p'])} | "
                  f"{fmt(c['kendall_tau'])} | {fmt(c['kendall_p'])} |")
    md.append("")

    # ----------------------------- RQ2 ------------------------------
    md.append("## RQ2 — Does language change the Token/s → Bit/s mapping?")
    md.append("See `RQ1.by_language` table above — each row is the Token/s ↔ "
              "Bit/s Spearman/Kendall correlation restricted to a single language. "
              "A high within-language correlation with a low cross-language correlation "
              "would indicate that language affects the mapping.")
    md.append("")
    md.append("ANOVA decomposition of Bit/s:")
    a_b = analysis["ANOVA_bit_per_second"]
    md.append(f"- Language factor: F = {fmt(a_b['F_language'])}, "
              f"p = {fmt(a_b['p_language'])}, partial η² = "
              f"{fmt(a_b['partial_eta2_language'])}")
    md.append(f"- Model factor:    F = {fmt(a_b['F_model'])}, "
              f"p = {fmt(a_b['p_model'])}, partial η² = "
              f"{fmt(a_b['partial_eta2_model'])}")
    md.append(f"- Model × Language interaction: F = {fmt(a_b['F_interaction'])}, "
              f"p = {fmt(a_b['p_interaction'])}, partial η² = "
              f"{fmt(a_b['partial_eta2_interaction'])}")
    md.append("")

    # ----------------------------- RQ3 ------------------------------
    md.append("## RQ3 — How much of Token/s vs Bit/s is explained by tokenizer?")
    rq3 = analysis["RQ3_tokenizer_explains_gap"]
    if "linear_regression_Bit_per_s_on_Token_per_s" in rq3:
        lr = rq3["linear_regression_Bit_per_s_on_Token_per_s"]
        md.append(f"- Linear fit: Bit/s ≈ {fmt(lr['slope'])}·Token/s + "
                  f"{fmt(lr['intercept'])} (R² = {fmt(lr['r2'])}, p = {fmt(lr['p'])})")
    if "spearman_BytesPerToken_vs_BitsPerToken" in rq3:
        s = rq3["spearman_BytesPerToken_vs_BitsPerToken"]
        md.append(f"- Bytes/Token ↔ Bits/Token Spearman ρ = "
                  f"{fmt(s['spearman_r'])} (p = {fmt(s['spearman_p'])})")
    md.append("")

    # ----------------------------- RQ4 ------------------------------
    md.append("## RQ4 — Any model that flips between Token/s and Bit/s ranking?")
    rq4 = analysis["RQ4_ranking_flips"]
    md.append(f"Cells compared: {rq4['n_cells_compared']}, significant flips: "
              f"{rq4['n_significant']}.")
    if rq4["significant_flips"]:
        md.append("")
        md.append("| Model | Language | Token-rank | Bit-rank | Δ |")
        md.append("|---|---|---|---|---|")
        for f in rq4["significant_flips"][:30]:
            md.append(f"| {f['model']} | {f['language']} | {f['token_rank']} | "
                      f"{f['bit_rank']} | {f['rank_diff']:+d} |")
    md.append("")

    # ----------------------------- RQ5 ------------------------------
    md.append("## RQ5 — Does model size matter?")
    md.append("")
    md.append("| Size bin | Models | n | mean Token/s | mean Bit/s | mean Bytes/Token |")
    md.append("|---|---|---|---|---|---|")
    for bin_, v in analysis["RQ5_model_size_effect"].items():
        md.append(f"| {bin_} | {', '.join(v['models'])} | {v['n']} | "
                  f"{fmt(v['token_per_second_mean'])} | {fmt(v['bit_per_second_mean'])} | "
                  f"{fmt(v['bytes_per_token_mean'])} |")
    md.append("")

    # ----------------------------- RQ6 ------------------------------
    md.append("## RQ6 — Language-level systematic effects on Bytes/Token")
    rq6 = analysis["RQ6_language_systematic_effect"]
    md.append(f"One-way ANOVA on Bytes/Token across languages: "
              f"F = {fmt(rq6['one_way_anova_F'])}, p = {fmt(rq6['one_way_anova_p'])}")
    md.append("")
    md.append("| Language | mean Bytes/Token | n |")
    md.append("|---|---|---|")
    for lang, m in rq6["means"].items():
        md.append(f"| {lang} | {fmt(m)} | {rq6['n_per_language'][lang]} |")
    md.append("")
    md.append("Top 10 largest pairwise Cohen's d (Bytes/Token):")
    pairs = sorted(rq6["pairwise"], key=lambda x: -abs(x["cohens_d"]) if x["cohens_d"] == x["cohens_d"] else 0)
    md.append("")
    md.append("| Lang A | Lang B | mean A | mean B | Cohen's d | Welch p |")
    md.append("|---|---|---|---|---|---|")
    for p in pairs[:10]:
        md.append(f"| {p['lang_a']} | {p['lang_b']} | {fmt(p['mean_a'])} | "
                  f"{fmt(p['mean_b'])} | {fmt(p['cohens_d'])} | {fmt(p['welch_p'])} |")
    md.append("")

    # ----------------------------- Per-cell ------------------------------
    md.append("## Appendix — Per-cell summary (mean ± std)")
    md.append("Selected columns: experiment, model, language, metric, n, mean, median, std, p5, p95, ci95")
    md.append("")
    md.append("| exp | model | language | metric | n | mean | median | std | p5 | p95 | ci95 |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in summary_rows[:200]:
        md.append(f"| {r['experiment']} | {r['model']} | {r['language']} | "
                  f"{r['metric']} | {r['n']} | {fmt(float(r['mean']))} | "
                  f"{fmt(float(r['median']))} | {fmt(float(r['std']))} | "
                  f"{fmt(float(r['p5']))} | {fmt(float(r['p95']))} | {fmt(float(r['ci95']))} |")
    md.append("")
    md.append("## Figures")
    md.append("- `figures/fig1_token_vs_bit_scatter.png` — Token/s vs Bit/s scatter")
    md.append("- `figures/fig2_token_per_second_box.png` — Token/s by language")
    md.append("- `figures/fig3_bit_per_second_box.png` — Bit/s by language")
    md.append("- `figures/fig4_bytes_per_token_bar.png` — Bytes/Token per (model, language)")
    md.append("- `figures/fig5_rank_comparison.png` — Rank comparison")
    md.append("- `figures/fig6_token_to_bit_mapping.png` — Token/s → Bit/s mapping")
    md.append("- `figures/fig7{a,b,c}_*_heatmap.png` — Heatmaps")

    out = root / "results" / "REPORT.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"[report] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
