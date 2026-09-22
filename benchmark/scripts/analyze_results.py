"""
analyze_results.py — Statistical analysis answering the research questions
listed in md §14 and §15.

  * RQ1: Are Token/s and Bit/s rankings consistent?
        -> Spearman ρ, Kendall τ across (model × language) mean cells.
  * RQ2: Does language change the Token/s → Bit/s mapping?
        -> Per-language correlation ρ.
  * RQ3: How much of the Token/s vs Bit/s gap is explained by tokenizer?
        -> Correlation between Bytes/Token and the (Bit/s ÷ Token/s)
           ratio, plus a regression R².
  * RQ4: Are there models that flip ranking between Token/s and Bit/s?
        -> Sign of (Bit/s_rank - Token/s_rank) per cell.
  * RQ5: Does model size matter?
        -> Group models by parameter_size bins, compare the two metrics.
  * RQ6: Do different scripts have systematic Bytes/Token differences?
        -> One-way ANOVA + η² + Cohen's d between language pairs.

Statistical tests (§15):
  * Spearman / Kendall rank correlation
  * Two-way ANOVA  (factor A = model, factor B = language) on Token/s and
                   on Bit/s, plus model × language interaction
  * Mixed effects   Bit/s ~ Model + Language + (1 | sample)  (random=sample)
  * η², partial η²
  * Cohen's d for pairwise model contrasts on the same language

Outputs:
  results/derived/analysis.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from scipy import stats


# --------------------------------------------------------------------------- #
def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def read_runs(root: Path, runs_dir: Optional[str] = None) -> List[dict]:
    if runs_dir:
        p = Path(runs_dir) / "derived" / "all_runs.csv"
        if not p.is_absolute():
            p = root / p
    else:
        p = root / "results" / "derived" / "all_runs.csv"
    with p.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_summary(root: Path, runs_dir: Optional[str] = None) -> List[dict]:
    if runs_dir:
        p = Path(runs_dir) / "derived" / "per_cell_summary.csv"
        if not p.is_absolute():
            p = root / p
    else:
        p = root / "results" / "derived" / "per_cell_summary.csv"
    with p.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def diagnose_empty_responses(runs: List[dict]) -> dict:
    """Return a per-model count of runs whose response_text was empty.
    These runs MUST be excluded from any throughput interpretation (the model
    used its entire token budget on internal thinking, never producing
    visible content)."""
    out: dict = {}
    for r in runs:
        if str(r.get("response_text_empty", "0")) == "1":
            out.setdefault(r["model"], 0)
            out[r["model"]] += 1
    return {"empty_response_counts": out,
            "n_empty_total": sum(out.values()),
            "n_total": len(runs),
            "note": "Runs with empty response_text are kept in the raw JSON for "
                    "auditability but are excluded from mean-per-cell statistics "
                    "below because their Bit/s = 0 would be misleading."}


def mean_per_cell(rows: List[dict]) -> Dict[Tuple[str, str, str], dict]:
    """Mean per (experiment, model, language). Excludes runs whose
    response_text was empty — their Bit/s would be 0 and would distort
    averages. The diagnostics dict in analysis.json reports how many runs
    were excluded per model."""
    bucket: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    for r in rows:
        if str(r.get("response_text_empty", "0")) == "1":
            continue
        key = (r["experiment"], r["model"], r["language"])
        bucket[key].append(r)

    out = {}
    for k, group in bucket.items():
        def m(metric):
            xs = [float(g[metric]) for g in group if g[metric]]
            return statistics.mean(xs) if xs else float("nan")
        out[k] = {
            "experiment": k[0], "model": k[1], "language": k[2],
            "n": len(group),
            "token_per_second_mean": m("token_per_second"),
            "bit_per_second_mean": m("bit_per_second"),
            "bytes_per_token_mean": m("bytes_per_token"),
            "bits_per_token_mean": m("bits_per_token"),
            "token_per_char_mean": m("token_per_char"),
        }
    return out


# --------------------------------------------------------------------------- #
def spearman_kendall(xs: List[float], ys: List[float]) -> dict:
    if len(xs) < 3:
        return {"spearman_r": None, "spearman_p": None,
                "kendall_tau": None, "kendall_p": None,
                "note": "not enough points"}
    sp = stats.spearmanr(xs, ys)
    kt = stats.kendalltau(xs, ys)
    return {
        "spearman_r": float(sp.statistic),
        "spearman_p": float(sp.pvalue),
        "kendall_tau": float(kt.statistic),
        "kendall_p": float(kt.pvalue),
    }


def rq1_rank_consistency(cells: dict) -> dict:
    """Across all (model, lang) cells, do Token/s rank and Bit/s rank agree?"""
    keys = list(cells.keys())
    xs = [cells[k]["token_per_second_mean"] for k in keys]
    ys = [cells[k]["bit_per_second_mean"] for k in keys]
    overall = spearman_kendall(xs, ys)

    by_lang = {}
    langs = sorted({k[2] for k in keys})
    for lang in langs:
        sub = [k for k in keys if k[2] == lang]
        sx = [cells[k]["token_per_second_mean"] for k in sub]
        sy = [cells[k]["bit_per_second_mean"] for k in sub]
        by_lang[lang] = spearman_kendall(sx, sy)
    return {"overall": overall, "by_language": by_lang}


def rq4_ranking_flips(cells: dict) -> dict:
    """For each model, across languages, does Token/s rank match Bit/s rank?"""
    flips: List[dict] = []
    keys = list(cells.keys())
    langs = sorted({k[2] for k in keys})
    for lang in langs:
        sub = [k for k in keys if k[2] == lang]
        if len(sub) < 2:
            continue
        tok = sorted(sub, key=lambda k: -cells[k]["token_per_second_mean"])
        bit = sorted(sub, key=lambda k: -cells[k]["bit_per_second_mean"])
        tok_rank = {k: i for i, k in enumerate(tok)}
        bit_rank = {k: i for i, k in enumerate(bit)}
        for k in sub:
            d = bit_rank[k] - tok_rank[k]
            if abs(d) >= max(1, len(sub) // 2):
                flips.append({
                    "model": k[1],
                    "language": k[2],
                    "token_rank": tok_rank[k] + 1,
                    "bit_rank": bit_rank[k] + 1,
                    "rank_diff": d,
                })
    return {
        "significant_flips": flips,
        "n_significant": len(flips),
        "n_cells_compared": sum(1 for k in keys),
    }


# --------------------------------------------------------------------------- #
def two_way_anova(df: List[dict], metric: str) -> dict:
    """Minimal two-way ANOVA (Model × Language) with interaction.

    Implementation is hand-rolled (no statsmodels formula API needed for the
    simple balanced case we usually have). Returns η² and partial η².
    """
    rows = [d for d in df if d.get(metric) is not None]
    if not rows:
        return {"error": "no data"}
    models = sorted({d["model"] for d in rows})
    langs = sorted({d["language"] for d in rows})

    grand_mean = statistics.mean(d[metric] for d in rows)
    n = len(rows)

    # cell means
    cells = defaultdict(list)
    for d in rows:
        cells[(d["model"], d["language"])].append(d[metric])
    cell_means = {k: statistics.mean(v) for k, v in cells.items()}
    cell_ns = {k: len(v) for k, v in cells.items()}

    # main effects
    model_means = {m: statistics.mean(cell_means[(m, l)] for l in langs
                                     if (m, l) in cell_means)
                   for m in models}
    lang_means = {l: statistics.mean(cell_means[(m, l)] for m in models
                                    if (m, l) in cell_means)
                  for l in langs}

    SS_A = sum(len([1 for d in rows if d["model"] == m])
               * (model_means[m] - grand_mean) ** 2 for m in models)
    SS_B = sum(len([1 for d in rows if d["language"] == l])
               * (lang_means[l] - grand_mean) ** 2 for l in langs)
    SS_AB = sum(cell_ns[k] * (cell_means[k] - model_means[k[0]] - lang_means[k[1]] + grand_mean) ** 2
                for k in cell_means)
    SS_T = sum((d[metric] - grand_mean) ** 2 for d in rows)
    SS_W = SS_T - SS_A - SS_B - SS_AB

    df_A = len(models) - 1
    df_B = len(langs) - 1
    df_AB = df_A * df_B
    df_W = n - len(models) * len(langs)

    def safe_div(a, b): return a / b if b else float("nan")

    MS_A = safe_div(SS_A, df_A)
    MS_B = safe_div(SS_B, df_B)
    MS_AB = safe_div(SS_AB, df_AB)
    MS_W = safe_div(SS_W, df_W)

    F_A = safe_div(MS_A, MS_W)
    F_B = safe_div(MS_B, MS_W)
    F_AB = safe_div(MS_AB, MS_W)

    p_A = 1 - stats.f.cdf(F_A, df_A, df_W) if F_A == F_A else None
    p_B = 1 - stats.f.cdf(F_B, df_B, df_W) if F_B == F_B else None
    p_AB = 1 - stats.f.cdf(F_AB, df_AB, df_W) if F_AB == F_AB else None

    def eta2(ss): return safe_div(ss, SS_T)
    def partial_eta2(ss, df_eff):
        return safe_div(ss, ss + SS_W) if (ss + SS_W) else float("nan")

    return {
        "metric": metric,
        "n_observations": n,
        "df_model": df_A,
        "df_language": df_B,
        "df_interaction": df_AB,
        "df_within": df_W,
        "F_model": F_A, "p_model": p_A,
        "F_language": F_B, "p_language": p_B,
        "F_interaction": F_AB, "p_interaction": p_AB,
        "eta2_model": eta2(SS_A),
        "eta2_language": eta2(SS_B),
        "eta2_interaction": eta2(SS_AB),
        "partial_eta2_model": partial_eta2(SS_A, df_A),
        "partial_eta2_language": partial_eta2(SS_B, df_B),
        "partial_eta2_interaction": partial_eta2(SS_AB, df_AB),
        "ss_total": SS_T,
        "ss_within": SS_W,
    }


# --------------------------------------------------------------------------- #
def cohens_d(a: List[float], b: List[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    sp = math.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    return (ma - mb) / sp if sp else float("nan")


def rq6_language_pairs(rows: List[dict]) -> dict:
    """Compare Bytes/Token between every pair of languages."""
    by_lang = defaultdict(list)
    for r in rows:
        v = r.get("bytes_per_token")
        if v:
            by_lang[r["language"]].append(float(v))
    langs = sorted(by_lang)
    pairs = []
    for a, b in combinations(langs, 2):
        d = cohens_d(by_lang[a], by_lang[b])
        # Welch's t-test for unequal variance
        if len(by_lang[a]) >= 2 and len(by_lang[b]) >= 2:
            t = stats.ttest_ind(by_lang[a], by_lang[b], equal_var=False)
            p = float(t.pvalue)
            tval = float(t.statistic)
        else:
            tval, p = float("nan"), float("nan")
        pairs.append({
            "lang_a": a, "lang_b": b,
            "mean_a": statistics.mean(by_lang[a]),
            "mean_b": statistics.mean(by_lang[b]),
            "cohens_d": d,
            "welch_t": tval,
            "welch_p": p,
        })
    # one-way omnibus
    groups = [by_lang[l] for l in langs]
    f = stats.f_oneway(*groups) if all(len(g) >= 2 for g in groups) else None
    return {
        "n_per_language": {l: len(by_lang[l]) for l in langs},
        "means": {l: statistics.mean(by_lang[l]) for l in langs},
        "one_way_anova_F": float(f.statistic) if f else None,
        "one_way_anova_p": float(f.pvalue) if f else None,
        "pairwise": pairs,
    }


def rq5_model_size(rows: List[dict], models_cfg: list) -> dict:
    """Group models by parameter_size bins and compare metrics."""
    bins = {
        "tiny_<2B": [],
        "small_2-4B": [],
        "medium_4-8B": [],
        "large_>8B": [],
    }
    size_lookup = {m["name"]: m["parameter_size"] for m in models_cfg}

    def bin_for(psize: str) -> str:
        n = float(psize.rstrip("BMm").replace("B", "").replace("M", "e-3") or 0)
        if n < 2: return "tiny_<2B"
        if n < 4: return "small_2-4B"
        if n < 8: return "medium_4-8B"
        return "large_>8B"

    by_bin = defaultdict(list)
    for r in rows:
        ps = size_lookup.get(r["model"])
        if not ps:
            continue
        b = bin_for(ps)
        by_bin[b].append(r)

    out = {}
    for b, group in by_bin.items():
        if not group:
            continue
        out[b] = {
            "n": len(group),
            "models": sorted({r["model"] for r in group}),
            "token_per_second_mean": statistics.mean(float(r["token_per_second"]) for r in group),
            "bit_per_second_mean": statistics.mean(float(r["bit_per_second"]) for r in group),
            "bytes_per_token_mean": statistics.mean(float(r["bytes_per_token"]) for r in group),
        }
    return out


def rq3_tokenizer_explains_gap(cells: dict) -> dict:
    """Fit Bit/s = a*Token/s + b and measure R²; also correlate
    Bytes/Token with the (Bit/s ÷ Token/s) ratio (= Bits/Token)."""
    keys = list(cells.keys())
    xs = [cells[k]["token_per_second_mean"] for k in keys]
    ys = [cells[k]["bit_per_second_mean"] for k in keys]
    bpt = [cells[k]["bytes_per_token_mean"] for k in keys]
    bit = [cells[k]["bits_per_token_mean"] for k in keys]
    if len(xs) < 3:
        return {"note": "too few points"}
    slope, intercept, r, p, se = stats.linregress(xs, ys)
    sp_bpt_bit = spearman_kendall(bpt, bit)
    return {
        "linear_regression_Bit_per_s_on_Token_per_s": {
            "slope": float(slope),
            "intercept": float(intercept),
            "r": float(r),
            "r2": float(r ** 2),
            "p": float(p),
            "se": float(se),
            "note": "If R² ≈ 1, Token/s alone is enough. If R² < 1, "
                    "Bytes/Token explains the residual — Bit/s carries "
                    "extra information.",
        },
        "spearman_BytesPerToken_vs_BitsPerToken": sp_bpt_bit,
    }


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml", type=Path)
    parser.add_argument("--runs-dir", default="",
                        help="path to a specific results/raw/new-XXX/ "
                             "folder. Defaults to results/derived.")
    args = parser.parse_args()

    root = _project_root()
    cfg = load_config(args.config)
    runs = read_runs(root, args.runs_dir or None)
    if not runs:
        print("[analysis] no derived/all_runs.csv found at "
              f"{args.runs_dir or 'results/derived'} — run calculate_metrics first")
        return 1

    # Cast numeric fields
    for r in runs:
        for k in ("token_per_second", "bit_per_second", "byte_per_second",
                  "bytes_per_token", "bits_per_token",
                  "token_per_char", "bytes_per_char", "expansion_ratio",
                  "identity_err", "eval_duration_s", "char_count", "utf8_bytes"):
            if k in r and r[k] != "":
                try:
                    r[k] = float(r[k])
                except ValueError:
                    r[k] = None

    cells = mean_per_cell(runs)
    diagnostics = diagnose_empty_responses(runs)

    out = {
        "diagnostics": diagnostics,
        "RQ1_ranking_consistency": rq1_rank_consistency(cells),
        "RQ2_language_changes_mapping": rq1_rank_consistency(cells)["by_language"],
        "RQ3_tokenizer_explains_gap": rq3_tokenizer_explains_gap(cells),
        "RQ4_ranking_flips": rq4_ranking_flips(cells),
        "RQ5_model_size_effect": rq5_model_size(runs, cfg["models"]),
        "RQ6_language_systematic_effect": rq6_language_pairs(runs),
        "ANOVA_token_per_second": two_way_anova(runs, "token_per_second"),
        "ANOVA_bit_per_second": two_way_anova(runs, "bit_per_second"),
        "ANOVA_bytes_per_token": two_way_anova(runs, "bytes_per_token"),
    }

    out_path = root / "results" / "derived" / "analysis.json"
    if args.runs_dir:
        out_path = Path(args.runs_dir) / "derived" / "analysis.json"
        if not out_path.is_absolute():
            out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")
    print(f"[analysis] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
