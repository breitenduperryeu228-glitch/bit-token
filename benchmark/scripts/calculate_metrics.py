"""
calculate_metrics.py — Convert raw Ollama JSON to derived metrics.

Outputs (per-cell summary + cross-cutting aggregate stats):
  * results/derived/all_runs.csv             one row per (experiment, model,
                                             language, sample, repeat)
  * results/derived/per_cell_summary.csv     mean/median/std/p5/p95/ci95 per
                                             (exp, model, lang) cell
  * results/derived/text_and_perf_stats.json text-level + cross-cutting
                                             perf stats (see below)

Per md §7–10 base metrics (Token/s, Byte/s, Bit/s, Bytes/Token, Bits/Token,
Token/Char, Bytes/Char, ExpansionRatio) are unchanged. The new file adds
four slices that the user asked for explicitly:

  * 常用单词长度 (word length) — per language + global
      min / median / max / mean / mode
  * token对应的bit数 (bits per token) — per model + per language + global
      min / median / max / mean / std
  * 每句话的长度 (sentence length) — per language + global
      min / median / max / mean / mode (counted in chars and UTF-8 bytes)
  * 每个语言的token/s和bit/s (per-language throughput) — per language
      min / median / max / mean / std for both metrics

Word segmentation:
  * Non-CJK scripts (Latin, Cyrillic, Hangul) — split on whitespace.
    This is the standard "word" for those scripts in FLORES-200.
  * CJK scripts (Han, Hiragana, Katakana) — no whitespace; each character
    is treated as a "word". The output marks the tokenization strategy so
    downstream consumers know not to compare CJK word-length distributions
    directly to Latin ones.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import yaml


_CJK_RE = re.compile(
    r"[\u3040-\u30ff"   # Hiragana + Katakana
    r"\u3400-\u4dbf"   # CJK Extension A
    r"\u4e00-\u9fff"   # CJK Unified Ideographs
    r"\uf900-\ufaff"   # CJK Compatibility Ideographs
    r"\uff66-\uff9f]"  # Half-width Katakana
)


# --------------------------------------------------------------------------- #
def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def utf8_metrics(text: str) -> dict:
    encoded = text.encode("utf-8")
    return {
        "utf8_bytes": len(encoded),
        "utf8_bits": 8 * len(encoded),
        "char_count": len(text),
    }


def is_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def segment_words(text: str) -> List[str]:
    """Whitespace tokenise Latin / Cyrillic / Hangul; per-character for CJK.

    Empty / whitespace-only inputs return [].
    """
    if is_cjk(text):
        return [ch for ch in text if not ch.isspace()]
    return text.split()


def _mode_or_none(xs: List) -> Optional[object]:
    """Return the first mode of `xs` (Python 3.8+ `statistics.mode`) or None."""
    if not xs:
        return None
    try:
        return statistics.mode(xs)
    except statistics.StatisticsError:
        return None


def distribution_stats(xs: List[float]) -> dict:
    """min/ mean/ max/ median/ std/ mode + n. Empty list returns n=0 + nan."""
    if not xs:
        return {"n": 0, "min": None, "median": None, "max": None,
                "mean": None, "std": None, "mode": None}
    return {
        "n": len(xs),
        "min": min(xs),
        "median": statistics.median(xs),
        "max": max(xs),
        "mean": statistics.mean(xs),
        "std": statistics.stdev(xs) if len(xs) > 1 else 0.0,
        "mode": _mode_or_none(xs),
    }


# --------------------------------------------------------------------------- #
def compute_row(raw: dict) -> dict | None:
    oll = raw.get("ollama", {})
    eval_count = oll.get("eval_count")
    eval_duration_ns = oll.get("eval_duration")
    if not eval_count or not eval_duration_ns:
        return None

    text = raw.get("response_text", "")
    u = utf8_metrics(text)
    eval_seconds = eval_duration_ns / 1e9

    token_per_s = eval_count / eval_seconds
    byte_per_s = u["utf8_bytes"] / eval_seconds
    bit_per_s = byte_per_s * 8

    bytes_per_token = u["utf8_bytes"] / eval_count
    bits_per_token = 8 * bytes_per_token
    token_per_char = eval_count / u["char_count"] if u["char_count"] else float("nan")
    bytes_per_char = u["utf8_bytes"] / u["char_count"] if u["char_count"] else float("nan")
    identity_err = abs(bit_per_s - token_per_s * bits_per_token)

    return {
        "experiment": raw["experiment"],
        "model": raw["model"],
        "language": raw["language"],
        "sample_id": raw["sample_id"],
        "repeat": raw["repeat"],
        "eval_count": eval_count,
        "eval_duration_ns": eval_duration_ns,
        "eval_duration_s": eval_seconds,
        "char_count": u["char_count"],
        "utf8_bytes": u["utf8_bytes"],
        "utf8_bits": u["utf8_bits"],
        "token_per_second": token_per_s,
        "byte_per_second": byte_per_s,
        "bit_per_second": bit_per_s,
        "bytes_per_token": bytes_per_token,
        "bits_per_token": bits_per_token,
        "token_per_char": token_per_char,
        "bytes_per_char": bytes_per_char,
        "identity_err": identity_err,
        "response_text_empty": int(len(text) == 0),
        "done_reason": oll.get("done_reason"),
    }


# --------------------------------------------------------------------------- #
def gather_all_runs(root: Path, runs_dir: Optional[str] = None) -> List[dict]:
    rows: List[dict] = []
    if runs_dir:
        raw_root = Path(runs_dir)
        if not raw_root.is_absolute():
            raw_root = root / runs_dir
    else:
        raw_root = root / "results" / "raw"
    if not raw_root.exists():
        return rows
    for path in raw_root.rglob("*.json"):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        row = compute_row(raw)
        if row is not None:
            rows.append(row)
    return rows


def attach_expansion_ratio(rows: List[dict], reference_model: str) -> None:
    by_key: Dict[tuple, dict] = defaultdict(dict)
    for r in rows:
        by_key[(r["experiment"], r["language"], r["sample_id"], r["repeat"])][r["model"]] = r
    for r in rows:
        key = (r["experiment"], r["language"], r["sample_id"], r["repeat"])
        ref = by_key[key].get(reference_model)
        r["expansion_ratio"] = (
            r["eval_count"] / ref["eval_count"] if ref and ref["eval_count"] else float("nan")
        )


# --------------------------------------------------------------------------- #
def percentile(xs: List[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] + (xs[c] - f) * (k - f)


def ci95(xs: List[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    m = statistics.mean(xs)
    s = statistics.stdev(xs)
    return 1.96 * s / math.sqrt(len(xs))


def summarise(rows: List[dict]) -> List[dict]:
    bucket: Dict[tuple, List[dict]] = defaultdict(list)
    for r in rows:
        bucket[(r["experiment"], r["model"], r["language"])].append(r)

    summary = []
    for (exp, model, lang), group in bucket.items():
        for metric in [
            "token_per_second", "bit_per_second", "bytes_per_token",
            "bits_per_token", "token_per_char", "bytes_per_char",
            "expansion_ratio",
        ]:
            xs = [g[metric] for g in group
                  if isinstance(g[metric], (int, float)) and not math.isnan(g[metric])]
            if not xs:
                continue
            summary.append({
                "experiment": exp,
                "model": model,
                "language": lang,
                "metric": metric,
                "n": len(xs),
                "mean": statistics.mean(xs),
                "median": statistics.median(xs),
                "std": statistics.stdev(xs) if len(xs) > 1 else 0.0,
                "p5": percentile(xs, 0.05),
                "p95": percentile(xs, 0.95),
                "ci95": ci95(xs),
            })
    return summary


# --------------------------------------------------------------------------- #
def gather_source_texts(root: Path, runs_dir: Optional[str] = None) -> List[dict]:
    """Yield one record per (language, sample_id) with the source text.

    Reads the same raw JSONs as `gather_all_runs` but extracts the
    `prompt.source_text` field, which the cell-level metrics don't carry.
    """
    if runs_dir:
        raw_root = Path(runs_dir)
        if not raw_root.is_absolute():
            raw_root = root / runs_dir
    else:
        raw_root = root / "results" / "raw"
    if not raw_root.exists():
        return []
    seen: set = set()
    out: List[dict] = []
    for path in raw_root.rglob("*.json"):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        key = (raw.get("language"), raw.get("sample_id"))
        if key in seen:
            continue
        seen.add(key)
        text = (raw.get("prompt") or {}).get("source_text") or ""
        out.append({
            "language": raw.get("language"),
            "sample_id": raw.get("sample_id"),
            "text": text,
        })
    return out


def compute_text_stats(records: List[dict]) -> dict:
    """Compute word / sentence length stats per language + globally.

    `records` is the output of `gather_source_texts()` — one record per
    unique (language, sample_id). For each record we split into words and
    compute per-word char length, plus per-sentence char length and
    per-sentence UTF-8 byte length.
    """
    by_lang_word: Dict[str, List[int]] = defaultdict(list)
    by_lang_sent_chars: Dict[str, List[int]] = defaultdict(list)
    by_lang_sent_bytes: Dict[str, List[int]] = defaultdict(list)
    tokenization_per_lang: Dict[str, str] = {}
    n_sentences_per_lang: Dict[str, int] = defaultdict(int)

    for rec in records:
        lang = rec["language"]
        text = rec["text"]
        if not text:
            continue
        tokenization_per_lang[lang] = "cjk" if is_cjk(text) else "whitespace"
        words = segment_words(text)
        by_lang_word[lang].extend(len(w) for w in words)
        by_lang_sent_chars[lang].append(len(text))
        by_lang_sent_bytes[lang].append(len(text.encode("utf-8")))
        n_sentences_per_lang[lang] += 1

    by_language: Dict[str, dict] = {}
    for lang in sorted(by_lang_word.keys()):
        word_xs = by_lang_word[lang]
        sent_char_xs = by_lang_sent_chars[lang]
        sent_byte_xs = by_lang_sent_bytes[lang]
        by_language[lang] = {
            "n_sentences": n_sentences_per_lang[lang],
            "tokenization": tokenization_per_lang.get(lang, "whitespace"),
            "word_length_chars": distribution_stats(word_xs),
            "sentence_length_chars": distribution_stats(sent_char_xs),
            "sentence_length_bytes": distribution_stats(sent_byte_xs),
        }

    all_word_xs: List[int] = []
    for xs in by_lang_word.values():
        all_word_xs.extend(xs)
    all_sent_char_xs: List[int] = []
    for xs in by_lang_sent_chars.values():
        all_sent_char_xs.extend(xs)
    all_sent_byte_xs: List[int] = []
    for xs in by_lang_sent_bytes.values():
        all_sent_byte_xs.extend(xs)

    return {
        "by_language": by_language,
        "global": {
            "n_sentences": sum(n_sentences_per_lang.values()),
            "word_length_chars": distribution_stats(all_word_xs),
            "sentence_length_chars": distribution_stats(all_sent_char_xs),
            "sentence_length_bytes": distribution_stats(all_sent_byte_xs),
        },
    }


def _safe_float_xs(rows: List[dict], key: str) -> List[float]:
    out: List[float] = []
    for r in rows:
        v = r.get(key)
        if isinstance(v, (int, float)) and not math.isnan(v):
            out.append(float(v))
    return out


def compute_perf_stats(rows: List[dict]) -> dict:
    """Bits/Token distribution + per-language and per-model throughput."""
    bits_per_token_xs = _safe_float_xs(rows, "bits_per_token")
    bpt_by_model: Dict[str, List[float]] = defaultdict(list)
    bpt_by_language: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        v = r.get("bits_per_token")
        if isinstance(v, (int, float)) and not math.isnan(v):
            bpt_by_model[r["model"]].append(float(v))
            bpt_by_language[r["language"]].append(float(v))

    bits_per_token: dict = {
        "global": distribution_stats(bits_per_token_xs),
        "by_model": {m: distribution_stats(xs) for m, xs in sorted(bpt_by_model.items())},
        "by_language": {l: distribution_stats(xs) for l, xs in sorted(bpt_by_language.items())},
    }

    tps_by_language: Dict[str, List[float]] = defaultdict(list)
    tps_by_model: Dict[str, List[float]] = defaultdict(list)
    bps_by_language: Dict[str, List[float]] = defaultdict(list)
    bps_by_model: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        tps = r.get("token_per_second")
        bps = r.get("bit_per_second")
        if isinstance(tps, (int, float)) and not math.isnan(tps):
            tps_by_language[r["language"]].append(float(tps))
            tps_by_model[r["model"]].append(float(tps))
        if isinstance(bps, (int, float)) and not math.isnan(bps):
            bps_by_language[r["language"]].append(float(bps))
            bps_by_model[r["model"]].append(float(bps))

    throughput_by_language: Dict[str, dict] = {}
    for lang in sorted(set(list(tps_by_language) + list(bps_by_language))):
        throughput_by_language[lang] = {
            "token_per_second": distribution_stats(tps_by_language.get(lang, [])),
            "bit_per_second": distribution_stats(bps_by_language.get(lang, [])),
        }

    throughput_by_model: Dict[str, dict] = {}
    for m in sorted(set(list(tps_by_model) + list(bps_by_model))):
        throughput_by_model[m] = {
            "token_per_second": distribution_stats(tps_by_model.get(m, [])),
            "bit_per_second": distribution_stats(bps_by_model.get(m, [])),
        }

    return {
        "bits_per_token": bits_per_token,
        "throughput_by_language": throughput_by_language,
        "throughput_by_model": throughput_by_model,
    }


def compute_text_and_perf_stats(root: Path, rows: List[dict], runs_dir: Optional[str]) -> dict:
    return {
        "text_stats": compute_text_stats(gather_source_texts(root, runs_dir)),
        **compute_perf_stats(rows),
    }


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml", type=Path)
    parser.add_argument("--runs-dir", default="",
                        help="path to a specific results/raw/new-XXX/ "
                             "folder. Defaults to results/raw (latest).")
    args = parser.parse_args()

    root = _project_root()
    cfg = load_config(args.config)
    rows = gather_all_runs(root, args.runs_dir or None)
    if not rows:
        print("[metrics] no raw JSON found under "
              f"{args.runs_dir or 'results/raw'} — run the benchmark first")
        return 1

    reference_model = cfg["models"][0]["name"]
    attach_expansion_ratio(rows, reference_model)

    out_dir = root / "results" / "derived"
    if args.runs_dir:
        out_dir = Path(args.runs_dir) / "derived"
        if not out_dir.is_absolute():
            out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_path = out_dir / "all_runs.csv"
    fields = list(rows[0].keys())
    with rows_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    summary = summarise(rows)
    sum_path = out_dir / "per_cell_summary.csv"
    if summary:
        with sum_path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
            w.writeheader()
            for s in summary:
                w.writerow(s)

    aggregate = compute_text_and_perf_stats(root, rows, args.runs_dir or None)
    agg_path = out_dir / "text_and_perf_stats.json"
    agg_path.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    identity = [r for r in rows if r["identity_err"] > 1e-6]
    print(f"[metrics] {len(rows)} runs -> {rows_path}")
    print(f"[metrics] summary -> {sum_path} ({len(summary)} rows)")
    print(f"[metrics] aggregate (text + perf slices) -> {agg_path}")
    print(f"[metrics] identity-check failures: {len(identity)} / {len(rows)}"
          f" (must be 0; Bit/s == Token/s * Bits/Token)")

    _print_aggregate_summary(aggregate)
    return 0


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def _print_aggregate_summary(agg: dict) -> None:
    print()
    print("=" * 78)
    print("  Aggregate stats (text_and_perf_stats.json)")
    print("=" * 78)

    text = agg["text_stats"]
    print()
    print("  Word length (chars) — by language:")
    print(f"    {'lang':<10} {'tokenizer':<12} {'sentences':>10} {'min':>5} {'median':>7} {'max':>5} {'mode':>5}")
    for lang in sorted(text["by_language"]):
        st = text["by_language"][lang]
        wl = st["word_length_chars"]
        print(
            f"    {lang:<10} {st['tokenization']:<12} {st['n_sentences']:>10} "
            f"{_fmt(wl['min']):>5} {_fmt(wl['median']):>7} {_fmt(wl['max']):>5} {_fmt(wl['mode']):>5}"
        )

    print()
    print("  Sentence length (chars) — by language:")
    print(f"    {'lang':<10} {'min':>5} {'median':>7} {'max':>5} {'mean':>7} {'mode':>5}")
    for lang in sorted(text["by_language"]):
        sl = text["by_language"][lang]["sentence_length_chars"]
        print(
            f"    {lang:<10} "
            f"{_fmt(sl['min']):>5} {_fmt(sl['median']):>7} {_fmt(sl['max']):>5} "
            f"{_fmt(sl['mean']):>7} {_fmt(sl['mode']):>5}"
        )

    print()
    print("  Bits / Token — global + by model:")
    bpt = agg["bits_per_token"]
    g = bpt["global"]
    print(
        f"    {'global':<40} min={_fmt(g['min']):>6} mean={_fmt(g['mean']):>6} "
        f"max={_fmt(g['max']):>6} std={_fmt(g['std']):>5} n={g['n']}"
    )
    for m, s in bpt["by_model"].items():
        print(
            f"    {m:<40} min={_fmt(s['min']):>6} mean={_fmt(s['mean']):>6} "
            f"max={_fmt(s['max']):>6} std={_fmt(s['std']):>5} n={s['n']}"
        )

    print()
    print("  Throughput by language — Token/s:")
    print(f"    {'lang':<10} {'min':>7} {'median':>8} {'max':>7} {'mean':>7}")
    for lang in sorted(agg["throughput_by_language"]):
        tps = agg["throughput_by_language"][lang]["token_per_second"]
        print(
            f"    {lang:<10} {_fmt(tps['min']):>7} {_fmt(tps['median']):>8} "
            f"{_fmt(tps['max']):>7} {_fmt(tps['mean']):>7}"
        )

    print()
    print("  Throughput by language — Bit/s:")
    print(f"    {'lang':<10} {'min':>9} {'median':>10} {'max':>9} {'mean':>9}")
    for lang in sorted(agg["throughput_by_language"]):
        bps = agg["throughput_by_language"][lang]["bit_per_second"]
        print(
            f"    {lang:<10} {_fmt(bps['min']):>9} {_fmt(bps['median']):>10} "
            f"{_fmt(bps['max']):>9} {_fmt(bps['mean']):>9}"
        )
    print("=" * 78)


if __name__ == "__main__":
    raise SystemExit(main())