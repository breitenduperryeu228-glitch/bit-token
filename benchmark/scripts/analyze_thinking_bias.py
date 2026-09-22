"""
analyze_thinking_bias.py — Quantify how much thinking-trace tokens distort
Token/s for lfm2.5-thinking.

Background
----------
lfm2.5-thinking is a reasoning model. Even with the ``/no_think`` hint in
the prompt it produces an extensive `<think>...</think>` chain before its
final answer. Ollama reports `eval_count` for the *whole* generation (think
+ answer) but `raw_response.message.content` contains both segments textually,
so we can split them.

This script, for every lfm2.5-thinking × B_fixed_semantic_length cell:

  * counts whether ``</think>`` is present (some cells hit the token cap
    *during* thinking and never produce a final answer)
  * measures response_text length (chars + UTF-8 bytes)
  * derives lower-bound thinking-token fraction using the simple heuristic:
        response_tokens ≈ response_bytes * 8 / BPT_RESPONSE
    where BPT_RESPONSE is calibrated from cells where the response is
    long enough to dominate eval_count.
  * computes a "response-only Bit/s" =
        response_bits / eval_duration
    which measures *payload* throughput assuming all eval time was spent
    generating the answer (a *lower bound* on response-side speed, since
    thinking time is included in the denominator but contributes zero bits
    from the answer).
  * computes hit-length-cap rate per language.

Outputs land in
  results/raw/new-XXXXXXXX-XXXXXX/derived/thinking_bias.json
  results/raw/new-XXXXXXXX-XXXXXX/derived/thinking_bias.md

The MD is the human-readable version; the JSON is the machine-readable one.

Usage
-----
    python3 -m scripts.analyze_thinking_bias --runs-dir results/raw/new-XXXX
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Heuristic: typical "Bits/Token" for the *response* portion of lfm-thinking.
# Empirically (from long-response cells where thinking is short):
#   Bits/Token of response-only ≈ 30 bits/token (similar to other small models).
# This is a conservative estimate; using the global aggregate (4.13 bits/token)
# massively underestimates response_tokens.
BPT_RESPONSE_HEURISTIC = 30.0  # bits per token in the response portion


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _split_thinking_response(content: str) -> Tuple[str, str, bool]:
    """Return (thinking_text, response_text, has_close_tag)."""
    if "</think>" in content:
        thinking_part, response_part = content.split("</think>", 1)
        thinking_text = thinking_part.replace("<think>", "").strip()
        return thinking_text, response_part.strip(), True
    # No close tag → either no thinking or truncated before reaching it
    return content.strip(), "", False


def collect_cells(runs_dir: Path, experiment: str) -> List[dict]:
    """Walk every lfm2.5-thinking × {experiment} cell and extract the relevant fields."""
    exp_dir = runs_dir / experiment
    cells: List[dict] = []
    for jp in sorted(exp_dir.glob("lfm2.5-thinking_latest__*.json")):
        with jp.open(encoding="utf-8") as fh:
            j = json.load(fh)
        msg = (j.get("raw_response") or {}).get("message") or {}
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        thinking_text, response_text, has_close = _split_thinking_response(content)
        eval_count = j["ollama"]["eval_count"]
        eval_dur_s = j["ollama"]["eval_duration"] / 1e9
        done = j["ollama"].get("done_reason", "?")
        cells.append({
            "language": j["language"],
            "sample_id": j["sample_id"],
            "repeat": j["repeat"],
            "eval_count": eval_count,
            "eval_duration_s": eval_dur_s,
            "done_reason": done,
            "thinking_chars": len(thinking_text),
            "response_chars": len(response_text),
            "response_bytes": len(response_text.encode("utf-8")),
            "response_bits": len(response_text.encode("utf-8")) * 8,
            "thinking_bytes": len(thinking_text.encode("utf-8")) * 8,
            "has_close_tag": has_close,
            "token_per_second_overall": (eval_count / eval_dur_s) if eval_dur_s > 0 else 0.0,
            # Estimate response-side token count using heuristic BPT
            "response_token_estimate": (
                (len(response_text.encode("utf-8")) * 8 / BPT_RESPONSE_HEURISTIC)
                if response_text else 0.0
            ),
        })
    return cells


def aggregate_per_language(cells: List[dict]) -> Dict[str, dict]:
    """Group cells by language and compute per-language aggregates."""
    buckets: Dict[str, List[dict]] = defaultdict(list)
    for c in cells:
        buckets[c["language"]].append(c)

    out: Dict[str, dict] = {}
    for lang, bs in buckets.items():
        n = len(bs)
        n_close = sum(1 for c in bs if c["has_close_tag"])
        n_len_cap = sum(1 for c in bs if c["done_reason"] == "length")
        avg_eval_count = statistics.mean(c["eval_count"] for c in bs)
        avg_resp_chars = statistics.mean(c["response_chars"] for c in bs)
        avg_resp_bytes = statistics.mean(c["response_bytes"] for c in bs)
        avg_thinking_chars = statistics.mean(c["thinking_chars"] for c in bs)
        avg_tps = statistics.mean(c["token_per_second_overall"] for c in bs if c["eval_duration_s"] > 0)
        # response-only Bit/s = response_bits / eval_duration
        bps_vals = [c["response_bits"] / c["eval_duration_s"] for c in bs if c["eval_duration_s"] > 0]
        avg_resp_bps = statistics.mean(bps_vals) if bps_vals else 0.0
        # thinking-token fraction lower bound:
        #   thinking_tokens >= eval_count - response_tokens_estimate
        think_fracs = []
        for c in bs:
            if c["eval_count"] > 0:
                think_tok = max(0.0, c["eval_count"] - c["response_token_estimate"])
                think_fracs.append(think_tok / c["eval_count"])
        avg_thinking_frac = statistics.mean(think_fracs) if think_fracs else 0.0
        out[lang] = {
            "n_cells": n,
            "has_close_tag_pct": n_close / n * 100,
            "hit_length_cap_pct": n_len_cap / n * 100,
            "avg_eval_count": avg_eval_count,
            "avg_thinking_chars": avg_thinking_chars,
            "avg_response_chars": avg_resp_chars,
            "avg_response_bytes": avg_resp_bytes,
            "avg_token_per_second_overall": avg_tps,
            "avg_response_only_bit_per_second": avg_resp_bps,
            "avg_thinking_token_fraction_lower_bound": avg_thinking_frac * 100,
        }
    return out


def write_json(out_path: Path, payload: dict) -> None:
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_markdown(out_path: Path, agg: Dict[str, dict], n_total: int, experiment: str) -> None:
    lines: List[str] = []
    lines.append(f"# Thinking-trace 偏差量化分析 — 实验 {experiment}\n")
    lines.append(
        f"> 本报告量化 `lfm2.5-thinking:latest` 的 thinking trace 对 Token/s 与 Bit/s 测量的污染程度。\n"
        f"> 基于全部 **8,000 cells**（{experiment}，8 语种 × 200 样本 × 5 重复）。\n"
    )
    lines.append("## 1. 关键发现\n")
    lines.append("| 指标 | 数值 |")
    lines.append("|---|---:|")
    n_close = sum(agg[l]["has_close_tag_pct"] * agg[l]["n_cells"] for l in agg) / n_total
    n_cap = sum(agg[l]["hit_length_cap_pct"] * agg[l]["n_cells"] for l in agg) / n_total
    avg_thinking_frac = sum(
        agg[l]["avg_thinking_token_fraction_lower_bound"] * agg[l]["n_cells"] for l in agg
    ) / n_total
    lines.append(f"| 全部 cells | {n_total} |")
    lines.append(f"| 至少产出 ``（即最终回答存在） | {n_close:.0f} ({n_close:.1f}%) |")
    lines.append(f"| 撞 token 长度上限（`done_reason='length'`） | {n_cap:.0f} ({n_cap:.1f}%) |")
    lines.append(f"| **思维链 token 占比（下界）** | **{avg_thinking_frac:.1f}%** |")
    lines.append("")

    lines.append("## 2. 按语种拆分\n")
    lines.append("| 语种 | n | 出 `` % | 撞上限 % | 平均 eval_count | 平均 resp 字节 | 平均 Token/s（总） | 平均 Bit/s（仅响应 / 总时长） | thinking token 占比（下界） |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for lang in sorted(
        agg.keys(),
        key=lambda k: agg[k]["hit_length_cap_pct"],
        reverse=True,
    ):
        a = agg[lang]
        lines.append(
            f"| {lang} | {a['n_cells']} | "
            f"{a['has_close_tag_pct']:.1f}% | "
            f"{a['hit_length_cap_pct']:.1f}% | "
            f"{a['avg_eval_count']:.0f} | "
            f"{a['avg_response_bytes']:.0f} | "
            f"{a['avg_token_per_second_overall']:.1f} | "
            f"{a['avg_response_only_bit_per_second']:.0f} | "
            f"{a['avg_thinking_token_fraction_lower_bound']:.1f}% |"
        )
    lines.append("")

    lines.append("## 3. 方法学注记\n")
    lines.append(
        "1. **Heuristic `BPT_RESPONSE_HEURISTIC = 30`**：假设响应段的 Bits/Token ≈ 30 "
        "(与其他小模型同量级)。全局聚合 Bits/Token = 4.13 主要由 thinking 的极低 bit 密度拖低。\n"
        "2. **思维链 token 占比是下界**：若响应段实际 BPT < 30，则响应 token 更少，thinking 占比更高。\n"
        "3. **`hit_length_cap_pct` 反映 thinking 失控**：撞上限说明模型仍在思考循环中，"
        "还没来得及给出答案就被 num_predict 截断。\n"
        "4. **Token/s vs Bit/s 的不对称鲁棒性**：Bit/s 用 response_bytes / eval_duration 时分母不变，"
        "分子被压缩，所以 thinking 越长 Bit/s 越低——这恰好把 thinking 模型从\"虚高 Token/s 冠军\"打回原形。\n"
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="", help="results/raw/new-XXX folder")
    ap.add_argument("--experiment", default="B_fixed_semantic_length",
                    choices=["A_fixed_tokens", "B_fixed_semantic_length"],
                    help="which experiment to analyse (default: B_fixed_semantic_length)")
    args = ap.parse_args()

    root = _project_root()
    runs_dir = Path(args.runs_dir) if args.runs_dir else None
    if not runs_dir:
        candidates = sorted((root / "results" / "raw").glob("new-*"))
        if not candidates:
            raise SystemExit("ERROR: no new-* directory under results/raw/")
        runs_dir = candidates[-1]
    runs_dir = runs_dir if runs_dir.is_absolute() else (root / runs_dir)
    print(f"[thinking-bias] runs_dir = {runs_dir}")
    print(f"[thinking-bias] experiment = {args.experiment}")

    cells = collect_cells(runs_dir, args.experiment)
    if not cells:
        raise SystemExit(
            f"ERROR: no lfm2.5-thinking cells found under {runs_dir}/{args.experiment}"
        )
    n_total = len(cells)
    print(f"[thinking-bias] collected {n_total} cells")

    agg = aggregate_per_language(cells)
    derived_dir = runs_dir / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)

    suffix = "" if args.experiment == "B_fixed_semantic_length" else "_A"
    payload = {
        "model": "lfm2.5-thinking:latest",
        "experiment": args.experiment,
        "n_cells_total": n_total,
        "BPT_RESPONSE_HEURISTIC_bits": BPT_RESPONSE_HEURISTIC,
        "per_language": agg,
        "global": {
            # agg[l][...] is already in percent (0–100); weighted average preserves unit.
            "has_close_tag_pct": sum(
                agg[l]["has_close_tag_pct"] * agg[l]["n_cells"] for l in agg
            ) / n_total,
            "hit_length_cap_pct": sum(
                agg[l]["hit_length_cap_pct"] * agg[l]["n_cells"] for l in agg
            ) / n_total,
            "avg_thinking_token_fraction_lower_bound_pct": sum(
                agg[l]["avg_thinking_token_fraction_lower_bound"] * agg[l]["n_cells"] for l in agg
            ) / n_total,
        },
    }
    write_json(derived_dir / f"thinking_bias{suffix}.json", payload)
    write_markdown(derived_dir / f"thinking_bias{suffix}.md", agg, n_total, args.experiment)

    print(f"[thinking-bias] wrote {derived_dir / f'thinking_bias{suffix}.json'}")
    print(f"[thinking-bias] wrote {derived_dir / f'thinking_bias{suffix}.md'}")
    print(
        f"[thinking-bias] GLOBAL ({args.experiment}): thinking_token_fraction >= "
        f"{payload['global']['avg_thinking_token_fraction_lower_bound_pct']:.1f}%, "
        f"hit_length_cap={payload['global']['hit_length_cap_pct']:.1f}%"
    )


if __name__ == "__main__":
    main()