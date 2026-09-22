#!/usr/bin/env bash
# post_run_q8.sh — Benchmark 跑完后一键出 Q8 报告 + Q4 vs Q8 对比
#
# 用法（在 benchmark/ 目录）：
#   ./scripts/post_run_q8.sh                          # 用默认的 new-q8-* 目录
#   ./scripts/post_run_q8.sh new-q8-20260918-102157   # 显式指定
#   Q4_DIR=new-20260906-202656 ./scripts/post_run_q8.sh  # 自定义 Q4 数据目录
#
# 做什么：
#   1) calculate_metrics（派生指标）
#   2) analyze_results（统计检验）
#   3) visualize（重画 fig1-fig7）
#   4) compare_q4_q8（Q4 vs Q8 对比表 + 对比图）
#   5) 打印摘要到终端

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
unset ALL_PROXY all_proxy 2>/dev/null || true

# ---- 0) 决定 Q8 out-dir ----
Q8_DIR="${1:-}"
if [[ -z "$Q8_DIR" ]]; then
    # 默认挑最新的 new-q8-* 目录
    Q8_DIR="$(ls -1dt results/raw/new-q8-* 2>/dev/null | head -1 | xargs -I{} basename {})"
    if [[ -z "$Q8_DIR" ]]; then
        echo "[post] ERROR: no new-q8-* directory under results/raw/"
        echo "[post]   pass it explicitly:  $0 <dir>"
        exit 1
    fi
fi
if [[ ! -d "results/raw/$Q8_DIR" ]]; then
    echo "[post] ERROR: results/raw/$Q8_DIR not found"
    exit 1
fi
echo "================================================================"
echo " Q8 out-dir: $Q8_DIR"
echo "================================================================"

# ---- 1) calculate_metrics ----
echo
echo "[post] step 1/4 — calculate_metrics"
python3 -m scripts.calculate_metrics --runs-dir "results/raw/$Q8_DIR" 2>&1 | tail -5

# ---- 2) analyze_results ----
echo
echo "[post] step 2/4 — analyze_results"
python3 -m scripts.analyze_results --runs-dir "results/raw/$Q8_DIR" 2>&1 | tail -5

# ---- 3) visualize ----
echo
echo "[post] step 3/4 — visualize"
python3 -m scripts.visualize --runs-dir "results/raw/$Q8_DIR" 2>&1 | tail -3
python3 -m scripts.visualize_vram --runs-dir "results/raw/$Q8_DIR" 2>&1 | tail -3 || true

# ---- 4) Q4 vs Q8 对比 ----
echo
echo "[post] step 4/4 — Q4 vs Q8 compare"
Q4_DIR="${Q4_DIR:-new-20260906-202656}"
if [[ -d "results/raw/$Q4_DIR/derived" ]]; then
    python3 -m scripts.compare_q4_q8 \
        --q4-dir "results/raw/$Q4_DIR" \
        --q8-dir "results/raw/$Q8_DIR" \
        --out-md  "results/raw/$Q8_DIR/q4_vs_q8.md" \
        --out-fig "results/raw/$Q8_DIR/q4_vs_q8.png" \
        2>&1 | tail -40
    echo
    echo "[post] written:"
    echo "   - results/raw/$Q8_DIR/q4_vs_q8.md"
    echo "   - results/raw/$Q8_DIR/q4_vs_q8.png"
else
    echo "[post] SKIP compare — results/raw/$Q4_DIR/derived not found"
fi

echo
echo "[post] done. See:"
echo "   - results/raw/$Q8_DIR/derived/all_runs.csv"
echo "   - results/raw/$Q8_DIR/derived/analysis.json"
echo "   - results/raw/$Q8_DIR/derived/text_and_perf_stats.json"
echo "   - figures/fig*.png"