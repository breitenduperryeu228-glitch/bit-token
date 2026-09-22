#!/usr/bin/env bash
# status_alt.sh — Print a one-shot progress snapshot of the alt benchmark
# process (the one whose PID is in `.run.alt.pid`, NOT the main `.run.pid`).
#
# Shows:
#   * PID alive or dead
#   * tail of logs/run.alt.log (last 15 lines)
#   * count of raw JSONs produced by the alt process's models
#
# Usage:
#   ./scripts/status_alt.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PIDFILE=".run.alt.pid"
LOG="logs/run.alt.log"

echo "============================================================"
echo "[status-alt] $ROOT"
if [[ -f "$PIDFILE" ]]; then
    PID="$(cat "$PIDFILE")"
    if kill -0 "$PID" 2>/dev/null; then
        echo "[status-alt] ALT PROCESS ALIVE  pid=$PID  cmdline=$(ps -p "$PID" -o args= 2>/dev/null || true)"
    else
        echo "[status-alt] ALT PROCESS DEAD    pidfile says $PID but no such process"
    fi
else
    echo "[status-alt] no .run.alt.pid — alt process never started"
fi

if [[ -f "$LOG" ]]; then
    echo "------------------------------------------------------------"
    echo "[status-alt] tail $LOG"
    echo "------------------------------------------------------------"
    tail -n 15 "$LOG" || true
fi

echo "------------------------------------------------------------"
echo "[status-alt] raw JSON counts by model (results/raw/*):"
echo "------------------------------------------------------------"
for exp in A_fixed_tokens B_fixed_semantic_length; do
    if [[ -d "results/raw/$exp" ]]; then
        echo "  [experiment $exp]"
        for m in qwen3.5_0.8b lfm2.5-thinking_latest granite4_350m-h LiquidAI_lfm2.5-350m_latest llama3.2_1b; do
            n=$(ls results/raw/$exp/ 2>/dev/null | grep -c "^${m}__" || true)
            printf "    %-40s  %4d cells\n" "$m" "$n"
        done
    fi
done
echo "============================================================"
