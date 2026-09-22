#!/usr/bin/env bash
# start_benchmark.sh — Launch the orchestrated v2 benchmark.
#
# What this differs from v1:
#   * Defaults to a fresh `new-YYYYMMDD-HHMMSS/` folder under results/raw
#     so this run does NOT clobber any earlier (polluted) data.
#   * Lets you pick a phase: A_fixed_tokens, B_fixed_semantic_length, or
#     both. "both" runs ALL models through A first, then ALL through B.
#   * Models are sorted by parameter count (smallest first) at runtime.
#   * Single-process lock is enforced inside run_benchmark.py (PID file
#     at the project root). A second invocation refuses to start.
#
# Usage:
#   ./scripts/start_benchmark.sh
#       # both phases, full matrix, default new-<date> out-dir
#
#   ./scripts/start_benchmark.sh --phase A
#       # only A_fixed_tokens across all models
#
#   ./scripts/start_benchmark.sh --phase both --out-dir new-20260906-1
#       # explicit out-dir (useful when you want two runs in different folders)
#
#   ./scripts/start_benchmark.sh --samples 3 --repeats 2 --phase A
#       # tiny matrix smoke test
#
#   ./scripts/start_benchmark.sh --status
#       # snapshot then exit (delegates to watch_progress.py)
#
#   ./scripts/start_benchmark.sh --no-resume
#       # wipe and restart from scratch (rarely needed; resume is default)
#
# The Python entry owns the PID lock. The shell wrapper only adds the
# background-launch convenience (nohup, setsid, log redirection) so you can
# `tail -F logs/run.log` while the benchmark runs unattended.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# This machine sets ALL_PROXY=socks://127.0.0.1:7897/, which the `ollama`
# Python client (httpx) cannot parse and crashes on import. ollama is served
# on localhost (already in no_proxy), so unset the socks proxy vars.
unset ALL_PROXY all_proxy 2>/dev/null || true

if [[ "${1:-}" == "--status" ]]; then
    shift
    exec python3 -m scripts.watch_progress "$@"
fi

EXTRA_ARGS=()
PHASE="both"
OUT_DIR=""
LOG_FILE="logs/run.log"

# Snapshot $@ into an array so that `shift` inside the loop doesn't leave
# stale args for later iterations.
ARGS=("$@")
i=0
while [[ $i -lt ${#ARGS[@]} ]]; do
    arg="${ARGS[$i]}"
    case "$arg" in
        --phase)        PHASE="${ARGS[$((i+1))]:-both}"; EXTRA_ARGS+=("--phase" "$PHASE"); i=$((i+2)) ;;
        --out-dir)      OUT_DIR="${ARGS[$((i+1))]:-}";    EXTRA_ARGS+=("--out-dir" "$OUT_DIR"); i=$((i+2)) ;;
        --log)          LOG_FILE="${ARGS[$((i+1))]:-logs/run.log}"; EXTRA_ARGS+=("--log" "$LOG_FILE"); i=$((i+2)) ;;
        --status|--help|-h)
            echo "Usage: $0 [--phase A|B|both] [--out-dir NAME] [--log FILE] [--samples N] [--repeats N] [--models m1,m2] [--languages c1,c2] [--status]"
            exit 0 ;;
        *)              EXTRA_ARGS+=("$arg"); i=$((i+1)) ;;
    esac
done

mkdir -p logs results/raw figures

if [[ -f .run.pid ]]; then
    OLD_PID="$(cat .run.pid)"
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "[start] ERROR: benchmark already running (PID=$OLD_PID)"
        echo "[start]   stop it with: ./scripts/stop_benchmark.sh"
        exit 1
    fi
    rm -f .run.pid
fi

CMD=(python3 -m scripts.run_benchmark "${EXTRA_ARGS[@]}")
echo "[start] launching: ${CMD[*]}"
echo "[start]   logs:      $ROOT/$LOG_FILE"
echo "[start]   progress:  python3 -m scripts.watch_progress"
echo "[start]   status:    $0 --status"
echo

setsid nohup "${CMD[@]}" > "$LOG_FILE" 2>&1 < /dev/null &
PID=$!
echo "$PID" > .run.pid

echo "[start] launched PID=$PID"
echo "[start] tail -F $LOG_FILE"