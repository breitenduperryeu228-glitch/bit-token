#!/usr/bin/env bash
# resume.sh — Cross-machine one-click resume of the v2 benchmark.
#
# Resumes the partially-complete run in `results/raw/new-20260906-202656/`.
# Because run_benchmark.py defaults to `--resume` (skips cells whose raw JSON
# already exists), this continues from exactly where the previous machine
# stopped, without re-running the 73,511 completed cells.
#
# Usage:
#   ./scripts/resume.sh
#
# Requirements (must hold on the new machine BEFORE running this):
#   * ollama daemon running (`ollama list` shows the 6 models)
#   * Python deps installed (pyyaml scipy psutil pynvml ollama pandas ...)
#   * the whole project directory copied over (including results/)
#
# What this script does:
#   1. Sanity-checks ollama is up and the 6 models are present.
#   2. Sanity-checks the data directory exists and counts completed cells.
#   3. Launches the benchmark in the background with the SAME out-dir
#      so `raw_exists()` picks up all completed cells.
#   4. Prints how to monitor.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# This machine sets ALL_PROXY=socks://127.0.0.1:7897/, which the `ollama`
# Python client (httpx) cannot parse and crashes on import. ollama is served
# on localhost (already in no_proxy), so unset the socks proxy vars.
unset ALL_PROXY all_proxy 2>/dev/null || true

OUT_DIR="new-20260906-202656"
RUNS_DIR="results/raw/$OUT_DIR"

MODELS=(
  "granite4:350m-h"
  "LiquidAI/lfm2.5-350m:latest"
  "qwen3:0.6b"
  "qwen3.5:0.8b"
  "llama3.2:1b"
  "lfm2.5-thinking:latest"
)

echo "=== resume.sh — preflight checks ==="

# 1. Ollama up?
if ! curl -s http://localhost:11434/api/ps >/dev/null 2>&1; then
    echo "ERROR: ollama daemon not reachable at http://localhost:11434"
    echo "       start it first:  ollama serve  (or systemctl start ollama)"
    exit 1
fi
echo "  [ok] ollama daemon reachable"

# 2. Models present?
#    NOTE: `grep -qF` inside a `set -o pipefail` pipeline mis-reports models as
#    missing because grep -q exits early and SIGPIPEs the `ollama list` producer.
#    Read the list once into a variable instead (no pipe, no SIGPIPE).
LIST="$(ollama list 2>/dev/null || true)"
MISSING=()
for m in "${MODELS[@]}"; do
    if ! grep -qF "$m" <<<"$LIST"; then
        MISSING+=("$m")
    fi
done
if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "ERROR: missing models — pull them first:"
    for m in "${MISSING[@]}"; do
        echo "         ollama pull \"$m\""
    done
    exit 1
fi
echo "  [ok] all 6 models present"

# 3. Data directory present?
if [[ ! -d "$RUNS_DIR" ]]; then
    echo "ERROR: $RUNS_DIR not found."
    echo "       Did you copy the whole project directory? The resume data lives at"
    echo "       $RUNS_DIR — it must be present for resume to skip completed cells."
    exit 1
fi
DONE=$(find "$RUNS_DIR" -name '*.json' 2>/dev/null | wc -l)
echo "  [ok] $RUNS_DIR present — $DONE cells already on disk (will be skipped)"

# 4. Single-process lock free?
if [[ -f .run.pid ]]; then
    OLD_PID="$(cat .run.pid)"
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "ERROR: benchmark already running (PID=$OLD_PID). Stop it first."
        exit 1
    fi
    rm -f .run.pid
fi
echo "  [ok] no running benchmark, lock free"

# 5. Launch
echo ""
echo "=== launching resume (both phases, out-dir=$OUT_DIR) ==="
setsid nohup python3 -m scripts.run_benchmark \
    --phase both \
    --out-dir "$OUT_DIR" \
    --log logs/run.log \
    > logs/resume.log 2>&1 < /dev/null &
PID=$!
disown "$PID"
echo "$PID" > .run.pid

echo ""
echo "  launched PID=$PID"
echo ""
echo "  Monitor progress:"
echo "    python3 -m scripts.watch_progress --runs-dir $RUNS_DIR"
echo "    tail -F logs/run.log"
echo ""
echo "  Stop:"
echo "    ./scripts/stop_benchmark.sh"
echo ""
echo "  Remaining work: ~22,499 cells (B_fixed_semantic_length:"
echo "    qwen3.5 6499 + llama3.2 8000 + lfm2.5-thinking 8000)"
