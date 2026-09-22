#!/usr/bin/env bash
# stop_benchmark.sh — Politely stop the benchmark started by
# start_benchmark.sh.  Sends SIGTERM and waits up to 30 s for the
# orchestrator to finish unloading the current model, then SIGKILL
# if needed. Cleans up .run.pid.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PIDFILE=".run.pid"
if [[ ! -f "$PIDFILE" ]]; then
    echo "[stop] no .run.pid file — benchmark was not started by start_benchmark.sh"
    exit 0
fi

PID="$(cat "$PIDFILE")"
if ! kill -0 "$PID" 2>/dev/null; then
    echo "[stop] PID $PID not running (stale pidfile)"
    rm -f "$PIDFILE"
    exit 0
fi

echo "[stop] SIGTERM -> $PID (waiting up to 30s for graceful unload)"
kill -TERM "$PID" 2>/dev/null || true
for i in $(seq 1 30); do
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "[stop] stopped after ${i}s"
        rm -f "$PIDFILE"
        exit 0
    fi
    sleep 1
done

echo "[stop] SIGKILL -> $PID"
kill -9 "$PID" 2>/dev/null || true
rm -f "$PIDFILE"
echo "[stop] killed"