#!/usr/bin/env bash
# stop_alt.sh — Stop the *alternative* benchmark process started by the same
# kind of invocation as `start_benchmark.sh`, but which writes its PID to
# `.run.alt.pid` instead of `.run.pid` (so it does NOT collide with the
# main `.run.pid` process). Sends SIGTERM, waits up to 5 s, then SIGKILL.
#
# Usage:
#   ./scripts/stop_alt.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PIDFILE=".run.alt.pid"
if [[ ! -f "$PIDFILE" ]]; then
    echo "[stop-alt] no .run.alt.pid — alt process was not started (or already stopped)"
    exit 0
fi

PID="$(cat "$PIDFILE")"
if ! kill -0 "$PID" 2>/dev/null; then
    echo "[stop-alt] PID $PID not running (stale pidfile)"
    rm -f "$PIDFILE"
    exit 0
fi

echo "[stop-alt] SIGTERM -> $PID"
kill "$PID" 2>/dev/null || true
for i in 1 2 3 4 5; do
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "[stop-alt] stopped (graceful)"
        rm -f "$PIDFILE"
        exit 0
    fi
    sleep 1
done

echo "[stop-alt] SIGKILL -> $PID"
kill -9 "$PID" 2>/dev/null || true
rm -f "$PIDFILE"
echo "[stop-alt] killed"
