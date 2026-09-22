#!/usr/bin/env bash
# q8_run.sh — Qwen3-0.6B Q8_0 专属 benchmark 一键脚本
#
# 设计原则：
#   1. 永远不会触碰 results/raw/new-20260906-202656/（原始 6 模型数据）
#   2. 只在 results/raw/new-q8-*/ 下写新数据
#   3. 默认 --resume（断点续跑），已完成 cell 自动跳过
#   4. 幂等：重复跑不会污染、不会覆盖已有数据
#
# 用法（都在 benchmark/ 目录下跑）：
#   ./scripts/q8_run.sh                # 自动：若无 new-q8-* 则新建；否则 resume 最新
#   ./scripts/q8_run.sh fresh          # 强制新建（用最新 new-q8-<时间戳> 目录）
#   ./scripts/q8_run.sh resume [DIR]   # 显式 resume（DIR 可省，默认 latest new-q8-*）
#   ./scripts/q8_run.sh status         # 看当前进度
#   ./scripts/q8_run.sh stop           # 停掉
#   ./scripts/q8_run.sh report [DIR]   # benchmark 完成后跑 metrics+visualize+对比
#
# 监控：
#   python3 -m scripts.watch_progress --runs-dir results/raw/<DIR>
#   tail -F logs/run_q8.log

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
unset ALL_PROXY all_proxy 2>/dev/null || true

Q8_MODEL="qwen3:0.6b-q8"
RAW_ROOT="results/raw"
PROTECTED_DIR="new-20260906-202656"   # 原始 6 模型数据，绝对不能动
LOG_FILE="logs/run_q8.log"
PID_FILE=".run.pid"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
die()  { echo "[q8] ERROR: $*" >&2; exit 1; }
info() { echo "[q8] $*" >&2; }

latest_q8_dir() {
    # 最新 new-q8-* 目录名（不含路径），按字典序倒排第一项
    ls -1dt "$RAW_ROOT"/new-q8-* 2>/dev/null | head -1 | xargs -I{} basename {}
}

q8_dir_path() {
    # 接受一个 DIR 参数：可以是 basename 或绝对路径；返回带前缀的路径
    local d="$1"
    [[ "$d" = /* ]] && echo "$d" || echo "$RAW_ROOT/$d"
}

preflight() {
    info "preflight checks..."

    # 1) ollama 在跑
    if ! curl -sf http://localhost:11434/api/ps >/dev/null 2>&1; then
        die "ollama daemon not reachable at http://localhost:11434"
    fi

    # 2) Q8 model 已注册（注意：不用 pipe + grep -q 避免 SIGPIPE 误报）
    local list; list="$(ollama list 2>/dev/null || true)"
    if ! grep -qF "$Q8_MODEL" <<<"$list"; then
        die "model '$Q8_MODEL' not registered.  run:
   cd '/home/hope/桌面/qwen3 0.6B q8' && ./download_qwen3_0.6b_q8.sh"
    fi
    info "  [ok] ollama up"
    info "  [ok] $Q8_MODEL registered"

    # 3) 不能在原始 6 模型数据上跑
    if [[ "${Q8_OUT_DIR:-}" == "$PROTECTED_DIR" ]]; then
        die "refusing to use --out-dir=$PROTECTED_DIR (that is the original 6-model dataset, must not be touched)"
    fi
}

lock_check() {
    if [[ -f "$PID_FILE" ]]; then
        local old; old="$(cat "$PID_FILE")"
        if kill -0 "$old" 2>/dev/null; then
            die "benchmark already running (PID=$old).  run:  $0 stop"
        fi
        rm -f "$PID_FILE"
    fi
}

# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
cmd_fresh() {
    preflight
    lock_check

    local stamp; stamp="$(date +%Y%m%d-%H%M%S)"
    Q8_OUT_DIR="new-q8-${stamp}"

    info "============================================================"
    info " launching FRESH Q8 run"
    info "   model : $Q8_MODEL"
    info "   out   : $RAW_ROOT/$Q8_OUT_DIR  (new dir, will not collide with anything)"
    info "   log   : $LOG_FILE"
    info "   resume: ON  (default; no completed cells yet so it's a no-op)"
    info "============================================================"

    launch
}

cmd_resume() {
    preflight
    lock_check

    local arg="${1:-}"
    if [[ -n "$arg" ]]; then
        Q8_OUT_DIR="$arg"
    else
        Q8_OUT_DIR="$(latest_q8_dir)"
    fi
    [[ -z "$Q8_OUT_DIR" ]] && die "no new-q8-* directory found and no DIR argument given"

    local full; full="$(q8_dir_path "$Q8_OUT_DIR")"
    [[ ! -d "$full" ]] && die "$full does not exist"

    local done; done="$(find "$full" -name '*.json' 2>/dev/null | wc -l)"
    info "============================================================"
    info " resuming Q8 run"
    info "   model : $Q8_MODEL"
    info "   out   : $full"
    info "   already on disk: $done cells (will be skipped)"
    info "   log   : $LOG_FILE"
    info "============================================================"

    launch
}

launch() {
    # 把 --models qwen3:0.6b-q8 --phase both --out-dir Q8_OUT_DIR 透传给 run_benchmark
    # run_benchmark 默认 --resume=True，所以已完成的 cell 会自动跳过
    mkdir -p logs "$RAW_ROOT/$Q8_OUT_DIR"

    setsid nohup python3 -m scripts.run_benchmark \
        --models "$Q8_MODEL" \
        --phase  both \
        --out-dir "$Q8_OUT_DIR" \
        --log     "$LOG_FILE" \
        > "logs/launch_q8.log" 2>&1 < /dev/null &
    local pid=$!
    disown "$pid" 2>/dev/null || true
    echo "$pid" > "$PID_FILE"

    info "launched PID=$pid"
    info ""
    info "monitor:"
    info "  python3 -m scripts.watch_progress --runs-dir $RAW_ROOT/$Q8_OUT_DIR"
    info "  tail -F $LOG_FILE"
    info ""
    info "stop:    $0 stop"
    info "report:  $0 report $Q8_OUT_DIR"
}

cmd_status() {
    echo "============================================================"
    echo " Q8 benchmark status  ($(date '+%Y-%m-%d %H:%M:%S'))"
    echo "============================================================"

    # 进程
    if [[ -f "$PID_FILE" ]]; then
        local pid; pid="$(cat "$PID_FILE")"
        if kill -0 "$pid" 2>/dev/null; then
            local etime; etime="$(ps -p "$pid" -o etime= 2>/dev/null | xargs)"
            echo "  PID $pid  alive  ($etime)"
        else
            echo "  PID $pid  DEAD (stale .run.pid — clean up with: rm $PID_FILE)"
        fi
    else
        echo "  no PID file  (benchmark not running)"
    fi

    # cell 进度（find 缺失目录会返回非零，但 set -e 会杀掉整个脚本，所以手动检查）
    local d; d="$(latest_q8_dir)"
    if [[ -z "$d" ]]; then
        echo "  no new-q8-* directory yet"
        return 0
    fi
    local full="$RAW_ROOT/$d"
    if [[ ! -d "$full" ]]; then
        echo "  dir vanished: $full"
        return 0
    fi
    local a=0 b=0
    [[ -d "$full/A_fixed_tokens" ]] && a="$(find "$full/A_fixed_tokens" -name '*.json' 2>/dev/null | wc -l)"
    [[ -d "$full/B_fixed_semantic_length" ]] && b="$(find "$full/B_fixed_semantic_length" -name '*.json' 2>/dev/null | wc -l)"
    echo "  latest dir: $d"
    echo "  A cells : $a / 8000"
    echo "  B cells : $b / 8000"
    echo "  total   : $((a+b)) / 16000  ($(( (a+b)*100/16000 ))%)"
}

cmd_stop() {
    if [[ ! -f "$PID_FILE" ]]; then
        info "no .run.pid — nothing to stop"
        return 0
    fi
    local pid; pid="$(cat "$PID_FILE")"
    if kill -0 "$pid" 2>/dev/null; then
        info "killing PID $pid (SIGTERM)..."
        kill -TERM "$pid" 2>/dev/null || true
        for _ in 1 2 3 4 5; do
            sleep 2
            if ! kill -0 "$pid" 2>/dev/null; then
                info "stopped"
                rm -f "$PID_FILE"
                return 0
            fi
        done
        info "still alive after 10s, sending SIGKILL"
        kill -KILL "$pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
}

cmd_report() {
    local arg="${1:-}"
    local dir="${arg:-$(latest_q8_dir)}"
    [[ -z "$dir" ]] && die "no new-q8-* directory found and no DIR argument given"

    info "running post-run analysis on $dir..."
    "$ROOT/scripts/post_run_q8.sh" "$dir"
}

# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
cmd="${1:-auto}"
shift || true

case "$cmd" in
    fresh)   cmd_fresh  "$@" ;;
    resume)  cmd_resume "$@" ;;
    status)  cmd_status  ;;
    stop)    cmd_stop    ;;
    report)  cmd_report "$@" ;;
    auto|"")
        # 自动：若无 new-q8-* 则 fresh；否则 resume latest
        if [[ -z "$(latest_q8_dir)" ]]; then
            cmd_fresh
        else
            cmd_resume
        fi
        ;;
    -h|--help|help)
        sed -n '2,30p' "$0"
        ;;
    *)
        die "unknown command: $cmd  (run '$0 help' for usage)"
        ;;
esac