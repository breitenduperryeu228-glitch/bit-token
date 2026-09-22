#!/usr/bin/env bash
# =============================================================================
# monitor.sh — 每小时刷新的基准测试进度监控
#
# 在终端里运行本脚本，会立即打印一份详细进度报告，之后每隔 INTERVAL 秒
# （默认 3600 = 1 小时）自动刷新一次。每份快照同时追加到
#   benchmark/logs/progress_history.md
# 方便回看历史。
#
# 用法：
#   ./monitor.sh              # 每小时刷新（默认）
#   ./monitor.sh 600          # 每 10 分钟刷新
#   ./monitor.sh --once       # 只打印一次后退出
#   ./monitor.sh 3600 --once  # 同上
#   Ctrl-C 退出
#
# 可选：指定数据目录
#   RUNS_DIR=results/raw/new-20260906-202656 ./monitor.sh
# =============================================================================

set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
BENCH="$ROOT/benchmark"
HIST="$BENCH/logs/progress_history.md"

# ollama 的 python 客户端在本机 socks:// 代理下会崩溃；本脚本不依赖它，
# 但顺手清掉以防万一。
unset ALL_PROXY all_proxy 2>/dev/null || true

INTERVAL=3600
ONCE=0
RUNS_DIR="${RUNS_DIR:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --once)   ONCE=1; shift ;;
        --help|-h)
            sed -n '2,30p' "$0"; exit 0 ;;
        ''|*[!0-9]*)
            echo "未知参数: $1" >&2
            echo "用法: $0 [秒数] [--once]" >&2
            exit 1 ;;
        *)        INTERVAL="$1"; shift ;;
    esac
done

if [[ ! -d "$BENCH" ]]; then
    echo "错误: 找不到 benchmark 目录 ($BENCH)" >&2
    exit 1
fi

PY="python3"
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "错误: 找不到 python3" >&2
    exit 1
fi

render() {
    local args=()
    [[ -n "$RUNS_DIR" ]] && args+=(--runs-dir "$RUNS_DIR")
    (
        cd "$BENCH" || exit 1
        "$PY" -m scripts.progress_report "${args[@]}"
    )
}

human_interval() {
    local s="$1"
    if (( s % 3600 == 0 )); then echo "$((s / 3600)) 小时"
    elif (( s % 60 == 0 )); then echo "$((s / 60)) 分钟"
    else echo "$s 秒"; fi
}

if [[ "$ONCE" == "1" ]]; then
    render
    exit 0
fi

# 循环刷新
trap 'echo; echo "[monitor] 已退出"; exit 0' INT TERM
while true; do
    # 仅在交互式终端里清屏
    if [[ -t 1 ]]; then clear; fi
    echo "刷新时间: $(date '+%Y-%m-%d %H:%M:%S')   每 $(human_interval "$INTERVAL") 刷新一次（Ctrl-C 退出）"
    echo "================================================================================"
    snap="$(render)"
    printf '%s\n' "$snap"

    # 追加历史快照
    mkdir -p "$(dirname "$HIST")"
    {
        printf '\n---\n<!-- snapshot %s -->\n' "$(date '+%F %T')"
        printf '%s\n' "$snap"
    } >> "$HIST"

    sleep "$INTERVAL"
done
