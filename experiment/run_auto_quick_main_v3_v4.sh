#!/usr/bin/env bash
set -Eeuo pipefail

# 自动完成 main -> ZipCache v3 -> ZipCache v4 的轻量化对比测试。
#
# 默认执行 graph-off 公平对比：
#   bash experiment/run_auto_quick_main_v3_v4.sh
#
# 如果要测试 CUDA Graph，对 main/v3/v4 使用同一个 graph 上限：
#   ENABLE_CUDA_GRAPH=1 CUDA_GRAPH_MAX_BS=16 bash experiment/run_auto_quick_main_v3_v4.sh
#
# 常用覆盖项：
#   MODEL_PATH=/path/to/model
#   V3_COMPRESSED_POOL_MB=32768
#   V4_COMPRESSED_POOL_MB=32768
#   V3_NORMAL_POOL_PAGES=32768
#   V4_NORMAL_POOL_PAGES=32768
#
# 最终结果直接保存到 experiment/logs/<时间>_<mode>/，每个版本一个独立目录。
# 每个版本目录会保存 server_command.txt、quick_test_command.txt 和本脚本快照。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/modelscope-cache/models/Qwen/Qwen3-0___6B}"
if [[ $# -ge 1 ]]; then
  MODEL_PATH="$1"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
MAIN_BRANCH="${MAIN_BRANCH:-main}"
ZIPCACHE_BRANCH="${ZIPCACHE_BRANCH:-ZipCache}"

SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
CLIENT_HOST="${CLIENT_HOST:-127.0.0.1}"
MAIN_PORT="${MAIN_PORT:-30000}"
ZIPCACHE_PORT="${ZIPCACHE_PORT:-30001}"

MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-16}"
MAX_PREFILL_LENGTH="${MAX_PREFILL_LENGTH:-4096}"
GPU_SAMPLE_INTERVAL="${GPU_SAMPLE_INTERVAL:-0.5}"
BENCH_TIMEOUT="${BENCH_TIMEOUT:-900}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-900}"

ENABLE_CUDA_GRAPH="${ENABLE_CUDA_GRAPH:-0}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-16}"

V3_NORMAL_POOL_PAGES="${V3_NORMAL_POOL_PAGES:-32768}"
V3_COMPRESSED_POOL_MB="${V3_COMPRESSED_POOL_MB:-40960}"
V4_NORMAL_POOL_PAGES="${V4_NORMAL_POOL_PAGES:-32768}"
V4_COMPRESSED_POOL_MB="${V4_COMPRESSED_POOL_MB:-40960}"

ZIPCACHE_UNIMPORTANT_RATIO="${ZIPCACHE_UNIMPORTANT_RATIO:-0.4}"
ZIPCACHE_K_IMPORTANT_BIT="${ZIPCACHE_K_IMPORTANT_BIT:-4}"
ZIPCACHE_K_UNIMPORTANT_BIT="${ZIPCACHE_K_UNIMPORTANT_BIT:-2}"
ZIPCACHE_V_IMPORTANT_BIT="${ZIPCACHE_V_IMPORTANT_BIT:-4}"
ZIPCACHE_V_UNIMPORTANT_BIT="${ZIPCACHE_V_UNIMPORTANT_BIT:-2}"
ZIPCACHE_MIN_RESTORE_TOKENS="${ZIPCACHE_MIN_RESTORE_TOKENS:-0}"
ZIPCACHE_STATS_INTERVAL="${ZIPCACHE_STATS_INTERVAL:-30}"

LOG_BASE="${LOG_BASE:-experiment/logs}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
GRAPH_LABEL="nograph"
if [[ "$ENABLE_CUDA_GRAPH" == "1" ]]; then
  GRAPH_LABEL="cg${CUDA_GRAPH_MAX_BS}"
fi
AUTO_PREFIX="${RUN_STAMP}_auto_quick_main_v3_v4_${GRAPH_LABEL}"
CONTROLLER_LOG="${CONTROLLER_LOG:-$LOG_BASE/${AUTO_PREFIX}_controller.log}"
SUMMARY_MD="${SUMMARY_MD:-$LOG_BASE/${AUTO_PREFIX}_summary.md}"
SERVER_TMP_DIR="${SERVER_TMP_DIR:-${TMPDIR:-/tmp}/${AUTO_PREFIX}_server_tmp}"

SERVER_PID=""
SERVER_USES_SETSID=0

mkdir -p "$LOG_BASE" "$SERVER_TMP_DIR"

export PYTHONPATH="$REPO_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"

log() {
  local msg="$*"
  printf '[%s] %s\n' "$(date '+%F %T')" "$msg" | tee -a "$CONTROLLER_LOG"
}

die() {
  log "ERROR: $*"
  exit 1
}

run_cmd() {
  log "+ $*"
  "$@" 2>&1 | tee -a "$CONTROLLER_LOG"
}

check_workload_files() {
  [[ -f experiment/workloads/gsm8k_public_correctness.jsonl ]] || die "缺少 GSM8K workload。"
  [[ -f experiment/workloads/longbench_long_context_pressure.jsonl ]] || die "缺少 LongBench pressure workload。"
  [[ -f experiment/workloads/public_shared_prefix.jsonl ]] || die "缺少 shared-prefix workload。"
}

check_prerequisites() {
  git rev-parse --show-toplevel >/dev/null 2>&1 || die "当前目录不是 git 仓库。"
  [[ -d python/minisgl ]] || die "未找到 python/minisgl，请在 miniSGLang 仓库根目录运行。"
  [[ -f experiment/run_all_experiments.py ]] || die "未找到 experiment/run_all_experiments.py。"
  check_workload_files

  if [[ "$MODEL_PATH" == /* && ! -e "$MODEL_PATH" ]]; then
    die "MODEL_PATH 是绝对路径但不存在：$MODEL_PATH"
  fi

  git show-ref --verify --quiet "refs/heads/$MAIN_BRANCH" || die "本地不存在分支：$MAIN_BRANCH"
  git show-ref --verify --quiet "refs/heads/$ZIPCACHE_BRANCH" || die "本地不存在分支：$ZIPCACHE_BRANCH"

  if [[ "${ALLOW_DIRTY:-0}" != "1" ]]; then
    git diff --quiet || die "当前存在未提交的 tracked 文件修改。请先提交/暂存，或设置 ALLOW_DIRTY=1。"
    git diff --cached --quiet || die "当前存在 staged 修改。请先提交/取消暂存，或设置 ALLOW_DIRTY=1。"
  fi
}

switch_branch() {
  local branch="$1"
  log "Switching to branch: $branch"
  run_cmd git switch "$branch"
  log "Current commit: $(git rev-parse --short HEAD) ($(git branch --show-current))"
}

wait_for_server() {
  local base_url="$1"
  local pid="$2"
  local server_log="$3"
  local deadline=$((SECONDS + SERVER_START_TIMEOUT))

  log "Waiting for server: $base_url/v1"
  while (( SECONDS < deadline )); do
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      tail -n 120 "$server_log" | tee -a "$CONTROLLER_LOG" || true
      die "server 进程提前退出，日志见：$server_log"
    fi

    if "$PYTHON_BIN" - "$base_url/v1" >/dev/null 2>&1 <<'PY'
import sys
import urllib.request

url = sys.argv[1]
with urllib.request.urlopen(url, timeout=3) as resp:
    resp.read(64)
PY
    then
      log "Server is ready: $base_url"
      return 0
    fi
    sleep 5
  done

  tail -n 120 "$server_log" | tee -a "$CONTROLLER_LOG" || true
  die "等待 server 启动超时：$base_url"
}

stop_server() {
  if [[ -z "$SERVER_PID" ]]; then
    return 0
  fi

  log "Stopping server pid=$SERVER_PID"
  if [[ "$SERVER_USES_SETSID" == "1" ]]; then
    kill -TERM "-$SERVER_PID" >/dev/null 2>&1 || true
  else
    kill -TERM "$SERVER_PID" >/dev/null 2>&1 || true
  fi

  for _ in $(seq 1 30); do
    if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
      wait "$SERVER_PID" >/dev/null 2>&1 || true
      SERVER_PID=""
      return 0
    fi
    sleep 1
  done

  log "Server did not stop after SIGTERM, sending SIGKILL."
  if [[ "$SERVER_USES_SETSID" == "1" ]]; then
    kill -KILL "-$SERVER_PID" >/dev/null 2>&1 || true
  else
    kill -KILL "$SERVER_PID" >/dev/null 2>&1 || true
  fi
  wait "$SERVER_PID" >/dev/null 2>&1 || true
  SERVER_PID=""
}

on_exit() {
  local status=$?
  stop_server
  if [[ $status -eq 0 ]]; then
    log "Auto quick comparison finished successfully. Summary: $SUMMARY_MD"
  else
    log "Auto quick comparison failed with status=$status. Controller log: $CONTROLLER_LOG"
  fi
  exit "$status"
}
trap on_exit EXIT

graph_args() {
  if [[ "$ENABLE_CUDA_GRAPH" == "1" ]]; then
    printf '%s\n' "--cuda-graph-max-bs" "$CUDA_GRAPH_MAX_BS"
  else
    printf '%s\n' "--cuda-graph-max-bs" "0"
  fi
}

zipcache_graph_args() {
  if [[ "$ENABLE_CUDA_GRAPH" == "1" ]]; then
    printf '%s\n' "--enable-zipcache-cuda-graph" "--cuda-graph-max-bs" "$CUDA_GRAPH_MAX_BS"
  else
    printf '%s\n' "--cuda-graph-max-bs" "0"
  fi
}

start_server() {
  local mode="$1"
  local port="$2"
  local server_log="$3"
  shift 3

  local -a cmd=(
    "$PYTHON_BIN" -m minisgl
    --model-path "$MODEL_PATH"
    --host "$SERVER_HOST"
    --port "$port"
    --cache-type radix
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --max-prefill-length "$MAX_PREFILL_LENGTH"
  )
  cmd+=("$@")

  log "Starting $mode server on port $port"
  printf '%q ' "${cmd[@]}" > "$SERVER_TMP_DIR/${mode}_server_command.txt"
  printf '\n' >> "$SERVER_TMP_DIR/${mode}_server_command.txt"

  if command -v setsid >/dev/null 2>&1; then
    SERVER_USES_SETSID=1
    setsid "${cmd[@]}" >"$server_log" 2>&1 &
  else
    SERVER_USES_SETSID=0
    "${cmd[@]}" >"$server_log" 2>&1 &
  fi
  SERVER_PID=$!
  echo "$SERVER_PID" > "$SERVER_TMP_DIR/${mode}_server.pid"
  wait_for_server "http://$CLIENT_HOST:$port" "$SERVER_PID" "$server_log"
}

latest_report_for_mode() {
  local mode="$1"
  local latest=""
  local d
  for d in "$LOG_BASE"/*_"$mode"; do
    if [[ -d "$d" ]]; then
      latest="$d"
    fi
  done
  if [[ -n "$latest" && -f "$latest/report.md" ]]; then
    printf '%s\n' "$latest/report.md"
  fi
}

run_quick_tests() {
  local mode="$1"
  local port="$2"
  local server_log="$3"
  local parse_zipcache="$4"

  local -a cmd=(
    "$PYTHON_BIN" experiment/run_all_experiments.py
    --mode "$mode"
    --base-url "http://$CLIENT_HOST:$port"
    --log-root "$LOG_BASE"
    --preset quick
    --gpu-sample-interval "$GPU_SAMPLE_INTERVAL"
    --timeout "$BENCH_TIMEOUT"
  )
  if [[ "$parse_zipcache" == "1" ]]; then
    cmd+=(--server-log "$server_log")
  fi
  if [[ -n "${MAX_SAMPLES:-}" ]]; then
    cmd+=(--max-samples "$MAX_SAMPLES")
  fi

  printf '%q ' "${cmd[@]}" > "$SERVER_TMP_DIR/${mode}_quick_test_command.txt"
  printf '\n' >> "$SERVER_TMP_DIR/${mode}_quick_test_command.txt"

  log "Running quick experiments: $mode"
  run_cmd "${cmd[@]}"

  local report
  report="$(latest_report_for_mode "$mode" || true)"
  if [[ -n "$report" ]]; then
    local run_dir
    run_dir="$(dirname "$report")"
    cp "$server_log" "$run_dir/${mode}_server.log"
    cp "$SERVER_TMP_DIR/${mode}_server_command.txt" "$run_dir/server_command.txt"
    cp "$SERVER_TMP_DIR/${mode}_quick_test_command.txt" "$run_dir/quick_test_command.txt"
    cp "$SERVER_TMP_DIR/${mode}_server.pid" "$run_dir/server.pid" 2>/dev/null || true
    cp "${BASH_SOURCE[0]}" "$run_dir/auto_quick_launcher.sh"
    {
      printf 'MODEL_PATH=%q\n' "$MODEL_PATH"
      printf 'PYTHON_BIN=%q\n' "$PYTHON_BIN"
      printf 'MAIN_BRANCH=%q\n' "$MAIN_BRANCH"
      printf 'ZIPCACHE_BRANCH=%q\n' "$ZIPCACHE_BRANCH"
      printf 'ENABLE_CUDA_GRAPH=%q\n' "$ENABLE_CUDA_GRAPH"
      printf 'CUDA_GRAPH_MAX_BS=%q\n' "$CUDA_GRAPH_MAX_BS"
      printf 'V3_NORMAL_POOL_PAGES=%q\n' "$V3_NORMAL_POOL_PAGES"
      printf 'V3_COMPRESSED_POOL_MB=%q\n' "$V3_COMPRESSED_POOL_MB"
      printf 'V4_NORMAL_POOL_PAGES=%q\n' "$V4_NORMAL_POOL_PAGES"
      printf 'V4_COMPRESSED_POOL_MB=%q\n' "$V4_COMPRESSED_POOL_MB"
      printf 'MAX_RUNNING_REQUESTS=%q\n' "$MAX_RUNNING_REQUESTS"
      printf 'MAX_PREFILL_LENGTH=%q\n' "$MAX_PREFILL_LENGTH"
      printf 'GPU_SAMPLE_INTERVAL=%q\n' "$GPU_SAMPLE_INTERVAL"
      printf 'BENCH_TIMEOUT=%q\n' "$BENCH_TIMEOUT"
    } > "$run_dir/auto_quick_env.txt"
    {
      printf -- '- `%s`: `%s`\n' "$mode" "$report"
      printf '  - server log: `%s`\n' "$run_dir/${mode}_server.log"
      printf '  - server command: `%s`\n' "$run_dir/server_command.txt"
      printf '  - test command: `%s`\n' "$run_dir/quick_test_command.txt"
      printf '  - launcher snapshot: `%s`\n' "$run_dir/auto_quick_launcher.sh"
    } >> "$SUMMARY_MD"
  else
    printf -- '- `%s`: report not found, server log `%s`\n' "$mode" "$server_log" >> "$SUMMARY_MD"
  fi
}

run_stage() {
  local branch="$1"
  local mode="$2"
  local port="$3"
  local parse_zipcache="$4"
  shift 4

  switch_branch "$branch"
  [[ -f experiment/run_all_experiments.py ]] || die "分支 $branch 缺少 experiment/run_all_experiments.py"
  check_workload_files

  local server_log="$SERVER_TMP_DIR/${mode}_server.log"
  start_server "$mode" "$port" "$server_log" "$@"
  run_quick_tests "$mode" "$port" "$server_log" "$parse_zipcache"
  stop_server
}

main() {
  check_prerequisites

  {
    printf '# miniSGLang auto quick main/v3/v4 comparison\n\n'
    printf -- '- started_at: `%s`\n' "$(date '+%F %T')"
    printf -- '- model_path: `%s`\n' "$MODEL_PATH"
    printf -- '- graph: `%s`\n' "$GRAPH_LABEL"
    printf -- '- log_root: `%s`\n' "$LOG_BASE"
    printf -- '- temporary_server_log_dir: `%s`\n\n' "$SERVER_TMP_DIR"
    printf '## Reports\n\n'
  } > "$SUMMARY_MD"

  local -a main_extra
  mapfile -t main_extra < <(graph_args)

  local -a zip_graph_extra
  mapfile -t zip_graph_extra < <(zipcache_graph_args)

  local main_mode="main_quick_${GRAPH_LABEL}"
  local v3_mode="zipcache_v3_quick_${GRAPH_LABEL}"
  local v4_mode="zipcache_v4_quick_${GRAPH_LABEL}"

  run_stage "$MAIN_BRANCH" "$main_mode" "$MAIN_PORT" 0 "${main_extra[@]}"

  local -a v3_extra=(
    --enable-zipcache-v3
    --zipcache-v3-normal-pool-pages "$V3_NORMAL_POOL_PAGES"
    --zipcache-v3-compressed-pool-mb "$V3_COMPRESSED_POOL_MB"
    --zipcache-unimportant-ratio "$ZIPCACHE_UNIMPORTANT_RATIO"
    --zipcache-k-important-bit "$ZIPCACHE_K_IMPORTANT_BIT"
    --zipcache-k-unimportant-bit "$ZIPCACHE_K_UNIMPORTANT_BIT"
    --zipcache-v-important-bit "$ZIPCACHE_V_IMPORTANT_BIT"
    --zipcache-v-unimportant-bit "$ZIPCACHE_V_UNIMPORTANT_BIT"
    --zipcache-v3-min-restore-tokens "$ZIPCACHE_MIN_RESTORE_TOKENS"
    --zipcache-stats-interval "$ZIPCACHE_STATS_INTERVAL"
  )
  v3_extra+=("${zip_graph_extra[@]}")
  run_stage "$ZIPCACHE_BRANCH" "$v3_mode" "$ZIPCACHE_PORT" 1 "${v3_extra[@]}"

  local -a v4_extra=(
    --enable-zipcache-v4
    --zipcache-v4-normal-pool-pages "$V4_NORMAL_POOL_PAGES"
    --zipcache-v4-compressed-pool-mb "$V4_COMPRESSED_POOL_MB"
    --zipcache-v4-use-kernel-compress
    --zipcache-v4-use-kernel-restore
    --zipcache-unimportant-ratio "$ZIPCACHE_UNIMPORTANT_RATIO"
    --zipcache-k-important-bit "$ZIPCACHE_K_IMPORTANT_BIT"
    --zipcache-k-unimportant-bit "$ZIPCACHE_K_UNIMPORTANT_BIT"
    --zipcache-v-important-bit "$ZIPCACHE_V_IMPORTANT_BIT"
    --zipcache-v-unimportant-bit "$ZIPCACHE_V_UNIMPORTANT_BIT"
    --zipcache-v4-min-restore-tokens "$ZIPCACHE_MIN_RESTORE_TOKENS"
    --zipcache-stats-interval "$ZIPCACHE_STATS_INTERVAL"
  )
  v4_extra+=("${zip_graph_extra[@]}")
  run_stage "$ZIPCACHE_BRANCH" "$v4_mode" "$ZIPCACHE_PORT" 1 "${v4_extra[@]}"

  log "All reports are listed in: $SUMMARY_MD"
  log "The final git branch is: $(git branch --show-current)"
}

main "$@"
