#!/usr/bin/env bash
#
# ZipCache V3 vs Main — Qwen3-8B equal-memory benchmark
#
# Runs 4 experiment groups:
#   Phase 1: main baseline (shared_prefix, longbench, gsm8k)
#   Phase 2: ZipCache V3 default 20GB compressed pool (same 3 experiments)
#   Phase 3: ZipCache V3 scaling (5GB / 10GB compressed pool, shared_prefix only)
#   Phase 4: summary
#
# Usage:
#   bash experiment/scripts/run_zipcache_benchmark.sh [MODEL_PATH] [ZIPCACHE_REF]
#
# Example:
#   bash experiment/scripts/run_zipcache_benchmark.sh \
#     /root/autodl-tmp/modelscope-cache/models/Qwen/Qwen3-8B \
#     zipcache-v3-quick-results
#
# Run in background with screen:
#   screen -S zipcache_bench -L
#   conda activate minisgl-ZipCache
#   bash experiment/scripts/run_zipcache_benchmark.sh
#

set -uo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH="${1:-/root/autodl-tmp/modelscope-cache/models/Qwen/Qwen3-8B}"
ZIPCACHE_REF="${2:-zipcache-v3-quick-results}"
MAIN_BRANCH="main"

PORT_MAIN=30000
PORT_ZIPCACHE=30001
HOST="0.0.0.0"

MAX_RUNNING_REQUESTS=8
MAX_PREFILL_LENGTH=4096
MEMORY_RATIO_MAIN="0.52"
NORMAL_POOL_PAGES=35000
COMPRESSED_POOL_DEFAULT_MB=20480
COMPRESSED_POOL_SIZES_MB=(5120 10240)
STATS_INTERVAL=10

GPU_SAMPLE_INTERVAL=0.5
EXPERIMENT_TIMEOUT=1200

LOG_ROOT="experiment/logs"
BENCH_ID="bench_8b_$(date +%Y%m%d_%H%M%S)"
BENCH_DIR="${LOG_ROOT}/${BENCH_ID}"

SERVER_PID=""
CURRENT_SERVER_PORT=""
STARTING_BRANCH=""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

die() {
    log "ERROR: $*"
    cleanup_server
    if [[ -n "${STARTING_BRANCH}" ]]; then
        git checkout "${STARTING_BRANCH}" 2>/dev/null || true
    fi
    exit 1
}

cleanup_server() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        log "Stopping server (PID ${SERVER_PID})..."
        kill "${SERVER_PID}" 2>/dev/null || true
        sleep 5
        kill -9 "${SERVER_PID}" 2>/dev/null || true
        SERVER_PID=""
    fi
    # Fallback: kill anything on the ports
    if command -v fuser >/dev/null 2>&1; then
        fuser -k "${PORT_MAIN}/tcp" 2>/dev/null || true
        fuser -k "${PORT_ZIPCACHE}/tcp" 2>/dev/null || true
    fi
    sleep 3
}

wait_for_server() {
    local port="$1"
    local timeout="${2:-300}"
    local start=$(date +%s)
    log "Waiting for server on port ${port} (timeout ${timeout}s)..."
    while true; do
        if curl -sf "http://127.0.0.1:${port}/v1" 2>/dev/null | grep -q "ok"; then
            log "Server ready on port ${port}."
            return 0
        fi
        local elapsed=$(( $(date +%s) - start ))
        if (( elapsed > timeout )); then
            log "Server failed to start within ${timeout}s."
            return 1
        fi
        if [[ -n "${SERVER_PID}" ]] && ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            log "Server process died."
            return 1
        fi
        sleep 5
    done
}

start_main_server() {
    local log_file="$1"
    log "Starting main server..."
    log "  model: ${MODEL_PATH}"
    log "  memory_ratio: ${MEMORY_RATIO_MAIN}"
    log "  port: ${PORT_MAIN}"

    PYTHONPATH=python nohup python -m minisgl \
        --model-path "${MODEL_PATH}" \
        --host "${HOST}" \
        --port "${PORT_MAIN}" \
        --cache-type radix \
        --max-running-requests "${MAX_RUNNING_REQUESTS}" \
        --max-prefill-length "${MAX_PREFILL_LENGTH}" \
        --memory-ratio "${MEMORY_RATIO_MAIN}" \
        > "${log_file}" 2>&1 &
    SERVER_PID=$!
    CURRENT_SERVER_PORT="${PORT_MAIN}"
    wait_for_server "${PORT_MAIN}" 300 || die "main server failed to start"
}

start_zipcache_server() {
    local log_file="$1"
    local compressed_pool_mb="$2"
    log "Starting ZipCache V3 server..."
    log "  model: ${MODEL_PATH}"
    log "  normal_pool_pages: ${NORMAL_POOL_PAGES}"
    log "  compressed_pool_mb: ${compressed_pool_mb}"
    log "  port: ${PORT_ZIPCACHE}"

    PYTHONPATH=python nohup python -m minisgl \
        --model-path "${MODEL_PATH}" \
        --host "${HOST}" \
        --port "${PORT_ZIPCACHE}" \
        --cache-type radix \
        --max-running-requests "${MAX_RUNNING_REQUESTS}" \
        --max-prefill-length "${MAX_PREFILL_LENGTH}" \
        --enable-zipcache-v3 \
        --zipcache-v3-normal-pool-pages "${NORMAL_POOL_PAGES}" \
        --zipcache-v3-compressed-pool-mb "${compressed_pool_mb}" \
        --zipcache-stats-interval "${STATS_INTERVAL}" \
        > "${log_file}" 2>&1 &
    SERVER_PID=$!
    CURRENT_SERVER_PORT="${PORT_ZIPCACHE}"
    wait_for_server "${PORT_ZIPCACHE}" 300 || die "ZipCache server failed to start (pool=${compressed_pool_mb}MB)"
}

run_experiment() {
    local mode="$1"
    local base_url="$2"
    local server_log="$3"
    local experiment_name="$4"
    local max_samples="$5"

    log "Running experiment: ${experiment_name} (mode=${mode}, max_samples=${max_samples})"

    python experiment/run_all_experiments.py \
        --mode "${mode}" \
        --base-url "${base_url}" \
        --server-log "${server_log}" \
        --only "${experiment_name}" \
        --max-samples "${max_samples}" \
        --log-root "${LOG_ROOT}" \
        --gpu-sample-interval "${GPU_SAMPLE_INTERVAL}" \
        --timeout "${EXPERIMENT_TIMEOUT}"

    local rc=$?
    if (( rc != 0 )); then
        log "WARNING: experiment ${experiment_name} exited with code ${rc}, continuing..."
    fi
    return ${rc}
}

safe_checkout() {
    local ref="$1"
    log "Switching to ${ref}..."
    git stash --include-untracked -q 2>/dev/null || true
    if ! git checkout "${ref}" 2>/dev/null; then
        git stash pop -q 2>/dev/null || true
        die "Failed to checkout ${ref}"
    fi
    git stash pop -q 2>/dev/null || true
    log "Now on: $(git branch --show-current || git rev-parse --short HEAD)"
}

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

log "=== ZipCache V3 Benchmark — ${BENCH_ID} ==="
log "Model: ${MODEL_PATH}"
log "ZipCache ref: ${ZIPCACHE_REF}"
log "Main branch: ${MAIN_BRANCH}"
log "Output: ${BENCH_DIR}"

mkdir -p "${BENCH_DIR}"

# Check model exists
if [[ ! -d "${MODEL_PATH}" ]]; then
    die "Model path not found: ${MODEL_PATH}"
fi

# Check GPU
if ! command -v nvidia-smi >/dev/null 2>&1; then
    die "nvidia-smi not found. Are you on a GPU server?"
fi
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# Check conda env has python
if ! python -c "import torch" 2>/dev/null; then
    die "PyTorch not available. Activate minisgl-ZipCache env first: conda activate minisgl-ZipCache"
fi

# Record starting branch
STARTING_BRANCH=$(git branch --show-current 2>/dev/null || git rev-parse HEAD)
log "Starting branch/commit: ${STARTING_BRANCH}"

# Check for uncommitted changes (warn only)
if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
    log "WARNING: You have uncommitted changes. They will be stashed during branch switches."
fi

trap 'cleanup_server; [[ -n "${STARTING_BRANCH}" ]] && git checkout "${STARTING_BRANCH}" 2>/dev/null || true' EXIT

# ---------------------------------------------------------------------------
# Phase 1: Main baseline
# ---------------------------------------------------------------------------

log ""
log "=========================================="
log "PHASE 1: Main baseline (equal memory budget)"
log "=========================================="

safe_checkout "${MAIN_BRANCH}"

MAIN_SERVER_LOG="${BENCH_DIR}/main_server.log"
start_main_server "${MAIN_SERVER_LOG}"

run_experiment "main_8b_25gb" "http://127.0.0.1:${PORT_MAIN}" "${MAIN_SERVER_LOG}" \
    "public_shared_prefix_serial" "64"

run_experiment "main_8b_25gb" "http://127.0.0.1:${PORT_MAIN}" "${MAIN_SERVER_LOG}" \
    "longbench_long_context_pressure" "32"

run_experiment "main_8b_25gb" "http://127.0.0.1:${PORT_MAIN}" "${MAIN_SERVER_LOG}" \
    "gsm8k_public_correctness" "64"

cleanup_server
log "Phase 1 complete."

# ---------------------------------------------------------------------------
# Phase 2: ZipCache V3 default (20GB compressed pool)
# ---------------------------------------------------------------------------

log ""
log "=========================================="
log "PHASE 2: ZipCache V3 (20GB compressed pool)"
log "=========================================="

safe_checkout "${ZIPCACHE_REF}"

ZIPCACHE_SERVER_LOG="${BENCH_DIR}/zipcache_v3_20gb_server.log"
start_zipcache_server "${ZIPCACHE_SERVER_LOG}" "${COMPRESSED_POOL_DEFAULT_MB}"

run_experiment "zipcache_v3_8b_20gb" "http://127.0.0.1:${PORT_ZIPCACHE}" "${ZIPCACHE_SERVER_LOG}" \
    "public_shared_prefix_serial" "64"

run_experiment "zipcache_v3_8b_20gb" "http://127.0.0.1:${PORT_ZIPCACHE}" "${ZIPCACHE_SERVER_LOG}" \
    "longbench_long_context_pressure" "32"

run_experiment "zipcache_v3_8b_20gb" "http://127.0.0.1:${PORT_ZIPCACHE}" "${ZIPCACHE_SERVER_LOG}" \
    "gsm8k_public_correctness" "64"

cleanup_server
log "Phase 2 complete."

# ---------------------------------------------------------------------------
# Phase 3: ZipCache V3 scaling (smaller compressed pools)
# ---------------------------------------------------------------------------

log ""
log "=========================================="
log "PHASE 3: ZipCache V3 compressed pool scaling"
log "=========================================="

for POOL_MB in "${COMPRESSED_POOL_SIZES_MB[@]}"; do
    log ""
    log "--- Compressed pool: ${POOL_MB}MB ---"

    SCALING_SERVER_LOG="${BENCH_DIR}/zipcache_v3_${POOL_MB}mb_server.log"
    start_zipcache_server "${SCALING_SERVER_LOG}" "${POOL_MB}"

    run_experiment "zipcache_v3_8b_${POOL_MB}mb" \
        "http://127.0.0.1:${PORT_ZIPCACHE}" \
        "${SCALING_SERVER_LOG}" \
        "public_shared_prefix_serial" "64"

    cleanup_server
    log "Pool ${POOL_MB}MB test complete."
done

log "Phase 3 complete."

# ---------------------------------------------------------------------------
# Phase 4: Summary
# ---------------------------------------------------------------------------

log ""
log "=========================================="
log "PHASE 4: Summary"
log "=========================================="

SUMMARY_FILE="${BENCH_DIR}/summary.txt"
{
    echo "ZipCache V3 Benchmark Summary"
    echo "Generated: $(date)"
    echo "Model: ${MODEL_PATH}"
    echo ""
    echo "=== Main 25GB (memory_ratio=${MEMORY_RATIO_MAIN}) ==="
    echo "Server log: ${MAIN_SERVER_LOG}"
    echo ""
    echo "=== ZipCache 25GB total (${NORMAL_POOL_PAGES} pages normal + ${COMPRESSED_POOL_DEFAULT_MB}MB compressed) ==="
    echo "Server log: ${ZIPCACHE_SERVER_LOG}"
    echo ""
    echo "=== ZipCache Scaling ==="
    for POOL_MB in "${COMPRESSED_POOL_SIZES_MB[@]}"; do
        echo "  Pool ${POOL_MB}MB: ${BENCH_DIR}/zipcache_v3_${POOL_MB}mb_server.log"
    done
    echo ""
    echo "=== All experiment logs ==="
    ls -la "${LOG_ROOT}" | grep "${BENCH_ID}" || true
} > "${SUMMARY_FILE}"

log "Summary written to: ${SUMMARY_FILE}"
log ""
log "=== BENCHMARK COMPLETE ==="
log "All results in: ${BENCH_DIR}/"
log ""
log "To view results:"
log "  ls ${LOG_ROOT}/ | grep 8b"
log "  cat ${LOG_ROOT}/*/report.md"

# Return to starting branch
if [[ -n "${STARTING_BRANCH}" && "${STARTING_BRANCH}" != "$(git branch --show-current 2>/dev/null || git rev-parse --short HEAD)" ]]; then
    safe_checkout "${STARTING_BRANCH}"
fi

log "Done."
