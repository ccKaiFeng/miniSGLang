#!/usr/bin/env bash
# Usage: bash experiment/scripts/run_zipcache_benchmark.sh [MODEL_PATH] [ZIPCACHE_REF]
# Omit ZIPCACHE_REF to test the current working tree, including uncommitted fixes.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python -u "${SCRIPT_DIR}/run_zipcache_benchmark.py" "$@"
