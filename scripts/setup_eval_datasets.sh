#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ $# -gt 0 ]]; then
  ROOT="$1"
  shift
else
  ROOT="./experiment"
fi

python -m pip install -U "datasets==2.19.2" "huggingface_hub<1.0" "pandas<3" pyarrow tqdm html2text
python scripts/download_eval_datasets.py --root "${ROOT}" "$@"
