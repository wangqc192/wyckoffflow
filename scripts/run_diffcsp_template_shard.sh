#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "Usage: $0 PYTHON GPU_ID DIFFCSP_WORKTREE MODEL_PATH INPUT_CSV SAVE_PATH LOG_PATH" >&2
  exit 2
fi

python_bin=$1
gpu_id=$2
diffcsp_worktree=$3
model_path=$4
input_csv=$5
save_path=$6
log_path=$7

mkdir -p "$(dirname "$save_path")" "$(dirname "$log_path")"
cd "$diffcsp_worktree"

CUDA_VISIBLE_DEVICES="$gpu_id" \
OMP_NUM_THREADS=14 \
MKL_NUM_THREADS=14 \
MPLCONFIGDIR=/tmp/diffcsp-mpl \
PYTHONUNBUFFERED=1 \
/usr/bin/time -f 'WALL=%e MAXRSS_KB=%M' \
  "$python_bin" scripts/sample.py \
  --model_path "$model_path" \
  -N 100 \
  --anneal_coords \
  --anneal_slope 5 \
  --batch_size 50 \
  -F "$input_csv" \
  -d "$save_path" \
  >"$log_path" 2>&1
