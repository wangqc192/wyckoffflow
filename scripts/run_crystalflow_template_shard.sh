#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 CRYSTALFLOW_DIR" >&2
  exit 2
fi

: "${CRYSTALFLOW_PYTHON:?Set CRYSTALFLOW_PYTHON before running this script}"
: "${CRYSTALFLOW_GPU_ID:?Set CRYSTALFLOW_GPU_ID before running this script}"
: "${CRYSTALFLOW_REPO:?Set CRYSTALFLOW_REPO before running this script}"
: "${CRYSTALFLOW_CHECKPOINT:?Set CRYSTALFLOW_CHECKPOINT before running this script}"

workflow_repo=$(cd "$(dirname "$0")/.." && pwd)
crystalflow_dir=$1
mkdir -p "$crystalflow_dir"
crystalflow_dir=$(cd "$crystalflow_dir" && pwd)
input_csv="$crystalflow_dir/wyckoff_info.csv"
query_path="$crystalflow_dir/diffcsp_queries.json"
sample_path="$crystalflow_dir/samples.pt"
log_path="$crystalflow_dir/sample.log"

"$CRYSTALFLOW_PYTHON" "$workflow_repo/scripts/prepare_crystalflow_queries.py" \
  --input "$input_csv" \
  --output "$query_path" \
  >"$log_path" 2>&1

CUDA_VISIBLE_DEVICES="$CRYSTALFLOW_GPU_ID" \
OMP_NUM_THREADS=14 \
MKL_NUM_THREADS=14 \
MPLCONFIGDIR=/tmp/diffcsp-mpl \
PYTHONUNBUFFERED=1 \
/usr/bin/time -f 'WALL=%e MAXRSS_KB=%M' \
  "$CRYSTALFLOW_PYTHON" "$workflow_repo/scripts/run_crystalflow_symmetry.py" \
  --diffcsp-repo "$CRYSTALFLOW_REPO" \
  --checkpoint-dir "$CRYSTALFLOW_CHECKPOINT" \
  --selected-json "$query_path" \
  --output "$sample_path" \
  --batch-size 128 \
  --ode-int-steps 100 \
  --anneal-slope 5 \
  --device cuda \
  --overwrite \
  >>"$log_path" 2>&1
