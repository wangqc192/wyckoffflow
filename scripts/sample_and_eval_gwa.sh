#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 3 || $# -gt 5 ]]; then
    echo "Usage: $0 MODEL_PATH NUM_EVALS RESULT_NAME [INPUT_CSV] [TARGET_CSV]" >&2
    exit 2
fi

model_path=$1
num_evals=$2
result_name=$3
input_csv=${4:-example/input_test.csv}
target_csv=${5:-data/mp20/test.csv}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/.." && pwd)
cd "$repo_root"

result_dir="results/${result_name}"
sample_path="${result_dir}/top-${num_evals}"
template_csv="${result_dir}/top-${num_evals}.csv"
evaluation_csv="${result_dir}/top-${num_evals}_gwa.csv"
summary_json="${result_dir}/top-${num_evals}_gwa.json"

mkdir -p "$result_dir"

echo "[1/3] Sampling ${num_evals} templates per target"
uv run python scripts/sample_wy.py \
    --model_path "$model_path" \
    --formula_file "$input_csv" \
    --num_evals "$num_evals" \
    --batch_size 128 \
    --save_path "$sample_path"

echo "[2/3] Extracting generated templates"
uv run python scripts/extract_wyckoff_samples.py \
    --input_pt "${sample_path}.pt" \
    --output_csv "$template_csv" \
    --top_k "$num_evals"

echo "[3/3] Evaluating G-W-A Top-${num_evals}"
uv run python scripts/eval_gwa.py \
    --target_path "$target_csv" \
    --gen_path "$template_csv" \
    --top_k "$num_evals" \
    --output_path "$evaluation_csv" \
    --summary_path "$summary_json"
