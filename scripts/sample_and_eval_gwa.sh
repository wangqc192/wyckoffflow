#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 3 || $# -gt 6 ]]; then
    echo "Usage: $0 MODEL_PATH NUM_EVALS RESULT_NAME [INPUT_CSV] [TARGET_CSV] [FLOW_STEPS]" >&2
    exit 2
fi

model_path=$1
num_evals=$2
result_name=$3
flow_steps=${4:-}
input_csv=${5:-example/input_test.csv}
target_csv=${6:-data/mp20/test.csv}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/.." && pwd)
cd "$repo_root"

if [[ -d "$model_path" ]]; then
    run_dir=${model_path%/}
elif [[ $(basename -- "$(dirname -- "$model_path")") == "checkpoints" ]]; then
    run_dir=$(dirname -- "$(dirname -- "$model_path")")
else
    run_dir=$(dirname -- "$model_path")
fi

if [[ -z "$flow_steps" ]]; then
    effective_flow_steps=$(uv run python - "$run_dir/hparams.yaml" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
print(int(config["model"]["model_config"]["flow_steps"]))
PY
    )
else
    effective_flow_steps=$flow_steps
fi

result_dir="${run_dir}/${result_name}"
result_stem="top-${num_evals}_flow_steps-${effective_flow_steps}"
sample_path="${result_dir}/${result_stem}"
template_csv="${result_dir}/${result_stem}.csv"
evaluation_csv="${result_dir}/${result_stem}_gwa.csv"
summary_json="${result_dir}/${result_stem}_gwa.json"

mkdir -p "$result_dir"

echo "[1/3] Sampling ${num_evals} templates per target"
sample_args=(
    --model_path "$model_path" \
    --formula_file "$input_csv" \
    --num_evals "$num_evals" \
    --batch_size 128 \
    --save_path "$sample_path"
)
if [[ -n "$flow_steps" ]]; then
    sample_args+=(--flow_steps "$flow_steps")
fi
uv run python scripts/sample_wy.py "${sample_args[@]}"

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
