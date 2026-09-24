#!/usr/bin/env bash

set -euo pipefail

reuse_logits=false
cpu_workers=52
sampling_mode=
sampling_mode_explicit=false
fixed_site_beam_size=
composition_args=()
positional=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --reuse-logits|--skip-sample-logits)
            reuse_logits=true
            shift
            ;;
        --enforce-composition|--no-enforce-composition)
            composition_args=("$1")
            shift
            ;;
        --cpu-workers)
            cpu_workers=$2
            shift 2
            ;;
        --cpu-workers=*)
            cpu_workers=${1#*=}
            shift
            ;;
        --sampling-mode|--sample-mode)
            sampling_mode=$2
            sampling_mode_explicit=true
            shift 2
            ;;
        --sampling-mode=*|--sample-mode=*)
            sampling_mode=${1#*=}
            sampling_mode_explicit=true
            shift
            ;;
        --fixed-site-beam-size|--topn-beam-size)
            fixed_site_beam_size=$2
            shift 2
            ;;
        --fixed-site-beam-size=*|--topn-beam-size=*)
            fixed_site_beam_size=${1#*=}
            shift
            ;;
        *)
            positional+=("$1")
            shift
            ;;
    esac
done
set -- "${positional[@]}"

if [[ $# -lt 3 || $# -gt 6 ]]; then
    echo "Usage: $0 MODEL_PATH NUM_SAMPLES RESULT_NAME [FLOW_STEPS] [INPUT_CSV] [TARGET_CSV] [--sampling-mode n-shot|top-n|greedy] [--enforce-composition|--no-enforce-composition] [--reuse-logits] [--cpu-workers N] [--fixed-site-beam-size N]" >&2
    exit 2
fi

model_path=$1
num_samples=$2
result_name=$3
flow_steps=${4:-100}
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

result_dir="${run_dir}/${result_name}"
mode_suffix=""
if [[ -z "$sampling_mode" ]]; then
    if [[ "$reuse_logits" == false ]]; then
        sampling_mode=n-shot
    fi
elif [[ "$sampling_mode" == "top-n" || "$sampling_mode" == "greedy" ]]; then
    mode_suffix="-${sampling_mode}"
elif [[ "$sampling_mode" != "n-shot" ]]; then
    echo "Unsupported sampling mode: $sampling_mode" >&2
    exit 2
fi
result_stem="top-${num_samples}_flow_steps-${flow_steps}${mode_suffix}"
sample_path="${result_dir}/${result_stem}"
logits_path="${sample_path}.logits.pt"
template_csv="${result_dir}/${result_stem}.csv"
evaluation_csv="${result_dir}/${result_stem}_gwa.csv"
summary_json="${result_dir}/${result_stem}_gwa.json"

mkdir -p "$result_dir"

if [[ "$reuse_logits" == true ]]; then
    if [[ ! -f "$logits_path" ]]; then
        candidates=()
        if [[ "$sampling_mode_explicit" == true ]]; then
            mapfile -t candidates < <(
                find "$result_dir" -maxdepth 1 -type f \
                    -name "top-${num_samples}_flow_steps-*${mode_suffix}.logits.pt" \
                    -print | sort
            )
            if [[ "$sampling_mode" == "n-shot" ]]; then
                filtered_candidates=()
                for candidate in "${candidates[@]}"; do
                    if [[ "$candidate" != *-top-n.logits.pt && "$candidate" != *-greedy.logits.pt ]]; then
                        filtered_candidates+=("$candidate")
                    fi
                done
                candidates=("${filtered_candidates[@]}")
            fi
        else
            nshot_pattern="${result_dir}/top-${num_samples}_flow_steps-${flow_steps}.logits.pt"
            topn_pattern="${result_dir}/top-${num_samples}_flow_steps-${flow_steps}-top-n.logits.pt"
            greedy_pattern="${result_dir}/top-${num_samples}_flow_steps-${flow_steps}-greedy.logits.pt"
            if [[ -f "$nshot_pattern" ]]; then
                candidates+=("$nshot_pattern")
            fi
            if [[ -f "$topn_pattern" ]]; then
                candidates+=("$topn_pattern")
            fi
            if [[ -f "$greedy_pattern" ]]; then
                candidates+=("$greedy_pattern")
            fi
        fi
        if [[ ${#candidates[@]} -eq 1 ]]; then
            logits_path=${candidates[0]}
            result_stem=$(basename -- "$logits_path" .logits.pt)
            sample_path="${result_dir}/${result_stem}"
            template_csv="${result_dir}/${result_stem}.csv"
            evaluation_csv="${result_dir}/${result_stem}_gwa.csv"
            summary_json="${result_dir}/${result_stem}_gwa.json"
        elif [[ ${#candidates[@]} -eq 0 ]]; then
            echo "No saved logits found at ${logits_path}" >&2
            exit 1
        else
            echo "No exact saved logits found at ${logits_path}; multiple candidates exist:" >&2
            printf '  %s\n' "${candidates[@]}" >&2
            echo "Pass FLOW_STEPS to select one." >&2
            exit 1
        fi
    fi

    echo "[1/3] Skipping GPU flow; decoding saved logits: ${logits_path}"
    reuse_args=(
        --model_path "$model_path"
        --save_path "$sample_path"
        --num-samples "$num_samples"
        --logits_path "$logits_path"
        --cpu-workers "$cpu_workers"
        --reuse-logits
    )
    if [[ "$sampling_mode_explicit" == true ]]; then
        reuse_args+=(--sampling-mode "$sampling_mode")
    fi
    if [[ -n "$fixed_site_beam_size" ]]; then
        reuse_args+=(--fixed-site-beam-size "$fixed_site_beam_size")
    fi
    uv run python scripts/sample_wy.py "${reuse_args[@]}" "${composition_args[@]}"
else
    sampling_mode=${sampling_mode:-n-shot}
    echo "[1/3] Sampling ${num_samples} templates per target"
    sample_args=(
        --model_path "$model_path" \
        --formula_file "$input_csv" \
        --num-samples "$num_samples" \
        --batch_size 128 \
        --flow_steps "$flow_steps" \
        --cpu-workers "$cpu_workers" \
        --sampling-mode "$sampling_mode" \
        --save_path "$sample_path"
    )
    if [[ -n "$fixed_site_beam_size" ]]; then
        sample_args+=(--fixed-site-beam-size "$fixed_site_beam_size")
    fi
    uv run python scripts/sample_wy.py "${sample_args[@]}" "${composition_args[@]}"
fi

echo "[2/3] Extracting generated templates"
uv run python scripts/extract_wyckoff_samples.py \
    --input_pt "${sample_path}.pt" \
    --output_csv "$template_csv" \
    --top_k "$num_samples"

echo "[3/3] Evaluating G-W-A Top-${num_samples}"
uv run python scripts/eval_gwa.py \
    --target_path "$target_csv" \
    --gen_path "$template_csv" \
    --top_k "$num_samples" \
    --output_path "$evaluation_csv" \
    --summary_path "$summary_json"
