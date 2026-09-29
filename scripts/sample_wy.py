"""CLI for Wyckoff sampling and decoding saved final logits."""

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from models.common.checkpoint import load_model
from models.common.composition import formula_to_counts
from models.pl_models.chemical_sg import load_chemical_space_group_model
from models.pl_models.count_conserving import DEFAULT_CPU_TASK_SIZE, DEFAULT_CPU_WORKERS
from models.sampling import (
    SAMPLING_MODES,
    collect_flow_logits,
    decode_logit_batches,
    make_condition,
    predict_space_group_conditions,
)


def _decode_logits_payload(
    payload,
    *,
    cpu_workers=DEFAULT_CPU_WORKERS,
    cpu_task_size=DEFAULT_CPU_TASK_SIZE,
    sampling_mode=None,
    num_samples=None,
    enforce_composition=None,
    fixed_site_beam_size=None,
):
    """Decode cached trajectories, preserving unspecified sampling settings."""
    logit_batches = payload["logit_batches"]
    settings = dict(payload.get("args", {}))
    saved_mode = settings.get("sampling_mode") or "n-shot"
    saved_num_samples = int(settings.get("num_samples", settings.get("num_evals", 1)))
    if sampling_mode is not None and sampling_mode != saved_mode:
        raise ValueError("changing sampling_mode requires new flow trajectories")
    sampling_mode = saved_mode
    num_samples = saved_num_samples if num_samples is None else num_samples
    if sampling_mode != "top-n" and num_samples != saved_num_samples:
        raise ValueError(
            f"changing num_samples for {sampling_mode} requires new flow trajectories"
        )
    if enforce_composition is None:
        enforce_composition = settings.get(
            "enforce_composition", settings.get("count_conserving", True)
        )
    if fixed_site_beam_size is None:
        fixed_site_beam_size = settings.get(
            "fixed_site_beam_size", settings.get("topn_beam_size")
        )
    if fixed_site_beam_size is None:
        fixed_site_beam_size = max(256, 8 * num_samples)
    for old_key in ("num_evals", "count_conserving", "topn_beam_size"):
        settings.pop(old_key, None)
    settings.update(
        num_samples=num_samples,
        sampling_mode=sampling_mode,
        enforce_composition=enforce_composition,
        fixed_site_beam_size=fixed_site_beam_size,
        cpu_workers=cpu_workers,
        cpu_task_size=cpu_task_size,
    )
    max_variable_count = int(logit_batches[0][2].shape[-1] - 1) if logit_batches else 0
    started = time.time()
    result = decode_logit_batches(
        logit_batches,
        max_variable_count,
        **{
            key: settings[key]
            for key in (
                "sampling_mode",
                "num_samples",
                "enforce_composition",
                "fixed_site_beam_size",
                "cpu_workers",
                "cpu_task_size",
            )
        },
    )
    cpu_decode_time = time.time() - started
    gpu_flow_time = float(payload.get("gpu_flow_time", payload.get("time", 0)))
    output = {
        "time": gpu_flow_time + cpu_decode_time,
        "gpu_flow_time": gpu_flow_time,
        "cpu_decode_time": cpu_decode_time,
        "generated_samples": result.samples,
        "infeasible_graph_indices": result.infeasible_graph_indices,
        "args": settings,
    }
    return output, result


def decode_logits_file(logits_path, save_path, **decode_options):
    """Decode a saved logits file without loading the model or running flow."""
    payload = torch.load(logits_path, map_location="cpu", weights_only=False)
    output, result = _decode_logits_payload(payload, **decode_options)
    output["reused_logits_path"] = str(logits_path)
    torch.save(output, save_path)
    return result, output["cpu_decode_time"]


def load_formula_tabular_file(formula_file):
    with open(formula_file, "r") as f:
        line = f.readline().strip()
        if "formula" not in line:
            print("First line inferred NOT a HEADER, assume no header line")
            header = None
        else:
            header = 0
    formula_tabular = pd.read_csv(formula_file, header=header)
    if header is None:
        columns = list(formula_tabular.columns)
        print("Assume first column as formulas")
        columns[0] = "formula"
        formula_tabular.columns = columns

    formula_list = formula_tabular["formula"].tolist()
    # assume any other columns are conditions
    keys = [
        key
        for key in formula_tabular.columns
        if key not in {"formula", "num_samples", "num_evals"}
    ]
    conditions_list = list(formula_tabular[keys].T.to_dict().values())

    return formula_list, conditions_list


def main(args):
    save_path = str(args.save_path) + ".pt"
    logits_path = args.logits_path or str(args.save_path) + ".logits.pt"
    if args.reuse_logits:
        result, elapsed = decode_logits_file(
            logits_path,
            save_path,
            cpu_workers=args.cpu_workers,
            cpu_task_size=args.cpu_task_size,
            sampling_mode=args.sampling_mode,
            num_samples=args.num_samples,
            enforce_composition=args.enforce_composition,
            fixed_site_beam_size=args.fixed_site_beam_size,
        )
        print(
            f"Decoded {len(result.samples)} cached samples in {elapsed:.2f}s: {save_path}"
        )
        return

    args.num_samples = 1 if args.num_samples is None else args.num_samples
    args.sampling_mode = args.sampling_mode or "n-shot"
    args.enforce_composition = (
        True if args.enforce_composition is None else args.enforce_composition
    )
    if args.fixed_site_beam_size is None:
        args.fixed_site_beam_size = max(256, 8 * args.num_samples)
    if args.sampling_mode == "top-n" and not args.enforce_composition:
        raise ValueError("top-n sampling requires enforce_composition=True")
    model, _, _ = load_model(Path(args.model_path), load_data=False)
    if torch.cuda.is_available():
        model.to("cuda")
    if args.formula_file is not None:
        formulas, conditions = load_formula_tabular_file(args.formula_file)
    else:
        formulas, conditions = args.formula, [{}] * len(args.formula)
    started = time.time()
    if args.space_group_top_k is not None:
        if args.space_group is not None or any("space_group" in c for c in conditions):
            raise ValueError("choose either supplied or predicted space groups")
        group_model = (
            load_chemical_space_group_model(args.space_group_model_path, model.device)
            if args.space_group_model_path
            else model
        )
        records = []
        for start in range(0, len(formulas), args.batch_size):
            stop = min(start + args.batch_size, len(formulas))
            compositions = torch.stack(
                [formula_to_counts(f, model.num_elements) for f in formulas[start:stop]]
            )
            records.extend(
                predict_space_group_conditions(
                    group_model,
                    compositions,
                    [conditions[i].get("target_index", i) for i in range(start, stop)],
                    args.space_group_top_k,
                )
            )
    else:
        records = [
            make_condition(
                formula,
                condition.get("space_group", args.space_group),
                model.num_elements,
                condition.get("target_index", index),
            )
            for index, (formula, condition) in enumerate(zip(formulas, conditions))
        ]
    logit_batches = collect_flow_logits(
        DataLoader(records, batch_size=args.batch_size),
        model,
        flow_steps=args.flow_steps,
        num_samples=args.num_samples,
        sampling_mode=args.sampling_mode,
    )
    gpu_flow_time = time.time() - started
    payload = {
        "gpu_flow_time": gpu_flow_time,
        "logit_batches": logit_batches,
        "args": vars(args),
    }
    torch.save(payload, logits_path)
    output, result = _decode_logits_payload(
        payload,
        cpu_workers=args.cpu_workers,
        cpu_task_size=args.cpu_task_size,
    )
    torch.save(output, save_path)
    print(f"Saved {len(result.samples)} samples to {save_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Sample formula-conditioned Wyckoff graphs"
    )
    parser.add_argument("--model_path")
    parser.add_argument("--formula", action="append")
    parser.add_argument("--formula_file")
    parser.add_argument(
        "--num-samples",
        "--num_samples",
        "--num_evals",
        dest="num_samples",
        type=int,
        help="samples per formula/space-group condition; default 1, or cached value on reuse",
    )
    parser.add_argument("--space_group", type=int)
    parser.add_argument(
        "--space-group-model-path",
        help="optional chemical space-group checkpoint; used with --space-group-top-k",
    )
    parser.add_argument(
        "--space-group-top-k",
        type=int,
        help="predict up to K feasible space groups using a joint checkpoint; "
        "num-samples is the number of templates per selected group",
    )
    parser.add_argument(
        "--sampling-mode",
        "--sampling_mode",
        "--sample-mode",
        choices=SAMPLING_MODES,
        default=None,
        help=(
            "n-shot runs random trajectories; top-n searches final-logit candidates; "
            "greedy takes argmax at each step and decodes one sample per trajectory"
        ),
    )
    parser.add_argument(
        "--fixed-site-beam-size",
        "--topn-beam-size",
        "--topn_beam_size",
        dest="fixed_site_beam_size",
        type=int,
        help="beam width for fixed Wyckoff sites; default max(256, 8*num_samples)",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--cpu-workers", "--cpu_workers", type=int, default=DEFAULT_CPU_WORKERS
    )
    parser.add_argument(
        "--cpu-task-size", "--cpu_task_size", type=int, default=DEFAULT_CPU_TASK_SIZE
    )
    parser.add_argument(
        "--flow_steps",
        type=int,
        default=100,
        help="inference flow steps (default: 100)",
    )
    parser.add_argument(
        "--enforce-composition",
        "--count_conserving",
        dest="enforce_composition",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enforce the target composition; enabled by default, or use cached value on reuse",
    )
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--logits_path")
    parser.add_argument("--reuse-logits", "--reuse_logits", action="store_true")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if not args.reuse_logits:
        if args.model_path is None:
            parser.error("provide --model_path")
        if args.formula_file is None and not args.formula:
            parser.error("provide --formula or --formula_file")
        if args.formula and args.space_group is None and args.space_group_top_k is None:
            parser.error("provide --space_group or --space-group-top-k with --formula")
        if args.space_group_model_path and args.space_group_top_k is None:
            parser.error("--space-group-model-path requires --space-group-top-k")
    if args.formula_file is not None and args.formula:
        parser.error("--formula and --formula_file are mutually exclusive")
    main(args)
