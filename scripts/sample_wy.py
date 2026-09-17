import argparse
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from models.common.checkpoint import load_model
from models.common.composition import formula_to_counts
from models.common.lookup_tables import chemical_symbols
from models.pl_models.count_conserving import (
    DEFAULT_CPU_TASK_SIZE,
    DEFAULT_CPU_WORKERS,
    cpu_dp_pool,
    cpu_dp_worker_count,
    sample_batch_to_compositions,
)


class SamplingData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "target_index":
            return 0
        return super().__inc__(key, value, *args, **kwargs)


def _format_formula(counts):
    counts = counts.detach().cpu().reshape(-1)
    return "".join(
        symbol if count == 1 else f"{symbol}{count}"
        for atomic_number, (symbol, count) in enumerate(
            zip(
                chemical_symbols,
                (int(round(float(value))) for value in counts.tolist()),
            )
        )
        if atomic_number > 0 and count > 0
    )


def _sample_or_skip(batch, model, count_conserving, flow_steps):
    try:
        return model.sample(
            batch,
            count_conserving=count_conserving,
            flow_steps=flow_steps,
        ).to_data_list()
    except ValueError as error:
        if str(error) != "Space group cannot realize the requested composition":
            raise
        if isinstance(batch, Batch) and batch.num_graphs > 1:
            generated_samples = []
            for target in batch.to_data_list():
                generated_samples.extend(
                    _sample_or_skip(target, model, count_conserving, flow_steps)
                )
            return generated_samples

        formula = _format_formula(batch.formula)
        space_group = int(batch.space_group.reshape(-1)[0])
        tqdm.write(f"Skipping formula={formula}, space_group={space_group}: {error}")
        return []


def _sample_logits_or_skip(batch, model, flow_steps):
    try:
        data, zero_logits, inf_logits = model.sample_logits(
            batch,
            flow_steps=flow_steps,
        )
        return [(data.cpu(), zero_logits, inf_logits)]
    except ValueError as error:
        if str(error) != "Space group cannot realize the requested composition":
            raise
        if isinstance(batch, Batch) and batch.num_graphs > 1:
            generated_batches = []
            for target in batch.to_data_list():
                generated_batches.extend(
                    _sample_logits_or_skip(target, model, flow_steps)
                )
            return generated_batches

        formula = _format_formula(batch.formula)
        space_group = int(batch.space_group.reshape(-1)[0])
        tqdm.write(f"Skipping formula={formula}, space_group={space_group}: {error}")
        return []


@torch.inference_mode()
def flow_logits(loader, model, flow_steps=None):
    """Run GPU flow for all batches and retain only their final decoder logits."""
    logit_batches = []
    with tqdm(
        total=len(loader.dataset), desc="Sampling logits", unit="target"
    ) as progress:
        for batch in loader:
            logit_batches.extend(_sample_logits_or_skip(batch, model, flow_steps))
            progress.update(batch.num_graphs)
    return logit_batches


def _infeasible_graph_message(data, graph_index):
    formula = _format_formula(data.composition[graph_index])
    space_group = int(data.space_group.reshape(-1)[graph_index])
    return (
        f"Skipping formula={formula}, space_group={space_group}: "
        "Space group cannot realize the requested composition"
    )


def _repair_logit_batch(
    logit_batch,
    max_variable_count,
    cpu_workers,
    cpu_task_size,
    executor=None,
    progress=None,
    candidate_count=None,
    zero_beam_size=256,
):
    data, zero_logits, inf_logits = logit_batch
    sampled, _, infeasible_graph_indices = sample_batch_to_compositions(
        data,
        zero_logits,
        inf_logits,
        max_variable_count,
        zero_beam_size=zero_beam_size,
        stochastic=candidate_count is None,
        cpu_workers=cpu_workers,
        cpu_task_size=cpu_task_size,
        executor=executor,
        progress=progress,
        announce=False,
        return_infeasible=True,
        candidate_count=candidate_count,
    )
    skipped_messages = Counter(
        _infeasible_graph_message(data, graph_index)
        for graph_index in infeasible_graph_indices
    )
    for message, count in skipped_messages.items():
        suffix = "" if count == 1 else f" ({count} samples)"
        tqdm.write(message + suffix)
    if candidate_count is not None:
        return sampled
    infeasible = set(infeasible_graph_indices)
    return [
        sample
        for graph_index, sample in enumerate(sampled.to_data_list())
        if graph_index not in infeasible
    ]


def repair_saved_logits(
    logit_batches,
    max_variable_count,
    cpu_workers=DEFAULT_CPU_WORKERS,
    cpu_task_size=DEFAULT_CPU_TASK_SIZE,
    candidate_count=None,
    zero_beam_size=256,
):
    """Decode saved final logits by batch with one shared CPU worker pool."""
    batch_graph_counts = [data.num_graphs for data, _, _ in logit_batches]
    total_graphs = sum(batch_graph_counts)
    if total_graphs == 0:
        return []

    worker_count = cpu_dp_worker_count(cpu_workers, total_graphs)
    print(
        f"[count_conserving] repair {total_graphs}/{total_graphs} graphs "
        f"with CPU DP using {worker_count} workers"
    )
    print(
        f"[count_conserving] {len(batch_graph_counts)} saved batches sent to repair; "
        f"graphs/batch min={min(batch_graph_counts)}, max={max(batch_graph_counts)}, "
        f"last={batch_graph_counts[-1]}, task_size={cpu_task_size}"
    )
    effective_beam_size = (
        max(256, 8 * (candidate_count or 1))
        if zero_beam_size is None
        else int(zero_beam_size)
    )
    generated_samples = []
    with cpu_dp_pool(worker_count, total_graphs) as (executor, _):
        with tqdm(total=total_graphs, desc="CPU DP", unit="graph") as progress:
            for logit_batch in logit_batches:
                generated_samples.extend(
                    _repair_logit_batch(
                        logit_batch,
                        max_variable_count,
                        worker_count,
                        cpu_task_size,
                        executor,
                        progress,
                        candidate_count,
                        effective_beam_size,
                    )
                )
    if candidate_count is not None:
        target_ids = []
        for batch_index, (data, _, _) in enumerate(logit_batches):
            if hasattr(data, "target_index"):
                target_ids.extend(
                    int(value) for value in data.target_index.reshape(-1).tolist()
                )
            else:
                start = sum(batch_graph_counts[:batch_index])
                target_ids.extend(range(start, start + data.num_graphs))
        target_ids = list(dict.fromkeys(target_ids))
        generated_by_target = Counter(
            int(sample.target_index.reshape(-1)[0])
            for sample in generated_samples
            if hasattr(sample, "target_index")
        )
        requested_targets = len(target_ids)
        reached = sum(
            generated_by_target.get(target_index, 0) >= candidate_count
            for target_index in target_ids
        )
        print(
            f"[top-n] unique templates per target: "
            f"{candidate_count} requested; {reached}/{requested_targets} targets reached; "
            f"generated {len(generated_samples)} unique templates"
        )
        insufficient = [
            (target_index, generated_by_target.get(target_index, 0))
            for target_index in target_ids
            if generated_by_target.get(target_index, 0) < candidate_count
        ]
        if insufficient:
            print(
                f"[top-n] {len(insufficient)} targets returned fewer than requested "
                "unique templates"
            )
    return generated_samples


def repair_logits_file(
    logits_path,
    save_path,
    cpu_workers=DEFAULT_CPU_WORKERS,
    cpu_task_size=DEFAULT_CPU_TASK_SIZE,
    sampling_mode=None,
    num_evals=None,
    zero_beam_size=None,
):
    """Decode a saved final-logits payload without rerunning GPU flow."""
    saved_logits = torch.load(logits_path, map_location="cpu", weights_only=False)
    logit_batches = saved_logits["logit_batches"]
    if not logit_batches:
        max_variable_count = 0
    else:
        max_variable_count = int(logit_batches[0][2].shape[-1] - 1)

    saved_args = saved_logits.get("args", {})
    saved_mode = saved_args.get("sampling_mode") or "n-shot"
    if sampling_mode is not None and sampling_mode != saved_mode:
        raise ValueError(
            f"saved logits use sampling mode {saved_mode!r}, but "
            f"{sampling_mode!r} was requested; pass the matching mode or use "
            "--reuse-logits without --sampling-mode"
        )
    effective_mode = sampling_mode or saved_mode
    effective_num_evals = int(
        num_evals if num_evals is not None else saved_args.get("num_evals", 1)
    )
    candidate_count = effective_num_evals if effective_mode == "top-n" else None
    saved_beam_size = saved_args.get("topn_beam_size")
    if zero_beam_size is None:
        zero_beam_size = saved_beam_size
    effective_beam_size = int(
        zero_beam_size
        if zero_beam_size is not None
        else max(256, 8 * (candidate_count or 1))
    )
    repair_start_time = time.time()
    generated_samples = repair_saved_logits(
        logit_batches,
        max_variable_count,
        cpu_workers,
        cpu_task_size,
        candidate_count,
        effective_beam_size,
    )
    cpu_dp_time = time.time() - repair_start_time
    gpu_flow_time = float(
        saved_logits.get("gpu_flow_time", saved_logits.get("time", 0))
    )
    output = {
        "time": gpu_flow_time + cpu_dp_time,
        "gpu_flow_time": gpu_flow_time,
        "cpu_dp_time": cpu_dp_time,
        "generated_samples": generated_samples,
        "args": saved_args,
        "reused_logits_path": str(logits_path),
    }
    torch.save(output, save_path)
    return generated_samples, cpu_dp_time


def flow(loader, model, count_conserving=True, flow_steps=None):
    """Generate Wyckoff graphs for every formula in ``loader``."""
    generated_samples = []
    with tqdm(total=len(loader.dataset), desc="Sampling", unit="target") as progress:
        for batch in loader:
            generated_samples.extend(
                _sample_or_skip(batch, model, count_conserving, flow_steps)
            )
            progress.update(batch.num_graphs)
    return generated_samples


class SampleDataset(Dataset):
    def __init__(
        self,
        formula,
        num_evals,
        space_group=None,
        num_elements=118,
        target_index=0,
    ):
        super().__init__()
        self.num_evals = int(num_evals)
        self.space_group = int(space_group)
        self.formula = formula_to_counts(formula, num_elements)
        self.target_index = target_index

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return SamplingData(
            space_group=torch.tensor(self.space_group, dtype=torch.long),
            num_evals=torch.tensor(self.num_evals, dtype=torch.long),
            formula=self.formula.unsqueeze(0).clone(),
            target_index=torch.tensor(self.target_index, dtype=torch.long),
        )


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
        key for key in formula_tabular.columns if key not in {"formula", "num_evals"}
    ]
    conditions_list = list(formula_tabular[keys].T.to_dict().values())

    return formula_list, conditions_list


def main(args):
    save_path = str(args.save_path) + ".pt"
    logits_path = args.logits_path or str(args.save_path) + ".logits.pt"
    if args.reuse_logits:
        print(f"Reusing saved final logits from: {logits_path}")
        generated_samples, cpu_dp_time = repair_logits_file(
            logits_path,
            save_path,
            args.cpu_workers,
            args.cpu_task_size,
            args.sampling_mode,
            args.num_evals,
            args.topn_beam_size,
        )
        print("CPU DP time:", cpu_dp_time)
        print(f"Saved {len(generated_samples)} repaired samples to: {save_path}")
        return

    args.sampling_mode = args.sampling_mode or "n-shot"

    print("Loading model...")
    model_path = Path(args.model_path)
    model, _, cfg = load_model(model_path, load_data=False)
    num_elements = int(cfg.model.model_config.num_elements)
    if args.flow_steps is None:
        args.flow_steps = int(cfg.model.model_config.flow_steps)

    if args.formula_file is not None:
        print(f"Trying reading sampling formulas from '{args.formula_file}'...")
        formula_list, conditions_list = load_formula_tabular_file(args.formula_file)
    else:
        formula_list = args.formula
        conditions_list = [{}] * len(formula_list)

    if torch.cuda.is_available():
        model.to("cuda")

    if args.sampling_mode == "top-n" and not args.count_conserving:
        raise ValueError("top-n sampling requires count_conserving=True")
    flow_num_evals = 1 if args.sampling_mode == "top-n" else args.num_evals
    test_set = ConcatDataset(
        SampleDataset(
            formula,
            flow_num_evals,
            space_group=conditions.get("space_group", args.space_group),
            num_elements=num_elements,
            target_index=conditions.get("target_index", target_index),
        )
        for target_index, (formula, conditions) in enumerate(
            zip(formula_list, conditions_list)
        )
    )
    print(test_set[0])
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    start_time = time.time()
    if args.count_conserving:
        logit_batches = flow_logits(
            test_loader,
            model,
            flow_steps=args.flow_steps,
        )
        logits_stop_time = time.time()
        print("GPU flow time:", logits_stop_time - start_time)

        # Persist all batches before starting CPU decoding.  This keeps the GPU
        # generation phase independent from the CPU-parallel dynamic program.
        torch.save(
            {
                "time": logits_stop_time - start_time,
                "logit_batches": logit_batches,
                "args": vars(args),
            },
            logits_path,
        )
        print(f"Saved all final logits to: {logits_path}")
        repair_start_time = time.time()
        generated_samples = repair_saved_logits(
            logit_batches,
            model.max_num_atoms,
            args.cpu_workers,
            args.cpu_task_size,
            args.num_evals if args.sampling_mode == "top-n" else None,
            args.topn_beam_size,
        )
        stop_time = time.time()
        print("CPU DP time:", stop_time - repair_start_time)
        output = {
            "time": stop_time - start_time,
            "gpu_flow_time": logits_stop_time - start_time,
            "cpu_dp_time": stop_time - repair_start_time,
            "generated_samples": generated_samples,
            "args": vars(args),
        }
    else:
        generated_samples = flow(
            test_loader,
            model,
            count_conserving=False,
            flow_steps=args.flow_steps,
        )
        stop_time = time.time()
        print("Model time:", stop_time - start_time)
        output = {
            "time": stop_time - start_time,
            "generated_samples": generated_samples,
            "args": vars(args),
        }

    torch.save(output, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample formula-conditioned Wyckoff graphs"
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--formula", action="append")
    parser.add_argument("--formula_file")
    parser.add_argument(
        "--num_evals",
        type=int,
        default=1,
        help="number of samples to generate for every formula",
    )
    parser.add_argument("--space_group", type=int)
    parser.add_argument(
        "--sampling_mode",
        "--sampling-mode",
        "--sample-mode",
        choices=("n-shot", "top-n"),
        default=None,
        help="n-shot samples with replacement; top-n returns distinct DP candidates",
    )
    parser.add_argument(
        "--topn_beam_size",
        "--topn-beam-size",
        type=int,
        default=None,
        help="fixed-site DP beam for top-n; defaults to max(256, 8*num_evals)",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--cpu_workers",
        "--cpu-workers",
        type=int,
        default=DEFAULT_CPU_WORKERS,
        help="CPU DP worker processes (default: 52)",
    )
    parser.add_argument(
        "--cpu_task_size",
        "--cpu-task-size",
        type=int,
        default=DEFAULT_CPU_TASK_SIZE,
        help="graphs decoded sequentially in each CPU pool task (default: 32)",
    )
    parser.add_argument(
        "--flow_steps",
        type=int,
        help="number of inference flow steps; defaults to the checkpoint value",
    )
    parser.add_argument(
        "--count_conserving",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enforce the conditioned composition",
    )
    parser.add_argument("--save_path", required=True)
    parser.add_argument(
        "--logits_path",
        help="intermediate file for all final logits; defaults to <save_path>.logits.pt",
    )
    parser.add_argument(
        "--reuse_logits",
        "--reuse-logits",
        action="store_true",
        help="skip GPU flow and decode an existing --logits_path on CPU",
    )
    args = parser.parse_args()
    if not args.reuse_logits and args.formula_file is None and not args.formula:
        parser.error("provide --formula or --formula_file")
    if args.formula_file is not None and args.formula:
        parser.error("--formula and --formula_file are mutually exclusive")
    main(args)
