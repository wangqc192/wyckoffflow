"""Shared Wyckoff sampling: flow trajectories followed by final decoding."""

from collections import Counter

import torch
from torch.distributions import Categorical
from torch_geometric.data import Batch, Data
from tqdm.auto import tqdm

from models.common.composition import formula_to_counts
from models.common.lookup_tables import chemical_symbols
from models.pl_models.count_conserving import (
    DEFAULT_CPU_TASK_SIZE,
    DEFAULT_CPU_WORKERS,
    DecodingResult,
    cpu_dp_pool,
    decode_composition_logits,
)
from models.pl_models.model_utils import create_x_matrix

SAMPLING_MODES = ("n-shot", "top-n", "greedy")


class SamplingData(Data):
    """A formula/space-group condition with identifiers preserved by PyG batching."""

    def __inc__(self, key, value, *args, **kwargs):
        if key in {"target_index", "sampling_group"}:
            return 0
        return super().__inc__(key, value, *args, **kwargs)


def make_condition(formula, space_group, num_elements, target_index):
    return SamplingData(
        formula=formula_to_counts(formula, num_elements).unsqueeze(0),
        space_group=torch.tensor(space_group, dtype=torch.long),
        target_index=torch.tensor(target_index, dtype=torch.long),
    )


def _num_trajectories(num_samples, sampling_mode):
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if sampling_mode not in SAMPLING_MODES:
        raise ValueError(f"unsupported sampling_mode: {sampling_mode}")
    return 1 if sampling_mode == "top-n" else num_samples


@torch.inference_mode()
def collect_flow_logits(
    loader, model, *, flow_steps, num_samples, sampling_mode="n-shot"
):
    """Collect final logits, with one or several trajectories per condition."""
    num_trajectories = _num_trajectories(num_samples, sampling_mode)
    logit_batches = []
    condition_offset = 0
    for batch in tqdm(loader, desc="Sampling logits", unit="batch"):
        trajectories = []
        for index, condition in enumerate(batch.to_data_list()):
            # Distinguish multiple space-group conditions for the same target.
            condition.sampling_group = torch.tensor(condition_offset + index)
            trajectories.extend(condition.clone() for _ in range(num_trajectories))
        condition_offset += batch.num_graphs
        logit_batches.append(
            model.sample_logits(
                Batch.from_data_list(trajectories),
                flow_steps=flow_steps,
                greedy=sampling_mode == "greedy",
            )
        )
    return logit_batches


def _infeasible_message(data, graph_index):
    counts = data.composition[graph_index].round().long().tolist()
    formula = "".join(
        chemical_symbols[element] + (str(count) if count != 1 else "")
        for element, count in enumerate(counts)
        if element > 0 and count > 0
    )
    space_group = int(data.space_group.reshape(-1)[graph_index])
    return (
        f"Skipping formula={formula}, space_group={space_group}: infeasible composition"
    )


def decode_independent_logits(data, zero_logits, inf_logits, *, greedy=False):
    """Decode each variable without enforcing the target atom counts."""
    sampled = data.clone().cpu()
    if greedy:
        sampled.x_0_dof = zero_logits.argmax(dim=-1)
        sampled.x_inf_dof = inf_logits.argmax(dim=-1)
    else:
        sampled.x_0_dof = (
            Categorical(logits=zero_logits).sample()
            if zero_logits.shape[0]
            else sampled.x_0_dof.long()
        )
        sampled.x_inf_dof = (
            Categorical(logits=inf_logits).sample()
            if inf_logits.shape[0]
            else sampled.x_inf_dof.long()
        )
    sampled.x = create_x_matrix(sampled.x_inf_dof, sampled.x_0_dof, sampled.zero_dof)
    return DecodingResult(sampled.to_data_list(), [])


def decode_logit_batches(
    logit_batches,
    max_variable_count,
    *,
    sampling_mode="n-shot",
    num_samples=1,
    enforce_composition=True,
    cpu_workers=DEFAULT_CPU_WORKERS,
    cpu_task_size=DEFAULT_CPU_TASK_SIZE,
    fixed_site_beam_size=None,
):
    """Decode final logits with a single worker pool and a fixed result type.

    In n-shot and greedy modes each stored trajectory gives one sample. Greedy
    decoding is deterministic. In top-n mode each condition has one trajectory
    and yields up to num_samples candidates.
    Failure indices refer to the concatenation of all input graph batches.
    """
    _num_trajectories(num_samples, sampling_mode)
    if sampling_mode == "top-n" and not enforce_composition:
        raise ValueError("top-n sampling requires enforce_composition=True")
    total_graphs = sum(data.num_graphs for data, _, _ in logit_batches)
    result = DecodingResult([], [])
    if total_graphs == 0:
        return result
    beam_size = (
        max(256, 8 * num_samples)
        if fixed_site_beam_size is None
        else fixed_site_beam_size
    )
    workers = cpu_workers if enforce_composition else 1
    graph_offset = 0
    with cpu_dp_pool(workers, total_graphs) as (executor, worker_count):
        with tqdm(total=total_graphs, desc="Decoding", unit="graph") as progress:
            for data, zero_logits, inf_logits in logit_batches:
                if enforce_composition:
                    decoded = decode_composition_logits(
                        data,
                        zero_logits,
                        inf_logits,
                        max_variable_count,
                        stochastic=sampling_mode == "n-shot",
                        num_candidates=(
                            num_samples if sampling_mode == "top-n" else None
                        ),
                        fixed_site_beam_size=beam_size,
                        cpu_workers=worker_count,
                        cpu_task_size=cpu_task_size,
                        executor=executor,
                        progress=progress,
                    )
                else:
                    decoded = decode_independent_logits(
                        data, zero_logits, inf_logits, greedy=sampling_mode == "greedy"
                    )
                    progress.update(data.num_graphs)
                messages = Counter(
                    _infeasible_message(data, index)
                    for index in decoded.infeasible_graph_indices
                )
                for message, count in messages.items():
                    tqdm.write(
                        message + (f" ({count} trajectories)" if count > 1 else "")
                    )
                result.samples.extend(decoded.samples)
                result.infeasible_graph_indices.extend(
                    graph_offset + index for index in decoded.infeasible_graph_indices
                )
                graph_offset += data.num_graphs
    return result


def sample_batch(
    model,
    batch,
    *,
    flow_steps,
    num_samples,
    sampling_mode="n-shot",
    enforce_composition=True,
    cpu_workers=DEFAULT_CPU_WORKERS,
    cpu_task_size=DEFAULT_CPU_TASK_SIZE,
    fixed_site_beam_size=None,
):
    logit_batches = collect_flow_logits(
        [batch],
        model,
        flow_steps=flow_steps,
        num_samples=num_samples,
        sampling_mode=sampling_mode,
    )
    return decode_logit_batches(
        logit_batches,
        model.max_num_atoms,
        sampling_mode=sampling_mode,
        num_samples=num_samples,
        enforce_composition=enforce_composition,
        cpu_workers=cpu_workers,
        cpu_task_size=cpu_task_size,
        fixed_site_beam_size=fixed_site_beam_size,
    )
