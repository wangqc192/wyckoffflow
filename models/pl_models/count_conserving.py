"""Exact composition constraints for formula-conditioned decoding."""

import math
import multiprocessing
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from functools import lru_cache

import torch
from wyckoff_generation.common.composition import decode_composition

from ..common import lookup_tables
from .model_utils import create_x_matrix

DEFAULT_CPU_WORKERS = 52
DEFAULT_CPU_TASK_SIZE = 32


def _space_group_multiplicities(space_group):
    key = str(int(space_group))
    multiplicities = reversed(lookup_tables.spg_wyckoff_multiplicities[key].values())
    dofs = reversed(lookup_tables.spg_wyckoff_degrees_of_freedom[key].values())
    grouped = ([], [])
    for multiplicity, dof in zip(multiplicities, dofs):
        grouped[dof != 0].append(int(multiplicity))
    return tuple(grouped[0]), tuple(grouped[1])


@lru_cache(maxsize=None)
def _reachable_counts(multiplicities, max_count, limit):
    reachable = [False] * (limit + 1)
    reachable[0] = True
    for multiplicity in multiplicities:
        updated = [False] * (limit + 1)
        for subtotal, valid in enumerate(reachable):
            if valid:
                for count in range(
                    min(max_count, (limit - subtotal) // multiplicity) + 1
                ):
                    updated[subtotal + count * multiplicity] = True
        reachable = updated
    return tuple(reachable)


def _fixed_assignment(targets, multiplicities, reachable):
    @lru_cache(maxsize=None)
    def search(position, remaining):
        if all(reachable[count] for count in remaining):
            return (-1,) * (len(multiplicities) - position)
        if position == len(multiplicities):
            return None

        result = search(position + 1, remaining)
        if result is not None:
            return (-1,) + result

        multiplicity = multiplicities[position]
        for element, count in enumerate(remaining):
            if count >= multiplicity:
                next_remaining = list(remaining)
                next_remaining[element] -= multiplicity
                result = search(position + 1, tuple(next_remaining))
                if result is not None:
                    return (element,) + result
        return None

    return search(0, tuple(targets))


@lru_cache(maxsize=65536)
def formula_supported_by_space_group(space_group, positive_counts, max_variable_count):
    counts = tuple(sorted(int(count) for count in positive_counts if count > 0))
    if not counts:
        return False
    fixed, variable = _space_group_multiplicities(space_group)
    reachable = _reachable_counts(variable, max_variable_count, max(counts))
    return _fixed_assignment(counts, fixed, reachable) is not None


@lru_cache(maxsize=8192)
def _feasible_space_groups(positive_counts, max_variable_count):
    return (False,) + tuple(
        formula_supported_by_space_group(
            space_group, positive_counts, max_variable_count
        )
        for space_group in range(1, 231)
    )


def formula_space_group_mask(compositions, max_variable_count):
    """Return the feasible space groups for each composition."""

    masks = []
    for composition in compositions.detach().cpu().round():
        counts = tuple(
            sorted(int(count) for count in composition[1:].tolist() if count > 0)
        )
        masks.append(_feasible_space_groups(counts, max_variable_count))
    return torch.tensor(masks, dtype=torch.bool, device=compositions.device)


def _gumbel_like(values):
    uniform = torch.rand_like(values).clamp_(1e-8, 1 - 1e-8)
    return -torch.log(-torch.log(uniform))


def _variable_count_dp(scores, multiplicities, target):
    layers = [{0: (0.0, -1, -1)}]
    for position, multiplicity in enumerate(multiplicities):
        current = {}
        for subtotal, (base_score, _, _) in layers[-1].items():
            max_count = min(
                scores.shape[1] - 1,
                (target - subtotal) // multiplicity,
            )
            for count in range(max_count + 1):
                total = subtotal + count * multiplicity
                score = base_score + float(scores[position, count])
                if total not in current or score > current[total][0]:
                    current[total] = (score, subtotal, count)
        layers.append(current)
    return layers


def _backtrack_counts(layers, target):
    counts = []
    for layer in reversed(layers[1:]):
        _, target, count = layer[target]
        counts.append(count)
    return list(reversed(counts))


def _variable_count_topk(scores, multiplicities, target, candidate_count):
    """Return the best distinct count assignments for one element.

    A variable-DoF Wyckoff position can contain more than one copy of the same
    element.  The ordinary decoder only keeps the best assignment for each
    subtotal.  Top-N decoding keeps a short list instead, which is enough to
    enumerate the best complete composition-conserving templates without
    sampling the same template again.
    """

    layers = [{0: [(0.0, ())]}]
    for position, multiplicity in enumerate(multiplicities):
        current = {}
        for subtotal, paths in layers[-1].items():
            max_count = min(
                scores.shape[1] - 1,
                (target - subtotal) // multiplicity,
            )
            for base_score, base_path in paths:
                for count in range(max_count + 1):
                    total = subtotal + count * multiplicity
                    current.setdefault(total, []).append(
                        (
                            base_score + float(scores[position, count]),
                            base_path + (count,),
                        )
                    )
        for subtotal, paths in current.items():
            paths.sort(key=lambda item: item[0], reverse=True)
            current[subtotal] = paths[:candidate_count]
        layers.append(current)
    return layers[-1]


def _decode_single_candidates_with_beam(
    zero_logits,
    inf_logits,
    zero_multiplicities,
    inf_multiplicities,
    target_composition,
    max_variable_count,
    beam_size,
    candidate_count,
):
    """Decode the highest-scoring distinct exact-composition assignments."""

    target_composition = target_composition.round().long()
    elements = torch.nonzero(target_composition[1:] > 0).flatten() + 1
    targets = tuple(int(target_composition[element]) for element in elements.tolist())
    zero_columns = torch.cat((torch.zeros(1, dtype=torch.long), elements))
    zero_scores = zero_logits[:, zero_columns]
    inf_scores = inf_logits[:, elements - 1]

    reachable = _reachable_counts(
        tuple(inf_multiplicities), max_variable_count, max(targets)
    )
    fallback = _fixed_assignment(targets, tuple(zero_multiplicities), reachable)
    if fallback is None:
        raise ValueError("Space group cannot realize the requested composition")

    variable_paths = [
        _variable_count_topk(
            inf_scores[:, index],
            inf_multiplicities,
            target,
            candidate_count,
        )
        for index, target in enumerate(targets)
    ]

    # Keep the best candidate_count fixed-site paths for every used-count
    # vector.  beam_size limits the number of vectors, as in the n-shot DP.
    states = {tuple(0 for _ in targets): [(0.0, ())]}
    fallback_used = [0] * len(targets)
    for position, multiplicity in enumerate(zero_multiplicities):
        expanded = {}
        for used, paths in states.items():
            options = [(-1, 0)] + [
                (element, element + 1)
                for element in range(len(targets))
                if used[element] + multiplicity <= targets[element]
            ]
            for base_score, base_choices in paths:
                for element, column in options:
                    next_used = list(used)
                    if element >= 0:
                        next_used[element] += multiplicity
                    next_used = tuple(next_used)
                    expanded.setdefault(next_used, []).append(
                        (
                            base_score + float(zero_scores[position, column]),
                            base_choices + (element,),
                        )
                    )

        for used, paths in expanded.items():
            paths.sort(key=lambda item: item[0], reverse=True)
            expanded[used] = paths[:candidate_count]
        if beam_size is not None and len(expanded) > beam_size:
            expanded = dict(
                sorted(
                    expanded.items(),
                    key=lambda item: item[1][0][0],
                    reverse=True,
                )[:beam_size]
            )

        fallback_element = fallback[position]
        if fallback_element >= 0:
            fallback_used[fallback_element] += multiplicity
        fallback_key = tuple(fallback_used)
        if fallback_key not in expanded:
            fallback_choices = fallback[: position + 1]
            fallback_score = sum(
                float(zero_scores[index, max(choice + 1, 0)])
                for index, choice in enumerate(fallback_choices)
            )
            expanded[fallback_key] = [(fallback_score, fallback_choices)]
        states = expanded

    complete = []
    for used, fixed_paths in states.items():
        residuals = tuple(target - subtotal for target, subtotal in zip(targets, used))
        if not all(
            residual in paths for residual, paths in zip(residuals, variable_paths)
        ):
            continue
        for fixed_score, zero_choices in fixed_paths:
            variable_combinations = [(0.0, ())]
            for paths, residual in zip(variable_paths, residuals):
                variable_combinations = [
                    (base_score + path_score, base_paths + (path_counts,))
                    for base_score, base_paths in variable_combinations
                    for path_score, path_counts in paths[residual]
                ]
                variable_combinations.sort(key=lambda item: item[0], reverse=True)
                variable_combinations = variable_combinations[:candidate_count]
            for variable_score, variable_counts in variable_combinations:
                complete.append(
                    (
                        fixed_score + variable_score,
                        zero_choices,
                        variable_counts,
                    )
                )

    complete.sort(key=lambda item: item[0], reverse=True)
    candidates = []
    seen = set()
    for _, zero_choices, variable_counts in complete:
        key = (zero_choices, variable_counts)
        if key in seen:
            continue
        seen.add(key)
        decoded_zero = torch.zeros(zero_logits.shape[0], dtype=torch.long)
        for position, element in enumerate(zero_choices):
            if element >= 0:
                decoded_zero[position] = elements[element]

        decoded_inf = torch.zeros(
            inf_logits.shape[0], inf_logits.shape[1], dtype=torch.long
        )
        for index, counts in enumerate(variable_counts):
            decoded_inf[:, elements[index] - 1] = torch.tensor(counts)
        candidates.append((decoded_zero, decoded_inf))
        if len(candidates) == candidate_count:
            break
    return candidates


def _fixed_state_count(zero_multiplicities, targets, limit):
    """Count reachable fixed-site usage states, stopping after ``limit``."""

    states = {tuple(0 for _ in targets)}
    for multiplicity in zero_multiplicities:
        expanded = set()
        for used in states:
            expanded.add(used)
            for element, target in enumerate(targets):
                if used[element] + multiplicity <= target:
                    next_used = list(used)
                    next_used[element] += multiplicity
                    expanded.add(tuple(next_used))
        if len(expanded) > limit:
            return limit + 1
        states = expanded
    return len(states)


def _decode_single_candidates(
    zero_logits,
    inf_logits,
    zero_multiplicities,
    inf_multiplicities,
    target_composition,
    max_variable_count,
    beam_size,
    candidate_count,
):
    """Return up to ``candidate_count`` distinct exact-composition candidates.

    The beam is only a performance limit.  If it hides feasible candidates, the
    beam is expanded until the requested number is found or all reachable
    fixed-site states have been searched.  Thus a small ``topn_beam_size`` can
    affect runtime, but it cannot cause an existing candidate to be duplicated
    to fill Top-N.
    """

    target_composition = target_composition.round().long()
    elements = torch.nonzero(target_composition[1:] > 0).flatten() + 1
    targets = tuple(int(target_composition[element]) for element in elements.tolist())
    beam = max(1, int(beam_size)) if beam_size is not None else None

    while True:
        candidates = _decode_single_candidates_with_beam(
            zero_logits,
            inf_logits,
            zero_multiplicities,
            inf_multiplicities,
            target_composition,
            max_variable_count,
            beam,
            candidate_count,
        )
        if len(candidates) >= candidate_count or beam is None:
            return candidates[:candidate_count]

        reachable_states = _fixed_state_count(
            zero_multiplicities,
            targets,
            beam,
        )
        if reachable_states <= beam:
            return candidates

        next_beam = max(beam + 1, beam * 2, candidate_count)
        beam = min(next_beam, reachable_states)


def _decode_single(
    zero_logits,
    inf_logits,
    zero_multiplicities,
    inf_multiplicities,
    target_composition,
    max_variable_count,
    beam_size,
    stochastic=True,
):
    target_composition = target_composition.round().long()
    elements = torch.nonzero(target_composition[1:] > 0).flatten() + 1
    targets = tuple(int(target_composition[element]) for element in elements.tolist())

    zero_columns = torch.cat((torch.zeros(1, dtype=torch.long), elements))
    zero_scores = zero_logits[:, zero_columns]
    inf_scores = inf_logits[:, elements - 1]
    if stochastic:
        zero_scores = zero_scores + _gumbel_like(zero_scores)
        inf_scores = inf_scores + _gumbel_like(inf_scores)

    reachable = _reachable_counts(
        tuple(inf_multiplicities), max_variable_count, max(targets)
    )
    fallback = _fixed_assignment(targets, tuple(zero_multiplicities), reachable)
    if fallback is None:
        raise ValueError("Space group cannot realize the requested composition")

    variable_layers = [
        _variable_count_dp(inf_scores[:, index], inf_multiplicities, target)
        for index, target in enumerate(targets)
    ]
    variable_scores = [
        {subtotal: state[0] for subtotal, state in layers[-1].items()}
        for layers in variable_layers
    ]

    states = {tuple(0 for _ in targets): (0.0, ())}
    fallback_used = [0] * len(targets)
    for position, multiplicity in enumerate(zero_multiplicities):
        expanded = {}
        for used, (score, choices) in states.items():
            options = [(-1, 0)] + [
                (element, element + 1)
                for element in range(len(targets))
                if used[element] + multiplicity <= targets[element]
            ]
            for element, column in options:
                next_used = list(used)
                if element >= 0:
                    next_used[element] += multiplicity
                next_used = tuple(next_used)
                next_score = score + float(zero_scores[position, column])
                if next_used not in expanded or next_score > expanded[next_used][0]:
                    expanded[next_used] = (
                        next_score,
                        choices + (element,),
                    )

        if len(expanded) > beam_size:
            expanded = dict(
                sorted(
                    expanded.items(),
                    key=lambda item: item[1][0],
                    reverse=True,
                )[:beam_size]
            )

        fallback_element = fallback[position]
        if fallback_element >= 0:
            fallback_used[fallback_element] += multiplicity
        fallback_key = tuple(fallback_used)
        if fallback_key not in expanded:
            fallback_choices = fallback[: position + 1]
            fallback_score = sum(
                float(zero_scores[index, max(choice + 1, 0)])
                for index, choice in enumerate(fallback_choices)
            )
            expanded[fallback_key] = (fallback_score, fallback_choices)
        states = expanded

    candidates = []
    for used, (fixed_score, choices) in states.items():
        residuals = tuple(target - subtotal for target, subtotal in zip(targets, used))
        if all(
            residual in scores for residual, scores in zip(residuals, variable_scores)
        ):
            score = fixed_score + sum(
                scores[residual] for residual, scores in zip(residuals, variable_scores)
            )
            candidates.append((score, choices, residuals))
    _, zero_choices, residuals = max(candidates, key=lambda candidate: candidate[0])

    decoded_zero = torch.zeros(zero_logits.shape[0], dtype=torch.long)
    for position, element in enumerate(zero_choices):
        if element >= 0:
            decoded_zero[position] = elements[element]

    decoded_inf = torch.zeros(
        inf_logits.shape[0], inf_logits.shape[1], dtype=torch.long
    )
    for index, residual in enumerate(residuals):
        decoded_inf[:, elements[index] - 1] = torch.tensor(
            _backtrack_counts(variable_layers[index], residual)
        )
    return decoded_zero, decoded_inf


def _decode_worker_init():
    torch.set_num_threads(1)


def cpu_dp_worker_count(cpu_workers, num_graphs):
    requested = DEFAULT_CPU_WORKERS if cpu_workers is None else int(cpu_workers)
    return min(requested, os.cpu_count() or 1, num_graphs)


@contextmanager
def cpu_dp_pool(cpu_workers, num_graphs):
    worker_count = cpu_dp_worker_count(cpu_workers, num_graphs)
    if worker_count <= 1:
        yield None, worker_count
        return

    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_decode_worker_init,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        yield executor, worker_count


def _decode_single_task(
    zero_logits,
    inf_logits,
    task,
    max_variable_count,
    zero_beam_size,
    stochastic,
    candidate_count=None,
):
    (
        graph_index,
        zero_start,
        zero_stop,
        inf_start,
        inf_stop,
        zero_multiplicities,
        inf_multiplicities,
        positive_targets,
        seed,
    ) = task
    if seed is not None:
        torch.manual_seed(seed)
    target_composition = torch.zeros(inf_logits.shape[1] + 1)
    for element, count in positive_targets:
        target_composition[element] = count
    try:
        if candidate_count is None:
            decoded = [
                _decode_single(
                    zero_logits[zero_start:zero_stop],
                    inf_logits[inf_start:inf_stop],
                    zero_multiplicities,
                    inf_multiplicities,
                    target_composition,
                    max_variable_count,
                    zero_beam_size,
                    stochastic=stochastic,
                )
            ]
        else:
            decoded = _decode_single_candidates(
                zero_logits[zero_start:zero_stop],
                inf_logits[inf_start:inf_stop],
                zero_multiplicities,
                inf_multiplicities,
                target_composition,
                max_variable_count,
                zero_beam_size,
                candidate_count,
            )
    except ValueError as error:
        if str(error) != "Space group cannot realize the requested composition":
            raise
        return graph_index, None
    return graph_index, [
        (decoded_zero.numpy(), decoded_inf.numpy())
        for decoded_zero, decoded_inf in decoded
    ]


def _decode_task_chunk(task):
    (
        zero_logits,
        inf_logits,
        graph_tasks,
        max_variable_count,
        zero_beam_size,
        stochastic,
        candidate_count,
    ) = task
    return [
        _decode_single_task(
            zero_logits,
            inf_logits,
            graph_task,
            max_variable_count,
            zero_beam_size,
            stochastic,
            candidate_count,
        )
        for graph_task in graph_tasks
    ]


def _graph_offsets(graph_assignment, num_graphs):
    counts = torch.bincount(graph_assignment, minlength=num_graphs)
    return torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(dim=0)))


def _share_for_workers(tensor, executor):
    tensor = tensor.contiguous()
    if executor is not None and not tensor.is_shared():
        tensor.share_memory_()
    return tensor


def _decode_graphs(
    data,
    zero_logits,
    inf_logits,
    graph_indices,
    max_variable_count,
    zero_beam_size,
    stochastic,
    cpu_workers=DEFAULT_CPU_WORKERS,
    cpu_task_size=DEFAULT_CPU_TASK_SIZE,
    executor=None,
    progress=None,
    candidate_count=None,
):
    if candidate_count is not None and candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    graph_indices = [int(index) for index in graph_indices]
    if not graph_indices:
        return data.x_0_dof, data.x_inf_dof, []

    targets = data.composition.detach().cpu().round().long()
    space_groups = data.space_group.detach().cpu().reshape(-1)
    positive_targets = {}
    feasible_graph_indices = []
    infeasible_graph_indices = []
    for graph_index in graph_indices:
        graph_targets = tuple(
            (element, int(count))
            for element, count in enumerate(targets[graph_index].tolist())
            if element > 0 and count > 0
        )
        positive_targets[graph_index] = graph_targets
        if formula_supported_by_space_group(
            int(space_groups[graph_index]),
            tuple(count for _, count in graph_targets),
            max_variable_count,
        ):
            feasible_graph_indices.append(graph_index)
        else:
            infeasible_graph_indices.append(graph_index)

    if progress is not None and infeasible_graph_indices:
        progress.update(len(infeasible_graph_indices))
    if not feasible_graph_indices:
        if candidate_count is not None:
            return {}, infeasible_graph_indices
        return data.x_0_dof, data.x_inf_dof, infeasible_graph_indices

    owns_executor = executor is None
    if owns_executor:
        pool = cpu_dp_pool(cpu_workers, len(feasible_graph_indices))
        executor, worker_count = pool.__enter__()
    else:
        worker_count = cpu_dp_worker_count(cpu_workers, len(feasible_graph_indices))

    try:
        zero_logits = _share_for_workers(zero_logits.detach().float().cpu(), executor)
        inf_logits = _share_for_workers(inf_logits.detach().float().cpu(), executor)
        zero_dof = data.zero_dof.detach().cpu()
        graph_zero = data.batch[data.zero_dof].detach().cpu()
        graph_inf = data.batch[~data.zero_dof].detach().cpu()
        multiplicities = data.multiplicities.detach().cpu().long()
        zero_multiplicities = multiplicities[zero_dof]
        inf_multiplicities = multiplicities[~zero_dof]

        num_graphs = data.composition.shape[0]
        zero_offsets = _graph_offsets(graph_zero, num_graphs)
        inf_offsets = _graph_offsets(graph_inf, num_graphs)
        seeds = (
            torch.randint(
                0,
                2**63 - 1,
                (len(feasible_graph_indices),),
                dtype=torch.long,
            ).tolist()
            if stochastic
            else [None] * len(feasible_graph_indices)
        )

        effective_task_size = max(
            cpu_task_size,
            math.ceil(len(feasible_graph_indices) / (worker_count * 4)),
        )

        def task_chunks():
            for chunk_start in range(
                0, len(feasible_graph_indices), effective_task_size
            ):
                chunk = []
                chunk_indices = feasible_graph_indices[
                    chunk_start : chunk_start + effective_task_size
                ]
                chunk_seeds = seeds[chunk_start : chunk_start + effective_task_size]
                for graph_index, seed in zip(chunk_indices, chunk_seeds):
                    zero_start = int(zero_offsets[graph_index])
                    zero_stop = int(zero_offsets[graph_index + 1])
                    inf_start = int(inf_offsets[graph_index])
                    inf_stop = int(inf_offsets[graph_index + 1])
                    chunk.append(
                        (
                            graph_index,
                            zero_start,
                            zero_stop,
                            inf_start,
                            inf_stop,
                            tuple(zero_multiplicities[zero_start:zero_stop].tolist()),
                            tuple(inf_multiplicities[inf_start:inf_stop].tolist()),
                            positive_targets[graph_index],
                            seed,
                        )
                    )
                yield (
                    zero_logits,
                    inf_logits,
                    chunk,
                    max_variable_count,
                    zero_beam_size,
                    stochastic,
                    candidate_count,
                )

        repaired_zero = data.x_0_dof.detach().cpu().long().clone()
        repaired_inf = data.x_inf_dof.detach().cpu().long().clone()
        candidate_results = {}

        def store(decoded_chunk):
            for graph_index, decoded in decoded_chunk:
                if decoded is None:
                    infeasible_graph_indices.append(graph_index)
                    continue
                if candidate_count is None:
                    decoded_zero, decoded_inf = decoded[0]
                    zero_start = int(zero_offsets[graph_index])
                    zero_stop = int(zero_offsets[graph_index + 1])
                    inf_start = int(inf_offsets[graph_index])
                    inf_stop = int(inf_offsets[graph_index + 1])
                    repaired_zero[zero_start:zero_stop] = torch.from_numpy(decoded_zero)
                    repaired_inf[inf_start:inf_stop] = torch.from_numpy(decoded_inf)
                else:
                    candidate_results[graph_index] = decoded
            if progress is not None:
                progress.update(len(decoded_chunk))

        chunks = iter(task_chunks())
        if executor is None:
            for task_chunk in chunks:
                store(_decode_task_chunk(task_chunk))
        else:
            pending = set()
            for _ in range(worker_count * 2):
                task_chunk = next(chunks, None)
                if task_chunk is None:
                    break
                pending.add(executor.submit(_decode_task_chunk, task_chunk))

            while pending:
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    store(future.result())
                    task_chunk = next(chunks, None)
                    if task_chunk is not None:
                        pending.add(executor.submit(_decode_task_chunk, task_chunk))

        if candidate_count is not None:
            return candidate_results, sorted(infeasible_graph_indices)
        return (
            repaired_zero.to(data.x_0_dof.device),
            repaired_inf.to(data.x_inf_dof.device),
            sorted(infeasible_graph_indices),
        )
    finally:
        if owns_executor:
            pool.__exit__(None, None, None)


def _candidate_data_list(data, candidate_results):
    graph_data = data.to_data_list()
    samples = []
    seen_by_group = {}
    for graph_index in sorted(candidate_results):
        graph = graph_data[graph_index]
        if hasattr(graph, "sampling_group"):
            group = int(graph.sampling_group.reshape(-1)[0])
        elif hasattr(graph, "target_index"):
            group = int(graph.target_index.reshape(-1)[0])
        else:
            group = graph_index
        seen = seen_by_group.setdefault(group, set())
        for decoded_zero, decoded_inf in candidate_results[graph_index]:
            signature = (decoded_zero.tobytes(), decoded_inf.tobytes())
            if signature in seen:
                continue
            seen.add(signature)
            graph_copy = graph.clone()
            graph_copy.x_0_dof = torch.from_numpy(decoded_zero).to(
                graph_copy.x_0_dof.device
            )
            graph_copy.x_inf_dof = torch.from_numpy(decoded_inf).to(
                graph_copy.x_inf_dof.device
            )
            graph_copy.x = create_x_matrix(
                graph_copy.x_inf_dof,
                graph_copy.x_0_dof,
                graph_copy.zero_dof,
            )
            samples.append(graph_copy)
    return samples


def sample_batch_to_compositions(
    data,
    zero_logits,
    inf_logits,
    max_variable_count,
    zero_beam_size=256,
    stochastic=True,
    cpu_workers=DEFAULT_CPU_WORKERS,
    cpu_task_size=DEFAULT_CPU_TASK_SIZE,
    executor=None,
    progress=None,
    announce=True,
    return_infeasible=False,
    candidate_count=None,
):
    """Decode graphs with exact composition constraints.

    ``candidate_count=None`` keeps the original n-shot behavior: each input
    graph produces one candidate, stochastically when ``stochastic=True``.
    Passing any positive integer enables deterministic top-N decoding and
    returns up to that many distinct candidates per input graph.  In particular,
    ``candidate_count=1`` is deterministic Top-1 rather than n-shot sampling.
    """

    num_graphs = data.composition.shape[0]
    if announce:
        worker_count = cpu_dp_worker_count(cpu_workers, num_graphs)
        print(
            f"[count_conserving] repair {num_graphs}/{num_graphs} graphs "
            f"with CPU DP using {worker_count} workers"
        )
    decoded = _decode_graphs(
        data,
        zero_logits,
        inf_logits,
        range(num_graphs),
        max_variable_count,
        zero_beam_size,
        stochastic,
        cpu_workers,
        cpu_task_size,
        executor,
        progress,
        candidate_count,
    )
    if candidate_count is not None:
        candidate_results, infeasible_graph_indices = decoded
        samples = _candidate_data_list(data, candidate_results)
        if return_infeasible:
            return samples, len(samples), infeasible_graph_indices
        if infeasible_graph_indices:
            raise ValueError("Space group cannot realize the requested composition")
        return samples, len(samples)

    data.x_0_dof, data.x_inf_dof, infeasible_graph_indices = decoded
    data.x = create_x_matrix(data.x_inf_dof, data.x_0_dof, data.zero_dof)
    if return_infeasible:
        return (
            data,
            num_graphs - len(infeasible_graph_indices),
            infeasible_graph_indices,
        )
    if infeasible_graph_indices:
        raise ValueError("Space group cannot realize the requested composition")
    return data, num_graphs


def repair_batch_to_compositions(
    data,
    zero_logits,
    inf_logits,
    max_variable_count,
    zero_beam_size=256,
    stochastic=True,
    cpu_workers=DEFAULT_CPU_WORKERS,
    cpu_task_size=DEFAULT_CPU_TASK_SIZE,
    executor=None,
    progress=None,
):
    """Replace mismatching samples with exact count-conserving assignments."""

    targets = data.composition
    decoded = decode_composition(data, targets.shape[1] - 1)
    repair_indices = torch.nonzero(
        torch.any(decoded.round() != targets.round(), dim=1)
    ).flatten()
    num_repairs = repair_indices.numel()
    print(f"[count_conserving] repair {num_repairs}/{targets.shape[0]} graphs")
    if num_repairs == 0:
        return data, 0

    data.x_0_dof, data.x_inf_dof, infeasible_graph_indices = _decode_graphs(
        data,
        zero_logits,
        inf_logits,
        repair_indices.tolist(),
        max_variable_count,
        zero_beam_size,
        stochastic,
        cpu_workers,
        cpu_task_size,
        executor,
        progress,
    )
    if infeasible_graph_indices:
        raise ValueError("Space group cannot realize the requested composition")
    data.x = create_x_matrix(data.x_inf_dof, data.x_0_dof, data.zero_dof)
    return data, num_repairs
