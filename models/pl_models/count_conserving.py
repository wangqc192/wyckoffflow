"""Exact composition constraints for formula-conditioned decoding."""

from functools import lru_cache

import torch
from wyckoff_generation.common.composition import decode_composition

from ..common import lookup_tables


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


def _decode_single(
    zero_logits,
    inf_logits,
    zero_multiplicities,
    inf_multiplicities,
    target_composition,
    max_variable_count,
    beam_size,
):
    target_composition = target_composition.round().long()
    elements = torch.nonzero(target_composition[1:] > 0).flatten() + 1
    targets = tuple(int(target_composition[element]) for element in elements.tolist())

    zero_columns = torch.cat((torch.zeros(1, dtype=torch.long), elements))
    zero_scores = zero_logits[:, zero_columns] + _gumbel_like(
        zero_logits[:, zero_columns]
    )
    inf_scores = inf_logits[:, elements - 1] + _gumbel_like(inf_logits[:, elements - 1])

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


def repair_batch_to_compositions(
    data,
    zero_logits,
    inf_logits,
    max_variable_count,
    zero_beam_size=256,
):
    """Replace mismatching samples with exact count-conserving assignments."""

    targets = data.composition
    decoded = decode_composition(data, targets.shape[1] - 1)
    repair_indices = torch.nonzero(
        torch.any(decoded.round() != targets.round(), dim=1)
    ).flatten()
    if repair_indices.numel() == 0:
        return data, 0

    zero_logits = zero_logits.detach().float().cpu()
    inf_logits = inf_logits.detach().float().cpu()
    targets = targets.detach().float().cpu()
    zero_dof = data.zero_dof.detach().cpu()
    graph_zero = data.batch[data.zero_dof].detach().cpu()
    graph_inf = data.batch[~data.zero_dof].detach().cpu()
    multiplicities = data.multiplicities.detach().cpu().long()

    repaired_zero = data.x_0_dof.detach().cpu().long().clone()
    repaired_inf = data.x_inf_dof.detach().cpu().long().clone()
    for graph_index in repair_indices.tolist():
        zero_rows = torch.nonzero(graph_zero == graph_index).flatten()
        inf_rows = torch.nonzero(graph_inf == graph_index).flatten()
        decoded_zero, decoded_inf = _decode_single(
            zero_logits[zero_rows],
            inf_logits[inf_rows],
            multiplicities[zero_dof][zero_rows].tolist(),
            multiplicities[~zero_dof][inf_rows].tolist(),
            targets[graph_index],
            max_variable_count,
            zero_beam_size,
        )
        repaired_zero[zero_rows] = decoded_zero
        repaired_inf[inf_rows] = decoded_inf

    data.x_0_dof = repaired_zero.to(data.x_0_dof.device)
    data.x_inf_dof = repaired_inf.to(data.x_inf_dof.device)
    return data, repair_indices.numel()
