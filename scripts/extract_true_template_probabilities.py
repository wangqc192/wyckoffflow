"""Extract final-model probabilities assigned to ground-truth Wyckoff templates.

The input is a saved ``sample_wy.py`` final-logits file.  For every saved flow
sample, this script maps the ground-truth ``wyckoff_spglib`` template to the
model graph and computes the product of the masked categorical probabilities
for all fixed and variable Wyckoff degrees of freedom.

If ``wyckoff_spglib`` contains several equivalent settings, the output reports
both the highest-probability setting and the probability sum over all settings.
The default ``true_template_probability`` column is the highest-probability
setting, which is the quantity most directly comparable with a generated
candidate's ``exp(decoder_log_score)``. Both quantities are conditional on the
saved final flow state, not marginal probabilities over flow trajectories.
"""

from __future__ import annotations

import argparse
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm.auto import tqdm

from models.common.lookup_tables import (
    chemical_symbols,
    spg_wyckoff_degrees_of_freedom,
    wyckoff_label_to_index,
)
from models.common.wyckoff_template import WyckoffTemplate

_LABEL_RE = re.compile(r"^(?P<multiplicity>[1-9][0-9]*)(?P<letter>[A-Za-z])$")
_ELEMENT_TO_ATOMIC_NUMBER = {
    symbol: atomic_number
    for atomic_number, symbol in enumerate(chemical_symbols)
    if atomic_number > 0
}


def _scalar_int(value: Any) -> int:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "reshape"):
        value = value.reshape(-1)[0]
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def _as_graph_batch(data: Any) -> tuple[torch.Tensor, int]:
    """Return local graph ids for the nodes in one saved PyG batch."""
    if hasattr(data, "batch"):
        batch = data.batch.detach().cpu().long()
        return batch, int(data.num_graphs)
    zero_dof = data.zero_dof.detach().cpu()
    return torch.zeros(zero_dof.numel(), dtype=torch.long), 1


def _offsets(graph_ids: torch.Tensor, num_graphs: int) -> list[int]:
    counts = torch.bincount(graph_ids, minlength=num_graphs).tolist()
    result = [0]
    for count in counts:
        result.append(result[-1] + int(count))
    return result


def _label_parts(label: str) -> tuple[int, str]:
    match = _LABEL_RE.fullmatch(str(label))
    if match is None:
        raise ValueError(f"invalid multiplicity-qualified Wyckoff label: {label}")
    return int(match.group("multiplicity")), match.group("letter")


def _template_assignments(
    template: WyckoffTemplate,
    site_positions: torch.Tensor,
    zero_mask: torch.Tensor,
    positive_elements: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map one template to the graph's fixed/variable categorical labels."""
    site_positions = site_positions.detach().cpu().long()
    zero_mask = zero_mask.detach().cpu().bool()
    positive_elements = positive_elements.detach().cpu().long()
    num_sites = int(site_positions.numel())
    if zero_mask.numel() != num_sites:
        raise ValueError("site positions and zero-DoF mask have different lengths")

    # ``wyckoff_pos_idx`` is local in the current model output, but normalising
    # by the minimum also accepts older batched files that stored a graph offset.
    positions = site_positions - site_positions.min() if num_sites else site_positions
    if sorted(positions.tolist()) != list(range(num_sites)):
        raise ValueError(
            "saved graph Wyckoff positions must be consecutive; "
            f"got {site_positions.tolist()}"
        )
    position_to_row = {int(position): row for row, position in enumerate(positions)}
    zero_rows = torch.nonzero(zero_mask, as_tuple=False).flatten().tolist()
    inf_rows = torch.nonzero(~zero_mask, as_tuple=False).flatten().tolist()
    zero_row_index = {row: index for index, row in enumerate(zero_rows)}
    inf_row_index = {row: index for index, row in enumerate(inf_rows)}
    element_to_column = {
        int(element) + 1: column
        for column, element in enumerate(positive_elements.tolist())
    }

    degrees = spg_wyckoff_degrees_of_freedom[str(template.spacegroup_number)]
    zero_values = torch.zeros(len(zero_rows), dtype=torch.long)
    inf_values = torch.zeros(
        len(inf_rows),
        len(positive_elements),
        dtype=torch.long,
    )
    for label, element in zip(template.wyckoff_letters, template.atom_types):
        _, letter = _label_parts(label)
        position_index = wyckoff_label_to_index.get(letter)
        if position_index is None:
            raise ValueError(f"unknown Wyckoff letter: {letter}")
        position = int(position_index) - 1
        try:
            row = position_to_row[position]
        except KeyError as error:
            raise ValueError(
                f"Wyckoff position {letter!r} is absent from space group "
                f"{template.spacegroup_number}"
            ) from error

        expected_zero = int(degrees[letter]) == 0
        if bool(zero_mask[row]) != expected_zero:
            raise ValueError(
                "saved graph DoF mask does not match the ground-truth Wyckoff "
                f"setting at position {letter!r} in space group "
                f"{template.spacegroup_number}"
            )

        try:
            atomic_number = _ELEMENT_TO_ATOMIC_NUMBER[element]
        except KeyError as error:
            raise ValueError(f"unknown chemical element: {element}") from error

        if expected_zero:
            zero_index = zero_row_index[row]
            previous = int(zero_values[zero_index])
            if previous not in (0, atomic_number):
                raise ValueError(
                    f"fixed Wyckoff position {label} has multiple elements"
                )
            zero_values[zero_index] = atomic_number
        else:
            try:
                element_column = element_to_column[atomic_number]
            except KeyError as error:
                raise ValueError(
                    f"element {element} is absent from the graph composition"
                ) from error
            inf_values[inf_row_index[row], element_column] += 1

    return zero_values, inf_values

def _template_log_probability(
    zero_logits: torch.Tensor,
    inf_logits: torch.Tensor,
    zero_values: torch.Tensor,
    inf_values: torch.Tensor,
    positive_elements: torch.Tensor,
) -> float:
    """Return the same product-model log probability used by Top-N decoding."""
    zero_values = zero_values.long()
    inf_values = inf_values.long()
    positive_elements = positive_elements.long()
    if zero_logits.shape[0] != zero_values.numel():
        raise ValueError(
            "zero-logit rows do not match the graph's zero-DoF positions: "
            f"{zero_logits.shape[0]} != {zero_values.numel()}"
        )
    if inf_logits.shape[0] != inf_values.shape[0]:
        raise ValueError(
            "variable-logit rows do not match the graph's variable positions: "
            f"{inf_logits.shape[0]} != {inf_values.shape[0]}"
        )
    if positive_elements.numel() and inf_logits.shape[1] <= int(positive_elements.max()):
        raise ValueError("saved variable logits do not contain all composition elements")
    if inf_logits.shape[1] != len(chemical_symbols) - 1:
        raise ValueError(
            "saved variable logits have an unexpected element dimension: "
            f"{inf_logits.shape[1]} != {len(chemical_symbols) - 1}"
        )
    if inf_values.numel() and int(inf_values.max()) >= inf_logits.shape[-1]:
        raise ValueError(
            "ground-truth variable occupation exceeds the saved logits' count "
            f"range: max={int(inf_values.max())}, classes={inf_logits.shape[-1]}"
        )

    log_probability = zero_logits.new_zeros(())
    if zero_values.numel():
        zero_log_probs = torch.log_softmax(zero_logits.float(), dim=-1)
        log_probability = log_probability + zero_log_probs.gather(
            1, zero_values.unsqueeze(-1)
        ).sum()
    if inf_values.numel():
        selected_logits = inf_logits[:, positive_elements, :]
        inf_log_probs = torch.log_softmax(selected_logits.float(), dim=-1)
        log_probability = log_probability + inf_log_probs.gather(
            -1, inf_values.unsqueeze(-1)
        ).sum()
    return float(log_probability.item())


def _unique_templates(value: str) -> tuple[WyckoffTemplate, ...]:
    templates = {}
    for template in WyckoffTemplate.from_protostructure_set(value):
        templates[template.occupancy_key()] = template
    return tuple(templates.values())


def _target_info(target_df: pd.DataFrame, target_index: int) -> dict[str, Any]:
    if target_index < 0 or target_index >= len(target_df):
        raise IndexError(
            f"target_index={target_index} is outside ground-truth rows "
            f"[0, {len(target_df)})"
        )
    row = target_df.iloc[target_index]
    templates = _unique_templates(str(row["wyckoff_spglib"]))
    return {
        "material_id": row.get("material_id", ""),
        "target_formula": row.get("pretty_formula", ""),
        "target_space_group": int(templates[0].spacegroup_number),
        "templates": templates,
    }


def _load_condition_rows(
    input_path: Path | None,
    saved_args: Mapping[str, Any],
) -> pd.DataFrame | None:
    if input_path is None:
        saved_formula_file = saved_args.get("formula_file")
        if saved_formula_file:
            candidate = Path(str(saved_formula_file))
            if candidate.exists():
                input_path = candidate
    if input_path is None:
        return None
    rows = pd.read_csv(input_path)
    if len(rows) == 0:
        raise ValueError(f"input condition file is empty: {input_path}")
    return rows


def _condition_target_index(row: pd.Series, fallback: int) -> int:
    if "target_index" not in row.index:
        return fallback
    return _scalar_int(row["target_index"])


def _condition_space_group(row: pd.Series, fallback: int) -> int:
    for column in ("space_group", "generated_space_group"):
        if column in row.index and pd.notna(row[column]):
            return _scalar_int(row[column])
    return fallback


def _graph_rows(
    data: Any,
    zero_logits: torch.Tensor,
    inf_logits: torch.Tensor,
) -> list[dict[str, Any]]:
    batch, num_graphs = _as_graph_batch(data)
    zero_mask_all = data.zero_dof.detach().cpu().bool()
    graph_zero = batch[zero_mask_all]
    graph_inf = batch[~zero_mask_all]
    zero_offsets = _offsets(graph_zero, num_graphs)
    inf_offsets = _offsets(graph_inf, num_graphs)
    if zero_logits is None or inf_logits is None:
        raise ValueError("saved logits file contains missing final logits")
    if zero_logits.shape[0] != zero_offsets[-1]:
        raise ValueError(
            "saved zero logits and graph data have different row counts: "
            f"{zero_logits.shape[0]} != {zero_offsets[-1]}"
        )
    if inf_logits.shape[0] != inf_offsets[-1]:
        raise ValueError(
            "saved variable logits and graph data have different row counts: "
            f"{inf_logits.shape[0]} != {inf_offsets[-1]}"
        )

    positions_all = (
        data.wyckoff_pos_idx.detach().cpu().long()
        if hasattr(data, "wyckoff_pos_idx")
        else None
    )
    rows = []
    for graph_index in range(num_graphs):
        node_mask = batch == graph_index
        local_zero_mask = zero_mask_all[node_mask]
        positions = (
            positions_all[node_mask]
            if positions_all is not None
            else torch.arange(int(node_mask.sum()))
        )
        zero_start, zero_stop = zero_offsets[graph_index : graph_index + 2]
        inf_start, inf_stop = inf_offsets[graph_index : graph_index + 2]
        composition = (
            data.composition[graph_index].detach().cpu().round().long()
        )
        rows.append(
            {
                "graph_index": graph_index,
                "space_group": _scalar_int(data.space_group[graph_index]),
                "composition": composition,
                "positive_elements": torch.nonzero(
                    composition[1:] > 0, as_tuple=False
                ).flatten(),
                "zero_mask": local_zero_mask,
                "positions": positions,
                "zero_logits": zero_logits[zero_start:zero_stop],
                "inf_logits": inf_logits[inf_start:inf_stop],
                "model_target_index": (
                    _scalar_int(data.target_index[graph_index])
                    if hasattr(data, "target_index")
                    else None
                ),
            }
        )
    return rows


def _composition_matches(
    composition: torch.Tensor,
    templates: tuple[WyckoffTemplate, ...],
) -> bool:
    expected = {index: int(value) for index, value in enumerate(composition.tolist())}
    expected = {index: count for index, count in expected.items() if index > 0 and count}
    for template in templates:
        actual = {
            _ELEMENT_TO_ATOMIC_NUMBER[element]: count
            for element, count in template.formula_counts.items()
        }
        if actual == expected:
            return True
    return False


def _score_graph(
    graph: dict[str, Any],
    target: dict[str, Any],
) -> dict[str, Any]:
    templates = target["templates"]
    result = {
        "status": "ok",
        "target_template_count": len(templates),
        "equivalent_templates": "|".join(template.to_gwa() for template in templates),
        "best_equivalent_template": "",
        "true_template_log_probability": None,
        "true_template_probability": None,
        "equivalent_template_log_probability": None,
        "equivalent_template_probability": None,
    }
    if graph["space_group"] != int(target["target_space_group"]):
        result["status"] = "space_group_mismatch"
        return result
    if not _composition_matches(graph["composition"], templates):
        result["status"] = "composition_mismatch"
        return result

    scored = []
    for template in templates:
        zero_values, inf_values = _template_assignments(
            template,
            graph["positions"],
            graph["zero_mask"],
            graph["positive_elements"],
        )
        log_probability = _template_log_probability(
            graph["zero_logits"],
            graph["inf_logits"],
            zero_values,
            inf_values,
            graph["positive_elements"],
        )
        scored.append((log_probability, template))

    if not scored:
        result["status"] = "no_valid_equivalent_template"
        return result

    best_log_probability, best_template = max(scored, key=lambda item: item[0])
    log_probabilities = torch.tensor([item[0] for item in scored], dtype=torch.float64)
    sum_log_probability = float(torch.logsumexp(log_probabilities, dim=0).item())
    result.update(
        {
            "best_equivalent_template": best_template.to_gwa(),
            "true_template_log_probability": best_log_probability,
            "true_template_probability": math.exp(best_log_probability)
            if best_log_probability > -745
            else 0.0,
            "equivalent_template_log_probability": sum_log_probability,
            "equivalent_template_probability": math.exp(sum_log_probability)
            if sum_log_probability > -745
            else 0.0,
        }
    )
    return result


def extract_probabilities(
    logits_path: str | Path,
    target_path: str | Path,
    output_path: str | Path,
    input_path: str | Path | None = None,
) -> pd.DataFrame:
    """Extract probabilities for every graph stored in a final-logits file."""
    payload = torch.load(logits_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "logit_batches" not in payload:
        raise ValueError(f"not a saved final-logits payload: {logits_path}")
    target_df = pd.read_csv(target_path)
    condition_df = _load_condition_rows(
        Path(input_path) if input_path is not None else None,
        payload.get("args", {}),
    )
    total_graphs = sum(int(data.num_graphs) for data, _, _ in payload["logit_batches"])
    repeat = None
    if condition_df is not None:
        if total_graphs % len(condition_df) != 0:
            raise ValueError(
                f"saved graph count {total_graphs} is not a multiple of input rows "
                f"{len(condition_df)}; pass the exact formula/condition CSV"
            )
        repeat = total_graphs // len(condition_df)

    target_cache: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    global_graph_index = 0
    for data, zero_logits, inf_logits in tqdm(
        payload["logit_batches"], desc="Extracting true-template probabilities", unit="batch"
    ):
        for graph in _graph_rows(data, zero_logits, inf_logits):
            if condition_df is not None:
                condition_index = global_graph_index // int(repeat)
                eval_index = global_graph_index % int(repeat)
                condition = condition_df.iloc[condition_index]
                target_index = _condition_target_index(condition, condition_index)
                condition_space_group = _condition_space_group(
                    condition, graph["space_group"]
                )
                condition_formula = condition.get("formula", "")
            else:
                condition_index = None
                eval_index = 0
                target_index = (
                    graph["model_target_index"]
                    if graph["model_target_index"] is not None
                    else global_graph_index
                )
                condition_space_group = graph["space_group"]
                condition_formula = ""

            if target_index not in target_cache:
                target_cache[target_index] = _target_info(target_df, target_index)
            target = target_cache[target_index]
            scored = _score_graph(graph, target)
            row = {
                "graph_index": global_graph_index,
                "condition_index": condition_index,
                "eval_index": eval_index,
                "target_index": target_index,
                "material_id": target["material_id"],
                "target_formula": target["target_formula"],
                "condition_formula": condition_formula,
                "condition_space_group": condition_space_group,
                "model_space_group": graph["space_group"],
                "target_space_group": target["target_space_group"],
                "model_target_index": graph["model_target_index"],
            }
            row.update(scored)
            rows.append(row)
            global_graph_index += 1

    if global_graph_index != total_graphs:
        raise AssertionError("internal graph-count mismatch")
    result = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def main(args: argparse.Namespace) -> None:
    result = extract_probabilities(
        args.logits,
        args.gt_file,
        args.output,
        args.input_file,
    )
    valid = result[result["status"] == "ok"]
    print(f"Wrote {len(result)} graph rows to {args.output}")
    print(f"Valid true-template probabilities: {len(valid)}/{len(result)}")
    if len(valid):
        print(
            "Best-equivalent probability: "
            f"mean={valid.true_template_probability.mean():.6e}, "
            f"median={valid.true_template_probability.median():.6e}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract model probabilities of MP20 ground-truth Wyckoff templates"
    )
    parser.add_argument("--logits", required=True, help="saved *.logits.pt file")
    parser.add_argument("--gt-file", required=True, help="ground-truth MP20 CSV")
    parser.add_argument(
        "--input-file",
        help=(
            "formula/condition CSV used to create the logits; defaults to the "
            "formula_file recorded in the logits payload"
        ),
    )
    parser.add_argument("--output", required=True, help="output probability CSV")
    main(parser.parse_args())
