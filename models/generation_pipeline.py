"""End-to-end helpers for formula → space group → Wyckoff → structure generation.

The module deliberately keeps external model loading behind functions.  This lets
unit tests exercise formula/template conversion without importing NextCrystal or
DiffCSP, both of which have heavier optional dependencies.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import chemparse
import pandas as pd
import torch
from torch_geometric.data import Batch, Data

from models.common.checkpoint import load_model
from models.common.lookup_tables import chemical_symbols
from models.common.wyckoff_template import WyckoffTemplate
from models.pl_models.count_conserving import formula_supported_by_space_group
from models.sampling import SAMPLING_MODES, make_condition, sample_batch


def parse_formula_counts(formula: str) -> dict[str, int]:
    """Parse a complete formula without reducing its integer counts."""

    counts = chemparse.parse_formula(str(formula))
    if not counts or any(
        float(value) <= 0 or int(value) != value for value in counts.values()
    ):
        raise ValueError(f"formula must contain positive integer counts: {formula!r}")
    return {element: int(value) for element, value in counts.items()}


def format_formula(counts: dict[str, int]) -> str:
    """Format counts in atomic-number order, retaining the complete formula."""

    by_number = {
        chemical_symbols.index(element): (element, int(count))
        for element, count in counts.items()
    }
    return "".join(
        element if count == 1 else f"{element}{count}"
        for _, (element, count) in sorted(by_number.items())
        if count > 0
    )


def formula_atom_count(formula: str) -> int:
    return sum(parse_formula_counts(formula).values())


def complete_formula_from_cif(cif: str) -> str:
    """Extract the complete conventional-cell formula from an embedded CIF."""

    match = re.search(r"^_chemical_formula_sum\s+(.+)$", str(cif), flags=re.MULTILINE)
    if match is None:
        raise ValueError("CIF does not contain _chemical_formula_sum")
    formula = match.group(1).strip().strip('"').strip("'").replace(" ", "")
    return format_formula(parse_formula_counts(formula))


def structure_sequence(data: Data) -> str:
    """Return a graph as the complete ``G-W-A-W-A-…`` sequence."""

    return WyckoffTemplate.from_model_output(data).to_gwa()


def exact_counts_from_sequence(sequence: str) -> dict[str, int]:
    return WyckoffTemplate.from_gwa(sequence).formula_counts


def wyckoff_template_from_sequence(sequence: str) -> str:
    """Convert ``G-W-A-…`` to CrystalFlow's ``G_A1x1a_…`` format."""

    return WyckoffTemplate.from_gwa(sequence).to_crystalflow()


def query_from_sequence(sequence: str) -> dict[str, Any]:
    """Return the API query used by DiffCSP++ (``diffcsppp``)."""

    return WyckoffTemplate.from_gwa(sequence).to_diffcsppp_query()


def _sample_one_space_group(
    model, formula, space_group, num_samples, *, flow_steps, sampling_mode="n-shot"
):
    condition = make_condition(formula, space_group, model.num_elements, target_index=0)
    return sample_batch(
        model,
        Batch.from_data_list([condition]),
        flow_steps=flow_steps,
        num_samples=num_samples,
        sampling_mode=sampling_mode,
    ).samples


def _rank_template_samples(samples, target_counts, sampling_mode):
    frequencies = Counter()
    representatives = {}
    for sample in samples:
        sequence = structure_sequence(sample)
        if exact_counts_from_sequence(sequence) != target_counts:
            continue
        frequencies[sequence] += 1
        representatives.setdefault(sequence, sample.cpu())
    if sampling_mode == "top-n":
        ranked = sorted(
            representatives, key=lambda key: int(representatives[key].candidate_rank)
        )
    else:
        ranked = sorted(representatives, key=lambda key: (-frequencies[key], key))
    return ranked, frequencies, representatives


def sample_wyckoff_templates(
    flow_model,
    formula: str,
    ranked_space_groups: list[int],
    *,
    templates_per_space_group: int = 4,
    template_pool_size: int = 16,
    flow_steps: int,
    sampling_mode: str = "n-shot",
) -> list[dict[str, Any]]:
    """Sample and deduplicate exact-composition templates for each SG rank."""

    target_formula = format_formula(parse_formula_counts(formula))
    target_counts = parse_formula_counts(target_formula)
    if sampling_mode not in SAMPLING_MODES:
        raise ValueError(f"unsupported sampling_mode: {sampling_mode}")
    rows: list[dict[str, Any]] = []
    for sg_rank, space_group in enumerate(ranked_space_groups, start=1):
        samples = _sample_one_space_group(
            flow_model,
            target_formula,
            int(space_group),
            (
                templates_per_space_group
                if sampling_mode == "top-n"
                else template_pool_size
            ),
            flow_steps=flow_steps,
            sampling_mode=sampling_mode,
        )
        ranked, frequencies, representatives = _rank_template_samples(
            samples, target_counts, sampling_mode
        )
        ranked = ranked[:templates_per_space_group]
        for template_rank, sequence in enumerate(ranked, start=1):
            query = query_from_sequence(sequence)
            rows.append(
                {
                    "material_index": 0,
                    "candidate_index": (sg_rank - 1) * templates_per_space_group
                    + template_rank
                    - 1,
                    "space_group_rank": sg_rank,
                    "template_rank": template_rank,
                    "target_formula": target_formula,
                    "target_space_group": "",
                    "target_structure_sequence": "",
                    "generated_formula": format_formula(
                        exact_counts_from_sequence(sequence)
                    ),
                    "generated_space_group": int(space_group),
                    "generated_structure_sequence": sequence,
                    "wyckoff_template": wyckoff_template_from_sequence(sequence),
                    "frequency": frequencies[sequence],
                    "query": query,
                    "sample": representatives[sequence],
                }
            )
    return rows


def predict_nextcrystal_space_groups(
    formula: str,
    *,
    nextcrystal_root: Path,
    checkpoint: Path,
    input_csv: Path,
    top_k: int,
    device: str = "cpu",
) -> pd.DataFrame:
    """Run NextCrystal's released SG predictor for one complete formula."""

    nextcrystal_root = Path(nextcrystal_root).resolve()
    checkpoint = Path(checkpoint).resolve()
    input_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "formula": format_formula(parse_formula_counts(formula)),
                "NAtoms": formula_atom_count(formula),
            }
        ]
    ).to_csv(input_csv, index=False)
    sys.path.insert(0, str(nextcrystal_root))
    from src.predictors.sg_predictor import SGPredictConfig, SpaceGroupPredictor

    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_config = checkpoint_payload.get("model_config", {})
    config = SGPredictConfig(
        model_path=str(checkpoint),
        input_csv=str(input_csv),
        output_csv=str(input_csv.with_name("nextcrystal_sg.csv")),
        wyckoff_template_path=str(nextcrystal_root / "data/wyckoff_template.json"),
        top_k=int(top_k),
        max_atoms=int(model_config.get("max_atoms", 512)),
        d_model=int(model_config.get("d_model", 256)),
        nhead=int(model_config.get("nhead", 8)),
        num_layers=int(model_config.get("num_layers", 6)),
        dim_feedforward=int(model_config.get("dim_feedforward", 512)),
        dropout=float(model_config.get("dropout", 0.1)),
        use_moe=bool(model_config.get("use_moe", True)),
        moe_layers=model_config.get("moe_layers", "all"),
        num_experts=int(model_config.get("num_experts", 8)),
        capacity_factor=float(model_config.get("capacity_factor", 1.5)),
        router_noisy_std=float(model_config.get("router_noisy_std", 0.5)),
        moe_loss_coef=float(model_config.get("moe_loss_coef", 0.002)),
        moe_top_k=int(model_config.get("moe_top_k", 2)),
    )
    predictor = SpaceGroupPredictor(config, device=torch.device(device))
    result = predictor.predict_dataframe()
    result.to_csv(config.output_csv, index=False)
    return result


def load_flow_model(checkpoint: Path, device: str):
    model, _, config = load_model(checkpoint, load_data=False, device=device)
    return model, config


def write_selected_queries(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = [row["query"] for row in rows]
    path.write_text(json.dumps(selected, indent=2), encoding="utf-8")


def _sample_record_batch(
    model, records, num_samples, *, flow_steps, sampling_mode="n-shot"
):
    """Sample all condition pairs once, preserving empty groups for failed inputs."""
    if not records:
        return []
    batch = Batch.from_data_list(
        [
            make_condition(
                record["formula"],
                record["generated_space_group"],
                model.num_elements,
                record["target_index"],
            )
            for record in records
        ]
    )
    result = sample_batch(
        model,
        batch,
        flow_steps=flow_steps,
        num_samples=num_samples,
        sampling_mode=sampling_mode,
    )
    by_group = {index: [] for index in range(len(records))}
    for sample in result.samples:
        by_group[int(sample.sampling_group)].append(sample)
    return [(record, by_group[index]) for index, record in enumerate(records)]


def sample_wyckoff_templates_for_targets(
    flow_model,
    targets: pd.DataFrame,
    space_group_predictions: pd.DataFrame,
    *,
    templates_per_space_group: int,
    template_pool_size: int,
    pair_batch_size: int = 32,
    flow_steps: int,
    sampling_mode: str = "n-shot",
) -> list[dict[str, Any]]:
    """Generate ranked exact-composition templates for a target CSV hierarchy."""

    target_columns = {"target_index", "formula"}
    prediction_columns = {
        "target_index",
        "Spacegroup Number",
        "SG_Rank",
        "SG_Prob",
    }
    missing = target_columns.difference(targets.columns)
    if missing:
        raise ValueError(f"targets are missing columns: {sorted(missing)}")
    missing = prediction_columns.difference(space_group_predictions.columns)
    if missing:
        raise ValueError(
            f"space-group predictions are missing columns: {sorted(missing)}"
        )
    if template_pool_size < templates_per_space_group:
        raise ValueError(
            "template_pool_size must be at least templates_per_space_group"
        )
    if sampling_mode not in SAMPLING_MODES:
        raise ValueError(f"unsupported sampling_mode: {sampling_mode}")

    target_records = targets.set_index("target_index").to_dict("index")
    pair_records = []
    for prediction in space_group_predictions.sort_values(
        ["target_index", "SG_Rank"], kind="stable"
    ).to_dict("records"):
        target_index = int(prediction["target_index"])
        target = target_records[target_index]
        formula = format_formula(parse_formula_counts(target["formula"]))
        space_group = int(prediction["Spacegroup Number"])
        counts = tuple(parse_formula_counts(formula).values())
        if not formula_supported_by_space_group(
            space_group, counts, int(flow_model.max_num_atoms)
        ):
            continue
        pair_records.append(
            {
                "target_index": target_index,
                "material_id": target.get("material_id", target_index),
                "formula": formula,
                "target_space_group": target.get("target_space_group", ""),
                "generated_space_group": space_group,
                "space_group_rank": int(prediction["SG_Rank"]),
                "space_group_probability": float(prediction["SG_Prob"]),
            }
        )

    sampled_pairs = []
    for start in range(0, len(pair_records), pair_batch_size):
        sampled_pairs.extend(
            _sample_record_batch(
                flow_model,
                pair_records[start : start + pair_batch_size],
                (
                    templates_per_space_group
                    if sampling_mode == "top-n"
                    else template_pool_size
                ),
                flow_steps=flow_steps,
                sampling_mode=sampling_mode,
            )
        )

    rows: list[dict[str, Any]] = []
    for record, samples in sampled_pairs:
        target_counts = parse_formula_counts(record["formula"])
        ranked_sequences, frequencies, representatives = _rank_template_samples(
            samples, target_counts, sampling_mode
        )
        if not ranked_sequences:
            continue
        selected_sequences = ranked_sequences[:templates_per_space_group]
        if len(selected_sequences) < templates_per_space_group:
            print(
                f"[templates] target={record['target_index']} "
                f"space_group={record['generated_space_group']}: requested "
                f"{templates_per_space_group} unique templates, got "
                f"{len(selected_sequences)}"
            )
        for template_rank, sequence in enumerate(selected_sequences, start=1):
            rows.append(
                {
                    "target_index": record["target_index"],
                    "material_id": record["material_id"],
                    "candidate_index": (
                        (record["space_group_rank"] - 1) * templates_per_space_group
                        + template_rank
                        - 1
                    ),
                    "space_group_rank": record["space_group_rank"],
                    "template_rank": template_rank,
                    "space_group_probability": record["space_group_probability"],
                    "target_formula": record["formula"],
                    "target_space_group": record["target_space_group"],
                    "generated_formula": format_formula(
                        exact_counts_from_sequence(sequence)
                    ),
                    "generated_space_group": record["generated_space_group"],
                    "generated_structure_sequence": sequence,
                    "wyckoff_template": wyckoff_template_from_sequence(sequence),
                    "frequency": frequencies[sequence],
                    "duplicate_template_of_rank": "",
                    "query": query_from_sequence(sequence),
                    "sample": representatives[sequence],
                }
            )

    rows.sort(
        key=lambda row: (
            row["target_index"],
            row["space_group_rank"],
            row["template_rank"],
        )
    )
    for template_index, row in enumerate(rows):
        row["template_index"] = template_index
    return rows


def predict_nextcrystal_space_groups_for_targets(
    targets: pd.DataFrame,
    *,
    nextcrystal_root: Path,
    checkpoint: Path,
    input_csv: Path,
    top_k: int,
    max_variable_count: int,
    candidate_top_k: int = 32,
    device: str = "cpu",
    batch_size: int = 256,
) -> pd.DataFrame:
    """Return each target's highest-ranked composition-realizable space groups."""

    required = {"target_index", "formula", "NAtoms"}
    missing = required.difference(targets.columns)
    if missing:
        raise ValueError(f"targets are missing columns: {sorted(missing)}")
    if candidate_top_k < top_k:
        raise ValueError("candidate_top_k must be at least top_k")

    input_csv.parent.mkdir(parents=True, exist_ok=True)
    predictor_input = targets[["target_index", "formula", "NAtoms"]].rename(
        columns={"target_index": "id"}
    )
    predictor_input.to_csv(input_csv, index=False)

    nextcrystal_root = Path(nextcrystal_root).resolve()
    checkpoint = Path(checkpoint).resolve()
    sys.path.insert(0, str(nextcrystal_root))
    from src.predictors.sg_predictor import SGPredictConfig, SpaceGroupPredictor

    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_config = checkpoint_payload.get("model_config", {})
    config = SGPredictConfig(
        model_path=str(checkpoint),
        input_csv=str(input_csv),
        output_csv=str(input_csv.with_name("nextcrystal_sg.csv")),
        wyckoff_template_path=str(nextcrystal_root / "data/wyckoff_template.json"),
        batch_size=int(batch_size),
        top_k=int(candidate_top_k),
        max_atoms=int(model_config.get("max_atoms", 512)),
        d_model=int(model_config.get("d_model", 256)),
        nhead=int(model_config.get("nhead", 8)),
        num_layers=int(model_config.get("num_layers", 6)),
        dim_feedforward=int(model_config.get("dim_feedforward", 512)),
        dropout=float(model_config.get("dropout", 0.1)),
        use_moe=bool(model_config.get("use_moe", True)),
        moe_layers=model_config.get("moe_layers", "all"),
        num_experts=int(model_config.get("num_experts", 8)),
        capacity_factor=float(model_config.get("capacity_factor", 1.5)),
        router_noisy_std=float(model_config.get("router_noisy_std", 0.5)),
        moe_loss_coef=float(model_config.get("moe_loss_coef", 0.002)),
        moe_top_k=int(model_config.get("moe_top_k", 2)),
    )
    predictor = SpaceGroupPredictor(config, device=torch.device(device))
    raw = predictor.predict_dataframe()
    raw["target_index"] = pd.to_numeric(raw["cif_name"]).astype(int)

    formulas = targets.set_index("target_index")["formula"].to_dict()

    def select(predictions: pd.DataFrame) -> tuple[list[dict[str, Any]], list[int]]:
        selected: list[dict[str, Any]] = []
        insufficient: list[int] = []
        grouped = {
            int(index): group
            for index, group in predictions.groupby("target_index", sort=False)
        }
        for target_index in targets["target_index"].astype(int):
            counts = tuple(parse_formula_counts(formulas[target_index]).values())
            selected_rank = 0
            group = grouped.get(target_index, pd.DataFrame())
            for prediction in group.sort_values("SG_Rank", kind="stable").to_dict(
                "records"
            ):
                space_group = int(prediction["Spacegroup Number"])
                if not 1 <= space_group <= 230:
                    continue
                if not formula_supported_by_space_group(
                    space_group, counts, max_variable_count
                ):
                    continue
                selected_rank += 1
                prediction["NextCrystal_Rank"] = int(prediction["SG_Rank"])
                prediction["SG_Rank"] = selected_rank
                selected.append(prediction)
                if selected_rank == top_k:
                    break
            if selected_rank < top_k:
                insufficient.append(target_index)
        return selected, insufficient

    selected, insufficient = select(raw)
    if insufficient and candidate_top_k < 231:
        fallback_input = input_csv.with_name(f"{input_csv.stem}_fallback.csv")
        fallback_targets = targets[targets["target_index"].isin(insufficient)]
        fallback_targets[["target_index", "formula", "NAtoms"]].rename(
            columns={"target_index": "id"}
        ).to_csv(fallback_input, index=False)
        predictor.cfg.input_csv = str(fallback_input)
        predictor.cfg.top_k = 231
        fallback = predictor.predict_dataframe()
        fallback["target_index"] = pd.to_numeric(fallback["cif_name"]).astype(int)
        raw = pd.concat(
            [raw[~raw["target_index"].isin(insufficient)], fallback],
            ignore_index=True,
        )
        selected, insufficient = select(raw)

    raw.sort_values(["target_index", "SG_Rank"], kind="stable").to_csv(
        config.output_csv, index=False
    )
    if insufficient:
        raise RuntimeError(
            f"fewer than {top_k} composition-realizable space groups for target "
            f"indices {insufficient[:10]}"
        )
    return pd.DataFrame(selected).reset_index(drop=True)
