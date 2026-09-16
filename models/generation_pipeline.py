"""End-to-end helpers for formula → space group → Wyckoff → structure generation.

The module deliberately keeps external model loading behind functions.  This lets
unit tests exercise formula/template conversion without importing NextCrystal or
DiffCSP, both of which have heavier optional dependencies.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import chemparse
import pandas as pd
import torch
from torch_geometric.data import Batch, Data

from models.common.checkpoint import load_model
from models.common.composition import formula_to_counts
from models.common.lookup_tables import chemical_symbols, spg_wyckoff_multiplicities
from models.common.wyckoff import (
    decode_wyckoff_elements,
    query_from_gwa_sequence,
    wyckoff_multiplicity,
)


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


def structure_sequence(data: Data) -> str:
    """Return a graph as the complete ``G-W-A-W-A-…`` sequence."""

    elements, labels = decode_wyckoff_elements(data.x)
    space_group = int(data.space_group.reshape(-1)[0])
    multiplicities = spg_wyckoff_multiplicities[str(space_group)]
    tokens = [str(space_group)]
    for element, label in zip(elements, labels):
        tokens.extend((f"{multiplicities[label]}{label}", element))
    return "-".join(tokens)


def exact_counts_from_sequence(sequence: str) -> dict[str, int]:
    query = query_from_gwa_sequence(sequence)
    counts: Counter[str] = Counter()
    for label, element in zip(query["wyckoff_letters"], query["atom_types"]):
        counts[element] += wyckoff_multiplicity(label)
    return dict(counts)


def wyckoff_template_from_sequence(sequence: str) -> str:
    """Convert ``G-W-A-…`` into DiffCSP's ``G_A1x1a_…`` format.

    Repeated identical ``(element, orbit)`` tokens are combined into the
    occupation integer expected by the symmetry-aware DiffCSP input parser.
    """

    query = query_from_gwa_sequence(sequence)
    grouped: dict[tuple[str, str], int] = {}
    order: list[tuple[str, str]] = []
    for label, element in zip(query["wyckoff_letters"], query["atom_types"]):
        key = (element, label)
        if key not in grouped:
            order.append(key)
            grouped[key] = 0
        grouped[key] += 1
    orbits = [
        f"{element}{grouped[(element, label)]}x{label}" for element, label in order
    ]
    return "_".join([str(query["spacegroup_number"]), *orbits])


def query_from_sequence(sequence: str) -> dict[str, Any]:
    """Return the API query used by DiffCSP-PP for one Wyckoff sequence."""

    query = query_from_gwa_sequence(sequence)
    return {
        "spacegroup_number": int(query["spacegroup_number"]),
        "wyckoff_letters": list(query["wyckoff_letters"]),
        "atom_types": list(query["atom_types"]),
    }


def _sample_one_space_group(
    model, formula: str, space_group: int, pool_size: int, *, flow_steps: int | None
):
    num_elements = int(model.num_elements)
    data = Data(
        formula=formula_to_counts(formula, num_elements).unsqueeze(0),
        num_evals=torch.tensor(pool_size, dtype=torch.long),
        space_group=torch.tensor(int(space_group), dtype=torch.long),
    )
    batch = Batch.from_data_list([data])
    try:
        generated = model.sample(
            batch,
            count_conserving=True,
            flow_steps=flow_steps,
        )
    except ValueError as error:
        if str(error) == "Space group cannot realize the requested composition":
            return []
        raise
    return generated.to_data_list()


def sample_wyckoff_templates(
    flow_model,
    formula: str,
    ranked_space_groups: list[int],
    *,
    templates_per_space_group: int = 4,
    template_pool_size: int = 16,
    flow_steps: int | None = None,
) -> list[dict[str, Any]]:
    """Sample and deduplicate exact-composition templates for each SG rank."""

    target_formula = format_formula(parse_formula_counts(formula))
    target_counts = parse_formula_counts(target_formula)
    rows: list[dict[str, Any]] = []
    for sg_rank, space_group in enumerate(ranked_space_groups, start=1):
        samples = _sample_one_space_group(
            flow_model,
            target_formula,
            int(space_group),
            template_pool_size,
            flow_steps=flow_steps,
        )
        frequencies: Counter[str] = Counter()
        representatives: dict[str, Data] = {}
        for sample in samples:
            sequence = structure_sequence(sample)
            candidate_counts = exact_counts_from_sequence(sequence)
            if candidate_counts != target_counts:
                continue
            frequencies[sequence] += 1
            representatives.setdefault(sequence, sample.cpu())
        ranked = sorted(
            representatives,
            key=lambda value: (-frequencies[value], value),
        )[:templates_per_space_group]
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
