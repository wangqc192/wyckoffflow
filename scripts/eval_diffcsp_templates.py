"""Evaluate DiffCSP structures generated from WyckoffFlow templates."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure

MATCHER_KWARGS = {
    "ltol": 0.3,
    "stol": 0.5,
    "angle_tol": 10,
    "primitive_cell": True,
    "scale": True,
}

_TARGET_STRUCTURES: list[Structure] = []
_TARGET_INDICES: list[int] = []
_TARGET_COMPOSITION_VALID: list[bool] = []
_TARGET_STRUCTURE_VALID: list[bool] = []
_MATCHER: StructureMatcher | None = None


def crystal_to_structure(crystal: dict) -> Structure:
    lengths = np.asarray(crystal["lengths"], dtype=float)
    angles = np.asarray(crystal["angles"], dtype=float)
    frac_coords = np.asarray(crystal["frac_coords"], dtype=float)
    if not (
        np.isfinite(lengths).all()
        and np.isfinite(angles).all()
        and np.isfinite(frac_coords).all()
        and (lengths > 0).all()
    ):
        raise ValueError("non-finite or non-positive lattice/coordinates")
    return Structure(
        Lattice.from_parameters(*lengths, *angles),
        crystal["atom_types"],
        frac_coords,
        coords_are_cartesian=False,
    )


def geometry_is_valid(structure: Structure) -> bool:
    lengths = np.asarray(structure.lattice.abc)
    angles = np.asarray(structure.lattice.angles)
    atom_scale = len(structure) ** (1 / 3)
    volume_per_atom = structure.volume / len(structure)
    return bool(
        structure.volume >= 0.1
        and (angles >= 10).all()
        and (angles <= 170).all()
        and (lengths / atom_scale <= 20).all()
        and (lengths / atom_scale >= 0.1).all()
        and 0.1 < volume_per_atom < 100
    )


def structure_is_valid(structure: Structure, cutoff: float = 0.5) -> bool:
    distance_matrix = structure.distance_matrix
    padded = distance_matrix + np.diag(
        np.ones(distance_matrix.shape[0]) * (cutoff + 10)
    )
    return bool(padded.min() >= cutoff and structure.volume >= 0.1)


def load_smact_validity(diffcsp_scripts: Path) -> Callable:
    diffcsp_scripts = diffcsp_scripts.resolve()
    sys.path[:0] = [str(diffcsp_scripts), str(diffcsp_scripts.parent)]
    from eval_utils import smact_validity

    return smact_validity


def composition_key(structure: Structure) -> tuple[tuple[int, ...], tuple[int, ...]]:
    composition = structure.composition
    elements = sorted(composition.elements, key=lambda element: element.Z)
    counts = np.array([int(round(composition[element])) for element in elements])
    counts //= np.gcd.reduce(counts)
    return tuple(element.Z for element in elements), tuple(counts.tolist())


def target_composition_validity(
    structures: list[Structure],
    smact_validity: Callable,
) -> list[bool]:
    cache = {}
    results = []
    for structure in structures:
        elements, counts = composition_key(structure)
        key = (elements, counts)
        if key not in cache:
            cache[key] = len(elements) < 8 and bool(
                smact_validity(elements, np.asarray(counts))
            )
        results.append(cache[key])
    return results


def evaluate_crystal(
    crystal: dict,
    target: Structure,
    matcher: StructureMatcher,
    target_composition_valid: bool = True,
    target_structure_valid: bool = True,
) -> tuple[bool, bool, bool, bool, bool, bool, float | None, str]:
    try:
        structure = crystal_to_structure(crystal)
    except Exception as exc:
        return False, False, False, False, False, False, None, str(exc)

    geometry_valid = geometry_is_valid(structure)
    structure_valid = geometry_valid and structure_is_valid(structure)
    composition_valid = (
        composition_key(structure) == composition_key(target)
        and target_composition_valid
    )
    if not geometry_valid:
        return (
            True,
            False,
            False,
            composition_valid,
            False,
            False,
            None,
            "invalid geometry",
        )

    try:
        rms = matcher.get_rms_dist(structure, target)
    except Exception as exc:
        return (
            True,
            True,
            structure_valid,
            composition_valid,
            False,
            False,
            None,
            str(exc),
        )
    if rms is None:
        return (
            True,
            True,
            structure_valid,
            composition_valid,
            False,
            False,
            None,
            "",
        )
    matched = structure_valid and composition_valid and target_structure_valid
    return (
        True,
        True,
        structure_valid,
        composition_valid,
        True,
        matched,
        float(rms[0]),
        "",
    )


def _initialize_workers(
    targets: list[Structure],
    target_indices: list[int],
    target_composition_valid: list[bool],
    target_structure_valid: list[bool],
) -> None:
    global _TARGET_STRUCTURES, _TARGET_INDICES
    global _TARGET_COMPOSITION_VALID, _TARGET_STRUCTURE_VALID, _MATCHER
    _TARGET_STRUCTURES = targets
    _TARGET_INDICES = target_indices
    _TARGET_COMPOSITION_VALID = target_composition_valid
    _TARGET_STRUCTURE_VALID = target_structure_valid
    _MATCHER = StructureMatcher(**MATCHER_KWARGS)


def _evaluate_indexed_crystal(item: tuple[int, dict]):
    index, crystal = item
    target_index = _TARGET_INDICES[index]
    target = _TARGET_STRUCTURES[target_index]
    assert _MATCHER is not None
    return evaluate_crystal(
        crystal,
        target,
        _MATCHER,
        _TARGET_COMPOSITION_VALID[target_index],
        _TARGET_STRUCTURE_VALID[target_index],
    )


def load_crystals(sample_paths: list[Path]) -> list[dict]:
    crystals = []
    for sample_path in sample_paths:
        sample = torch.load(sample_path, map_location="cpu", weights_only=False)
        crystals.extend(sample["crystal_list"])
    return crystals


def evaluate_all(
    crystals: list[dict],
    target_structures: list[Structure],
    target_indices: list[int],
    target_composition_valid: list[bool],
    target_structure_valid: list[bool],
    workers: int,
) -> list[tuple[bool, bool, bool, bool, bool, bool, float | None, str]]:
    indexed_crystals = enumerate(crystals)
    if workers == 1:
        matcher = StructureMatcher(**MATCHER_KWARGS)
        return [
            evaluate_crystal(
                crystal,
                target_structures[target_indices[index]],
                matcher,
                target_composition_valid[target_indices[index]],
                target_structure_valid[target_indices[index]],
            )
            for index, crystal in indexed_crystals
        ]

    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_initialize_workers,
        initargs=(
            target_structures,
            target_indices,
            target_composition_valid,
            target_structure_valid,
        ),
    ) as executor:
        return list(
            executor.map(_evaluate_indexed_crystal, indexed_crystals, chunksize=8)
        )


def build_summary(details: pd.DataFrame, total_targets: int, top_k: int = 20) -> dict:
    grouped = details.groupby("target_index", sort=False)
    material_hits = grouped["matched"].any()
    matcher_hits = grouped["matcher_matched"].any()
    return {
        "metric": f"DiffCSP StructureMatcher Top-{top_k}",
        "top_k": top_k,
        "matcher": MATCHER_KWARGS,
        "generated_structures": len(details),
        "constructed_structures": int(details["constructed"].sum()),
        "geometry_valid_structures": int(details["geometry_valid"].sum()),
        "structure_valid_structures": int(details["structure_valid"].sum()),
        "composition_valid_structures": int(details["composition_valid"].sum()),
        "matched_candidates": int(details["matched"].sum()),
        "materials_with_candidates": int(details["target_index"].nunique()),
        "valid_target_materials": int(grouped["target_valid"].first().sum()),
        "matched_materials": int(material_hits.sum()),
        "total_materials": total_targets,
        "match_rate": float(material_hits.sum() / total_targets),
        "raw_matcher_matched_candidates": int(details["matcher_matched"].sum()),
        "raw_matcher_matched_materials": int(matcher_hits.sum()),
        "raw_matcher_match_rate": float(matcher_hits.sum() / total_targets),
    }


def main(args) -> None:
    manifest = pd.read_csv(args.manifest)
    targets = pd.read_csv(args.targets)
    crystals = load_crystals([Path(path) for path in args.samples])
    if len(crystals) != len(manifest):
        raise ValueError(
            f"Generated structures ({len(crystals)}) and manifest rows "
            f"({len(manifest)}) differ"
        )

    target_indices = manifest["target_index"].astype(int).tolist()
    if target_indices and max(target_indices) >= len(targets):
        raise ValueError("Manifest target_index is outside the target CSV")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        target_structures = [
            Structure.from_str(cif, fmt="cif") for cif in targets["cif.conv"]
        ]
    smact_validity = load_smact_validity(args.diffcsp_scripts)
    target_composition_valid = target_composition_validity(
        target_structures,
        smact_validity,
    )
    target_structure_valid = [
        geometry_is_valid(structure) and structure_is_valid(structure)
        for structure in target_structures
    ]
    results = evaluate_all(
        crystals,
        target_structures,
        target_indices,
        target_composition_valid,
        target_structure_valid,
        args.workers,
    )

    details = manifest.copy()
    result_columns = [
        "constructed",
        "geometry_valid",
        "structure_valid",
        "composition_valid",
        "matcher_matched",
        "matched",
        "rms_dist",
        "error",
    ]
    details[result_columns] = results
    details["target_valid"] = [
        target_composition_valid[index] and target_structure_valid[index]
        for index in target_indices
    ]
    details["material_id"] = targets.iloc[target_indices]["material_id"].to_numpy()
    summary = build_summary(details, len(targets), args.top_k)

    output_path = Path(args.output)
    summary_path = Path(args.summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    details.to_csv(output_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(
        f"DiffCSP StructureMatcher Top-{args.top_k}: {summary['matched_materials']}/"
        f"{summary['total_materials']} ({summary['match_rate']:.2%})"
    )
    print(
        f"Raw StructureMatcher Top-{args.top_k}: "
        f"{summary['raw_matcher_matched_materials']}/"
        f"{summary['total_materials']} ({summary['raw_matcher_match_rate']:.2%})"
    )
    print(
        f"Candidates: {summary['matched_candidates']} validity-filtered matches, "
        f"{summary['geometry_valid_structures']} geometry-valid, "
        f"{summary['generated_structures']} generated"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--diffcsp-scripts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=20)
    main(parser.parse_args())
