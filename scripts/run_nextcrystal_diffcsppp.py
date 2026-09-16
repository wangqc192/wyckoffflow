#!/usr/bin/env python3
"""Prepare and sample DiffCSP++ structures from symmetry templates."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Lattice, Structure

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.common.wyckoff import (  # noqa: E402
    query_from_gwa_sequence,
    wyckoff_multiplicity,
)

_SPACE_GROUP_TOP_K = 5
_TEMPLATES_PER_SPACE_GROUP = 4
_CANDIDATES_PER_MATERIAL = _SPACE_GROUP_TOP_K * _TEMPLATES_PER_SPACE_GROUP
_MANIFEST_COLUMNS = [
    "input_index",
    "source_index",
    "material_index",
    "candidate_index",
    "material_id",
    "target_formula",
    "target_structure_sequence",
    "candidate_formula",
    "generated_structure_sequence",
    "spacegroup_number",
    "space_group_rank",
    "template_rank",
    "wyckoff_letters",
    "atom_types",
]


def template_reduced_formula(query: dict[str, Any]) -> str:
    counts: Counter[str] = Counter()
    wyckoff_letters = query["wyckoff_letters"]
    atom_types = query["atom_types"]
    if len(wyckoff_letters) != len(atom_types):
        raise ValueError("wyckoff_letters and atom_types must have equal length")
    for wyckoff, element in zip(wyckoff_letters, atom_types):
        counts[str(element)] += wyckoff_multiplicity(str(wyckoff))
    return Composition(counts).reduced_formula


def build_wyckoffflow_manifest(
    evaluation: pd.DataFrame, targets: pd.DataFrame
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Build a DiffCSP++ manifest from sampled or hierarchical CSV rows."""
    required_evaluation_columns = {
        "material_index",
        "candidate_index",
        "space_group_rank",
        "template_rank",
        "target_structure_sequence",
        "generated_structure_sequence",
    }
    missing = required_evaluation_columns - set(evaluation.columns)
    if missing:
        raise ValueError(
            f"WyckoffFlow evaluation CSV is missing columns: {sorted(missing)}"
        )
    required_target_columns = {"material_id", "pretty_formula"}
    missing = required_target_columns - set(targets.columns)
    if missing:
        raise ValueError(f"target CSV is missing columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    selected_queries: list[dict[str, Any]] = []
    seen_candidates: set[tuple[int, int]] = set()
    material_candidate_indices: dict[int, set[int]] = {}
    target_compositions = [
        Composition(str(formula)).reduced_composition
        for formula in targets["pretty_formula"]
    ]
    target_sequences: dict[int, str] = {}
    target_sequence_compositions: dict[int, Composition] = {}
    sequence_cache: dict[str, tuple[dict[str, Any], str, Composition]] = {}
    for source_index, row in enumerate(evaluation.itertuples(index=False)):
        material_index = int(getattr(row, "material_index"))
        candidate_index = int(getattr(row, "candidate_index"))
        space_group_rank = int(getattr(row, "space_group_rank"))
        template_rank = int(getattr(row, "template_rank"))
        if not 0 <= material_index < len(targets):
            raise ValueError(f"material index out of range: {material_index}")
        if not 1 <= space_group_rank <= _SPACE_GROUP_TOP_K:
            raise ValueError(f"invalid space_group_rank: {space_group_rank}")
        if not 1 <= template_rank <= _TEMPLATES_PER_SPACE_GROUP:
            raise ValueError(f"invalid template_rank: {template_rank}")
        expected_candidate_index = (
            (space_group_rank - 1) * _TEMPLATES_PER_SPACE_GROUP + template_rank - 1
        )
        if candidate_index != expected_candidate_index:
            raise ValueError(
                f"candidate {(material_index, candidate_index)} does not match "
                f"space-group/template ranks; expected {expected_candidate_index}"
            )
        candidate_key = (material_index, candidate_index)
        if candidate_key in seen_candidates:
            raise ValueError(f"duplicate material/candidate row: {candidate_key}")
        seen_candidates.add(candidate_key)
        material_candidate_indices.setdefault(material_index, set()).add(
            candidate_index
        )

        target_sequence = str(getattr(row, "target_structure_sequence")).strip()
        if not target_sequence or target_sequence.lower() == "nan":
            raise ValueError(
                f"candidate {candidate_key} has no target_structure_sequence"
            )
        previous_sequence = target_sequences.setdefault(material_index, target_sequence)
        if previous_sequence != target_sequence:
            raise ValueError(
                f"material {material_index} has inconsistent target sequences"
            )
        if material_index not in target_sequence_compositions:
            target_query = query_from_gwa_sequence(target_sequence)
            target_sequence_compositions[material_index] = Composition(
                template_reduced_formula(target_query)
            ).reduced_composition

        sequence = getattr(row, "generated_structure_sequence")
        if pd.isna(sequence) or not str(sequence).strip():
            continue
        sequence = str(sequence).strip()
        cached = sequence_cache.get(sequence)
        if cached is None:
            query = query_from_gwa_sequence(sequence)
            candidate_formula = template_reduced_formula(query)
            candidate_composition = Composition(candidate_formula).reduced_composition
            cached = (query, candidate_formula, candidate_composition)
            sequence_cache[sequence] = cached
        query, candidate_formula, candidate_composition = cached
        target = targets.iloc[material_index]
        target_composition = target_compositions[material_index]
        if target_sequence_compositions[material_index] != target_composition:
            raise ValueError(
                f"target structure sequence for material {material_index} "
                f"does not match target formula {target_composition.reduced_formula}"
            )
        if candidate_composition != target_composition:
            raise ValueError(
                f"candidate {candidate_key} formula {candidate_formula} does not "
                f"match target {target_composition.reduced_formula}"
            )

        input_index = len(selected_queries)
        selected_queries.append(query)
        rows.append(
            {
                "input_index": input_index,
                "source_index": source_index,
                "material_index": material_index,
                "candidate_index": candidate_index,
                "material_id": target["material_id"],
                "target_formula": target["pretty_formula"],
                "target_structure_sequence": target_sequence,
                "candidate_formula": candidate_formula,
                "generated_structure_sequence": sequence,
                "spacegroup_number": query["spacegroup_number"],
                "space_group_rank": space_group_rank,
                "template_rank": template_rank,
                "wyckoff_letters": json.dumps(query["wyckoff_letters"]),
                "atom_types": json.dumps(query["atom_types"]),
            }
        )

    expected_indices = set(range(_CANDIDATES_PER_MATERIAL))
    for material_index in range(len(targets)):
        actual_indices = material_candidate_indices.get(material_index, set())
        if actual_indices != expected_indices:
            raise ValueError(
                f"material {material_index} candidate slots are not exactly "
                f"0..{_CANDIDATES_PER_MATERIAL - 1}"
            )

    return pd.DataFrame(rows, columns=_MANIFEST_COLUMNS), selected_queries


def atomic_json_dump(value: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w") as handle:
        json.dump(value, handle)
    os.replace(temporary_path, output_path)


def prepare(args: argparse.Namespace) -> None:
    evaluation = pd.read_csv(args.evaluation_csv)
    targets = pd.read_csv(args.test_csv)
    manifest, selected_queries = build_wyckoffflow_manifest(evaluation, targets)

    args.manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = args.manifest_csv.with_suffix(
        args.manifest_csv.suffix + ".tmp"
    )
    manifest.to_csv(temporary_manifest, index=False)
    os.replace(temporary_manifest, args.manifest_csv)
    atomic_json_dump(selected_queries, args.selected_json)

    counts = manifest.groupby("material_index").size()
    print(f"evaluation rows:           {len(evaluation)}")
    print(f"selected templates:        {len(selected_queries)}")
    print(f"target materials:          {len(targets)}")
    print(f"materials with candidates: {counts.size}")
    print(f"materials without any:     {len(targets) - counts.size}")
    print(f"materials with full pool:  {(counts == _CANDIDATES_PER_MATERIAL).sum()}")
    print(f"manifest: {args.manifest_csv}")
    print(f"selected JSON: {args.selected_json}")


def configure_diffcsp_imports(repo_path: Path) -> None:
    repo_path = repo_path.resolve()
    os.environ.setdefault("PROJECT_ROOT", str(repo_path))
    os.environ.setdefault("HYDRA_JOBS", "/tmp/diffcsppp_hydra")
    os.environ.setdefault("WANDB_DIR", "/tmp")
    os.environ.setdefault("USE_WANDB_LOGGING", "0")
    sys.path.insert(0, str(repo_path / "scripts"))
    sys.path.insert(0, str(repo_path))


def load_model_compat(checkpoint_dir: Path, repo_path: Path, device: torch.device):
    configure_diffcsp_imports(repo_path)
    import hydra
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(
        config_dir=str(checkpoint_dir.resolve()), version_base="1.1"
    ):
        cfg = compose(config_name="hparams")
    model = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )

    checkpoints = sorted(checkpoint_dir.glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint found in {checkpoint_dir}")
    checkpoint_path = next(
        (path for path in checkpoints if "last" in path.name), checkpoints[-1]
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    model.to(device)
    return model, checkpoint_path


def shard_bounds(total: int, shard_index: int, num_shards: int) -> tuple[int, int]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    return total * shard_index // num_shards, total * (shard_index + 1) // num_shards


def empty_sample_payload(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "metadata": metadata,
        "input_indices": torch.empty(0, dtype=torch.long),
        "frac_coords": torch.empty((0, 3), dtype=torch.float32),
        "atom_types": torch.empty(0, dtype=torch.long),
        "lattices": torch.empty((0, 3, 3), dtype=torch.float32),
        "lengths": torch.empty((0, 3), dtype=torch.float32),
        "angles": torch.empty((0, 3), dtype=torch.float32),
        "num_atoms": torch.empty(0, dtype=torch.long),
    }


def save_sample_payload(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temporary_output)
    os.replace(temporary_output, output_path)


def sample(args: argparse.Namespace) -> None:
    if args.batch_size != 128 or args.num_shards != 4:
        raise ValueError("DiffCSP++ sampling requires --batch-size 128 --num-shards 4")
    # Hydra initialization in DiffCSP++ may change cwd; anchor all paths first.
    args.diffcsp_repo = args.diffcsp_repo.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.selected_json = args.selected_json.resolve()
    args.output = args.output.resolve()
    if args.output.exists() and not args.overwrite:
        print(f"output already exists, skipping: {args.output}")
        return

    with args.selected_json.open() as handle:
        all_queries = json.load(handle)
    start, end = shard_bounds(len(all_queries), args.shard_index, args.num_shards)
    queries = all_queries[start:end]
    query_indices = list(range(start, end))
    if not queries:
        save_sample_payload(
            empty_sample_payload(
                {
                    "checkpoint": str(args.checkpoint_dir),
                    "selected_json": str(args.selected_json),
                    "shard_index": args.shard_index,
                    "num_shards": args.num_shards,
                    "shard_start": start,
                    "shard_end": end,
                    "batch_size": args.batch_size,
                    "seed": args.seed,
                    "step_lr": args.step_lr,
                    "elapsed_seconds": 0.0,
                    "parse_errors": [],
                }
            ),
            args.output,
        )
        print(f"saved empty shard: {args.output}")
        return

    configure_diffcsp_imports(args.diffcsp_repo)
    import diffcsp.pl_modules.diffusion as diffusion_module
    from eval_utils import lattices_to_params_shape
    from sample_api import CustomDataset, get_data_from_syminfo
    from torch_geometric.loader import DataLoader

    data_list = []
    valid_query_indices = []
    parse_errors = []
    for query_index, query in zip(query_indices, queries):
        try:
            data_list.append(get_data_from_syminfo(**query))
            valid_query_indices.append(query_index)
        except Exception as error:
            parse_errors.append({"input_index": query_index, "error": repr(error)})
    if not data_list:
        raise RuntimeError("the shard contains no parseable templates")

    device = torch.device(args.device)
    torch.manual_seed(args.seed + args.shard_index)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + args.shard_index)
    model, checkpoint_path = load_model_compat(
        args.checkpoint_dir, args.diffcsp_repo, device
    )
    diffusion_module.tqdm = lambda iterable, *unused_args, **unused_kwargs: iterable

    loader = DataLoader(
        CustomDataset(data_list),
        batch_size=min(args.batch_size, len(data_list)),
        shuffle=False,
    )
    outputs: dict[str, list[torch.Tensor]] = {
        "frac_coords": [],
        "atom_types": [],
        "lattices": [],
        "lengths": [],
        "angles": [],
        "num_atoms": [],
    }
    offset = 0
    started_at = time.time()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            batch = batch.to(device)
            sampled, trajectory = model.sample(batch, step_lr=args.step_lr)
            lengths, angles = lattices_to_params_shape(sampled["lattices"])
            for key in ("frac_coords", "atom_types", "lattices", "num_atoms"):
                outputs[key].append(sampled[key].detach().cpu())
            outputs["lengths"].append(lengths.detach().cpu())
            outputs["angles"].append(angles.detach().cpu())
            offset += batch.num_graphs
            del trajectory, sampled, batch
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elapsed = time.time() - started_at
            print(
                f"shard {args.shard_index}: batch {batch_index + 1}/{len(loader)}, "
                f"samples {offset}/{len(data_list)}, elapsed {elapsed:.1f}s",
                flush=True,
            )

    result = {
        "metadata": {
            "checkpoint": str(checkpoint_path.resolve()),
            "selected_json": str(args.selected_json.resolve()),
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "shard_start": start,
            "shard_end": end,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "step_lr": args.step_lr,
            "elapsed_seconds": time.time() - started_at,
            "parse_errors": parse_errors,
        },
        "input_indices": torch.tensor(valid_query_indices, dtype=torch.long),
    }
    for key, tensors in outputs.items():
        result[key] = torch.cat(tensors, dim=0)

    save_sample_payload(result, args.output)
    print(f"saved shard: {args.output}")


def load_sample_records(sample_paths: list[Path]) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for sample_path in sample_paths:
        payload = torch.load(sample_path, map_location="cpu", weights_only=False)
        input_indices = payload["input_indices"].tolist()
        num_atoms = payload["num_atoms"].tolist()
        if len(input_indices) != len(num_atoms):
            raise ValueError(f"index/count length mismatch in {sample_path}")
        atom_offset = 0
        for local_index, (input_index, atom_count) in enumerate(
            zip(input_indices, num_atoms)
        ):
            if input_index in records:
                raise ValueError(f"duplicate sampled input_index: {input_index}")
            next_offset = atom_offset + int(atom_count)
            records[int(input_index)] = {
                "frac_coords": payload["frac_coords"][atom_offset:next_offset].numpy(),
                "atom_types": payload["atom_types"][atom_offset:next_offset].numpy(),
                "lengths": payload["lengths"][local_index].numpy(),
                "angles": payload["angles"][local_index].numpy(),
            }
            atom_offset = next_offset
        if atom_offset != len(payload["atom_types"]):
            raise ValueError(f"unused atom rows in {sample_path}")
    return records


_MATCHER: StructureMatcher | None = None
_SMACT_VALIDITY = None


def initialize_match_worker(
    diffcsp_repo: str, stol: float, angle_tol: float, ltol: float
) -> None:
    global _MATCHER, _SMACT_VALIDITY
    configure_diffcsp_imports(Path(diffcsp_repo))
    from eval_utils import smact_validity

    _MATCHER = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)
    _SMACT_VALIDITY = smact_validity


def structure_validity(structure: Structure, cutoff: float = 0.5) -> bool:
    distance_matrix = structure.distance_matrix
    padded = distance_matrix + np.diag(
        np.ones(distance_matrix.shape[0]) * (cutoff + 10.0)
    )
    return bool(padded.min() >= cutoff and structure.volume >= 0.1)


def composition_validity(structure: Structure) -> bool:
    if _SMACT_VALIDITY is None:
        raise RuntimeError("match worker was not initialized")
    composition = structure.composition
    elements = sorted(composition.elements, key=lambda element: element.Z)
    counts = np.array([int(round(composition[element])) for element in elements])
    counts //= np.gcd.reduce(counts)
    return bool(_SMACT_VALIDITY(tuple(element.Z for element in elements), counts))


def evaluate_material_task(task: dict[str, Any]) -> dict[str, Any]:
    if _MATCHER is None:
        raise RuntimeError("match worker was not initialized")
    result = {
        "material_index": task["material_index"],
        "structure_match": False,
        "matched_input_index": None,
        "matched_candidate_index": None,
        "matched_rms": None,
        "evaluated_candidate_count": 0,
        "valid_candidate_count": 0,
        "match_errors": 0,
    }
    try:
        target = Structure.from_str(task["target_cif"], fmt="cif")
        target_valid = composition_validity(target) and structure_validity(target)
    except Exception:
        result["match_errors"] += 1
        return result
    if not target_valid:
        return result

    for candidate in task["candidates"]:
        result["evaluated_candidate_count"] += 1
        try:
            lengths = candidate["lengths"]
            angles = candidate["angles"]
            if not np.isfinite(lengths).all() or not np.isfinite(angles).all():
                continue
            if np.min(lengths) <= 0:
                continue
            structure = Structure(
                lattice=Lattice.from_parameters(*(lengths.tolist() + angles.tolist())),
                species=candidate["atom_types"],
                coords=candidate["frac_coords"],
                coords_are_cartesian=False,
            )
            if not composition_validity(structure) or not structure_validity(structure):
                continue
            result["valid_candidate_count"] += 1
            rms_distance = _MATCHER.get_rms_dist(structure, target)
            if rms_distance is None:
                continue
            result.update(
                {
                    "structure_match": True,
                    "matched_input_index": candidate["input_index"],
                    "matched_candidate_index": candidate["candidate_index"],
                    "matched_rms": float(rms_distance[0]),
                }
            )
            break
        except Exception:
            result["match_errors"] += 1
    return result


def percentage(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else float("nan")


def evaluate(args: argparse.Namespace) -> None:
    manifest = pd.read_csv(args.manifest_csv)
    targets = pd.read_csv(args.test_csv)
    required_manifest_columns = {"target_structure_sequence"}
    missing = required_manifest_columns - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest is missing target columns: {sorted(missing)}")

    sample_records = load_sample_records(args.sample_files)
    expected_indices = set(manifest["input_index"].astype(int))
    sampled_indices = set(sample_records)
    unexpected_indices = sampled_indices - expected_indices
    if unexpected_indices:
        raise ValueError(
            f"sample files contain unexpected indices: {len(unexpected_indices)}"
        )
    missing_indices = expected_indices - sampled_indices
    if missing_indices and not args.allow_partial:
        raise ValueError(f"sample files are missing {len(missing_indices)} inputs")

    manifest = manifest[manifest["input_index"].isin(sampled_indices)].copy()
    manifest["space_group_match"] = [
        int(row.spacegroup_number)
        == int(str(row.target_structure_sequence).split("-", 1)[0])
        for row in manifest.itertuples()
    ]
    manifest["gwa_match"] = [
        str(row.generated_structure_sequence) == str(row.target_structure_sequence)
        for _, row in manifest.iterrows()
    ]

    grouped = {index: frame for index, frame in manifest.groupby("material_index")}
    tasks = []
    material_rows: dict[int, dict[str, Any]] = {}
    for material_index, target in targets.iterrows():
        frame = grouped.get(material_index)
        if frame is None:
            candidate_count = 0
            space_group_match = False
            gwa_match = False
            candidates = []
        else:
            frame = frame.sort_values("candidate_index")
            candidate_count = len(frame)
            space_group_match = bool(frame["space_group_match"].any())
            gwa_match = bool(frame["gwa_match"].any())
            candidates = []
            for row in frame.itertuples():
                record = dict(sample_records[int(row.input_index)])
                record.update(
                    {
                        "input_index": int(row.input_index),
                        "candidate_index": int(row.candidate_index),
                    }
                )
                candidates.append(record)
        material_rows[material_index] = {
            "material_index": material_index,
            "material_id": target["material_id"],
            "target_formula": target["pretty_formula"],
            "candidate_count": candidate_count,
            "space_group_match": space_group_match,
            "gwa_match": gwa_match,
        }
        if candidates:
            tasks.append(
                {
                    "material_index": material_index,
                    "target_cif": target["cif"],
                    "candidates": candidates,
                }
            )

    started_at = time.time()
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=initialize_match_worker,
        initargs=(str(args.diffcsp_repo), args.stol, args.angle_tol, args.ltol),
    ) as executor:
        results = executor.map(evaluate_material_task, tasks, chunksize=8)
        for completed, result in enumerate(results, start=1):
            material_rows[result["material_index"]].update(result)
            if completed % 100 == 0 or completed == len(tasks):
                print(
                    f"matched {completed}/{len(tasks)} materials, "
                    f"elapsed {time.time() - started_at:.1f}s",
                    flush=True,
                )

    for material_index, row in material_rows.items():
        if "structure_match" not in row:
            row.update(
                {
                    "structure_match": False,
                    "matched_input_index": None,
                    "matched_candidate_index": None,
                    "matched_rms": None,
                    "evaluated_candidate_count": 0,
                    "valid_candidate_count": 0,
                    "match_errors": 0,
                }
            )
    results_frame = pd.DataFrame(
        [material_rows[index] for index in range(len(targets))]
    )

    total = len(results_frame)
    formula_hits = int((results_frame["candidate_count"] > 0).sum())
    space_group_hits = int(results_frame["space_group_match"].sum())
    gwa_hits = int(results_frame["gwa_match"].sum())
    structure_hits = int(results_frame["structure_match"].sum())
    both = int((results_frame["gwa_match"] & results_frame["structure_match"]).sum())
    structure_only = int(
        ((~results_frame["gwa_match"]) & results_frame["structure_match"]).sum()
    )
    gwa_only = int(
        (results_frame["gwa_match"] & (~results_frame["structure_match"])).sum()
    )
    neither = total - both - structure_only - gwa_only
    summary = {
        "materials": total,
        "sampled_templates": len(sample_records),
        "missing_templates": len(missing_indices),
        "formula_top20": {
            "hits": formula_hits,
            "percent": percentage(formula_hits, total),
        },
        "space_group_top20": {
            "hits": space_group_hits,
            "percent": percentage(space_group_hits, total),
        },
        "gwa_top20": {"hits": gwa_hits, "percent": percentage(gwa_hits, total)},
        "structure_top20": {
            "hits": structure_hits,
            "percent": percentage(structure_hits, total),
        },
        "intersection": {
            "both": both,
            "structure_only": structure_only,
            "gwa_only": gwa_only,
            "neither": neither,
        },
        "structure_given_gwa_hit_percent": percentage(both, gwa_hits),
        "structure_given_gwa_miss_percent": percentage(
            structure_only, total - gwa_hits
        ),
        "matcher": {
            "stol": args.stol,
            "angle_tol": args.angle_tol,
            "ltol": args.ltol,
        },
        "elapsed_seconds": time.time() - started_at,
    }

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary_csv = args.output_csv.with_suffix(args.output_csv.suffix + ".tmp")
    results_frame.to_csv(temporary_csv, index=False)
    os.replace(temporary_csv, args.output_csv)
    atomic_json_dump(summary, args.summary_json)
    print(json.dumps(summary, indent=2))
    print(f"material results: {args.output_csv}")
    print(f"summary: {args.summary_json}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare", help="Convert WyckoffFlow candidates to DiffCSP++ inputs"
    )
    prepare_parser.add_argument("--evaluation-csv", type=Path, required=True)
    prepare_parser.add_argument("--test-csv", type=Path, required=True)
    prepare_parser.add_argument("--manifest-csv", type=Path, required=True)
    prepare_parser.add_argument("--selected-json", type=Path, required=True)
    prepare_parser.set_defaults(func=prepare)

    sample_parser = subparsers.add_parser("sample")
    sample_parser.add_argument("--diffcsp-repo", type=Path, required=True)
    sample_parser.add_argument("--checkpoint-dir", type=Path, required=True)
    sample_parser.add_argument("--selected-json", type=Path, required=True)
    sample_parser.add_argument("--output", type=Path, required=True)
    sample_parser.add_argument("--shard-index", type=int, default=0)
    sample_parser.add_argument("--num-shards", type=int, default=4)
    sample_parser.add_argument("--batch-size", type=int, default=128)
    sample_parser.add_argument("--seed", type=int, default=42)
    sample_parser.add_argument("--step-lr", type=float, default=1e-5)
    sample_parser.add_argument("--device", default="cuda")
    sample_parser.add_argument("--overwrite", action="store_true")
    sample_parser.set_defaults(func=sample)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--diffcsp-repo", type=Path, required=True)
    evaluate_parser.add_argument("--manifest-csv", type=Path, required=True)
    evaluate_parser.add_argument("--test-csv", type=Path, required=True)
    evaluate_parser.add_argument("--sample-files", type=Path, nargs="+", required=True)
    evaluate_parser.add_argument("--output-csv", type=Path, required=True)
    evaluate_parser.add_argument("--summary-json", type=Path, required=True)
    evaluate_parser.add_argument("--workers", type=int, default=16)
    evaluate_parser.add_argument("--stol", type=float, default=0.5)
    evaluate_parser.add_argument("--angle-tol", type=float, default=10.0)
    evaluate_parser.add_argument("--ltol", type=float, default=0.3)
    evaluate_parser.add_argument("--allow-partial", action="store_true")
    evaluate_parser.set_defaults(func=evaluate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
