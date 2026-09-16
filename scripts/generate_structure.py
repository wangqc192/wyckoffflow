#!/usr/bin/env python3
"""Generate symmetry-constrained crystal structures from a chemical formula.

Stages:
1. NextCrystal predicts Top-K space groups.
2. This repository's Wyckoff flow samples exact-composition templates.
3. DiffCSP expands every symmetry template into a concrete structure.

The default paths match the local repositories used to develop this pipeline,
but every external repository/checkpoint can be overridden from the command line.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from pymatgen.core import Lattice, Structure

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.generation_pipeline import (  # noqa: E402
    load_flow_model,
    predict_nextcrystal_space_groups,
    sample_wyckoff_templates,
    write_selected_queries,
)


def resolve_device(choice: str) -> str:
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return choice


def public_template_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    excluded = {"query", "sample"}
    return [
        {key: value for key, value in row.items() if key not in excluded}
        for row in rows
    ]


def write_diffcsp_template_csv(rows: list[dict[str, Any]], path: Path) -> None:
    pd.DataFrame(
        [
            {
                "formula": row["target_formula"],
                "num_evals": 1,
                "pressure": 0,
                "wyckoff": row["wyckoff_template"],
            }
            for row in rows
        ]
    ).to_csv(path, index=False)


def load_structure_records(sample_paths: list[Path]) -> dict[int, dict[str, Any]]:
    """Load standardized DiffCSP/DiffCSP++ sample payloads by input index."""

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
            input_index = int(input_index)
            if input_index in records:
                raise ValueError(f"duplicate sampled input_index: {input_index}")
            next_offset = atom_offset + int(atom_count)
            records[input_index] = {
                "lengths": payload["lengths"][local_index].tolist(),
                "angles": payload["angles"][local_index].tolist(),
                "atom_types": payload["atom_types"][atom_offset:next_offset].tolist(),
                "frac_coords": payload["frac_coords"][atom_offset:next_offset].tolist(),
                "sample_file": str(sample_path),
            }
            atom_offset = next_offset
        if atom_offset != len(payload["atom_types"]):
            raise ValueError(f"unused atom rows in {sample_path}")
    return records


def write_structure_files(
    sample_paths: list[Path], templates_csv: Path, output_dir: Path
) -> int:
    """Convert standardized sample payloads into indexed CIF and POSCAR files."""

    templates = pd.read_csv(templates_csv)
    records_by_index = load_structure_records(sample_paths)
    unexpected = set(records_by_index) - set(range(len(templates)))
    if unexpected:
        raise ValueError(f"sample output has unexpected input indices: {unexpected}")

    cif_dir = output_dir / "cif"
    poscar_dir = output_dir / "poscar"
    cif_dir.mkdir(parents=True, exist_ok=True)
    poscar_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for input_index, sample in sorted(records_by_index.items()):
        structure = Structure(
            lattice=Lattice.from_parameters(
                *sample["lengths"],
                *sample["angles"],
            ),
            species=sample["atom_types"],
            coords=sample["frac_coords"],
            coords_are_cartesian=False,
        )
        row = templates.iloc[input_index]
        stem = (
            f"candidate_{int(row.candidate_index):03d}_"
            f"sg{int(row.generated_space_group)}"
        )
        cif_path = cif_dir / f"{stem}.cif"
        poscar_path = poscar_dir / f"{stem}.vasp"
        structure.to(filename=str(cif_path), fmt="cif")
        structure.to(filename=str(poscar_path), fmt="poscar")
        records.append(
            {
                "input_index": input_index,
                "candidate_index": int(row.candidate_index),
                "space_group_rank": int(row.space_group_rank),
                "template_rank": int(row.template_rank),
                "generated_space_group": int(row.generated_space_group),
                "generated_structure_sequence": row.generated_structure_sequence,
                "sample_file": sample["sample_file"],
                "cif": str(cif_path),
                "poscar": str(poscar_path),
            }
        )
    pd.DataFrame(records).to_csv(output_dir / "structures.csv", index=False)
    return len(records)


def run_diffcsp(
    args: argparse.Namespace, selected_json: Path, output_dir: Path
) -> list[Path]:
    sample_path = output_dir / "diffcsp_sample.pt"
    command = [
        str(args.diffcsp_python),
        str(Path(__file__).with_name("run_diffcsp_symmetry.py")),
        "--diffcsp-repo",
        str(args.diffcsp_repo),
        "--checkpoint-dir",
        str(args.diffcsp_checkpoint),
        "--selected-json",
        str(selected_json),
        "--output",
        str(sample_path),
        "--batch-size",
        str(args.diffcsp_batch_size),
        "--ode-int-steps",
        str(args.diffcsp_ode_int_steps),
        "--anneal-slope",
        str(args.diffcsp_anneal_slope),
        "--seed",
        str(args.seed),
        "--device",
        args.diffcsp_device,
        "--overwrite",
    ]
    subprocess.run(command, check=True)
    return [sample_path]


def run_diffcsppp(
    args: argparse.Namespace, selected_json: Path, output_dir: Path
) -> list[Path]:
    sample_paths = [
        output_dir / f"diffcsppp_sample_shard{shard_index}.pt"
        for shard_index in range(4)
    ]
    for shard_index, sample_path in enumerate(sample_paths):
        command = [
            str(args.diffcsppp_python),
            str(Path(__file__).with_name("run_nextcrystal_diffcsppp.py")),
            "sample",
            "--diffcsp-repo",
            str(args.diffcsppp_repo),
            "--checkpoint-dir",
            str(args.diffcsppp_checkpoint),
            "--selected-json",
            str(selected_json),
            "--output",
            str(sample_path),
            "--shard-index",
            str(shard_index),
            "--num-shards",
            "4",
            "--batch-size",
            "128",
            "--seed",
            str(args.seed),
            "--device",
            resolve_device(args.diffcsp_device),
            "--overwrite",
        ]
        subprocess.run(command, check=True)
    return sample_paths


def main(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    print("[1/3] NextCrystal space-group prediction", flush=True)
    sg_predictions = predict_nextcrystal_space_groups(
        args.formula,
        nextcrystal_root=args.nextcrystal_root,
        checkpoint=args.nextcrystal_checkpoint,
        input_csv=output_dir / "nextcrystal_input.csv",
        top_k=args.space_group_top_k,
        device=device,
    )
    sg_predictions.to_csv(output_dir / "space_groups.csv", index=False)
    ranked_space_groups = [
        int(value)
        for value in sg_predictions.sort_values("SG_Rank")["Spacegroup Number"].tolist()
        if 1 <= int(value) <= 230
    ]
    if len(ranked_space_groups) < args.space_group_top_k:
        raise RuntimeError("NextCrystal did not return enough valid space groups")

    print("[2/3] Wyckoff template generation", flush=True)
    flow_model, flow_config = load_flow_model(args.flow_checkpoint, device)
    rows = sample_wyckoff_templates(
        flow_model,
        args.formula,
        ranked_space_groups[: args.space_group_top_k],
        templates_per_space_group=args.templates_per_space_group,
        template_pool_size=args.template_pool_size,
        flow_steps=args.flow_steps,
    )
    if not rows:
        raise RuntimeError("Wyckoff flow produced no exact-composition templates")
    templates_csv = output_dir / "templates.csv"
    pd.DataFrame(public_template_rows(rows)).to_csv(templates_csv, index=False)
    torch.save([row["sample"] for row in rows], output_dir / "templates.pt")
    write_diffcsp_template_csv(rows, output_dir / "diffcsp_templates.csv")
    selected_json = output_dir / "diffcsp_queries.json"
    write_selected_queries(rows, selected_json)

    effective_flow_steps = (
        args.flow_steps
        if args.flow_steps is not None
        else int(flow_config.model.model_config.flow_steps)
    )
    run_metadata = {
        "formula": args.formula,
        "seed": args.seed,
        "device": device,
        "nextcrystal_root": str(args.nextcrystal_root.resolve()),
        "nextcrystal_checkpoint": str(args.nextcrystal_checkpoint.resolve()),
        "flow_checkpoint": str(args.flow_checkpoint.resolve()),
        "flow_steps": effective_flow_steps,
        "space_group_top_k": args.space_group_top_k,
        "templates_per_space_group": args.templates_per_space_group,
        "template_pool_size": args.template_pool_size,
        "generated_templates": len(rows),
        "structure_backend": args.structure_backend,
        "structure_device": resolve_device(args.diffcsp_device),
    }
    if args.structure_backend == "diffcsp":
        run_metadata.update(
            {
                "structure_repo": str(args.diffcsp_repo.resolve()),
                "structure_checkpoint": str(args.diffcsp_checkpoint.resolve()),
                "structure_python": str(args.diffcsp_python.resolve()),
                "diffcsp_batch_size": args.diffcsp_batch_size,
                "diffcsp_ode_int_steps": args.diffcsp_ode_int_steps,
                "diffcsp_anneal_slope": args.diffcsp_anneal_slope,
            }
        )
    else:
        run_metadata.update(
            {
                "structure_repo": str(args.diffcsppp_repo.resolve()),
                "structure_checkpoint": str(args.diffcsppp_checkpoint.resolve()),
                "structure_python": str(args.diffcsppp_python.resolve()),
                "diffcsppp_batch_size": 128,
                "diffcsppp_num_shards": 4,
            }
        )
    (output_dir / "run.json").write_text(
        json.dumps(run_metadata, indent=2), encoding="utf-8"
    )

    if args.prepare_only:
        print(f"Prepared {len(rows)} templates in {output_dir}")
        return

    if args.structure_backend == "diffcsp":
        print("[3/3] DiffCSP symmetry-constrained structure generation", flush=True)
        sample_paths = run_diffcsp(args, selected_json, output_dir)
    else:
        print("[3/3] DiffCSP++ symmetry-constrained structure generation", flush=True)
        sample_paths = run_diffcsppp(args, selected_json, output_dir)
    count = write_structure_files(sample_paths, templates_csv, output_dir)
    print(f"Generated {count} structures in {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formula", required=True)
    parser.add_argument("--flow-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--nextcrystal-root",
        type=Path,
        default=Path("/home/wangqc/NextCrystal"),
    )
    parser.add_argument(
        "--nextcrystal-checkpoint",
        type=Path,
        default=Path("/home/wangqc/NextCrystal/artifacts/mp_20/spacegroup.ckpt"),
    )
    parser.add_argument(
        "--structure-backend",
        choices=["diffcsp", "diffcsppp"],
        default="diffcsp",
    )
    parser.add_argument(
        "--diffcsp-repo",
        type=Path,
        default=Path("/home/wangqc/DiffCSP"),
    )
    parser.add_argument(
        "--diffcsp-checkpoint",
        type=Path,
        default=Path("/home/wangqc/DiffCSP/ckpt/CSP-mp20-sym"),
    )
    parser.add_argument(
        "--diffcsp-python",
        type=Path,
        default=Path("/home/wangqc/miniconda3/envs/crystalflow/bin/python"),
    )
    parser.add_argument("--diffcsp-batch-size", type=int, default=50)
    parser.add_argument("--diffcsp-ode-int-steps", type=int, default=100)
    parser.add_argument("--diffcsp-anneal-slope", type=float, default=5.0)
    parser.add_argument(
        "--diffcsppp-repo",
        type=Path,
        default=Path("/home/wangqc/DiffCSP-PP"),
    )
    parser.add_argument(
        "--diffcsppp-checkpoint",
        type=Path,
        default=Path("/home/wangqc/DiffCSP-PP/checkpoint/mp_csp"),
    )
    parser.add_argument(
        "--diffcsppp-python",
        type=Path,
        default=Path("/home/wangqc/miniconda3/envs/nextcrystal/bin/python"),
    )
    parser.add_argument("--space-group-top-k", type=int, default=5)
    parser.add_argument("--templates-per-space-group", type=int, default=4)
    parser.add_argument("--template-pool-size", type=int, default=16)
    parser.add_argument("--flow-steps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--diffcsp-device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )
    parser.add_argument("--prepare-only", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
