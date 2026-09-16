#!/usr/bin/env python3
"""Generate concrete structures with DiffCSP's symmetry-aware CSP model.

This runner carries the symmetry-template parsing and ``SymData`` construction
used by ``/home/wangqc/DiffCSP`` into this repository. DiffCSP model code and
checkpoints remain external and are selected with command-line paths.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.common.lookup_tables import chemical_symbols  # noqa: E402
from models.common.wyckoff import wyckoff_multiplicity  # noqa: E402


class SymData(Data):
    """PyG graph whose symmetry-anchor indices need per-graph offsets."""

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if "batch" in key and isinstance(value, Tensor):
            return int(value.max()) + 1
        if "index" in key or key == "face":
            return self.num_nodes
        if key == "symm_map":
            return self.num_nodes
        return 0


def validate_query(query: dict[str, Any]) -> tuple[int, list[str], list[str]]:
    """Validate one DiffCSP symmetry query without importing PyXtal."""

    space_group = int(query["spacegroup_number"])
    if not 1 <= space_group <= 230:
        raise ValueError(f"space group must be in 1..230: {space_group}")
    wyckoff_letters = [str(value) for value in query["wyckoff_letters"]]
    atom_types = [str(value) for value in query["atom_types"]]
    if not wyckoff_letters or len(wyckoff_letters) != len(atom_types):
        raise ValueError(
            "wyckoff_letters and atom_types must have equal nonzero length"
        )
    for label in wyckoff_letters:
        wyckoff_multiplicity(label)
    for element in atom_types:
        if element not in chemical_symbols[1:]:
            raise ValueError(f"unknown element: {element}")
    return space_group, wyckoff_letters, atom_types


def query_to_data(query: dict[str, Any]) -> SymData:
    """Expand one space-group/Wyckoff query into DiffCSP symmetry tensors."""

    from pyxtal.symmetry import Wyckoff_position

    space_group, wyckoff_letters, atom_types = validate_query(query)
    multiplicities = [wyckoff_multiplicity(label) for label in wyckoff_letters]
    operations = []
    for label, multiplicity in zip(wyckoff_letters, multiplicities):
        orbit_operations = list(
            Wyckoff_position.from_group_and_letter(space_group, label)
        )
        if len(orbit_operations) != multiplicity:
            raise ValueError(
                f"space group {space_group} label {label} has "
                f"{len(orbit_operations)} operations, expected {multiplicity}"
            )
        operations.extend(operation.affine_matrix for operation in orbit_operations)

    atom_numbers = np.concatenate(
        [
            np.full(multiplicity, chemical_symbols.index(element), dtype=np.int64)
            for element, multiplicity in zip(atom_types, multiplicities)
        ]
    )
    ops = np.asarray(operations, dtype=np.float32)
    ops_inv = np.linalg.pinv(ops[:, :3, :3]).astype(np.float32)
    orbit_starts = np.cumsum([0, *multiplicities[:-1]])
    anchor_index = np.repeat(orbit_starts, multiplicities)
    num_atoms = int(sum(multiplicities))
    return SymData(
        atom_types=torch.from_numpy(atom_numbers),
        num_atoms=num_atoms,
        num_nodes=num_atoms,
        spacegroup=space_group,
        ops=torch.from_numpy(ops),
        ops_inv=torch.from_numpy(ops_inv),
        anchor_index=torch.from_numpy(anchor_index).long(),
    )


def configure_diffcsp_imports(repo_path: Path) -> None:
    repo_path = repo_path.resolve()
    os.environ.setdefault("PROJECT_ROOT", str(repo_path))
    os.environ.setdefault("HYDRA_JOBS", "/tmp/diffcsp_hydra")
    os.environ.setdefault("WANDB_DIR", "/tmp")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/diffcsp_mpl")
    sys.path.insert(0, str(repo_path / "scripts"))
    sys.path.insert(0, str(repo_path))


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(choice)


def empty_payload(metadata: dict[str, Any]) -> dict[str, Any]:
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


def save_payload(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, output_path)


def sample(args: argparse.Namespace) -> None:
    args.diffcsp_repo = args.diffcsp_repo.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.selected_json = args.selected_json.resolve()
    args.output = args.output.resolve()
    if args.output.exists() and not args.overwrite:
        print(f"output already exists, skipping: {args.output}")
        return

    with args.selected_json.open() as handle:
        queries = json.load(handle)
    if not queries:
        save_payload(
            empty_payload(
                {
                    "checkpoint": str(args.checkpoint_dir),
                    "selected_json": str(args.selected_json),
                    "seed": args.seed,
                    "ode_int_steps": args.ode_int_steps,
                    "parse_errors": [],
                    "elapsed_seconds": 0.0,
                }
            ),
            args.output,
        )
        return

    data_list = []
    valid_query_indices = []
    parse_errors = []
    for query_index, query in enumerate(queries):
        try:
            data_list.append(query_to_data(query))
            valid_query_indices.append(query_index)
        except Exception as error:
            parse_errors.append({"input_index": query_index, "error": repr(error)})
    if not data_list:
        raise RuntimeError("no symmetry template could be parsed")

    configure_diffcsp_imports(args.diffcsp_repo)
    from eval_utils import lattices_to_params_shape, load_model

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model, _, _ = load_model(args.checkpoint_dir, load_data=False)
    model = model.to(device).eval()

    loader = DataLoader(data_list, batch_size=args.batch_size, shuffle=False)
    outputs: dict[str, list[torch.Tensor]] = {
        "frac_coords": [],
        "atom_types": [],
        "lattices": [],
        "lengths": [],
        "angles": [],
        "num_atoms": [],
    }
    started_at = time.time()
    completed = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            batch = batch.to(device)
            sampled, trajectory = model.sample(
                batch,
                N=args.ode_int_steps,
                anneal_coords=True,
                anneal_slope=args.anneal_slope,
            )
            lengths, angles = lattices_to_params_shape(sampled["lattices"])
            for key in ("frac_coords", "atom_types", "lattices", "num_atoms"):
                outputs[key].append(sampled[key].detach().cpu())
            outputs["lengths"].append(lengths.detach().cpu())
            outputs["angles"].append(angles.detach().cpu())
            completed += batch.num_graphs
            del trajectory, sampled, batch
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(
                f"batch {batch_index + 1}/{len(loader)}, "
                f"samples {completed}/{len(data_list)}, "
                f"elapsed {time.time() - started_at:.1f}s",
                flush=True,
            )

    result = {
        "metadata": {
            "checkpoint": str(args.checkpoint_dir),
            "selected_json": str(args.selected_json),
            "batch_size": args.batch_size,
            "seed": args.seed,
            "ode_int_steps": args.ode_int_steps,
            "anneal_slope": args.anneal_slope,
            "parse_errors": parse_errors,
            "elapsed_seconds": time.time() - started_at,
        },
        "input_indices": torch.tensor(valid_query_indices, dtype=torch.long),
    }
    for key, tensors in outputs.items():
        result[key] = torch.cat(tensors, dim=0)
    save_payload(result, args.output)
    print(f"saved structures: {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diffcsp-repo",
        type=Path,
        default=Path("/home/wangqc/DiffCSP"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("/home/wangqc/DiffCSP/ckpt/CSP-mp20-sym"),
    )
    parser.add_argument("--selected-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--ode-int-steps", type=int, default=100)
    parser.add_argument("--anneal-slope", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    sample(build_parser().parse_args())
