"""Evaluate DiffCSP++ CSP samples stored in ``.pt`` files.

Only the reconstruction/CSP task is implemented.  DiffCSP++ stores the atom
coordinates for all crystals in one concatenated array, so this script keeps
its loader separate from the CrystalFlow loader while sharing the evaluator
in :mod:`eval_utils`.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from tqdm.auto import tqdm

try:
    from .eval_utils import (
        CrystalRecord,
        RecEval,
        load_targets,
        manifest_columns,
        resolve_pt_paths,
        structure_to_record,
        write_metrics,
    )
except ImportError:  # direct ``python scripts/compute_metrics_diffcsppp.py``
    from eval_utils import (  # type: ignore[no-redef]
        CrystalRecord,
        RecEval,
        load_targets,
        manifest_columns,
        resolve_pt_paths,
        structure_to_record,
        write_metrics,
    )


def load_diffcsppp_records(
    sample_paths: Path | str | Sequence[Path | str],
    show_progress: bool = True,
) -> dict[int, CrystalRecord]:
    """Load DiffCSP++ PT files keyed by their explicit ``input_indices``."""
    import torch

    paths = resolve_pt_paths(sample_paths)
    records: dict[int, CrystalRecord] = {}
    iterator = tqdm(
        paths,
        desc="Loading DiffCSP++ PT files",
        unit="file",
        disable=not show_progress,
    )
    for sample_path in iterator:
        payload = torch.load(sample_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError(f"sample payload is not a mapping: {sample_path}")
        required = (
            "input_indices",
            "frac_coords",
            "atom_types",
            "lengths",
            "angles",
            "num_atoms",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(
                f"DiffCSP++ sample is missing fields {missing}: {sample_path}"
            )

        input_indices = payload["input_indices"]
        if hasattr(input_indices, "detach"):
            input_indices = input_indices.detach().cpu().numpy()
        else:
            input_indices = np.asarray(input_indices)
        input_indices = np.asarray(input_indices).reshape(-1)

        num_atoms = payload["num_atoms"]
        if hasattr(num_atoms, "detach"):
            num_atoms = num_atoms.detach().cpu().numpy()
        else:
            num_atoms = np.asarray(num_atoms)
        num_atoms = np.asarray(num_atoms).reshape(-1)

        frac_coords = payload["frac_coords"]
        if hasattr(frac_coords, "detach"):
            frac_coords = frac_coords.detach().cpu().numpy()
        else:
            frac_coords = np.asarray(frac_coords)
        atom_types = payload["atom_types"]
        if hasattr(atom_types, "detach"):
            atom_types = atom_types.detach().cpu().numpy()
        else:
            atom_types = np.asarray(atom_types)
        lengths = payload["lengths"]
        if hasattr(lengths, "detach"):
            lengths = lengths.detach().cpu().numpy()
        else:
            lengths = np.asarray(lengths)
        angles = payload["angles"]
        if hasattr(angles, "detach"):
            angles = angles.detach().cpu().numpy()
        else:
            angles = np.asarray(angles)

        if len(input_indices) != len(num_atoms):
            raise ValueError(f"input_indices/num_atoms mismatch: {sample_path}")
        if len(frac_coords) != len(atom_types):
            raise ValueError(f"frac_coords/atom_types mismatch: {sample_path}")
        if len(lengths) != len(num_atoms) or len(angles) != len(num_atoms):
            raise ValueError(f"lattice parameter length mismatch: {sample_path}")

        atom_offset = 0
        for local_index, (input_index, atom_count) in enumerate(
            zip(input_indices, num_atoms)
        ):
            input_index = int(input_index)
            if input_index in records:
                raise ValueError(f"duplicate input_index {input_index}")
            atom_count = int(atom_count)
            if atom_count < 0:
                raise ValueError(f"negative num_atoms in {sample_path}")
            next_offset = atom_offset + atom_count
            if next_offset > len(frac_coords):
                raise ValueError(f"num_atoms exceeds concatenated arrays: {sample_path}")
            records[input_index] = {
                "frac_coords": np.asarray(frac_coords[atom_offset:next_offset]),
                "atom_types": np.asarray(atom_types[atom_offset:next_offset]),
                "lengths": np.asarray(lengths[local_index]),
                "angles": np.asarray(angles[local_index]),
            }
            atom_offset = next_offset
        if atom_offset != len(frac_coords):
            raise ValueError(f"unused atom rows in {sample_path}")
    return records


def _load_cif_records(
    sample_path: Path | str | Sequence[Path | str],
    manifest: pd.DataFrame,
    show_progress: bool,
) -> dict[int, CrystalRecord]:
    """Keep the old CIF-directory API usable for callers of compute_metrics."""
    if isinstance(sample_path, (str, Path)):
        path = Path(sample_path)
        paths = sorted(path.glob("*.cif")) if path.is_dir() else [path]
    else:
        paths = [Path(path) for path in sample_path]
    _, sample_column, _ = manifest_columns(manifest)
    manifest_indices = manifest[sample_column].astype(int).tolist()
    records: dict[int, CrystalRecord] = {}
    iterator = tqdm(
        paths,
        desc="Loading CIF files",
        unit="file",
        disable=not show_progress,
    )
    for position, path in enumerate(iterator):
        try:
            structure = Structure.from_file(path)
        except Exception:
            continue
        try:
            input_index = int(path.stem)
        except ValueError:
            if position >= len(manifest_indices):
                continue
            input_index = manifest_indices[position]
        records[input_index] = structure_to_record(structure)
    return records


def _is_pt_input(sample_path: Path | str | Sequence[Path | str]) -> bool:
    if isinstance(sample_path, (str, Path)):
        path = Path(sample_path)
        return path.suffix == ".pt" or (
            path.is_dir() and any(path.glob("*.pt"))
        )
    return any(Path(path).suffix == ".pt" for path in sample_path)


def _load_predictions(
    samples: Path | str | Sequence[Path | str] | Mapping[int, Any],
    manifest: pd.DataFrame,
    show_progress: bool,
) -> Mapping[int, Any] | Sequence[Any]:
    if isinstance(samples, Mapping):
        return samples
    if _is_pt_input(samples):
        return load_diffcsppp_records(samples, show_progress=show_progress)
    return _load_cif_records(samples, manifest, show_progress=show_progress)


def compute_metrics(
    manifest: pd.DataFrame,
    samples: Path | str | Sequence[Path | str] | Mapping[int, Any],
    targets: Sequence[Structure],
    target_composition_valid: Sequence[bool] | None = None,
    workers: int = 1,
    show_progress: bool = True,
    ltol: float = 0.3,
    stol: float = 0.5,
    angle_tol: float = 10.0,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Compute best-of-candidates reconstruction metrics."""
    predictions = _load_predictions(samples, manifest, show_progress)
    evaluator = RecEval(
        predictions,
        targets,
        manifest=manifest,
        target_composition_valid=target_composition_valid,
        workers=workers,
        show_progress=show_progress,
        ltol=ltol,
        stol=stol,
        angle_tol=angle_tol,
    )
    metrics = evaluator.get_metrics()
    assert evaluator.details is not None
    return metrics, evaluator.details


def _default_samples(root_path: Path) -> Path:
    for candidate in (root_path, root_path / "samples", root_path / "samples" / "cif"):
        if candidate.is_file() or candidate.is_dir() and (
            any(candidate.glob("*.pt")) or any(candidate.glob("*.cif"))
        ):
            return candidate
    return root_path


def main(args: argparse.Namespace) -> None:
    root_path = Path(args.root_path)
    manifest_path = Path(args.manifest) if args.manifest else root_path / "manifest.csv"
    manifest = pd.read_csv(manifest_path)
    targets = load_targets(args.gt_file, manifest, getattr(args, "cif_column", None))

    sample_files = getattr(args, "sample_files", None)
    samples = sample_files or getattr(args, "samples_dir", None) or _default_samples(root_path)
    metrics, details = compute_metrics(
        manifest,
        samples,
        targets,
        workers=int(getattr(args, "workers", 1)),
        show_progress=not bool(getattr(args, "no_progress", False)),
        ltol=float(getattr(args, "ltol", 0.3)),
        stol=float(getattr(args, "stol", 0.5)),
        angle_tol=float(getattr(args, "angle_tol", 10.0)),
    )

    output_path = Path(args.output) if args.output else root_path / "eval_metrics.json"
    details_path = (
        Path(args.details) if args.details else root_path / "eval_details.csv"
    )
    write_metrics(output_path, details_path, metrics, details)

    print(f"Matched: {metrics['num_matched']}/{metrics['num_targets']}")
    print(f"Match Rate: {metrics['match_rate']:.4%}")
    if metrics["rms_dist"] is not None:
        print(f"RMS Dist: {metrics['rms_dist']:.6f}")
    print(f"Wrote metrics to {output_path}")
    print(f"Wrote details to {details_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-path", "--root_path", required=True, type=Path)
    parser.add_argument("--gt-file", "--gt_file", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["csp"],
        choices=["csp"],
        help="Evaluation task. Only CSP reconstruction is supported.",
    )
    parser.add_argument("--sample-files", nargs="+", type=Path)
    parser.add_argument("--samples-dir", "--samples_dir", type=Path)
    parser.add_argument("--cif-column", "--cif_column")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--details", type=Path)
    parser.add_argument("--ltol", type=float, default=0.3)
    parser.add_argument("--stol", type=float, default=0.5)
    parser.add_argument("--angle-tol", "--angle_tol", type=float, default=10.0)
    parser.add_argument(
        "--workers",
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes used to evaluate target materials.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable PT loading and reconstruction progress bars.",
    )
    return parser


def cli() -> None:
    main(build_parser().parse_args())


if __name__ == "__main__":
    cli()
