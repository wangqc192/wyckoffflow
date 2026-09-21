"""Evaluate CrystalFlow CSP samples stored in ``.pt`` files.

CrystalFlow writes one crystal dictionary per entry in ``crystal_list`` and
normally does not write an input index for each entry.  This script therefore
maps the concatenated PT order to the manifest order.  DiffCSP++ uses a
separate loader because its PT files contain concatenated atom arrays and
explicit ``input_indices``.
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
except ImportError:  # direct ``python scripts/compute_metrics_crystalflow.py``
    from eval_utils import (  # type: ignore[no-redef]
        CrystalRecord,
        RecEval,
        load_targets,
        manifest_columns,
        resolve_pt_paths,
        structure_to_record,
        write_metrics,
    )


def load_crystalflow_records(
    sample_paths: Path | str | Sequence[Path | str],
    sample_indices: Sequence[int] | None = None,
    show_progress: bool = True,
) -> dict[int, CrystalRecord]:
    """Load CrystalFlow PT files keyed by manifest order or explicit indices."""
    import torch

    paths = resolve_pt_paths(sample_paths)
    crystals: list[CrystalRecord] = []
    explicit_indices: list[int] | None = None
    iterator = tqdm(
        paths,
        desc="Loading CrystalFlow PT files",
        unit="file",
        disable=not show_progress,
    )
    for sample_path in iterator:
        payload = torch.load(sample_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError(f"CrystalFlow sample is not a mapping: {sample_path}")

        payload_indices = payload.get("input_indices")
        if payload_indices is not None:
            if hasattr(payload_indices, "detach"):
                payload_indices = payload_indices.detach().cpu().numpy()
            else:
                payload_indices = np.asarray(payload_indices)
            current_indices = [
                int(index) for index in np.asarray(payload_indices).reshape(-1)
            ]
            if explicit_indices is None:
                explicit_indices = []
            explicit_indices.extend(current_indices)

        if "crystal_list" in payload:
            payload_crystals = payload["crystal_list"]
            for crystal in payload_crystals:
                if not isinstance(crystal, Mapping):
                    raise ValueError(f"invalid crystal_list entry: {sample_path}")
                missing = [
                    field
                    for field in ("frac_coords", "atom_types", "lengths", "angles")
                    if field not in crystal
                ]
                if missing:
                    raise ValueError(
                        f"CrystalFlow crystal is missing fields {missing}: {sample_path}"
                    )
                frac_coords = crystal["frac_coords"]
                atom_types = crystal["atom_types"]
                lengths = crystal["lengths"]
                angles = crystal["angles"]
                if hasattr(frac_coords, "detach"):
                    frac_coords = frac_coords.detach().cpu().numpy()
                else:
                    frac_coords = np.asarray(frac_coords)
                if hasattr(atom_types, "detach"):
                    atom_types = atom_types.detach().cpu().numpy()
                else:
                    atom_types = np.asarray(atom_types)
                if hasattr(lengths, "detach"):
                    lengths = lengths.detach().cpu().numpy()
                else:
                    lengths = np.asarray(lengths)
                if hasattr(angles, "detach"):
                    angles = angles.detach().cpu().numpy()
                else:
                    angles = np.asarray(angles)
                crystals.append(
                    {
                        "frac_coords": np.asarray(frac_coords),
                        "atom_types": np.asarray(atom_types),
                        "lengths": np.asarray(lengths),
                        "angles": np.asarray(angles),
                    }
                )
            continue

        required = ("frac_coords", "atom_types", "lengths", "angles", "num_atoms")
        missing = [field for field in required if field not in payload]
        if missing:
            raise ValueError(
                f"CrystalFlow sample is missing fields {missing}: {sample_path}"
            )
        frac_coords = payload["frac_coords"]
        atom_types = payload["atom_types"]
        lengths = payload["lengths"]
        angles = payload["angles"]
        num_atoms = payload["num_atoms"]
        if hasattr(frac_coords, "detach"):
            frac_coords = frac_coords.detach().cpu().numpy()
        else:
            frac_coords = np.asarray(frac_coords)
        if hasattr(atom_types, "detach"):
            atom_types = atom_types.detach().cpu().numpy()
        else:
            atom_types = np.asarray(atom_types)
        if hasattr(lengths, "detach"):
            lengths = lengths.detach().cpu().numpy()
        else:
            lengths = np.asarray(lengths)
        if hasattr(angles, "detach"):
            angles = angles.detach().cpu().numpy()
        else:
            angles = np.asarray(angles)
        if hasattr(num_atoms, "detach"):
            num_atoms = num_atoms.detach().cpu().numpy()
        else:
            num_atoms = np.asarray(num_atoms)

        frac_coords = np.asarray(frac_coords).reshape(-1, 3)
        atom_types = np.asarray(atom_types).reshape(-1)
        lengths = np.asarray(lengths).reshape(-1, 3)
        angles = np.asarray(angles).reshape(-1, 3)
        num_atoms = np.asarray(num_atoms).reshape(-1)
        if len(frac_coords) != len(atom_types):
            raise ValueError(f"frac_coords/atom_types mismatch: {sample_path}")
        if len(lengths) != len(num_atoms) or len(angles) != len(num_atoms):
            raise ValueError(f"lattice parameter length mismatch: {sample_path}")

        atom_offset = 0
        for local_index, atom_count in enumerate(num_atoms):
            atom_count = int(atom_count)
            if atom_count < 0:
                raise ValueError(f"negative num_atoms in {sample_path}")
            next_offset = atom_offset + atom_count
            if next_offset > len(frac_coords):
                raise ValueError(f"num_atoms exceeds concatenated arrays: {sample_path}")
            crystals.append(
                {
                    "frac_coords": frac_coords[atom_offset:next_offset],
                    "atom_types": atom_types[atom_offset:next_offset],
                    "lengths": lengths[local_index],
                    "angles": angles[local_index],
                }
            )
            atom_offset = next_offset
        if atom_offset != len(frac_coords):
            raise ValueError(f"unused atom rows in {sample_path}")

    if explicit_indices is not None:
        if len(explicit_indices) != len(crystals):
            raise ValueError("input_indices/sample length mismatch")
        indices = explicit_indices
    elif sample_indices is not None:
        indices = [int(index) for index in sample_indices]
        if len(indices) != len(crystals):
            raise ValueError(
                "CrystalFlow PT entries do not have the same length as the manifest"
            )
    else:
        indices = list(range(len(crystals)))

    records: dict[int, CrystalRecord] = {}
    for input_index, crystal in zip(indices, crystals):
        if input_index in records:
            raise ValueError(f"duplicate CrystalFlow sample index: {input_index}")
        records[input_index] = crystal
    return records


def _load_cif_records(
    sample_path: Path | str | Sequence[Path | str],
    manifest: pd.DataFrame,
    show_progress: bool,
) -> dict[int, CrystalRecord]:
    """Keep a small CIF compatibility path for programmatic callers."""
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
    _, sample_column, _ = manifest_columns(manifest)
    if _is_pt_input(samples):
        return load_crystalflow_records(
            samples,
            sample_indices=manifest[sample_column].astype(int).tolist(),
            show_progress=show_progress,
        )
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
    for candidate in (root_path, root_path / "samples"):
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


if __name__ == "__main__":
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
    main(parser.parse_args())
