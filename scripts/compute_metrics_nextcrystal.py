"""Evaluate NextCrystal CIF samples with the shared CSP reconstruction metric.

The NextCrystal run at ``~/NextCrystal/outputs/mp_20`` stores one CIF per
query in ``sample_structures/{query_index}.cif``.  When no prebuilt manifest is
provided, this script reconstructs the one-based query index and zero-based
source input index from NextCrystal's native assignment and query files.  The
target structures, validity checks, composition filter, StructureMatcher
settings, and best-candidate aggregation are exactly the same as the
CrystalFlow and DiffCSP++ evaluation scripts.
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd
from pymatgen.core import Structure
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module=r"pymatgen\..*")

try:
    from .eval_utils import (
        CrystalRecord,
        RecEval,
        load_targets,
        structure_to_record,
        write_metrics,
    )
except ImportError:  # direct ``python scripts/compute_metrics_nextcrystal.py``
    from eval_utils import (  # type: ignore[no-redef]
        CrystalRecord,
        RecEval,
        load_targets,
        structure_to_record,
        write_metrics,
    )


DEFAULT_ROOT = Path.home() / "NextCrystal" / "outputs" / "mp_20"
DEFAULT_TARGETS = Path("data/mp20/test.csv")
DEFAULT_ASSIGNMENTS = "postprocessed_assignments_from_top5_sg.csv"
DEFAULT_QUERY_FILES = ("mp_test.json", "mp_test.csv")


def build_manifest(nextcrystal_manifest: pd.DataFrame) -> pd.DataFrame:
    """Convert the NextCrystal manifest to the shared evaluator schema."""
    required = {"query_index", "source_input_id", "candidate_rank"}
    missing = sorted(required - set(nextcrystal_manifest.columns))
    if missing:
        raise ValueError(f"NextCrystal manifest is missing columns: {missing}")

    return pd.DataFrame(
        {
            "target_index": pd.to_numeric(
                nextcrystal_manifest["source_input_id"], errors="raise"
            ).astype(int),
            "input_index": pd.to_numeric(
                nextcrystal_manifest["query_index"], errors="raise"
            ).astype(int),
            "candidate_index": pd.to_numeric(
                nextcrystal_manifest["candidate_rank"], errors="raise"
            ).astype(int),
        }
    )


def build_manifest_from_outputs(
    assignment_csv: Path | str,
    query_file: Path | str,
) -> pd.DataFrame:
    """Reconstruct the candidate manifest from the native NextCrystal outputs.

    NextCrystal writes one assignment row per predicted space group and stores
    valid assignments inside the ``Assignments`` JSON list.  The sampled CIFs
    are flattened in that same valid-assignment order, so the query index is
    reconstructed by scanning those lists and the candidate rank is counted
    per source MP20 input.  This produces the same grouping as the optional
    NextCrystal manifest without requiring that extra file.
    """
    assignment_csv = Path(assignment_csv)
    query_file = Path(query_file)
    assignments = pd.read_csv(assignment_csv)
    required = {
        "cif_name",
        "Formula pretty",
        "NAtoms",
        "Spacegroup Number",
        "Assignments",
    }
    missing = sorted(required - set(assignments.columns))
    if missing:
        raise ValueError(f"NextCrystal assignment CSV is missing columns: {missing}")

    if query_file.suffix.lower() == ".json":
        query_records = json.loads(query_file.read_text(encoding="utf-8"))
        query_count = len(query_records)
    else:
        query_count = len(pd.read_csv(query_file))

    candidate_ranks: defaultdict[int, int] = defaultdict(int)
    manifest_rows: list[dict[str, object]] = []
    query_index = 0
    for row in assignments.to_dict(orient="records"):
        source_input_id = int(row["cif_name"])
        candidates = json.loads(row["Assignments"])
        for candidate in candidates:
            if str(candidate).startswith("Unable to find"):
                continue
            candidate_ranks[source_input_id] += 1
            query_index += 1
            manifest_rows.append(
                {
                    "query_index": query_index,
                    "input_number": source_input_id + 1,
                    "source_input_id": source_input_id,
                    "candidate_rank": candidate_ranks[source_input_id],
                    "formula": row["Formula pretty"],
                    "num_atoms": int(row["NAtoms"]),
                    "spacegroup_number": int(row["Spacegroup Number"]),
                }
            )

    if query_index != query_count:
        raise ValueError(
            "NextCrystal assignment/query count mismatch: "
            f"reconstructed {query_index} valid candidates, query file contains "
            f"{query_count}"
        )
    return pd.DataFrame(manifest_rows)


def load_nextcrystal_records(
    sample_dir: Path | str,
    nextcrystal_manifest: pd.DataFrame,
    show_progress: bool = True,
) -> dict[int, CrystalRecord]:
    """Load ``{query_index}.cif`` files keyed by one-based query index."""
    sample_dir = Path(sample_dir)
    records: dict[int, CrystalRecord] = {}
    iterator = tqdm(
        nextcrystal_manifest.itertuples(index=False),
        total=len(nextcrystal_manifest),
        desc="Loading NextCrystal CIF files",
        unit="file",
        disable=not show_progress,
    )
    for row in iterator:
        query_index = int(row.query_index)
        path = sample_dir / f"{query_index}.cif"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                structure = Structure.from_file(path)
        except Exception:
            continue
        records[query_index] = structure_to_record(structure)
    return records


def compute_metrics(
    nextcrystal_manifest: pd.DataFrame,
    sample_dir: Path | str,
    targets: list[Structure],
    workers: int = 1,
    show_progress: bool = True,
    ltol: float = 0.3,
    stol: float = 0.5,
    angle_tol: float = 10.0,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Compute the shared best-of-candidates CSP reconstruction metrics."""
    manifest = build_manifest(nextcrystal_manifest)
    predictions = load_nextcrystal_records(
        sample_dir,
        nextcrystal_manifest,
        show_progress=show_progress,
    )
    evaluator = RecEval(
        predictions,
        targets,
        manifest=manifest,
        workers=workers,
        show_progress=show_progress,
        ltol=ltol,
        stol=stol,
        angle_tol=angle_tol,
    )
    metrics = evaluator.get_metrics()
    assert evaluator.details is not None
    return metrics, evaluator.details


def _default_manifest(root_path: Path) -> Path | None:
    candidates = (
        root_path / "manifest.csv",
        root_path.parent.parent / "manifest.csv",
        Path.home() / "NextCrystal" / "manifest.csv",
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _default_query_file(root_path: Path) -> Path:
    for filename in DEFAULT_QUERY_FILES:
        candidate = root_path / filename
        if candidate.is_file():
            return candidate
    return root_path / DEFAULT_QUERY_FILES[0]


def _load_raw_manifest(args: argparse.Namespace, root_path: Path) -> pd.DataFrame:
    if args.manifest is not None:
        manifest_path = Path(args.manifest).expanduser()
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        return pd.read_csv(manifest_path)

    assignment_arg = getattr(args, "assignment_csv", None)
    query_arg = getattr(args, "query_file", None)
    assignment_csv = (
        Path(assignment_arg).expanduser()
        if assignment_arg is not None
        else root_path / DEFAULT_ASSIGNMENTS
    )
    query_file = (
        Path(query_arg).expanduser()
        if query_arg is not None
        else _default_query_file(root_path)
    )
    if assignment_csv.is_file() or query_file.is_file() or assignment_arg or query_arg:
        if not assignment_csv.is_file():
            raise FileNotFoundError(
                f"NextCrystal assignment CSV is missing: {assignment_csv}"
            )
        if not query_file.is_file():
            raise FileNotFoundError(f"NextCrystal query file is missing: {query_file}")
        print(f"Reconstructing NextCrystal manifest from {assignment_csv}")
        return build_manifest_from_outputs(assignment_csv, query_file)

    manifest_path = _default_manifest(root_path)
    if manifest_path is not None:
        return pd.read_csv(manifest_path)
    raise FileNotFoundError(
        "NextCrystal manifest and native output files were not found under "
        f"{root_path}"
    )


def main(args: argparse.Namespace) -> None:
    root_path = Path(args.root_path).expanduser()
    target_path = Path(args.gt_file).expanduser()
    sample_dir = (
        Path(args.samples_dir).expanduser()
        if args.samples_dir is not None
        else root_path / "sample_structures"
    )

    raw_manifest = _load_raw_manifest(args, root_path)
    manifest = build_manifest(raw_manifest)
    targets = load_targets(target_path, manifest, args.cif_column)
    metrics, details = compute_metrics(
        raw_manifest,
        sample_dir,
        targets,
        workers=args.workers,
        show_progress=not args.no_progress,
        ltol=args.ltol,
        stol=args.stol,
        angle_tol=args.angle_tol,
    )

    output_path = (
        Path(args.output).expanduser()
        if args.output is not None
        else root_path / "eval_metrics.json"
    )
    details_path = (
        Path(args.details).expanduser()
        if args.details is not None
        else root_path / "eval_details.csv"
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
    parser.add_argument("--root-path", "--root_path", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional prebuilt manifest; otherwise reconstruct it from NextCrystal outputs.",
    )
    parser.add_argument("--assignment-csv", "--assignment_csv", type=Path)
    parser.add_argument("--query-file", "--query_file", type=Path)
    parser.add_argument(
        "--gt-file",
        "--gt_file",
        type=Path,
        default=DEFAULT_TARGETS,
        help="Ground-truth MP20 CSV; the default is data/mp20/test.csv.",
    )
    parser.add_argument("--samples-dir", "--samples_dir", type=Path)
    parser.add_argument("--cif-column", "--cif_column")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--details", type=Path)
    parser.add_argument("--ltol", type=float, default=0.3)
    parser.add_argument("--stol", type=float, default=0.5)
    parser.add_argument("--angle-tol", "--angle_tol", type=float, default=10.0)
    parser.add_argument("--workers", "--num-workers", type=int, default=1)
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable CIF loading and reconstruction progress bars.",
    )
    return parser


def cli() -> None:
    main(build_parser().parse_args())


if __name__ == "__main__":
    cli()
