#!/usr/bin/env python3
"""Extract DiffCSP++ templates as CrystalFlow ``wyckoff_info.csv`` input.

The input PT file produced by ``csp_from_template.py`` stores the Wyckoff
labels in ``templates`` and the element identities in the corresponding
PyTorch-Geometric graph.  CrystalFlow expects one row per template with the
complete conventional-cell formula and a ``G_ElementNxMultiplicityLetter``
Wyckoff string.
"""

from __future__ import annotations

# isort: off
import argparse
import csv
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


# Add the repository root when this file is run by absolute path.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from models.common.lookup_tables import chemical_symbols  # noqa: E402
from models.common.wyckoff_template import WyckoffTemplate  # noqa: E402
# isort: on

DEFAULT_INPUT = Path(
    "/home/wangqc/DiffCSP-PP/checkpoint/mp_csp/eval_diff_template.pt"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/wangqc/wyckoffflow-pl/outputs/diffcsp_pp_template_crystalflow"
)
_INPUT_FIELDS = ["formula", "num_evals", "pressure", "wyckoff"]
_MANIFEST_FIELDS = [
    "input_index",
    "source_index",
    "formula",
    "target_formula",
    "target_wyckoff_spglib",
    "material_id",
    "wyckoff",
    "num_evals",
    "pressure",
]
_MISSING_FIELDS = [
    "source_index",
    "target_formula",
    "target_wyckoff_spglib",
    "material_id",
    "reason",
]


def _elements_by_anchor(data: Any) -> tuple[str, ...]:
    """Recover one chemical symbol for each unique DiffCSP++ anchor."""

    anchors = data.anchor_index.detach().cpu().reshape(-1).tolist()
    atom_types = data.atom_types.detach().cpu().reshape(-1).tolist()
    anchor_order = sorted({int(anchor) for anchor in anchors})

    elements: list[str] = []
    for anchor in anchor_order:
        atomic_numbers = {
            int(atom_type)
            for current_anchor, atom_type in zip(anchors, atom_types)
            if int(current_anchor) == anchor
        }
        if len(atomic_numbers) != 1:
            raise ValueError(
                f"anchor {anchor} has multiple element types: "
                f"{sorted(atomic_numbers)}"
            )
        atomic_number = next(iter(atomic_numbers))
        elements.append(chemical_symbols[atomic_number])
    return tuple(elements)


def template_from_record(record: Mapping[str, Any], data: Any) -> WyckoffTemplate:
    """Convert one valid DiffCSP++ PT record to a single template."""

    if not record.get("found", False):
        raise ValueError("DiffCSP++ template was not found")

    labels = tuple(str(label) for label in record["wyckoff_positions"])
    elements = _elements_by_anchor(data)
    if len(labels) != len(elements):
        raise ValueError(
            "number of Wyckoff labels does not match number of graph anchors: "
            f"{len(labels)} != {len(elements)}"
        )
    return WyckoffTemplate(int(record["spacegroup"]), labels, elements)


def _target_row(target_df: pd.DataFrame | None, source_index: int) -> dict[str, Any]:
    """Return optional MP20 metadata for one source index."""

    if target_df is None:
        return {
            "target_formula": "",
            "target_wyckoff_spglib": "",
            "material_id": "",
        }
    row = target_df.iloc[source_index]
    return {
        "target_formula": row.get("pretty_formula", ""),
        "target_wyckoff_spglib": row.get("wyckoff_spglib", ""),
        "material_id": row.get("material_id", ""),
    }


def extract_templates(
    input_path: str | Path,
    *,
    target_path: str | Path | None = None,
    num_evals: int = 1,
    pressure: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract valid templates and report PT records without a template.

    Rows keep the original PT order, but ``input_index`` is compacted after
    records with ``found=False`` are omitted because CrystalFlow cannot parse
    those records as Wyckoff input.
    """

    if num_evals < 1:
        raise ValueError("num_evals must be positive")

    payload = torch.load(input_path, map_location="cpu", weights_only=False)
    template_records = payload["templates"]
    data_list = payload["input_data_batch"].to_data_list()
    if len(template_records) != len(data_list):
        raise ValueError(
            "templates and input_data_batch have different lengths: "
            f"{len(template_records)} != {len(data_list)}"
        )

    target_df = None
    if target_path is not None:
        target_df = pd.read_csv(target_path)
        if len(target_df) != len(template_records):
            raise ValueError(
                "target CSV and PT have different numbers of rows: "
                f"{len(target_df)} != {len(template_records)}"
            )

    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for source_index, (record, data) in enumerate(zip(template_records, data_list)):
        metadata = _target_row(target_df, source_index)
        if not record.get("found", False):
            missing.append(
                {
                    "source_index": source_index,
                    **metadata,
                    "reason": "template_not_found",
                }
            )
            continue

        template = template_from_record(record, data)
        row = {
            "input_index": len(rows),
            "source_index": source_index,
            "formula": template.formula_with_counts,
            **metadata,
            "wyckoff": template.to_crystalflow(),
            "num_evals": num_evals,
            "pressure": pressure,
        }
        rows.append(row)
    return rows, missing


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    rows: list[dict[str, Any]],
    missing: list[dict[str, Any]],
    output_dir: str | Path,
) -> None:
    """Write CrystalFlow input, source-index manifest, and missing records."""

    output_dir = Path(output_dir)
    input_rows = [
        {field: row[field] for field in _INPUT_FIELDS}
        for row in rows
    ]
    manifest_rows = [{field: row[field] for field in _MANIFEST_FIELDS} for row in rows]
    _write_csv(output_dir / "wyckoff_info.csv", _INPUT_FIELDS, input_rows)
    _write_csv(output_dir / "manifest.csv", _MANIFEST_FIELDS, manifest_rows)
    _write_csv(output_dir / "missing_templates.csv", _MISSING_FIELDS, missing)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--target",
        type=Path,
        help="optional target CSV aligned with the PT records (for manifest only)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-evals", type=int, default=1)
    parser.add_argument("--pressure", type=int, default=0)
    args = parser.parse_args()

    rows, missing = extract_templates(
        args.input,
        target_path=args.target,
        num_evals=args.num_evals,
        pressure=args.pressure,
    )
    write_outputs(rows, missing, args.output_dir)
    print(f"Wrote {len(rows)} templates to {args.output_dir / 'wyckoff_info.csv'}")
    print(f"Skipped {len(missing)} records without a template")


if __name__ == "__main__":
    main()
