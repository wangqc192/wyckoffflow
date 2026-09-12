"""Extract human-readable Wyckoff samples from a sampling ``.pt`` file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from models.common.lookup_tables import (
    chemical_symbols,
    spg_wyckoff,
    spg_wyckoff_degrees_of_freedom,
)


def _scalar(value) -> int:
    """Convert a scalar tensor or number to a Python integer."""

    if isinstance(value, torch.Tensor):
        return int(value.reshape(-1)[0].item())
    return int(value)


def _format_formula(counts: dict[int, int]) -> str:
    """Format counts without reducing the complete chemical formula."""

    parts = []
    for atomic_number in sorted(counts):
        count = counts[atomic_number]
        if count <= 0:
            continue
        symbol = chemical_symbols[atomic_number]
        parts.append(symbol if count == 1 else f"{symbol}{count}")
    return "".join(parts) or "X"


def _canonical_occupancy(space_group: int, entries: list[tuple[str, str]]) -> str:
    entries = sorted(entries, key=lambda entry: (entry[0], entry[1]))
    return "_".join([str(space_group), *(value for _, value in entries)])


def _occupancy_key(value: str) -> tuple[str, ...]:
    space_group, *entries = value.split("_")
    return (space_group, *sorted(entries))


def _decode_sample(sample) -> dict[str, object]:
    space_group = _scalar(sample.space_group)
    labels = list(reversed(spg_wyckoff[str(space_group)]))
    fallback_dof = list(
        reversed(spg_wyckoff_degrees_of_freedom[str(space_group)].values())
    )

    x = sample.x.detach().cpu()
    multiplicities = sample.multiplicities.detach().cpu().reshape(-1)
    degrees = getattr(sample, "degrees_of_freedom", None)
    if degrees is None:
        degrees = torch.tensor(fallback_dof)
    else:
        degrees = degrees.detach().cpu().reshape(-1)

    formula_counts: dict[int, int] = {}
    occupancy_entries: list[tuple[str, str]] = []

    for row, (label, multiplicity, dof) in enumerate(
        zip(labels, multiplicities.tolist(), degrees.tolist())
    ):
        multiplicity = int(round(multiplicity))
        if dof == 0:
            atomic_number = int(round(float(x[row, 0].item())))
            if atomic_number <= 0:
                continue
            species = chemical_symbols[atomic_number]
            formula_counts[atomic_number] = (
                formula_counts.get(atomic_number, 0) + multiplicity
            )
            occupancy_entries.append((label, f"{species}1x{multiplicity}{label}"))
        else:
            values = x[row, 1 : len(chemical_symbols)]
            entries = 0
            for atomic_number, amount in enumerate(values.tolist(), start=1):
                amount = int(round(amount))
                if amount <= 0:
                    continue
                species = chemical_symbols[atomic_number]
                formula_counts[atomic_number] = (
                    formula_counts.get(atomic_number, 0) + multiplicity * amount
                )
                occupancy_entries.append(
                    (label, f"{species}{amount}x{multiplicity}{label}")
                )
                entries += 1
            if entries == 0:
                continue

    target_counts = getattr(sample, "composition", None)
    target_formula = ""
    if target_counts is not None:
        target = target_counts.detach().cpu().reshape(-1)
        target_formula = _format_formula(
            {
                atomic_number: int(round(float(amount)))
                for atomic_number, amount in enumerate(target.tolist())
                if atomic_number > 0 and amount > 0
            }
        )

    return {
        "space_group": space_group,
        "formula": _format_formula(formula_counts),
        "target_formula": target_formula,
        "wyckoff_occupancy": _canonical_occupancy(space_group, occupancy_entries),
    }


def _top_k(payload, override):
    if override is not None:
        return override
    return int(payload["args"]["num_evals"])


def extract_samples(
    input_path: str | Path,
    top_k: int | None = None,
) -> list[dict[str, object]]:
    """Read generated samples from ``input_path`` and decode them."""

    payload = torch.load(input_path, map_location="cpu", weights_only=False)
    top_k = _top_k(payload, top_k)
    rows = []
    occupancy_rows = {}
    for index, sample in enumerate(payload["generated_samples"]):
        target_index = (
            _scalar(sample.target_index)
            if hasattr(sample, "target_index")
            else index // top_k
        )
        row = {
            "sample_index": index,
            "target_index": target_index,
            **_decode_sample(sample),
        }
        key = (target_index, _occupancy_key(row["wyckoff_occupancy"]))
        row_index = occupancy_rows.get(key)
        if row_index is None:
            row["count"] = 1
            occupancy_rows[key] = len(rows)
            rows.append(row)
        else:
            rows[row_index]["count"] += 1
    return rows


def write_csv(rows: list[dict[str, object]], output_path: str | Path) -> None:
    fieldnames = [
        "sample_index",
        "target_index",
        "space_group",
        "formula",
        "target_formula",
        "wyckoff_occupancy",
        "count",
    ]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract human-readable Wyckoff samples from a .pt file"
    )
    parser.add_argument("--input_pt", required=True, help="sampling output .pt file")
    parser.add_argument("--output_csv", required=True, help="destination CSV file")
    parser.add_argument("--top_k", type=int, help="override samples per input record")
    args = parser.parse_args()

    rows = extract_samples(args.input_pt, args.top_k)
    write_csv(rows, args.output_csv)
    print(f"Wrote {len(rows)} samples to {args.output_csv}")


if __name__ == "__main__":
    main()
