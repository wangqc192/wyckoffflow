"""Extract human-readable Wyckoff samples from a sampling ``.pt`` file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from models.common.lookup_tables import chemical_symbols
from models.common.wyckoff_template import WyckoffTemplate


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


def _decode_sample(sample) -> dict[str, object]:
    template = WyckoffTemplate.from_model_output(sample)

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
        "space_group": template.spacegroup_number,
        "formula": template.formula,
        "target_formula": target_formula,
        "wyckoff_occupancy": template.to_crystalflow(),
    }


def _top_k(payload, override):
    if override is not None:
        return override
    return int(payload["args"]["num_evals"])


def extract_samples(
    input_path: str | Path,
    top_k: int | None = None,
) -> list[dict[str, object]]:
    """Read and decode every generated sample from ``input_path``.

    Duplicate Wyckoff occupancies are intentionally kept as separate rows.
    """

    payload = torch.load(input_path, map_location="cpu", weights_only=False)
    top_k = _top_k(payload, top_k)
    rows = []
    for index, sample in enumerate(payload["generated_samples"]):
        target_index = (
            _scalar(sample.target_index)
            if hasattr(sample, "target_index")
            else index // top_k
        )
        candidate_rank = (
            _scalar(sample.candidate_rank)
            if hasattr(sample, "candidate_rank")
            else index % top_k + 1
        )
        candidate_probability = (
            float(sample.candidate_probability.reshape(-1)[0].item())
            if hasattr(sample, "candidate_probability")
            else None
        )
        rows.append(
            {
                "sample_index": index,
                "target_index": target_index,
                "candidate_rank": candidate_rank,
                "candidate_probability": candidate_probability,
                **_decode_sample(sample),
                "count": 1,
            }
        )
    return rows


def write_csv(rows: list[dict[str, object]], output_path: str | Path) -> None:
    fieldnames = [
        "sample_index",
        "target_index",
        "candidate_rank",
        "candidate_probability",
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
