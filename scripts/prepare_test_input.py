"""Prepare a formula-conditioned sampling input from the MP20 test CSV."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import chemparse
import pandas as pd

from models.common.lookup_tables import chemical_symbols


def _formula_from_cif(cif: str) -> str:
    match = re.search(r"^_chemical_formula_sum\s+(.+)$", cif, flags=re.MULTILINE)
    value = match.group(1).strip().strip("'").strip('"')
    counts = chemparse.parse_formula(value.replace(" ", ""))
    ordered = sorted(
        (
            chemical_symbols.index(element),
            element,
            int(round(amount)),
        )
        for element, amount in counts.items()
    )
    return "".join(
        element if amount == 1 else f"{element}{amount}"
        for _, element, amount in ordered
    )


def prepare_test_input(
    input_csv: str | Path,
    output_csv: str | Path,
) -> pd.DataFrame:
    """Convert conventional CIFs embedded in ``input_csv`` to sampling rows."""

    source = pd.read_csv(input_csv, usecols=["cif.conv", "spacegroup.number.conv"])
    result = pd.DataFrame(
        {
            "target_index": range(len(source)),
            "formula": source["cif.conv"].map(_formula_from_cif),
            "space_group": source["spacegroup.number.conv"].astype(int),
        }
    )
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert MP20 test CIFs into formula-conditioned sampling input"
    )
    parser.add_argument("--input_csv", default="data/mp20/test.csv")
    parser.add_argument("--output_csv", default="example/input_test.csv")
    args = parser.parse_args()

    result = prepare_test_input(
        args.input_csv,
        args.output_csv,
    )
    print(f"Wrote {len(result)} test records to {args.output_csv}")


if __name__ == "__main__":
    main()
