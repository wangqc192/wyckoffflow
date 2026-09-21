"""Convert NextDiff symmetry-query JSON to CrystalFlow ``wyckoff_info.csv``."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from models.common.wyckoff_template import WyckoffTemplate

DEFAULT_INPUT = Path("/home/wangqc/NextCrystal/results/evaluate/nextdiff_input.json")
_FIELDNAMES = ["formula", "num_evals", "pressure", "wyckoff"]


def _convert_query(
    query: Mapping[str, Any],
    index: int,
    num_evals: int,
    pressure: int,
) -> dict[str, Any]:
    try:
        template = WyckoffTemplate.from_diffcsppp_query(query)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Query {index} is invalid: {error}") from error

    return {
        "formula": template.formula_with_counts,
        "num_evals": num_evals,
        "pressure": pressure,
        "wyckoff": template.to_crystalflow(),
    }


def convert_queries(
    queries: list[dict[str, Any]],
    *,
    num_evals: int = 1,
    pressure: int = 0,
) -> list[dict[str, Any]]:
    """Convert NextDiff query records to CrystalFlow CSV records.

    Each query is converted independently and input order is preserved.  Query
    rows are intentionally not deduplicated.
    """

    if num_evals < 1:
        raise ValueError("num_evals must be positive")

    rows: list[dict[str, Any]] = []
    for index, query in enumerate(queries):
        if not isinstance(query, Mapping):
            raise ValueError(f"Query {index} is not a JSON object")
        rows.append(_convert_query(query, index, num_evals, pressure))
    return rows


def convert_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    num_evals: int = 1,
    pressure: int = 0,
) -> int:
    """Read a NextDiff JSON file and write a CrystalFlow ``wyckoff_info.csv``."""

    input_path = Path(input_path)
    output_path = Path(output_path)
    with input_path.open("r", encoding="utf-8") as handle:
        queries = json.load(handle)
    if not isinstance(queries, list):
        raise ValueError("NextDiff input JSON must contain a top-level list")

    rows = convert_queries(queries, num_evals=num_evals, pressure=pressure)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="output CSV path; use wyckoff_info.csv for the CrystalFlow filename",
    )
    parser.add_argument("--num-evals", type=int, default=1)
    parser.add_argument("--pressure", type=int, default=0)
    args = parser.parse_args()

    row_count = convert_file(
        args.input,
        args.output,
        num_evals=args.num_evals,
        pressure=args.pressure,
    )
    print(f"Wrote {row_count} rows to {args.output}")


if __name__ == "__main__":
    main()
