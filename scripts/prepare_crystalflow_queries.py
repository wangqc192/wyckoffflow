#!/usr/bin/env python3
"""Convert CrystalFlow's Wyckoff CSV input to structured symmetry queries."""

from __future__ import annotations

# isort: off
import argparse
import json
import sys
from pathlib import Path


# Add the repository root when this file is run by absolute path.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402
from models.common.wyckoff_template import WyckoffTemplate  # noqa: E402
# isort: on


def wyckoff_to_query(value: str) -> dict[str, object]:
    """Convert one CrystalFlow template to a DiffCSP++ query."""

    return WyckoffTemplate.from_crystalflow(value).to_diffcsppp_query()


def prepare_queries(input_path: Path, output_path: Path) -> int:
    templates = pd.read_csv(input_path)
    required = {"formula", "wyckoff"}
    missing = required.difference(templates.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")
    queries = [wyckoff_to_query(value) for value in templates["wyckoff"]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(queries, indent=2) + "\n", encoding="utf-8")
    return len(queries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    count = prepare_queries(args.input, args.output)
    print(f"Wrote {count} CrystalFlow queries to {args.output}")


if __name__ == "__main__":
    main()
