"""Convert extracted Wyckoff templates to CrystalFlow input."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def prepare_templates(
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    start_index: int = 0,
    limit: int | None = None,
) -> tuple[int, int]:
    templates = pd.read_csv(input_path)
    required = {
        "sample_index",
        "target_index",
        "formula",
        "target_formula",
        "wyckoff_occupancy",
        "count",
    }
    missing = required.difference(templates.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")

    templates = templates.sort_values(
        ["target_index", "sample_index"],
        kind="stable",
    ).reset_index(drop=True)
    templates = templates.iloc[start_index:]
    if limit is not None:
        templates = templates.iloc[:limit]
    templates = templates.copy()
    if not templates["formula"].eq(templates["target_formula"]).all():
        raise ValueError("CrystalFlow input requires composition-conserving templates")

    manifest = templates.rename(columns={"wyckoff_occupancy": "wyckoff"}).copy()
    manifest.insert(0, "diffcsp_index", manifest.index)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest[["formula", "wyckoff"]].assign(
        num_evals=1,
        pressure=0,
    )[["formula", "num_evals", "pressure", "wyckoff"]].to_csv(
        output_path,
        index=False,
    )
    manifest.to_csv(manifest_path, index=False)
    return len(manifest), int(manifest["target_index"].nunique())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    templates, targets = prepare_templates(
        args.input,
        args.output,
        args.manifest,
        start_index=args.start_index,
        limit=args.limit,
    )
    print(f"Wrote {templates} templates for {targets} targets")


if __name__ == "__main__":
    main()
