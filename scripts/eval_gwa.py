"""Evaluate generated G-W-A templates against an MP20 test CSV."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from numbers import Number
from pathlib import Path

import pandas as pd

from models.common.wyckoff_template import WyckoffTemplate


def occupancy_key(value: object) -> tuple[str, ...]:
    """Return the canonical key for one generated template."""

    if isinstance(value, Number) and not isinstance(value, bool):
        return (str(int(value)),)
    return WyckoffTemplate.from_crystalflow(str(value)).occupancy_key()


def target_occupancy_keys(protostructure: str) -> set[tuple[str, ...]]:
    """Return canonical keys for all equivalent target Wyckoff settings."""

    return {
        template.occupancy_key()
        for template in WyckoffTemplate.from_protostructure_set(protostructure)
    }


def evaluate(target_df: pd.DataFrame, generated_df: pd.DataFrame, top_k: int):
    """Return per-target results and the exact Top-K G-W-A hit count."""

    generated_df = generated_df.copy()
    if "target_index" not in generated_df:
        generated_df["target_index"] = generated_df["sample_index"] // top_k

    candidates = defaultdict(set)
    generated_counts = defaultdict(int)
    has_count = "count" in generated_df.columns
    for row in generated_df.itertuples(index=False):
        target_index = int(row.target_index)
        candidates[target_index].add(occupancy_key(row.wyckoff_occupancy))
        generated_counts[target_index] += int(row.count) if has_count else 1

    details = []
    hits = 0
    for target_index, row in target_df.reset_index(drop=True).iterrows():
        target_keys = target_occupancy_keys(row["wyckoff_spglib"])
        matched = bool(target_keys & candidates[target_index])
        hits += int(matched)
        details.append(
            {
                "target_index": target_index,
                "material_id": row.get("material_id", ""),
                "formula": row.get("pretty_formula", ""),
                "generated_count": generated_counts[target_index],
                "matched": matched,
            }
        )
    return pd.DataFrame(details), hits


def evaluation_summary(details: pd.DataFrame, hits: int, top_k: int) -> dict:
    """Build the machine-readable G-W-A Top-K summary."""

    total = len(details)
    return {
        "metric": "G-W-A Top-K",
        "top_k": top_k,
        "matched_materials": hits,
        "total_materials": total,
        "match_rate": hits / total,
        "materials_with_generated_samples": int((details.generated_count > 0).sum()),
    }


def main(args) -> None:
    target_df = pd.read_csv(args.target_path)
    generated_df = pd.read_csv(args.gen_path)
    details, hits = evaluate(target_df, generated_df, args.top_k)
    total = len(details)

    print(f"G-W-A Top-{args.top_k}: {hits}/{total} ({hits / total:.2%})")
    print(
        f"Targets with generated samples: {(details.generated_count > 0).sum()}/{total}"
    )
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        details.to_csv(output_path, index=False)
        print(f"Wrote details to {output_path}")
    if args.summary_path:
        summary_path = Path(args.summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary = evaluation_summary(details, hits, args.top_k)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate exact G-W-A template Top-K match rate"
    )
    parser.add_argument("--target_path", required=True)
    parser.add_argument("--gen_path", required=True)
    parser.add_argument("--top_k", type=int, required=True)
    parser.add_argument("--output_path")
    parser.add_argument("--summary_path")
    main(parser.parse_args())
