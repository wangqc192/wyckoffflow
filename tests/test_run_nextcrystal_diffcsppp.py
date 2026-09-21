import argparse
import importlib.util
import json
from pathlib import Path

import torch

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "run_nextcrystal_diffcsppp.py"
SPEC = importlib.util.spec_from_file_location("run_nextcrystal_diffcsppp", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_shard_bounds_cover_all_inputs_without_overlap():
    bounds = [MODULE.shard_bounds(11, index, 4) for index in range(4)]
    assert bounds == [(0, 2), (2, 5), (5, 8), (8, 11)]


def test_query_from_gwa_sequence_builds_diffcsppp_query():
    assert MODULE.query_from_gwa_sequence("44-4c-Ga-4c-Te") == {
        "spacegroup_number": 44,
        "wyckoff_letters": ["4c", "4c"],
        "atom_types": ["Ga", "Te"],
    }


def test_empty_four_way_shard_writes_standard_payload(tmp_path: Path):
    selected_json = tmp_path / "queries.json"
    selected_json.write_text(
        json.dumps(
            [
                {
                    "spacegroup_number": 1,
                    "wyckoff_letters": ["1a"],
                    "atom_types": ["H"],
                }
            ]
        )
    )
    output = tmp_path / "shard0.pt"
    args = argparse.Namespace(
        batch_size=128,
        num_shards=4,
        diffcsppp_repo=tmp_path / "repo",
        checkpoint_dir=tmp_path / "checkpoint",
        selected_json=selected_json,
        output=output,
        overwrite=True,
        shard_index=0,
        seed=42,
        step_lr=1e-5,
        device="cpu",
    )

    MODULE.sample(args)

    payload = torch.load(output, map_location="cpu", weights_only=False)
    assert payload["input_indices"].shape == (0,)
    assert payload["frac_coords"].shape == (0, 3)
    assert payload["lattices"].shape == (0, 3, 3)
    assert payload["metadata"]["shard_start"] == 0
    assert payload["metadata"]["shard_end"] == 0
