import importlib.util
from pathlib import Path

import pandas as pd
import torch

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "generate_structure.py"
SPEC = importlib.util.spec_from_file_location("generate_structure", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_write_structure_files_merges_shards_by_global_input_index(tmp_path: Path):
    templates = pd.DataFrame(
        [
            {
                "candidate_index": 0,
                "space_group_rank": 1,
                "template_rank": 1,
                "generated_space_group": 1,
                "generated_structure_sequence": "1-1a-H",
            },
            {
                "candidate_index": 4,
                "space_group_rank": 2,
                "template_rank": 1,
                "generated_space_group": 2,
                "generated_structure_sequence": "2-1a-Li",
            },
        ]
    )
    templates_csv = tmp_path / "templates.csv"
    templates.to_csv(templates_csv, index=False)

    shard0 = tmp_path / "shard0.pt"
    torch.save(
        {
            "input_indices": torch.empty(0, dtype=torch.long),
            "frac_coords": torch.empty((0, 3)),
            "atom_types": torch.empty(0, dtype=torch.long),
            "lengths": torch.empty((0, 3)),
            "angles": torch.empty((0, 3)),
            "num_atoms": torch.empty(0, dtype=torch.long),
        },
        shard0,
    )
    shard1 = tmp_path / "shard1.pt"
    torch.save(
        {
            "input_indices": torch.tensor([1, 0]),
            "frac_coords": torch.tensor([[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]]),
            "atom_types": torch.tensor([3, 1]),
            "lengths": torch.tensor([[4.0, 4.0, 4.0], [3.0, 3.0, 3.0]]),
            "angles": torch.tensor([[90.0, 90.0, 90.0], [90.0, 90.0, 90.0]]),
            "num_atoms": torch.tensor([1, 1]),
        },
        shard1,
    )

    count = MODULE.write_structure_files(
        [shard0, shard1], templates_csv, tmp_path / "structures"
    )

    assert count == 2
    records = pd.read_csv(tmp_path / "structures" / "structures.csv")
    assert records.input_index.tolist() == [0, 1]
    assert records.candidate_index.tolist() == [0, 4]
    assert (tmp_path / "structures" / "cif" / "candidate_000_sg1.cif").exists()
    assert (tmp_path / "structures" / "poscar" / "candidate_004_sg2.vasp").exists()
