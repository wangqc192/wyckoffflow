from types import SimpleNamespace

import torch

from models.common.lookup_tables import chemical_symbols
from scripts.extract_wyckoff_samples import extract_samples


def test_extract_samples_keeps_duplicate_occupancies(tmp_path):
    x = torch.zeros(1, len(chemical_symbols))
    x[0, 14] = 1  # Si
    sample = SimpleNamespace(
        space_group=torch.tensor([1]),
        x=x,
        multiplicities=torch.tensor([1]),
        degrees_of_freedom=torch.tensor([3]),
        composition=torch.zeros(len(chemical_symbols)),
    )
    sample.composition[14] = 1
    input_path = tmp_path / "samples.pt"
    torch.save(
        {
            "args": {"num_evals": 2},
            "generated_samples": [sample, sample],
        },
        input_path,
    )

    rows = extract_samples(input_path)

    assert len(rows) == 2
    assert [row["sample_index"] for row in rows] == [0, 1]
    assert [row["target_index"] for row in rows] == [0, 0]
    assert [row["wyckoff_occupancy"] for row in rows] == [
        "1_Si1x1a",
        "1_Si1x1a",
    ]
    assert [row["count"] for row in rows] == [1, 1]
