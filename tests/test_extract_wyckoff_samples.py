from types import SimpleNamespace

import pandas as pd
import torch

from models.common.lookup_tables import chemical_symbols
from scripts.eval_gwa import evaluate
from scripts.extract_wyckoff_samples import extract_samples, write_csv


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
    assert [row["candidate_rank"] for row in rows] == [1, 2]
    assert [row["wyckoff_occupancy"] for row in rows] == [
        "1_Si1x1a",
        "1_Si1x1a",
    ]
    assert [row["count"] for row in rows] == [1, 1]


def test_unconstrained_export_keeps_empty_and_wrong_composition_samples(tmp_path):
    samples = []
    for index, count in enumerate([0, 2, 1]):
        x = torch.zeros(1, len(chemical_symbols))
        x[0, 14] = count
        composition = torch.zeros(len(chemical_symbols))
        composition[14] = 1
        samples.append(
            SimpleNamespace(
                target_index=torch.tensor(index),
                space_group=torch.tensor([1]),
                x=x,
                multiplicities=torch.tensor([1]),
                degrees_of_freedom=torch.tensor([3]),
                composition=composition,
            )
        )
    path = tmp_path / "samples.pt"
    torch.save({"args": {"num_samples": 1}, "generated_samples": samples}, path)
    rows = extract_samples(path)
    assert [row["formula"] for row in rows] == ["", "Si2", "Si"]
    assert [row["target_formula"] for row in rows] == ["Si"] * 3
    assert rows[0]["wyckoff_occupancy"] == "1"

    csv_path = tmp_path / "samples.csv"
    write_csv(rows, csv_path)
    targets = pd.DataFrame({"wyckoff_spglib": ["A_aP1_1_a:Si"] * 3})
    details, hits = evaluate(targets, pd.read_csv(csv_path), top_k=1)
    assert hits == 1
    assert details.generated_count.tolist() == [1, 1, 1]
    assert details.matched.tolist() == [False, False, True]
