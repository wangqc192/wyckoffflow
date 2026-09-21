import json

import pandas as pd

from scripts.convert_nextdiff_to_wyckoff_info import convert_file, convert_queries


def test_convert_queries_builds_formula_and_occupancy():
    rows = convert_queries(
        [
            {
                "spacegroup_number": 216,
                "wyckoff_letters": ["4d", "4a"],
                "atom_types": ["Ga", "Te"],
            },
            {
                "spacegroup_number": 156,
                "wyckoff_letters": ["1c", "1c", "1b"],
                "atom_types": ["Sb", "Sb", "Te"],
            },
        ]
    )

    assert rows == [
        {
            "formula": "Ga4Te4",
            "num_evals": 1,
            "pressure": 0,
            "wyckoff": "216_Ga1x4d_Te1x4a",
        },
        {
            "formula": "Sb2Te1",
            "num_evals": 1,
            "pressure": 0,
            "wyckoff": "156_Sb2x1c_Te1x1b",
        },
    ]


def test_convert_file_writes_diffcsp_columns_without_deduplicating(tmp_path):
    source = tmp_path / "nextdiff_input.json"
    output = tmp_path / "wyckoff_info.csv"
    query = {
        "spacegroup_number": 1,
        "wyckoff_letters": ["1a"],
        "atom_types": ["Si"],
    }
    source.write_text(json.dumps([query, query]), encoding="utf-8")

    assert convert_file(source, output) == 2
    result = pd.read_csv(output)
    assert result.columns.tolist() == ["formula", "num_evals", "pressure", "wyckoff"]
    assert result.to_dict("records") == [
        {
            "formula": "Si1",
            "num_evals": 1,
            "pressure": 0,
            "wyckoff": "1_Si1x1a",
        },
        {
            "formula": "Si1",
            "num_evals": 1,
            "pressure": 0,
            "wyckoff": "1_Si1x1a",
        },
    ]
