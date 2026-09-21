import pandas as pd
import torch
from torch_geometric.data import Batch, Data

from scripts.extract_diffcsppp_templates import extract_templates, write_outputs


def test_extract_diffcsppp_templates_keeps_valid_rows_and_reports_missing(tmp_path):
    payload = {
        "templates": [
            {
                "found": True,
                "spacegroup": 1,
                "wyckoff_positions": ["1a", "1a"],
            },
            {
                "found": False,
                "spacegroup": None,
                "wyckoff_positions": [],
            },
        ],
        "input_data_batch": Batch.from_data_list(
            [
                Data(
                    atom_types=torch.tensor([3, 8]),
                    anchor_index=torch.tensor([0, 1]),
                    num_nodes=2,
                ),
                Data(
                    atom_types=torch.tensor([6]),
                    anchor_index=torch.tensor([0]),
                    num_nodes=1,
                ),
            ]
        ),
    }
    input_path = tmp_path / "templates.pt"
    torch.save(payload, input_path)
    target_path = tmp_path / "target.csv"
    pd.DataFrame(
        {
            "pretty_formula": ["LiO", "C"],
            "wyckoff_spglib": ["AB_aP2_1_a_a:Li-O", "A_cP1_1_a:C"],
            "material_id": ["mp-1", "mp-2"],
        }
    ).to_csv(target_path, index=False)

    rows, missing = extract_templates(input_path, target_path=target_path)

    assert len(rows) == 1
    assert rows[0]["input_index"] == 0
    assert rows[0]["source_index"] == 0
    assert rows[0]["formula"] == "Li1O1"
    assert rows[0]["wyckoff"] == "1_Li1x1a_O1x1a"
    assert rows[0]["target_formula"] == "LiO"
    assert missing == [
        {
            "source_index": 1,
            "target_formula": "C",
            "target_wyckoff_spglib": "A_cP1_1_a:C",
            "material_id": "mp-2",
            "reason": "template_not_found",
        }
    ]

    output_dir = tmp_path / "output"
    write_outputs(rows, missing, output_dir)
    assert pd.read_csv(output_dir / "wyckoff_info.csv").to_dict("records") == [
        {
            "formula": "Li1O1",
            "num_evals": 1,
            "pressure": 0,
            "wyckoff": "1_Li1x1a_O1x1a",
        }
    ]
    assert len(pd.read_csv(output_dir / "manifest.csv")) == 1
    assert len(pd.read_csv(output_dir / "missing_templates.csv")) == 1
