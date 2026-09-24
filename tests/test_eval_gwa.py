import json
from types import SimpleNamespace

import pandas as pd

from scripts.eval_gwa import (
    evaluate,
    evaluation_summary,
    main,
    occupancy_key,
    target_occupancy_keys,
)


def test_occupancy_key_ignores_order_but_preserves_counts():
    expected = occupancy_key("194_Ga1x4f_Te1x4f")

    assert occupancy_key("194_Te1x4f_Ga1x4f") == expected
    assert occupancy_key("194_Ga2x4f_Te1x4f") != expected


def test_occupancy_key_accepts_numeric_csv_values():
    assert occupancy_key(194) == ("194",)
    assert occupancy_key("194") == ("194",)


def test_target_occupancy_keys_include_equivalent_settings():
    keys = target_occupancy_keys("AB_oC4_65_a_c:Cu-Ni")

    assert occupancy_key("65_Ni1x2a_Cu1x2c") in keys
    assert occupancy_key("65_Cu1x2a_Ni1x2c") in keys


def test_evaluate_groups_top_k_candidates_by_target_index():
    targets = pd.DataFrame(
        {
            "material_id": ["first", "second"],
            "pretty_formula": ["GaTe", "CuNi"],
            "wyckoff_spglib": [
                "AB_hP8_194_f_f:Ga-Te",
                "AB_oC4_65_a_c:Cu-Ni",
            ],
        }
    )
    generated = pd.DataFrame(
        {
            "sample_index": [0, 1, 20],
            "target_index": [0, 0, 1],
            "wyckoff_occupancy": [
                "194_Ga1x4f_Te1x4f",
                "194_Ga1x2a_Te1x2a",
                "65_Cu2x2b_Ni1x2d",
            ],
            "count": [1, 19, 20],
        }
    )

    details, hits = evaluate(targets, generated, top_k=20)

    assert hits == 1
    assert details["generated_count"].tolist() == [20, 20]
    assert details["matched"].tolist() == [True, False]


def test_evaluate_does_not_mix_targets_with_same_formula_and_space_group():
    targets = pd.DataFrame(
        {
            "material_id": ["first", "second"],
            "pretty_formula": ["CuNi", "CuNi"],
            "wyckoff_spglib": [
                "AB_oC4_65_a_c:Cu-Ni",
                "AB_oC8_65_g_h:Cu-Ni",
            ],
        }
    )
    generated = pd.DataFrame(
        {
            "sample_index": [0, 1],
            "target_index": [0, 1],
            "wyckoff_occupancy": [
                "65_Cu1x4g_Ni1x4h",
                "65_Cu1x2a_Ni1x2c",
            ],
        }
    )

    details, hits = evaluate(targets, generated, top_k=1)

    assert hits == 0
    assert details["matched"].tolist() == [False, False]


def test_evaluation_summary_reports_exact_top_k_rate():
    details = pd.DataFrame({"generated_count": [20, 0, 20]})

    assert evaluation_summary(details, hits=2, top_k=20) == {
        "metric": "G-W-A Top-K",
        "top_k": 20,
        "matched_materials": 2,
        "total_materials": 3,
        "match_rate": 2 / 3,
        "materials_with_generated_samples": 2,
    }


def test_main_writes_json_summary(tmp_path):
    target_path = tmp_path / "target.csv"
    generated_path = tmp_path / "generated.csv"
    summary_path = tmp_path / "summary.json"
    pd.DataFrame(
        {
            "material_id": ["first"],
            "pretty_formula": ["GaTe"],
            "wyckoff_spglib": ["AB_hP8_194_f_f:Ga-Te"],
        }
    ).to_csv(target_path, index=False)
    pd.DataFrame(
        {
            "sample_index": [0],
            "target_index": [0],
            "wyckoff_occupancy": ["194_Ga1x4f_Te1x4f"],
        }
    ).to_csv(generated_path, index=False)

    main(
        SimpleNamespace(
            target_path=target_path,
            gen_path=generated_path,
            top_k=1,
            output_path=None,
            summary_path=summary_path,
        )
    )

    assert json.loads(summary_path.read_text()) == {
        "metric": "G-W-A Top-K",
        "top_k": 1,
        "matched_materials": 1,
        "total_materials": 1,
        "match_rate": 1.0,
        "materials_with_generated_samples": 1,
    }
