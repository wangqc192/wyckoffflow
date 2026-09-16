import pandas as pd

from scripts.prepare_diffcsp_templates import prepare_templates


def test_prepare_templates_slices_rows_and_preserves_global_indices(tmp_path):
    source = tmp_path / "templates.csv"
    output = tmp_path / "wyckoff_info.csv"
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "sample_index": [2, 0, 1],
            "target_index": [1, 0, 0],
            "formula": ["Li1", "Na1", "K1"],
            "target_formula": ["Li1", "Na1", "K1"],
            "wyckoff_occupancy": ["2_Li1x1a", "1_Na1x1a", "1_K1x1a"],
            "count": [1, 1, 1],
        }
    ).to_csv(source, index=False)

    template_count, target_count = prepare_templates(
        source,
        output,
        manifest,
        start_index=1,
        limit=1,
    )

    assert (template_count, target_count) == (1, 1)
    assert pd.read_csv(output)["formula"].tolist() == ["K1"]
    assert pd.read_csv(manifest)["diffcsp_index"].tolist() == [1]
