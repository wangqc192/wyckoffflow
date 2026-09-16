from pathlib import Path

import torch
from torch_geometric.data import Data

import models.generation_pipeline as pipeline
from models.common.lookup_tables import chemical_symbols, wyckoff_label_to_index


def wyckoff_graph(space_group, sites):
    matrix = torch.zeros((max(wyckoff_label_to_index.values()), len(chemical_symbols)))
    for label, element, count in sites:
        row = wyckoff_label_to_index[label] - 1
        matrix[row, chemical_symbols.index(element)] = count
    return Data(x=matrix, space_group=torch.tensor(space_group))


def test_formula_parsing_keeps_complete_integer_counts():
    assert pipeline.parse_formula_counts("Ga4Te4") == {"Ga": 4, "Te": 4}
    assert pipeline.formula_atom_count("Ga4Te4") == 8
    assert pipeline.format_formula(pipeline.parse_formula_counts("Li2(PO4)")) == (
        "Li2O4P"
    )


def test_sequence_conversions_preserve_all_orbits():
    sequence = "194-4f-Ga-4f-Te"
    assert pipeline.exact_counts_from_sequence(sequence) == {"Ga": 4, "Te": 4}
    assert pipeline.query_from_sequence(sequence) == {
        "spacegroup_number": 194,
        "wyckoff_letters": ["4f", "4f"],
        "atom_types": ["Ga", "Te"],
    }
    assert pipeline.wyckoff_template_from_sequence(sequence) == ("194_Ga1x4f_Te1x4f")


def test_diffcsp_template_combines_repeated_element_orbit_pairs():
    assert (
        pipeline.wyckoff_template_from_sequence("1-1a-Li-1a-Li-1a-O")
        == "1_Li2x1a_O1x1a"
    )


def test_template_sampling_deduplicates_and_keeps_hierarchy_slots(monkeypatch):
    ga_te_194 = wyckoff_graph(194, [("f", "Ga", 1), ("f", "Te", 1)])
    ga_only_194 = wyckoff_graph(194, [("f", "Ga", 1)])
    ga_te_225 = wyckoff_graph(225, [("a", "Ga", 1), ("b", "Te", 1)])

    def fake_sample(model, formula, space_group, pool_size, *, flow_steps):
        assert formula == "Ga4Te4"
        assert pool_size == 16
        assert flow_steps == 7
        if space_group == 194:
            return [ga_te_194, ga_te_194.clone(), ga_only_194]
        return [ga_te_225]

    monkeypatch.setattr(pipeline, "_sample_one_space_group", fake_sample)
    rows = pipeline.sample_wyckoff_templates(
        object(),
        "Ga4Te4",
        [194, 225],
        templates_per_space_group=4,
        template_pool_size=16,
        flow_steps=7,
    )

    assert [row["candidate_index"] for row in rows] == [0, 4]
    assert [row["space_group_rank"] for row in rows] == [1, 2]
    assert [row["template_rank"] for row in rows] == [1, 1]
    assert [row["frequency"] for row in rows] == [2, 1]
    assert [row["generated_formula"] for row in rows] == ["Ga4Te4", "Ga4Te4"]


def test_write_selected_queries_keeps_template_order(tmp_path: Path):
    path = tmp_path / "queries.json"
    rows = [
        {"query": {"spacegroup_number": 1}},
        {"query": {"spacegroup_number": 2}},
    ]
    pipeline.write_selected_queries(rows, path)
    assert path.read_text() == (
        '[\n  {\n    "spacegroup_number": 1\n  },\n'
        '  {\n    "spacegroup_number": 2\n  }\n]'
    )
