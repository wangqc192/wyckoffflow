from pathlib import Path

import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

import models.generation_pipeline as pipeline
import models.sampling as sampling
from models.common.lookup_tables import chemical_symbols, wyckoff_label_to_index
from models.pl_models.count_conserving import DecodingResult


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


@pytest.mark.parametrize("sampling_mode", ["n-shot", "greedy"])
def test_template_sampling_deduplicates_and_keeps_hierarchy_slots(
    monkeypatch, sampling_mode
):
    ga_te_194 = wyckoff_graph(194, [("f", "Ga", 1), ("f", "Te", 1)])
    ga_only_194 = wyckoff_graph(194, [("f", "Ga", 1)])
    ga_te_225 = wyckoff_graph(225, [("a", "Ga", 1), ("b", "Te", 1)])

    def fake_sample(model, formula, space_group, pool_size, *, flow_steps, **kwargs):
        assert formula == "Ga4Te4"
        assert pool_size == 16
        assert flow_steps == 7
        assert kwargs["sampling_mode"] == sampling_mode
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
        sampling_mode=sampling_mode,
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


def test_top_n_batch_preserves_target_groups_and_indices(monkeypatch):
    records = [
        {
            "target_index": 10,
            "formula": "Ga4Te4",
            "generated_space_group": 194,
        },
        {
            "target_index": 10,
            "formula": "Ga4Te4",
            "generated_space_group": 225,
        },
        {
            "target_index": 20,
            "formula": "Ga4Te4",
            "generated_space_group": 194,
        },
    ]

    class FakeModel:
        num_elements = 118
        max_num_atoms = 8

        def sample_logits(self, batch, *, flow_steps, greedy):
            assert greedy is False
            assert batch.target_index.tolist() == [10, 10, 20]
            assert batch.sampling_group.tolist() == [0, 1, 2]
            return batch, None, None

    def fake_decode(data, zero_logits, inf_logits, max_num_atoms, **kwargs):
        return DecodingResult(data.to_data_list(), [])

    monkeypatch.setattr(sampling, "decode_composition_logits", fake_decode)
    sampled = pipeline._sample_record_batch(
        FakeModel(), records, 2, flow_steps=1, sampling_mode="top-n"
    )

    assert [record["target_index"] for record, _ in sampled] == [10, 10, 20]
    assert [record["generated_space_group"] for record, _ in sampled] == [194, 225, 194]
    assert [int(samples[0].target_index) for _, samples in sampled] == [10, 10, 20]
    assert [int(samples[0].sampling_group) for _, samples in sampled] == [0, 1, 2]


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("sampling_mode", ["top-n", "greedy"])
def test_template_ranking_in_both_entry_points(monkeypatch, batched, sampling_mode):
    high = wyckoff_graph(194, [("f", "Ga", 1), ("f", "Te", 1)])
    low = wyckoff_graph(194, [("e", "Ga", 1), ("f", "Te", 1)])
    high.candidate_rank = torch.tensor(1)
    high.decoder_log_score = torch.tensor(-0.1)
    low.candidate_rank = torch.tensor(2)
    low.decoder_log_score = torch.tensor(-2.0)
    samples = [high, low] if sampling_mode == "top-n" else [high, high.clone(), low]

    if batched:

        def fake_batch(model, records, num_samples, **kwargs):
            assert kwargs["sampling_mode"] == sampling_mode
            assert num_samples == (2 if sampling_mode == "top-n" else 3)
            return [(record, samples) for record in records]

        monkeypatch.setattr(pipeline, "_sample_record_batch", fake_batch)
        model = type("FakeModel", (), {"max_num_atoms": 8})()
        targets = pd.DataFrame([{"target_index": 0, "formula": "Ga4Te4"}])
        predictions = pd.DataFrame(
            [
                {
                    "target_index": 0,
                    "Spacegroup Number": 194,
                    "SG_Rank": 1,
                    "SG_Prob": 1.0,
                }
            ]
        )
        rows = pipeline.sample_wyckoff_templates_for_targets(
            model,
            targets,
            predictions,
            templates_per_space_group=2,
            template_pool_size=3,
            flow_steps=1,
            sampling_mode=sampling_mode,
        )
    else:

        def fake_single(model, formula, space_group, num_samples, **kwargs):
            assert kwargs["sampling_mode"] == sampling_mode
            assert num_samples == (2 if sampling_mode == "top-n" else 3)
            return samples

        monkeypatch.setattr(pipeline, "_sample_one_space_group", fake_single)
        rows = pipeline.sample_wyckoff_templates(
            object(),
            "Ga4Te4",
            [194],
            templates_per_space_group=2,
            template_pool_size=3,
            flow_steps=1,
            sampling_mode=sampling_mode,
        )
    assert [row["template_rank"] for row in rows] == [1, 2]
    assert [int(row["sample"].candidate_rank) for row in rows] == [1, 2]
    assert [row["frequency"] for row in rows] == (
        [1, 1] if sampling_mode == "top-n" else [2, 1]
    )
    assert rows[0]["generated_structure_sequence"] == "194-4f-Ga-4f-Te"
