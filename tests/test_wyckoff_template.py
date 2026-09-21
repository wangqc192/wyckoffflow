from types import SimpleNamespace

import pytest
import torch

from models.common.lookup_tables import chemical_symbols, wyckoff_label_to_index
from models.common.wyckoff_template import WyckoffTemplate, WyckoffTemplete


def test_round_trip_model_crystalflow_and_diffcsppp_formats():
    template = WyckoffTemplate.from_gwa("1-1a-Li-1a-Li-1a-O")

    assert template.formula_counts == {"Li": 2, "O": 1}
    assert template.formula == "Li2O"
    assert template.to_crystalflow() == "1_Li2x1a_O1x1a"
    assert template.to_diffcsppp_query() == {
        "spacegroup_number": 1,
        "wyckoff_letters": ["1a", "1a", "1a"],
        "atom_types": ["Li", "Li", "O"],
    }
    assert WyckoffTemplate.from_crystalflow(template.to_crystalflow()) == template
    assert (
        WyckoffTemplate.from_diffcsppp_query(template.to_diffcsppp_query())
        == template
    )


def test_template_equality_ignores_orbit_order_and_letter_case():
    first = WyckoffTemplate.from_crystalflow("47_O1x8A_Sr1x1b")
    second = WyckoffTemplate.from_gwa("47-1b-Sr-8A-O")

    assert first == second
    assert hash(first) == hash(second)
    assert first.to_diffcsppp_query()["wyckoff_letters"] == ["8A", "1b"]
    with pytest.raises(ValueError, match="expected 1"):
        WyckoffTemplate.from_gwa("47-1b-Sr-8a-O")


def test_protostructure_matching_checks_all_equivalent_settings():
    template = WyckoffTemplate.from_crystalflow("65_Ni1x2a_Cu1x2c")

    assert template.matches_protostructure("AB_oC4_65_a_c:Cu-Ni")
    assert not WyckoffTemplate.from_crystalflow(
        "65_Cu1x4g_Ni1x4h"
    ).matches_protostructure("AB_oC4_65_a_c:Cu-Ni")


def test_model_graph_with_truncated_element_channels_is_converted():
    matrix = torch.zeros((1, 101))
    matrix[0, chemical_symbols.index("Si")] = 1
    sample = SimpleNamespace(x=matrix, space_group=torch.tensor([1]))

    template = WyckoffTemplate.from_model_output(sample)

    assert template.to_gwa() == "1-1a-Si"
    assert template.to_crystalflow() == "1_Si1x1a"


def test_model_graph_is_converted_without_dataframe_context():
    matrix = torch.zeros(
        (max(wyckoff_label_to_index.values()), len(chemical_symbols))
    )
    matrix[wyckoff_label_to_index["f"] - 1, chemical_symbols.index("Ga")] = 1
    matrix[wyckoff_label_to_index["f"] - 1, chemical_symbols.index("Te")] = 1
    sample = SimpleNamespace(x=matrix, space_group=torch.tensor([194]))

    template = WyckoffTemplate.from_model_output(sample)

    assert template.to_gwa() == "194-4f-Ga-4f-Te"
    assert template.to_crystalflow() == "194_Ga1x4f_Te1x4f"


def test_original_misspelled_class_name_remains_available():
    assert WyckoffTemplete is WyckoffTemplate
