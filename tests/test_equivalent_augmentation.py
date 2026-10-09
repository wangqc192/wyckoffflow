import numpy as np
import torch

from aviary.wren.data import parse_protostructure_label
from models.common.wyckoff_template import WyckoffTemplate
from models.pl_data.dataset import CrystalDataset


def test_parser_without_augmentation_preserves_written_setting_and_counts():
    group, multiplicities, elements, settings = parse_protostructure_label(
        "AB2_cF12_225_b_2a:Cl-Na", augment=False
    )
    assert group == "225"
    assert multiplicities == [4.0, 4.0, 4.0]
    assert elements == ["Cl", "Na", "Na"]
    assert settings == [("b", "a", "a")]
    assert len(parse_protostructure_label("AB_cF8_225_b_a:Cl-Na")[3]) == 2


def test_disabling_augmentation_uses_label_not_first_cached_setting():
    dataset = CrystalDataset(
        {"wyckoff_spglib": "AB_cF8_225_b_a:Cl-Na"},
        num_elements=100,
        augment_equivalent_templates=False,
    )
    assert dataset.data[0]["wyckoff_set"][0] != ("b", "a")
    original = np.array(dataset.data[0]["wyckoff_element_matrix"], copy=True)
    expected = WyckoffTemplate.from_crystalflow("225_Cl1x4b_Na1x4a")
    for seed in range(8):
        torch.manual_seed(seed)
        graph = dataset[0]
        assert WyckoffTemplate.from_model_output(graph) == expected
        assert graph.composition[0, 11] == graph.composition[0, 17] == 4
        assert torch.equal(graph.x_0_dof, graph.x[graph.zero_dof, 0])
        assert torch.equal(graph.x_inf_dof, graph.x[~graph.zero_dof, 1:101])
    np.testing.assert_array_equal(
        original, dataset.data[0]["wyckoff_element_matrix"]
    )


def test_disabled_augmentation_preserves_control_rng_and_validation_views():
    augmented = CrystalDataset(
        {"wyckoff_spglib": "AB_cF8_225_b_a:Cl-Na"}, num_elements=100
    )
    fixed = CrystalDataset(
        augmented.data, num_elements=100, augment_equivalent_templates=False
    )
    observed = set()
    for seed in range(16):
        torch.manual_seed(seed)
        observed.add(WyckoffTemplate.from_model_output(augmented[0]).occupancy_key())
        expected_rng = torch.get_rng_state()
        torch.manual_seed(seed)
        fixed[0]
        assert torch.equal(torch.get_rng_state(), expected_rng)
    assert observed == {
        template.occupancy_key()
        for template in WyckoffTemplate.from_protostructure_set(
            "AB_cF8_225_b_a:Cl-Na"
        )
    }


def test_single_setting_is_unchanged_when_augmentation_disabled():
    augmented = CrystalDataset(
        {"wyckoff_spglib": "AB_aP3_1_2a_a:Li-O"}, num_elements=100
    )
    fixed = CrystalDataset(
        augmented.data, num_elements=100, augment_equivalent_templates=False
    )
    torch.manual_seed(42)
    before = torch.get_rng_state()
    assert torch.equal(augmented[0].x, fixed[0].x)
    assert torch.equal(torch.get_rng_state(), before)
