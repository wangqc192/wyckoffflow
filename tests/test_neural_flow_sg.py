import pytest
import torch

from models.pl_models.chemical_sg import (
    ChemicalSpaceGroupPredictor,
    load_chemical_space_group_model,
)
from models.pl_models.count_conserving import formula_space_group_mask
from models.pl_models.flow import DiscreteFlowModule
from models.pl_models.neural_flow_sg import FlowSpaceGroupScore
from models.pl_models.neural_fusion import NeuralFusion


def test_flow_scoring_checkpoint_uses_only_composition_and_preserves_proposals(
    tmp_path,
):
    torch.set_num_threads(2)
    config = dict(num_elements=10, width=16, layers=1, dropout=0)
    expert = ChemicalSpaceGroupPredictor(**config).eval()
    member = {"config": config, "state_dict": expert.state_dict()}
    fusion_config = dict(experts=1, width=16, dropout=0, num_elements=10)
    fusion = NeuralFusion(**fusion_config).eval()
    reference = {
        "architecture": "neural_fusion",
        "expert_groups": [[member]],
        "fusion_config": fusion_config,
        "fusion_state_dict": fusion.state_dict(),
    }
    flow_config = dict(
        optimizer_config={"_target_": "torch.optim.AdamW", "lr": 1e-4},
        decoder={
            "_target_": "models.pl_models.crystal_gnn.CrystalGNN",
            "hidden_dim": 16,
            "element_dim": 8,
            "num_gnn_layers": 1,
            "num_heads": 4,
            "dropout": 0,
        },
        num_elements=10,
        max_num_atoms=54,
        flow_source="zeros",
    )
    flow = DiscreteFlowModule(**flow_config).eval()
    score = FlowSpaceGroupScore(
        torch.zeros(32), torch.ones(32), dropout=0, num_elements=10
    )
    checkpoint = {
        "architecture": "neural_flow_sg",
        "reference": reference,
        "base_members": [member],
        "base_weights": [1],
        "flow_hyperparameters": flow_config,
        "flow_state_dict": flow.state_dict(),
        "score_state_dict": score.state_dict(),
        "score_config": {"dropout": 0},
        "proposal_groups": 2,
        "mixture_weight": 0.25,
    }
    path = tmp_path / "flow_sg.pt"
    torch.save(checkpoint, path)
    loaded = load_chemical_space_group_model(path)
    loaded.double()
    assert loaded.flow.dtype == torch.float64
    loaded.float()
    assert loaded.flow.dtype == torch.float32
    composition = torch.zeros(2, 11)
    composition[:, [3, 8]] = torch.tensor([[4.0, 4.0], [1.0, 1.0]])
    feasible = formula_space_group_mask(composition, 54)
    probability, proposals = loaded.predict_space_groups_with_base(composition)
    expected = expert(composition, feasible).softmax(-1)
    torch.testing.assert_close(probability, expected)
    torch.testing.assert_close(proposals, expected)
    assert not loaded.flow.decoder.output_norm._forward_hooks
    with torch.no_grad():
        score.score[-1].weight.normal_(std=0.1)
    checkpoint["score_state_dict"] = score.state_dict()
    torch.save(checkpoint, path)
    learned = load_chemical_space_group_model(path)
    result, proposals = learned.predict_space_groups_with_base(composition)
    assert not torch.allclose(result, expected)
    torch.testing.assert_close(proposals, expected)
    torch.testing.assert_close(result.sum(-1), torch.ones(2))
    assert (result[~feasible] == 0).all()
    torch.testing.assert_close(
        learned.predict_space_groups(composition[:1]), result[:1], atol=2e-5, rtol=1e-4
    )
    # Top-k padding may include index 0 when fewer candidates have positive mass.
    def one_group_only(values):
        p = torch.zeros(len(values), 231, device=values.device)
        p[:, 1] = 1
        return p

    learned.base.predict_space_groups = one_group_only
    learned.proposal_groups = 231
    result = learned.predict_space_groups(composition)
    expected_sparse = 0.75 * expected
    expected_sparse[:, 1] += 0.25
    torch.testing.assert_close(result, expected_sparse)
    checkpoint["flow_hyperparameters"]["flow_source"] = "uniform"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="all-zero"):
        load_chemical_space_group_model(path)
