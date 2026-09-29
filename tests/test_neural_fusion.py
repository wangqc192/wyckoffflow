import pytest
import torch
import torch.nn.functional as F

from models.pl_models.chemical_sg import (
    ChemicalSpaceGroupPredictor,
    load_chemical_space_group_model,
)
from models.pl_models.count_conserving import formula_space_group_mask
from models.pl_models.neural_fusion import NeuralFusion


@pytest.mark.parametrize("base_weights", [None, [0.2, 0.8]])
def test_fusion_starts_at_expert_average_and_learns_with_masked_classes(
    tmp_path, base_weights
):
    torch.set_num_threads(2)
    config = dict(num_elements=10, width=16, layers=1, dropout=0)
    experts = [ChemicalSpaceGroupPredictor(**config).eval() for _ in range(2)]
    composition = torch.zeros(2, 11)
    composition[:, [3, 8]] = torch.tensor([[4.0, 4.0], [2.0, 6.0]])
    feasible = formula_space_group_mask(composition, 54)
    p = torch.stack(
        [model(composition, feasible).softmax(-1).detach() for model in experts], dim=1
    )
    fusion_config = dict(
        experts=2, width=16, dropout=0, num_elements=10, base_weights=base_weights
    )
    fusion = NeuralFusion(**fusion_config).eval()
    logits, correction = fusion(composition, feasible, p)
    initial = p.mean(1) if base_weights is None else 0.2 * p[:, 0] + 0.8 * p[:, 1]
    torch.testing.assert_close(logits.softmax(-1), initial)
    assert torch.equal(correction, torch.zeros_like(correction))
    F.cross_entropy(logits, torch.tensor([1, 2])).backward()
    assert all(
        torch.isfinite(param.grad).all()
        for param in fusion.parameters()
        if param.grad is not None
    )
    assert fusion.score[-1].weight.grad.abs().sum() > 0
    with torch.no_grad():
        fusion.score[-1].weight.normal_(std=0.1)
    members = [
        {"config": config, "state_dict": expert.state_dict()} for expert in experts
    ]
    checkpoint = {
        "architecture": "neural_fusion",
        "expert_groups": [members],
        "fusion_config": fusion_config,
        "fusion_state_dict": fusion.state_dict(),
    }
    path = tmp_path / "fusion.pt"
    torch.save(checkpoint, path)
    loaded = load_chemical_space_group_model(path)
    expected = fusion(composition, feasible, p)[0].softmax(-1)
    result = loaded.predict_space_groups(composition)
    _, base = loaded.predict_space_groups_with_base(composition)
    torch.testing.assert_close(base, p.mean(1))
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(loaded.predict_space_groups(composition[:1]), result[:1])
    assert (result[~feasible] == 0).all()
    members[0]["prior_weight"] = 1.0
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="purely neural"):
        load_chemical_space_group_model(path)


def test_prototype_head_sums_learned_class_probabilities_by_group(tmp_path):
    torch.set_num_threads(2)
    config = dict(num_elements=10, width=16, layers=1, dropout=0, auxiliary_classes=4)
    expert = ChemicalSpaceGroupPredictor(**config).eval()
    fusion_config = dict(experts=1, width=16, dropout=0, num_elements=10)
    fusion = NeuralFusion(**fusion_config).eval()
    composition = torch.zeros(2, 11)
    composition[:, [3, 8]] = torch.tensor([[4.0, 4.0], [1.0, 1.0]])
    feasible = formula_space_group_mask(composition, 54)
    taxonomy = torch.tensor([1, 1, 2, 225])
    sg_logits, prototype_logits = expert(composition, feasible, return_auxiliary=True)
    classes = prototype_logits.masked_fill(~feasible[:, taxonomy], -torch.inf).softmax(
        -1
    )
    expected = 0.5 * sg_logits.softmax(-1)
    expected[:, 1] += 0.5 * classes[:, :2].sum(-1)
    expected[:, 2] += 0.5 * classes[:, 2]
    expected[:, 225] += 0.5 * classes[:, 3]
    checkpoint = {
        "architecture": "neural_fusion",
        "expert_groups": [[{"config": config, "state_dict": expert.state_dict()}]],
        "fusion_config": fusion_config,
        "fusion_state_dict": fusion.state_dict(),
        "prototype_head": {
            "member": 0,
            "weight": 0.5,
            "space_group_ids": taxonomy.tolist(),
        },
    }
    path = tmp_path / "hierarchy.pt"
    torch.save(checkpoint, path)
    model = load_chemical_space_group_model(path)
    result, base = model.predict_space_groups_with_base(composition)
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(base, sg_logits.softmax(-1))
    torch.testing.assert_close(result.sum(-1), torch.ones(2))
    assert (result[~feasible] == 0).all()
    torch.testing.assert_close(model.predict_space_groups(composition[:1]), result[:1])
