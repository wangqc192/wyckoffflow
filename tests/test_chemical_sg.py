import pytest
import torch
import torch.nn.functional as F

from models.pl_models.chemical_sg import (
    ChemicalSpaceGroupEnsemble,
    ChemicalSpaceGroupPredictor,
    load_chemical_space_group_model,
    masked_classification_loss,
)
from models.pl_models.count_conserving import formula_space_group_mask


def test_smoothing_uses_only_feasible_classes_and_has_finite_gradients():
    logits = torch.tensor([[10.0, 2.0, -1.0, 100.0]], requires_grad=True)
    mask = torch.tensor([[False, True, True, False]])
    loss = masked_classification_loss(logits, torch.tensor([1]), mask, 0.1)
    expected = F.cross_entropy(logits[:, 1:3], torch.tensor([0]), label_smoothing=0.1)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0] == logits.grad[0, 3] == 0


@pytest.mark.parametrize(
    "attention,conditioned", [(False, False), (True, False), (False, True)]
)
def test_composition_prediction_batch_independence_masking_and_roundtrip(
    attention, conditioned
):
    torch.set_num_threads(2)
    model = ChemicalSpaceGroupPredictor(
        num_elements=10,
        width=32,
        layers=1,
        dropout=0,
        attention=attention,
        count_embedding=attention,
        conditioned=conditioned,
        count_histogram=conditioned,
    ).eval()
    composition = torch.zeros(2, 11)
    composition[0, 3] = 1
    composition[1, [2, 3, 8]] = torch.tensor([2.0, 4.0, 64.0])
    feasible = formula_space_group_mask(composition, 54)
    logits = model(composition, feasible)
    single = model(composition[:1], feasible[:1])
    torch.testing.assert_close(logits[:1], single, atol=2e-6, rtol=2e-5)
    probabilities = logits.softmax(-1)
    assert bool((probabilities[~feasible] == 0).all())
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(2))
    assert probabilities[0, 225] == 0
    target = torch.tensor([1, 2])
    loss = masked_classification_loss(logits, target, feasible, 0.05)
    loss.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    other = ChemicalSpaceGroupPredictor(
        num_elements=10,
        width=32,
        layers=1,
        dropout=0,
        attention=attention,
        count_embedding=attention,
        conditioned=conditioned,
        count_histogram=conditioned,
    ).eval()
    other.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(other(composition, feasible), logits)


def test_ensemble_checkpoint_masks_and_handles_unseen_stoichiometry(tmp_path):
    config = dict(
        num_elements=10, width=32, layers=1, dropout=0.0, attention=False, role_groups=4
    )
    member = ChemicalSpaceGroupPredictor(**config).eval()
    checkpoint = {"config": config, "state_dict": member.state_dict()}
    composition = torch.zeros(2, 11)
    composition[0, 3] = 1
    composition[1, [3, 8]] = torch.tensor([2.0, 4.0])
    feasible = formula_space_group_mask(composition, 54)
    second = ChemicalSpaceGroupPredictor(**config).eval()
    other = {"config": config, "state_dict": second.state_dict()}
    model = ChemicalSpaceGroupEnsemble([checkpoint, other], [1, 3]).eval()
    expected = 0.25 * member(composition, feasible).softmax(-1) + 0.75 * second(
        composition, feasible
    ).softmax(-1)
    torch.testing.assert_close(model.predict_space_groups(composition), expected)
    assert bool((expected[~feasible] == 0).all())
    torch.testing.assert_close(expected.sum(-1), torch.ones(2))
    torch.testing.assert_close(
        model.predict_space_groups(composition[:1]), expected[:1]
    )
    path = tmp_path / "ensemble.pt"
    torch.save({"members": [checkpoint, other], "weights": [1, 3]}, path)
    loaded = load_chemical_space_group_model(path)
    torch.testing.assert_close(loaded.predict_space_groups(composition), expected)


@pytest.mark.parametrize("field", ["prior", "prior_exponent", "prior_weight"])
def test_prior_checkpoints_are_rejected(tmp_path, field):
    config = dict(num_elements=10, width=32, layers=1)
    member = ChemicalSpaceGroupPredictor(**config)
    checkpoint = {"config": config, "state_dict": member.state_dict(), field: 1.0}
    path = tmp_path / "prior.pt"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="purely neural"):
        load_chemical_space_group_model(path)


def test_unmasked_classifier_and_training_only_auxiliary_head():
    model = ChemicalSpaceGroupPredictor(
        num_elements=10,
        width=32,
        layers=1,
        dropout=0,
        use_feasibility=False,
        conditioned=True,
        auxiliary_classes=8,
    ).eval()
    composition = torch.zeros(2, 11)
    composition[:, 3] = 1
    restricted = formula_space_group_mask(composition, 54)
    logits, auxiliary = model(composition, restricted, return_auxiliary=True)
    # An unmasked network never consumes even an intentionally different feasibility mask.
    torch.testing.assert_close(logits, model(composition, torch.ones_like(restricted)))
    assert torch.isfinite(logits[:, 1:]).all()
    assert torch.isneginf(logits[:, 0]).all()
    F.cross_entropy(auxiliary, torch.tensor([1, 2])).backward()
    assert model.project[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("mixed_precision", [False, True])
def test_hierarchical_training_has_finite_gradients_and_checkpoint_inference(
    tmp_path, mixed_precision
):
    config = dict(
        num_elements=10,
        width=16,
        layers=1,
        dropout=0,
        auxiliary_classes=4,
        prototype_space_groups=[1, 1, 2, 225],
        prototype_weight=0.5,
    )
    model = ChemicalSpaceGroupPredictor(**config).eval()
    composition = torch.zeros(2, 11)
    composition[:, [3, 8]] = torch.tensor([[1.0, 1.0], [4.0, 4.0]])
    feasible = formula_space_group_mask(composition, 54)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=mixed_precision):
        logits, auxiliary = model(composition, feasible, return_auxiliary=True)
    loss = masked_classification_loss(logits, torch.tensor([1, 225]), feasible, 0.02)
    loss = loss + 0.3 * F.cross_entropy(auxiliary, torch.tensor([0, 3]))
    loss.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    assert model.auxiliary[-1].weight.grad.abs().sum() > 0
    assert model.output[-1].weight.grad.abs().sum() > 0
    path = tmp_path / "hierarchical.pt"
    torch.save({"config": config, "state_dict": model.state_dict()}, path)
    loaded = load_chemical_space_group_model(path)
    probability = loaded.predict_space_groups(composition)
    torch.testing.assert_close(probability, model(composition, feasible).softmax(-1))
    assert (probability[~feasible] == 0).all()
