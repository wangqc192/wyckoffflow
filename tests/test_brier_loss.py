import pytest
import torch

from models.pl_models.flow import categorical_brier_loss


def test_brier_stays_bounded_for_saturated_predictions_and_empty_branch():
    logits = torch.tensor([[100.0, -100.0], [-100.0, 100.0]], requires_grad=True)
    losses = categorical_brier_loss(logits, torch.tensor([0, 0]))
    torch.testing.assert_close(losses, torch.tensor([0.0, 2.0]))
    losses.sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert categorical_brier_loss(
        logits[:0], torch.empty(0, dtype=torch.long)
    ).shape == (0,)


@pytest.mark.parametrize("num_classes", [3, 55])
def test_expected_brier_is_minimized_at_true_posterior(num_classes):
    # Ambiguous endpoints should still learn their posterior, not a forced
    # winner or a uniformly smoothed version of the posterior.
    truth = torch.arange(1, num_classes + 1, dtype=torch.float32)
    truth = truth / truth.sum()
    logits = truth.log().detach().requires_grad_(True)
    targets = torch.arange(num_classes)
    loss = (
        categorical_brier_loss(logits.expand(num_classes, -1), targets) * truth
    ).sum()
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits), atol=1e-7, rtol=0)
    uniform = torch.zeros(num_classes, num_classes)
    assert (categorical_brier_loss(uniform, targets) * truth).sum() > loss


def test_masked_classes_have_zero_loss_gradient_and_no_probability_floor():
    logits = torch.tensor([[2.0, 1.0, 1000.0]], requires_grad=True)
    masked = logits.masked_fill(torch.tensor([[False, False, True]]), -torch.inf)
    loss = categorical_brier_loss(masked, torch.tensor([0]))
    loss.sum().backward()
    assert logits.grad[0, 0] < 0
    assert logits.grad[0, 1] > 0  # Gradient descent reduces wrong-class probability.
    assert logits.grad[0, 2] == 0
    expected = 2 * torch.sigmoid(torch.tensor(-1.0)).square()
    torch.testing.assert_close(loss[0], expected)


def test_brier_uses_float32_for_mixed_precision_probabilities():
    logits = torch.tensor([[8.0, 0.0, -5.0]], dtype=torch.bfloat16, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = categorical_brier_loss(logits, torch.tensor([0]))
    assert loss.dtype == torch.float32
    assert 0 < loss.item() < 1e-5
    loss.sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0] < 0
