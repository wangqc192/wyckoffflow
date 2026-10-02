import pytest
import torch
from torch import nn

from models.pl_models.mlp import get_mlp


def test_dropout_can_be_changed_without_changing_checkpoint_keys():
    torch.manual_seed(42)
    disabled = get_mlp(8, 3, 16, num_hidden_layers=3, layer_norm=True)
    enabled = get_mlp(8, 3, 16, num_hidden_layers=3, layer_norm=True, dropout=0.5)
    enabled.load_state_dict(disabled.state_dict(), strict=True)
    inputs = torch.randn(32, 8)
    expected = disabled(inputs)
    torch.testing.assert_close(disabled(inputs), expected, rtol=0, atol=0)
    assert not torch.equal(enabled(inputs), enabled(inputs))
    enabled.eval()
    torch.testing.assert_close(enabled(inputs), expected, rtol=0, atol=0)


@pytest.mark.parametrize("layer_norm", [False, True])
@pytest.mark.parametrize("dropout", [None, 0.0, 0.2])
@pytest.mark.parametrize("hidden_layers", [1, 3])
def test_legacy_checkpoint_preserves_train_and_eval_outputs(
    layer_norm, dropout, hidden_layers
):
    # Build the two historical Sequential layouts independently of get_mlp.
    layers = []
    for index in range(hidden_layers):
        layers.append(nn.Linear(8 if index == 0 else 16, 16))
        if layer_norm:
            layers.append(nn.LayerNorm(16))
        layers.append(nn.SiLU())
        if dropout is not None:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(16, 3))
    legacy = nn.Sequential(nn.Sequential(*layers))
    restored = nn.Sequential(
        get_mlp(8, 3, 16, hidden_layers, layer_norm=layer_norm, dropout=dropout or 0.0)
    )
    restored.load_state_dict(legacy.state_dict(), strict=True)
    # Some exports discard state_dict metadata; nested prefixes must still work.
    restored.load_state_dict(dict(legacy.state_dict()), strict=True)
    inputs = torch.randn(32, 8)
    for training in (True, False):
        legacy.train(training)
        restored.train(training)
        torch.manual_seed(7)
        expected = legacy(inputs)
        torch.manual_seed(7)
        torch.testing.assert_close(restored(inputs), expected, rtol=0, atol=0)


def test_dropout_does_not_drop_linear_output():
    model = get_mlp(8, 3, 16, dropout=1.0)
    inputs = torch.randn(32, 8)
    torch.testing.assert_close(model(inputs), model[-1].bias.expand(32, -1))
    linear = get_mlp(8, 3, 16, num_hidden_layers=0, dropout=1.0)
    torch.testing.assert_close(linear(inputs), linear[0](inputs))
