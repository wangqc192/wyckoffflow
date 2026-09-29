"""MLP building blocks shared by the model modules."""

import torch.nn as nn


def get_mlp(
    input_dim,
    output_dim,
    hidden_dim,
    num_hidden_layers=1,
    activation="SiLU",
    *,
    layer_norm=False,
    dropout=None,
):
    """Build exactly ``num_hidden_layers`` hidden blocks and a linear output.

    ``dropout=None`` omits the layer; explicit 0.0 retains Dropout(0) and its
    Sequential index so existing CrystalGNN checkpoints keep their keys.
    """
    activation_cls = getattr(nn, activation)
    layers = []
    current_dim = input_dim
    for _ in range(num_hidden_layers):
        layers.append(nn.Linear(current_dim, hidden_dim))
        current_dim = hidden_dim
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(activation_cls())
        if dropout is not None:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)
