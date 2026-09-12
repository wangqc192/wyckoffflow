"""MLP building blocks shared by the model modules."""

import torch.nn as nn


def get_mlp(input_dim, output_dim, hidden_dim, num_hidden_layers, activation):
    activation_cls = getattr(nn, activation)
    layers = [nn.Linear(input_dim, hidden_dim), activation_cls()]
    for _ in range(num_hidden_layers):
        layers.extend((nn.Linear(hidden_dim, hidden_dim), activation_cls()))
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)
