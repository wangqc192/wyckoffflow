"""Categorical source distributions."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CategoricalSource(nn.Module):
    def __init__(self, name, variable_dim, marginal):
        super().__init__()
        if name == "uniform":
            probabilities = torch.ones(variable_dim)
        elif name == "marginal":
            probabilities = F.pad(
                marginal.flatten()[:variable_dim],
                (0, max(0, variable_dim - marginal.numel())),
            )
        elif name in {"zeros", "zeros_init"}:
            probabilities = torch.zeros(variable_dim)
            probabilities[0] = 1
        else:
            raise ValueError(f"Unknown categorical source: {name}")
        self.register_buffer("probabilities", probabilities / probabilities.sum())

    def sample(self, shape):
        samples = torch.multinomial(
            self.probabilities.expand(math.prod(shape), -1),
            1,
        ).squeeze(-1)
        return samples.reshape(shape)
