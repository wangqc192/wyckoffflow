"""Continuous-time encoders for Wyckoff flow matching."""

import math

import torch
from torch import nn

from .mlp import get_mlp


class FlowTimeEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_frequencies=32,
        min_frequency=1.0,
        max_frequency=100.0,
        include_raw_time=True,
        time_scale=1.0,
    ):
        super().__init__()
        frequencies = torch.pi * torch.logspace(
            math.log10(min_frequency), math.log10(max_frequency), num_frequencies
        )
        self.register_buffer("frequencies", frequencies)
        self.include_raw_time = include_raw_time
        self.time_scale = time_scale
        input_dim = 2 * num_frequencies + 2 * include_raw_time
        self.projection = get_mlp(input_dim, hidden_dim, hidden_dim)

    def forward(self, time):
        time = time.float().reshape(-1, 1)
        angles = time * self.time_scale * self.frequencies
        # Raw time distinguishes the endpoints even for periodic features.
        features = [angles.sin(), angles.cos()]
        if self.include_raw_time:
            features = [time, 1 - time, *features]
        return self.projection(torch.cat(features, dim=-1))


class DiffCSPTimeEncoder(nn.Module):
    """DiffCSP's sinusoidal features for continuous time in [0, 1].

    The default 33 pairs match FlowTimeEncoder's 66-input projection. Time
    scaling changes the sinusoidal phase, with continuous time used by default.
    """

    def __init__(
        self, hidden_dim, num_frequencies=33, max_period=10000.0, time_scale=1.0
    ):
        super().__init__()
        frequencies = torch.exp(
            torch.arange(num_frequencies)
            * -(math.log(max_period) / (num_frequencies - 1))
        )
        self.register_buffer("frequencies", frequencies)
        self.time_scale = time_scale
        self.projection = get_mlp(2 * num_frequencies, hidden_dim, hidden_dim)

    def forward(self, time):
        angles = time.float().reshape(-1, 1) * self.time_scale * self.frequencies
        return self.projection(torch.cat((angles.sin(), angles.cos()), dim=-1))
