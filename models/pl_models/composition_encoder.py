"""Static chemical features shared by space-group and occupation models."""

from typing import NamedTuple

import torch
from torch import nn

from .mlp import get_mlp


def _load_legacy_global_composition(state_dict, prefix, *args):
    """Load global composition weights saved under previous module names."""
    new_prefix = f"{prefix}global_composition_encoder."
    for previous_name in ("total_encoder", "composition_stats_encoder"):
        old_prefix = f"{prefix}{previous_name}."
        for key in list(state_dict):
            if key.startswith(old_prefix):
                state_dict[f"{new_prefix}{key[len(old_prefix) :]}"] = state_dict.pop(key)


class CompositionFeatures(NamedTuple):
    element_embeddings: torch.Tensor
    element_tokens: torch.Tensor
    pooled: torch.Tensor


class CrystalCompositionEncoder(nn.Module):
    """Static chemical features shared by space-group and occupation prediction."""

    def __init__(self, num_elements, element_dim, hidden_dim):
        super().__init__()
        self.element_embedding = nn.Embedding(num_elements + 1, element_dim)
        self.token_encoder = get_mlp(element_dim + 2, hidden_dim, hidden_dim)
        self.global_composition_encoder = get_mlp(2, hidden_dim, hidden_dim)
        self._register_load_state_dict_pre_hook(_load_legacy_global_composition)

    def forward(self, composition):
        target = composition[:, 1:].float()
        present = target > 0
        total = target.sum(dim=-1, keepdim=True).clamp_min(1)
        species = present.sum(dim=-1, keepdim=True).clamp_min(1)
        embeddings = self.element_embedding.weight
        elements = embeddings[1:].unsqueeze(0).expand(target.shape[0], -1, -1)
        tokens = self.token_encoder(
            torch.cat(
                (
                    elements,
                    target.log1p().unsqueeze(-1),
                    (target / total).unsqueeze(-1),
                ),
                dim=-1,
            )
        )
        tokens = tokens * present.unsqueeze(-1)
        pooled = tokens.sum(dim=1) / species
        pooled = pooled + self.global_composition_encoder(
            torch.cat((total.log1p(), species.float().log1p()), dim=-1)
        )
        return CompositionFeatures(embeddings, tokens, pooled)
