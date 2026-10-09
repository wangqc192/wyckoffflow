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
    """Encode element counts into per-element and pooled static features.

    Defaults match the shared Joint encoder. MLP settings and the species-count
    feature also support the standalone SG encoder's existing parameter layout.
    """

    def __init__(
        self,
        num_elements,
        element_dim,
        hidden_dim,
        *,
        mlp_hidden_dim=None,
        num_hidden_layers=1,
        activation="SiLU",
        include_species_count=True,
    ):
        super().__init__()
        self.include_species_count = include_species_count
        if mlp_hidden_dim is None:
            mlp_hidden_dim = hidden_dim
        self.element_embedding = nn.Embedding(num_elements + 1, element_dim)
        self.token_encoder = get_mlp(
            element_dim + 2, hidden_dim, mlp_hidden_dim, num_hidden_layers, activation
        )
        self.global_composition_encoder = get_mlp(
            2 if include_species_count else 1,
            hidden_dim,
            mlp_hidden_dim,
            num_hidden_layers,
            activation,
        )
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
        global_features = total.log1p()
        if self.include_species_count:
            global_features = torch.cat(
                (global_features, species.float().log1p()), dim=-1
            )
        pooled = pooled + self.global_composition_encoder(global_features)
        return CompositionFeatures(embeddings, tokens, pooled)
