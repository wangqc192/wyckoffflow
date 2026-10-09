"""Composition-conditioned space-group prediction."""

import torch
import torch.nn as nn

from ..common import lookup_tables
from .composition_encoder import CrystalCompositionEncoder
from .mlp import get_mlp


class SpaceGroupSetEncoder(nn.Module):
    """Encode each space group from its Wyckoff positions."""

    def __init__(self, config, output_dim):
        super().__init__()
        max_positions = 27
        multiplicities = torch.zeros(231, max_positions, dtype=torch.long)
        degrees_of_freedom = torch.zeros_like(multiplicities)
        position_indices = torch.zeros_like(multiplicities)
        token_mask = torch.zeros(231, max_positions, dtype=torch.bool)
        for space_group in range(1, 231):
            key = str(space_group)
            multiplicity_values = list(
                reversed(lookup_tables.spg_wyckoff_multiplicities[key].values())
            )
            dof_values = list(
                reversed(lookup_tables.spg_wyckoff_degrees_of_freedom[key].values())
            )
            num_positions = len(multiplicity_values)
            multiplicities[space_group, :num_positions] = torch.tensor(
                multiplicity_values
            )
            degrees_of_freedom[space_group, :num_positions] = torch.tensor(dof_values)
            position_indices[space_group, :num_positions] = torch.arange(num_positions)
            token_mask[space_group, :num_positions] = True

        self.register_buffer("multiplicities", multiplicities, persistent=False)
        self.register_buffer(
            "degrees_of_freedom",
            degrees_of_freedom,
            persistent=False,
        )
        self.register_buffer("position_indices", position_indices, persistent=False)
        self.register_buffer("token_mask", token_mask, persistent=False)
        self.multiplicity_embedding = nn.Embedding(193, output_dim)
        self.dof_embedding = nn.Embedding(4, output_dim)
        self.position_embedding = nn.Embedding(max_positions, output_dim)
        self.token_encoder = get_mlp(
            output_dim,
            output_dim,
            config["hidden_dim"],
            config["mlp_hidden_layers"],
            config["mlp_activation"],
        )
        self.space_group_embedding = nn.Embedding(231, 16)
        self.space_group_projection = nn.Linear(16, output_dim)
        self.output_projection = get_mlp(
            output_dim,
            output_dim,
            config["hidden_dim"],
            config["mlp_hidden_layers"],
            config["mlp_activation"],
        )

    def forward(self):
        tokens = (
            self.multiplicity_embedding(self.multiplicities)
            + self.dof_embedding(self.degrees_of_freedom)
            + self.position_embedding(self.position_indices)
        )
        tokens = self.token_encoder(tokens) * self.token_mask.unsqueeze(-1)
        pooled = tokens.sum(dim=1) / self.token_mask.sum(dim=1, keepdim=True).clamp_min(
            1
        )
        space_group_indices = torch.arange(231, device=tokens.device)
        pooled += self.space_group_projection(
            self.space_group_embedding(space_group_indices)
        )
        return self.output_projection(pooled)


class SpaceGroupHead(nn.Module):
    """Score space groups from already encoded composition features."""

    def __init__(self, config, input_dim, *, composition_encoder=None):
        super().__init__()
        # Register the standalone encoder first to preserve optimizer checkpoints.
        self.composition_encoder = composition_encoder
        self.mlp = get_mlp(
            input_dim,
            231,
            2 * config["hidden_dim"],
            config["mlp_hidden_layers"],
            config["mlp_activation"],
            layer_norm=config.get("layer_norm", False),
            dropout=config.get("dropout", 0.0),
        )

        use_compatibility = config.get(
            "compatibility",
            config.get("spg_compatibility", False),
        )
        if use_compatibility:
            self.compatibility_encoder = SpaceGroupSetEncoder(config, input_dim)
            self.compatibility_mlp = get_mlp(
                3 * input_dim,
                1,
                config["hidden_dim"],
                config["mlp_hidden_layers"],
                config["mlp_activation"],
                layer_norm=config.get("layer_norm", False),
                dropout=config.get("dropout", 0.0),
            )
            nn.init.zeros_(self.compatibility_mlp[-1].weight)
            nn.init.zeros_(self.compatibility_mlp[-1].bias)
        else:
            self.compatibility_encoder = None
            self.compatibility_mlp = None

    def forward(self, composition_features):
        logits = self.mlp(composition_features)
        if self.compatibility_mlp is None:
            return logits

        space_group_features = self.compatibility_encoder().unsqueeze(0)
        composition_features = composition_features.unsqueeze(1).expand(-1, 231, -1)
        space_group_features = space_group_features.expand(
            composition_features.shape[0],
            -1,
            -1,
        )
        interaction = torch.cat(
            (
                composition_features,
                space_group_features,
                composition_features * space_group_features,
            ),
            dim=-1,
        )
        return logits + self.compatibility_mlp(interaction).squeeze(-1)


class SpaceGroupPredictor(SpaceGroupHead):
    """Standalone predictor retaining its own composition encoder."""

    def __init__(self, config):
        num_elements = config["num_elements"]
        encoder_dim = config.get(
            "spg_composition_encoder_dim", config.get("composition_encoder_dim")
        )
        composition_encoder = (
            CrystalCompositionEncoder(
                num_elements,
                encoder_dim,
                encoder_dim,
                mlp_hidden_dim=2 * encoder_dim,
                num_hidden_layers=config["mlp_hidden_layers"],
                activation=config["mlp_activation"],
                include_species_count=False,
            )
            if encoder_dim is not None
            else None
        )
        super().__init__(
            config,
            encoder_dim if encoder_dim is not None else num_elements + 1,
            composition_encoder=composition_encoder,
        )
        self.num_elements = num_elements

    def encode(self, composition):
        if self.composition_encoder is None:
            return composition.float().log1p()
        return self.composition_encoder(composition).pooled

    def forward(self, composition):
        return super().forward(self.encode(composition))
