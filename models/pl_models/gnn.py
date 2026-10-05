"""Graph neural network for Wyckoff-position prediction."""

import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax

from .composition_encoder import _load_legacy_global_composition
from .mlp import get_mlp


class ContinuousTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        frequencies = 2 * torch.pi * 2 ** torch.arange((embedding_dim + 1) // 2)
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, time):
        angles = time.float().unsqueeze(-1) * self.frequencies
        features = torch.cat((angles.sin(), angles.cos()), dim=-1)
        return self.projection(features[..., : self.projection[0].in_features])


class CompositionSetEncoder(nn.Module):
    """Encode element identities and counts as an unordered set."""

    def __init__(
        self, num_elements, output_dim, hidden_dim, num_hidden_layers, activation
    ):
        super().__init__()
        self.num_elements = num_elements
        self.element_embedding = nn.Embedding(num_elements + 1, output_dim)
        self.token_encoder = get_mlp(
            output_dim + 2,
            output_dim,
            hidden_dim,
            num_hidden_layers,
            activation,
        )
        self.global_composition_encoder = get_mlp(
            1,
            output_dim,
            hidden_dim,
            num_hidden_layers,
            activation,
        )
        self._register_load_state_dict_pre_hook(_load_legacy_global_composition)

    def forward(self, composition):
        counts = composition[:, 1:].float()
        present = counts > 0
        total = counts.sum(dim=1, keepdim=True).clamp_min(1)
        elements = torch.arange(
            1, self.num_elements + 1, device=composition.device
        ).expand(composition.shape[0], -1)
        tokens = torch.cat(
            (
                self.element_embedding(elements),
                counts.log1p().unsqueeze(-1),
                (counts / total).unsqueeze(-1),
            ),
            dim=-1,
        )
        encoded = self.token_encoder(tokens) * present.unsqueeze(-1)
        pooled = encoded.sum(dim=1) / present.sum(dim=1, keepdim=True).clamp_min(1)
        return pooled + self.global_composition_encoder(total.log1p())


class CompositionFiLM(nn.Module):
    def __init__(self, condition_dim, hidden_dim):
        super().__init__()
        self.projection = nn.Linear(condition_dim, 2 * hidden_dim)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, condition):
        return self.projection(condition).chunk(2, dim=-1)


class WyckoffGNNLayer(MessagePassing):
    def __init__(
        self,
        node_hidden_dim,
        condition_dim,
        hidden_dim,
        num_hidden_layers,
        activation,
        use_softmax,
    ):
        super().__init__("sum")
        message_dim = node_hidden_dim + condition_dim
        self.a_mlp = get_mlp(
            2 * message_dim,
            1,
            hidden_dim,
            num_hidden_layers,
            activation,
        )
        self.psi_mlp = get_mlp(
            message_dim,
            node_hidden_dim,
            hidden_dim,
            num_hidden_layers,
            activation,
        )
        self.use_softmax = use_softmax

    def message(self, h_i, h_j, index):
        weights = self.a_mlp(torch.cat((h_i, h_j), dim=1))
        if self.use_softmax:
            weights = softmax(weights, index)
        return weights * self.psi_mlp(h_j)

    def forward(self, hidden, condition, edge_index):
        messages = self.propagate(
            edge_index,
            h=torch.cat((hidden, condition), dim=1),
        )
        return hidden + messages


class WyckoffGNN(nn.Module):
    def __init__(
        self,
        num_elements,
        max_num_atoms,
        hidden_dim,
        dof_pos_sg_emb_size,
        num_gnn_layers,
        gnn_activation,
        mlp_hidden_layers,
        mlp_activation,
        composition_encoder_dim=None,
        composition_film=False,
        conditional_composition=False,
        continuous_time=False,
        t_max=None,
        no_multiplicity_encoding=True,
        binary_dof_encoding=True,
        no_softmax=True,
    ):
        super().__init__()
        condition_dim = dof_pos_sg_emb_size
        num_mlp_layers = mlp_hidden_layers
        activation = mlp_activation

        self.num_elements = num_elements
        self.max_atom_num = self.num_elements
        self.max_num_atoms = max_num_atoms
        self.binary_dof_encoding = binary_dof_encoding
        self.inf_dof_embedding = nn.Embedding(self.max_num_atoms + 1, 1)
        self.inf_dof_linear = nn.Linear(self.num_elements, hidden_dim)
        self.zero_dof_embedding = nn.Embedding(self.num_elements + 1, hidden_dim)
        self.dof_embedding = nn.Embedding(
            2 if self.binary_dof_encoding else 4,
            condition_dim,
        )
        self.pos_embedding = nn.Embedding(27, condition_dim)
        self.sg_embedding = nn.Embedding(231, condition_dim)
        self.time_embedding = (
            ContinuousTimeEmbedding(condition_dim)
            if continuous_time
            else nn.Embedding(t_max, condition_dim)
        )
        self.multiplicity_embedding = (
            None if no_multiplicity_encoding else nn.Embedding(193, condition_dim)
        )

        composition_dim = composition_encoder_dim
        if conditional_composition:
            if composition_dim is None:
                self.composition_encoder = get_mlp(
                    self.num_elements + 1,
                    condition_dim,
                    2 * condition_dim,
                    num_mlp_layers,
                    activation,
                )
                self.composition_condition_projection = nn.Identity()
            else:
                self.composition_encoder = CompositionSetEncoder(
                    self.num_elements,
                    composition_dim,
                    2 * composition_dim,
                    num_mlp_layers,
                    activation,
                )
                self.composition_condition_projection = get_mlp(
                    composition_dim,
                    condition_dim,
                    2 * condition_dim,
                    num_mlp_layers,
                    activation,
                )
        else:
            self.composition_encoder = None

        if composition_film and composition_dim is None:
            raise ValueError("composition_film requires a composition set encoder")
        self.layers = nn.ModuleList(
            WyckoffGNNLayer(
                hidden_dim,
                condition_dim,
                2 * (hidden_dim + condition_dim),
                num_mlp_layers,
                activation,
                use_softmax=not no_softmax,
            )
            for _ in range(num_gnn_layers)
        )
        self.composition_film_layers = (
            nn.ModuleList(
                CompositionFiLM(composition_dim, hidden_dim) for _ in self.layers
            )
            if composition_film
            else None
        )
        self.activation = getattr(nn, gnn_activation)()
        self.zero_df_out_mlp = get_mlp(
            hidden_dim,
            self.num_elements + 1,
            2 * hidden_dim,
            num_mlp_layers,
            activation,
        )
        self.inf_df_out_mlp = get_mlp(
            hidden_dim,
            self.num_elements * (self.max_num_atoms + 1),
            2 * hidden_dim,
            num_mlp_layers,
            activation,
        )

    def forward(self, data, time):
        zero_dof = data.zero_dof
        zero_hidden = self.zero_dof_embedding(data.x_0_dof.long())
        inf_hidden = self.inf_dof_linear(
            self.inf_dof_embedding(data.x_inf_dof.long()).squeeze(-1)
        )
        hidden = zero_hidden.new_empty(zero_dof.shape[0], zero_hidden.shape[-1])
        hidden[zero_dof] = zero_hidden
        hidden[~zero_dof] = inf_hidden

        composition = (
            self.composition_encoder(data.composition)
            if self.composition_encoder is not None
            else None
        )
        dof = zero_dof.long() if self.binary_dof_encoding else data.degrees_of_freedom
        condition = (
            self.dof_embedding(dof.long())
            + self.pos_embedding(data.wyckoff_pos_idx.long())
            + self.sg_embedding(data.space_group.long()).repeat_interleave(
                data.num_pos, dim=0
            )
            + self.time_embedding(time).repeat_interleave(data.num_pos, dim=0)
        )
        if self.multiplicity_embedding is not None:
            condition += self.multiplicity_embedding(data.multiplicities.long())
        if composition is not None:
            condition += self.composition_condition_projection(
                composition
            ).repeat_interleave(data.num_pos, dim=0)

        for index, layer in enumerate(self.layers):
            hidden = layer(hidden, condition, data.edge_index)
            if self.composition_film_layers is not None:
                gamma, beta = self.composition_film_layers[index](composition)
                gamma = gamma.repeat_interleave(data.num_pos, dim=0)
                beta = beta.repeat_interleave(data.num_pos, dim=0)
                hidden = (1 + gamma) * hidden + beta
            hidden = self.activation(hidden)

        zero_logits = self.zero_df_out_mlp(hidden[zero_dof])
        inf_logits = self.inf_df_out_mlp(hidden[~zero_dof]).reshape(
            -1, self.num_elements, self.max_num_atoms + 1
        )
        return zero_logits, inf_logits
