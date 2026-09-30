"""Composition-aware graph attention for Wyckoff template flow matching.

Nodes are Wyckoff orbit *types*, not atoms with coordinates. Multiplicities
and composition therefore use the conventional cell, as in preprocessing.
Only the current noisy occupations and the supplied conditions are read.
"""

import json
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import scatter, softmax

from ..common.lookup_tables import wyckoff_label_to_index
from .mlp import get_mlp


def _signed_log1p(value):
    return value.sign() * value.abs().log1p()


class CompositionFeatures(NamedTuple):
    element_embeddings: torch.Tensor
    element_tokens: torch.Tensor
    pooled: torch.Tensor


class CrystalCompositionEncoder(nn.Module):
    """Static chemical features shared by space-group and occupation prediction."""

    def __init__(self, num_elements, element_dim, hidden_dim):
        super().__init__()
        self.element_embedding = nn.Embedding(num_elements + 1, element_dim)
        self.token_encoder = get_mlp(
            element_dim + 2, hidden_dim, hidden_dim, dropout=0.0
        )
        self.total_encoder = get_mlp(2, hidden_dim, hidden_dim, dropout=0.0)

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
        pooled = pooled + self.total_encoder(
            torch.cat((total.log1p(), species.float().log1p()), dim=-1)
        )
        return CompositionFeatures(embeddings, tokens, pooled)


def _symmetry_table():
    """Load the Wren descriptors already shipped with the dataset dependency."""
    from aviary import PKG_DIR

    path = Path(PKG_DIR) / "embeddings/wyckoff/bra-alg-off.json"
    descriptors = json.loads(path.read_text())
    width = len(descriptors["1"]["a"])
    table = torch.zeros(231, 27, width)
    for group in range(1, 231):
        for letter, features in descriptors[str(group)].items():
            table[group, wyckoff_label_to_index[letter] - 1] = torch.tensor(features)
    return table


def occupation_counts(data, num_elements):
    """Orbit counts per node/element and multiplicity-weighted atom totals.

    Read the branch states rather than ``data.x``: callers can update the flow
    state without refreshing that convenience matrix.
    """
    counts = data.x_inf_dof.new_zeros((data.zero_dof.numel(), num_elements)).float()
    counts[data.zero_dof] = F.one_hot(
        data.x_0_dof.long(), num_classes=num_elements + 1
    )[:, 1:].float()
    counts[~data.zero_dof] = data.x_inf_dof.float()
    allocated = scatter(
        counts * data.multiplicities.float().unsqueeze(-1),
        data.batch,
        dim=0,
        dim_size=data.composition.shape[0],
        reduce="sum",
    )
    return counts, allocated


class FlowTimeEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.register_buffer("frequencies", torch.pi * torch.logspace(0, 2, 32))
        self.projection = get_mlp(66, hidden_dim, hidden_dim, dropout=0.0)

    def forward(self, time):
        time = time.float().reshape(-1, 1)
        angles = time * self.frequencies
        # Raw time distinguishes the endpoints even for periodic features.
        return self.projection(
            torch.cat((time, 1 - time, angles.sin(), angles.cos()), dim=-1)
        )


class CrystalAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout, edge_bias):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.condition_affine = nn.Linear(hidden_dim, 4 * hidden_dim)
        nn.init.zeros_(self.condition_affine.weight)
        nn.init.zeros_(self.condition_affine.bias)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.edge_bias = (
            get_mlp(7, num_heads, num_heads * 4, dropout=0.0) if edge_bias else None
        )
        self.attention_out = nn.Linear(hidden_dim, hidden_dim)
        self.ffn = get_mlp(hidden_dim, hidden_dim, 4 * hidden_dim, dropout=dropout)
        self.residual_scale = nn.Parameter(torch.full((2, hidden_dim), 0.1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, condition, edge_index, edge_features):
        a_scale, a_shift, f_scale, f_shift = self.condition_affine(condition).chunk(
            4, dim=-1
        )
        attention_input = self.attention_norm(hidden) * (1 + a_scale) + a_shift
        query, key, value = (
            self.qkv(attention_input)
            .reshape(-1, 3, self.num_heads, self.head_dim)
            .unbind(dim=1)
        )
        source, target = edge_index
        # Accumulate attention scores in float32 under mixed precision.
        score = (query[target].float() * key[source].float()).sum(dim=-1)
        score = score * self.head_dim**-0.5
        if self.edge_bias is not None:
            score = score + self.edge_bias(edge_features).float()
        weights = softmax(score, target, num_nodes=hidden.shape[0]).to(value.dtype)
        messages = scatter(
            weights.unsqueeze(-1) * value[source],
            target,
            dim=0,
            dim_size=hidden.shape[0],
            reduce="sum",
        ).flatten(1)
        hidden = hidden + self.residual_scale[0] * self.dropout(
            self.attention_out(messages)
        )
        ffn_input = self.ffn_norm(hidden) * (1 + f_scale) + f_shift
        return hidden + self.residual_scale[1] * self.dropout(self.ffn(ffn_input))


class CrystalGNN(nn.Module):
    """Drop-in decoder with shared element/count heads and explicit atom budgets.

    Composition residuals are soft features, not hard constraints on the noisy
    state: the flow must be able to remove incorrect occupied sites. Element
    masking and exact final composition decoding remain the caller's job.
    """

    def __init__(
        self,
        num_elements,
        max_num_atoms,
        hidden_dim=256,
        element_dim=128,
        num_gnn_layers=4,
        num_heads=8,
        dropout=0.1,
        use_symmetry_features=True,
        use_composition_residual=True,
        use_edge_bias=True,
        conditional_composition=True,
        continuous_time=True,
        external_composition=False,
        predict_all_elements=False,
    ):
        super().__init__()
        if not conditional_composition or not continuous_time:
            raise ValueError(
                "CrystalGNN requires composition-conditioned continuous flow"
            )
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_elements = num_elements
        self.max_num_atoms = max_num_atoms
        self.hidden_dim = hidden_dim
        self.element_dim = element_dim
        self.external_composition = external_composition
        self.predict_all_elements = predict_all_elements
        self.use_composition_residual = use_composition_residual
        self.element_embedding = (
            None
            if external_composition
            else nn.Embedding(num_elements + 1, element_dim)
        )
        self.count_embedding = nn.Embedding(max_num_atoms + 1, element_dim)
        self.state_encoder = get_mlp(
            2 * element_dim + 2, hidden_dim, hidden_dim, dropout=0.0
        )
        self.empty_state = nn.Parameter(torch.zeros(hidden_dim))
        self.budget_encoder = get_mlp(
            3 if external_composition else element_dim + 5,
            hidden_dim,
            hidden_dim,
            dropout=0.0,
        )
        self.total_encoder = get_mlp(
            1 if external_composition else 3, hidden_dim, hidden_dim, dropout=0.0
        )
        self.sg_embedding = nn.Embedding(231, hidden_dim)
        self.position_embedding = nn.Embedding(27, hidden_dim)
        self.dof_embedding = nn.Embedding(4, hidden_dim)
        self.multiplicity_embedding = nn.Embedding(193, hidden_dim)
        self.time_encoder = FlowTimeEncoder(hidden_dim)
        if use_symmetry_features:
            table = _symmetry_table()
            # Save descriptors in checkpoints so their values are reproducible.
            self.register_buffer("symmetry_features", table)
            self.symmetry_encoder = get_mlp(
                table.shape[-1], hidden_dim, hidden_dim, dropout=0.0
            )
        else:
            self.register_buffer("symmetry_features", None)
            self.symmetry_encoder = None
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList(
            CrystalAttentionLayer(hidden_dim, num_heads, dropout, use_edge_bias)
            for _ in range(num_gnn_layers)
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.pair_encoder = get_mlp(5, hidden_dim, hidden_dim, dropout=0.0)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.empty_head = get_mlp(hidden_dim, 1, hidden_dim, dropout=dropout)
        self.element_head = get_mlp(hidden_dim, 1, hidden_dim, dropout=dropout)
        self.count_head = get_mlp(
            hidden_dim, max_num_atoms + 1, hidden_dim, dropout=dropout
        )

    def _encode_state(self, data, counts, element_embeddings):
        nodes, elements = counts.nonzero(as_tuple=True)
        occupied = counts[nodes, elements]
        atoms = occupied * data.multiplicities[nodes]
        tokens = self.state_encoder(
            torch.cat(
                (
                    F.embedding(elements + 1, element_embeddings),
                    self.count_embedding(occupied.long()),
                    occupied.log1p().unsqueeze(-1),
                    atoms.log1p().unsqueeze(-1),
                ),
                dim=-1,
            )
        )
        hidden = scatter(tokens, nodes, dim=0, dim_size=counts.shape[0], reduce="sum")
        species = (counts > 0).sum(dim=-1, keepdim=True).clamp_min(1)
        return hidden / species.sqrt() + self.empty_state

    def _encode_budget(self, target, allocated, composition_features=None):
        present = target > 0
        total = target.sum(dim=-1, keepdim=True).clamp_min(1)
        residual = target - allocated
        dynamic = torch.stack(
            (
                allocated.log1p(),
                _signed_log1p(residual),
                residual / target.clamp_min(1),
            ),
            dim=-1,
        )
        if not self.use_composition_residual:
            dynamic = torch.zeros_like(dynamic)
        current_total = allocated.sum(dim=-1, keepdim=True).log1p()
        if not self.use_composition_residual:
            current_total = torch.zeros_like(current_total)
        if self.external_composition:
            # Only the residual branch reads the current noisy occupations.
            residual_tokens = self.budget_encoder(dynamic)
            pooled = (residual_tokens * present.unsqueeze(-1)).sum(dim=1)
            pooled = pooled / present.sum(dim=-1, keepdim=True).clamp_min(1)
            return (
                composition_features.element_tokens + residual_tokens,
                composition_features.pooled
                + pooled
                + self.total_encoder(current_total),
            )
        features = torch.cat(
            (target.log1p().unsqueeze(-1), (target / total).unsqueeze(-1), dynamic),
            dim=-1,
        )
        elements = (
            self.element_embedding.weight[1:]
            .unsqueeze(0)
            .expand(target.shape[0], -1, -1)
        )
        tokens = self.budget_encoder(torch.cat((elements, features), dim=-1))
        pooled = (tokens * present.unsqueeze(-1)).sum(dim=1)
        pooled = pooled / present.sum(dim=-1, keepdim=True).clamp_min(1)
        totals = torch.cat(
            (
                total.log1p(),
                present.sum(dim=-1, keepdim=True).float().log1p(),
                current_total,
            ),
            dim=-1,
        )
        return tokens, pooled + self.total_encoder(totals)

    @staticmethod
    def _edge_features(data):
        source, target = data.edge_index
        multiplicity = data.multiplicities.float()
        source_m, target_m = multiplicity[source], multiplicity[target]
        gcd = torch.gcd(source_m.long(), target_m.long()).float()
        dof = data.degrees_of_freedom.float() / 3
        return torch.stack(
            (
                (source_m / target_m).log(),
                gcd / source_m,
                gcd / target_m,
                (source_m == target_m).float(),
                dof[source],
                dof[target],
                (source == target).float(),
            ),
            dim=-1,
        )

    def forward(self, data, time, composition_features=None):
        counts, allocated = occupation_counts(data, self.num_elements)
        target = data.composition[:, 1:].float()
        element_tokens, composition = self._encode_budget(
            target, allocated, composition_features
        )
        element_embeddings = (
            composition_features.element_embeddings
            if self.external_composition
            else self.element_embedding.weight
        )
        group = data.space_group.long().reshape(-1)[data.batch]
        position = data.wyckoff_pos_idx.long()
        condition = (
            self.sg_embedding(group)
            + self.position_embedding(position)
            + self.dof_embedding(data.degrees_of_freedom.long())
            + self.multiplicity_embedding(data.multiplicities.long())
            + (composition + self.time_encoder(time))[data.batch]
        )
        if self.symmetry_encoder is not None:
            condition = condition + self.symmetry_encoder(
                self.symmetry_features[group, position]
            )
        hidden = self.input_norm(
            self._encode_state(data, counts, element_embeddings) + condition
        )
        edge_features = self._edge_features(data)
        for layer in self.layers:
            hidden = layer(hidden, condition, data.edge_index, edge_features)
        hidden = self.output_norm(hidden)

        # Unmasked training needs learned predictions for absent elements too.
        # Otherwise their channels remain placeholders for the caller to mask.
        if self.predict_all_elements:
            nodes = torch.arange(
                hidden.shape[0], device=hidden.device
            ).repeat_interleave(self.num_elements)
            elements = torch.arange(self.num_elements, device=hidden.device).repeat(
                hidden.shape[0]
            )
        else:
            nodes, elements = (target[data.batch] > 0).nonzero(as_tuple=True)
        graphs = data.batch[nodes]
        multiplicity = data.multiplicities[nodes].float()
        atom_target = target[graphs, elements]
        current = counts[nodes, elements]
        residual = atom_target - allocated[graphs, elements]
        if not self.use_composition_residual:
            residual = torch.zeros_like(residual)
        pair_features = torch.stack(
            (
                current.log1p(),
                current * multiplicity / atom_target.clamp_min(1),
                (atom_target / multiplicity).log1p(),
                torch.remainder(atom_target, multiplicity) / multiplicity,
                _signed_log1p(residual / multiplicity),
            ),
            dim=-1,
        )
        query = self.query_norm(
            hidden[nodes]
            + element_tokens[graphs, elements]
            + self.pair_encoder(pair_features)
        )
        fixed = data.zero_dof[nodes]
        # Construct both outputs in the branch ordering expected by the flow.
        zero_logits = hidden.new_zeros((hidden.shape[0], self.num_elements + 1))
        zero_logits[:, 0] = self.empty_head(hidden).squeeze(-1)
        zero_logits[nodes[fixed], elements[fixed] + 1] = (
            self.element_head(query[fixed]).squeeze(-1).to(hidden.dtype)
        )
        inf_logits = hidden.new_zeros(
            (data.x_inf_dof.shape[0], self.num_elements, self.max_num_atoms + 1)
        )
        inf_index = (~data.zero_dof).long().cumsum(dim=0) - 1
        inf_logits[inf_index[nodes[~fixed]], elements[~fixed]] = self.count_head(
            query[~fixed]
        ).to(hidden.dtype)
        return zero_logits[data.zero_dof], inf_logits
