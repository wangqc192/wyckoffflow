"""Conditioned graph attention and feed-forward residual block."""

import torch
from torch import nn

from .attention import GraphAttention
from .mlp import get_mlp


def modulate(hidden, scale, shift):
    return hidden * (1 + scale) + shift


def _load_legacy_gnn_block(state_dict, prefix, *args):
    """Translate parameter paths from blocks saved before module extraction."""
    for old, new in (
        ("qkv", "attention.qkv"),
        ("edge_bias", "attention.edge_bias"),
        ("attention_out", "attention.attention_out"),
        ("condition_affine", "adaLN_modulation.1"),
    ):
        old_prefix = f"{prefix}{old}."
        for key in list(state_dict):
            if key.startswith(old_prefix):
                state_dict[f"{prefix}{new}.{key[len(old_prefix):]}"] = state_dict.pop(
                    key
                )


class GnnBlock(nn.Module):
    """Dropout on FFN hidden activations and both residual branch outputs.

    Attention weights and conditioning features are not dropped.
    """

    def __init__(
        self,
        hidden_dim,
        num_heads,
        dropout,
        edge_bias,
        use_residual_scale=True,
        use_condition_silu=False,
    ):
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU() if use_condition_silu else nn.Identity(),
            nn.Linear(hidden_dim, 4 * hidden_dim, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        self.attention = GraphAttention(hidden_dim, num_heads, edge_bias)
        self.ffn = get_mlp(hidden_dim, hidden_dim, 4 * hidden_dim, dropout=dropout)
        self.residual_scale = (
            nn.Parameter(torch.full((2, hidden_dim), 0.1))
            if use_residual_scale
            else None
        )
        self.attention_residual_dropout = nn.Dropout(dropout)
        self.ffn_residual_dropout = nn.Dropout(dropout)
        self._register_load_state_dict_pre_hook(_load_legacy_gnn_block)

    def forward(self, hidden, condition, edge_index, edge_features):
        residual_scale = (
            self.residual_scale if self.residual_scale is not None else (1.0, 1.0)
        )
        a_scale, a_shift, f_scale, f_shift = self.adaLN_modulation(condition).chunk(
            4, dim=-1
        )
        attention_input = modulate(self.attention_norm(hidden), a_scale, a_shift)
        hidden = hidden + residual_scale[0] * self.attention_residual_dropout(
            self.attention(attention_input, edge_index, edge_features)
        )
        ffn_input = modulate(self.ffn_norm(hidden), f_scale, f_shift)
        return hidden + residual_scale[1] * self.ffn_residual_dropout(
            self.ffn(ffn_input)
        )
