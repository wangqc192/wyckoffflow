"""Sparse scaled dot-product attention over graph edges."""

from torch import nn
from torch_geometric.utils import scatter, softmax

from .mlp import get_mlp


class GraphAttention(nn.Module):
    """Multi-head graph attention with learned edge biases and projections.

    Input and output have shape (nodes, hidden_dim). Edge features have shape
    (edges, 7). Conditioning, normalization and dropout belong to the block.
    """

    def __init__(self, hidden_dim, num_heads, edge_bias):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.edge_bias = get_mlp(7, num_heads, num_heads * 4) if edge_bias else None
        self.attention_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden, edge_index, edge_features):
        query, key, value = (
            self.qkv(hidden).reshape(-1, 3, self.num_heads, self.head_dim).unbind(dim=1)
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
        return self.attention_out(messages)
