from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanMessageLayer(nn.Module):
    """Small DGL message-passing layer that deliberately ignores edge labels."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.neigh_linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, graph, x: torch.Tensor) -> torch.Tensor:
        import dgl.function as fn

        with graph.local_scope():
            h = self.dropout(x)
            graph.ndata["h"] = h
            graph.update_all(fn.copy_u("h", "m"), fn.mean("m", "neigh"))
            neigh = graph.ndata.get("neigh", torch.zeros_like(h))
            out = self.self_linear(h) + self.neigh_linear(neigh)
            out = self.norm(out)
            return F.relu(out)


class SafeGraphEncoder(nn.Module):
    """Directed graph encoder for the three-expert baseline.

    It uses node features and graph structure only. Edge labels/types are never accepted
    as an argument, which keeps the edge-type expert from reading its target.
    """

    uses_edge_type = False

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        dims: List[int] = [in_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList(
            [MeanMessageLayer(dims[i], dims[i + 1], dropout=dropout) for i in range(num_layers)]
        )
        self.output_dim = hidden_dim

    def forward(self, graph, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(graph, h)
        return h
