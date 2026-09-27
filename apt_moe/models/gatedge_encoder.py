from __future__ import annotations

import torch
import torch.nn as nn

from graphmae.models.gatedge import GATEdge


class ProvFusionGATEdgeEncoder(nn.Module):
    """Small wrapper around ProvFusion's GATEdge encoder path."""

    uses_edge_type = True

    def __init__(
        self,
        in_dim: int,
        edge_in_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.1,
        attn_drop: float = 0.0,
        negative_slope: float = 0.2,
        residual: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        self.in_dim = int(in_dim)
        self.edge_in_dim = int(edge_in_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.encoder = GATEdge(
            in_dim=in_dim,
            edge_in_dim=edge_in_dim,
            num_hidden=hidden_dim // num_heads,
            out_dim=hidden_dim // num_heads,
            num_layers=num_layers,
            nhead=num_heads,
            nhead_out=num_heads,
            activation=nn.PReLU(),
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=nn.LayerNorm,
            concat_out=True,
            encoding=True,
        )

    def forward(self, graph, node_features: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        return self.encoder(graph, node_features.float(), edge_features.float())
