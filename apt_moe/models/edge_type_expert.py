from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_encoder import SafeGraphEncoder


class EdgeTypeExpert(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        edge_type_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        multilabel: bool = False,
    ) -> None:
        super().__init__()
        self.edge_type_dim = edge_type_dim
        self.multilabel = multilabel
        self.encoder = SafeGraphEncoder(node_feat_dim, hidden_dim, num_layers, dropout)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, edge_type_dim),
        )

    @property
    def uses_edge_type(self) -> bool:
        return False

    def forward(self, graph, node_features: torch.Tensor, edges: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.encoder(graph, node_features.float())
        if edges is None:
            src, dst = graph.edges()
            edges = torch.stack([src, dst], dim=1)
        src_h = h[edges[:, 0]]
        dst_h = h[edges[:, 1]]
        return self.decoder(torch.cat([src_h, dst_h], dim=-1))

    def per_edge_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.multilabel:
            return F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none").mean(dim=-1)
        return F.cross_entropy(logits, targets.long(), reduction="none")

    def loss(self, graph, node_features: torch.Tensor, edges: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = self.forward(graph, node_features, edges)
        return self.per_edge_loss(logits, targets).mean()

    def edge_scores(self, graph, node_features: torch.Tensor, edges: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = self.forward(graph, node_features, edges)
        return self.per_edge_loss(logits, targets)
