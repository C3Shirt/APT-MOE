from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ExpertOutput
from .gatedge_encoder import ProvFusionGATEdgeEncoder


class GraphExpert(nn.Module):
    """ProvFusion-compatible structure expert over node type, topology, and edge type."""

    def __init__(
        self,
        node_type_dim: int,
        edge_type_dim: int,
        hidden_dim: int,
        expert_dim: int,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.node_type_dim = int(node_type_dim)
        self.edge_type_dim = int(edge_type_dim)
        self.hidden_dim = int(hidden_dim)
        self.expert_dim = int(expert_dim)
        self.encoder = ProvFusionGATEdgeEncoder(
            in_dim=node_type_dim,
            edge_in_dim=edge_type_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.mask_token = nn.Parameter(torch.zeros(1, node_type_dim))
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_type_dim),
        )
        self.residual_projector = nn.Linear(node_type_dim, expert_dim)

    def masked_input(self, node_type_one_hot: torch.Tensor, mask_nodes: torch.Tensor) -> torch.Tensor:
        x = node_type_one_hot.clone()
        x[mask_nodes] = 0.0
        x[mask_nodes] = x[mask_nodes] + self.mask_token
        return x

    def forward(
        self,
        graph,
        node_type_one_hot: torch.Tensor,
        edge_features: torch.Tensor,
        mask_nodes: Optional[torch.Tensor] = None,
    ) -> ExpertOutput:
        x = node_type_one_hot.float()
        if mask_nodes is not None:
            x = self.masked_input(x, mask_nodes)
        hidden = self.encoder(graph, x, edge_features)
        logits = self.decoder(hidden)
        probs = torch.softmax(logits, dim=-1)
        raw_residual = node_type_one_hot.float() - probs
        target = node_type_one_hot.argmax(dim=1).long()
        score = F.cross_entropy(logits, target, reduction="none")
        residual = self.residual_projector(raw_residual)
        if mask_nodes is None:
            loss = score.mean()
        else:
            loss = score[mask_nodes].mean()
        return ExpertOutput(
            residual=residual,
            score=score,
            loss=loss,
            aux={"logits": logits, "raw_residual": raw_residual, "hidden": hidden, "per_node_loss": score},
        )

    def node_scores(self, graph, node_type_one_hot: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        return self.forward(graph, node_type_one_hot, edge_features).score
