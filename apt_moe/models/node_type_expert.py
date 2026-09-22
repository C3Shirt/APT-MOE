from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_encoder import SafeGraphEncoder


class NodeTypeExpert(nn.Module):
    def __init__(
        self,
        num_node_types: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_node_types = num_node_types
        self.encoder = SafeGraphEncoder(num_node_types, hidden_dim, num_layers, dropout)
        self.mask_token = nn.Parameter(torch.zeros(1, num_node_types))
        self.classifier = nn.Linear(hidden_dim, num_node_types)

    def masked_input(self, node_type_one_hot: torch.Tensor, mask_nodes: torch.Tensor) -> torch.Tensor:
        x = node_type_one_hot.clone()
        x[mask_nodes] = 0.0
        x[mask_nodes] = x[mask_nodes] + self.mask_token
        return x

    def forward(self, graph, node_type_one_hot: torch.Tensor, mask_nodes: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask_nodes is not None:
            node_type_one_hot = self.masked_input(node_type_one_hot, mask_nodes)
        h = self.encoder(graph, node_type_one_hot.float())
        return self.classifier(h)

    def loss(self, graph, node_type_one_hot: torch.Tensor, target: torch.Tensor, mask_nodes: torch.Tensor) -> torch.Tensor:
        logits = self.forward(graph, node_type_one_hot, mask_nodes)
        return F.cross_entropy(logits[mask_nodes], target[mask_nodes], reduction="mean")

    def node_scores(self, graph, node_type_one_hot: torch.Tensor, target: torch.Tensor, mask_folds: int = 8) -> torch.Tensor:
        device = node_type_one_hot.device
        num_nodes = graph.num_nodes()
        scores = torch.empty(num_nodes, device=device)
        folds = torch.arange(num_nodes, device=device).chunk(max(1, min(mask_folds, num_nodes)))
        for mask_nodes in folds:
            logits = self.forward(graph, node_type_one_hot, mask_nodes)
            log_probs = F.log_softmax(logits[mask_nodes], dim=-1)
            scores[mask_nodes] = -log_probs.gather(1, target[mask_nodes].view(-1, 1)).squeeze(1)
        return scores
