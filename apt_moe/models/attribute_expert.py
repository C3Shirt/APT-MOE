from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_encoder import SafeGraphEncoder


class AttributeExpert(nn.Module):
    def __init__(
        self,
        node_type_dim: int,
        attr_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        lambda_cos: float = 0.5,
        lambda_mse: float = 0.5,
    ) -> None:
        super().__init__()
        if attr_dim <= 0:
            raise ValueError("AttributeExpert requires attr_dim > 0.")
        self.node_type_dim = node_type_dim
        self.attr_dim = attr_dim
        self.lambda_cos = lambda_cos
        self.lambda_mse = lambda_mse
        self.encoder = SafeGraphEncoder(node_type_dim + attr_dim, hidden_dim, num_layers, dropout)
        self.mask_token = nn.Parameter(torch.zeros(1, attr_dim))
        self.decoder = nn.Linear(hidden_dim, attr_dim)

    def masked_input(self, node_type_one_hot: torch.Tensor, attrs: torch.Tensor, mask_nodes: torch.Tensor) -> torch.Tensor:
        masked_attrs = attrs.clone()
        masked_attrs[mask_nodes] = 0.0
        masked_attrs[mask_nodes] = masked_attrs[mask_nodes] + self.mask_token
        return torch.cat([node_type_one_hot.float(), masked_attrs.float()], dim=1)

    def forward(
        self,
        graph,
        node_type_one_hot: torch.Tensor,
        attrs: torch.Tensor,
        mask_nodes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask_nodes is None:
            x = torch.cat([node_type_one_hot.float(), attrs.float()], dim=1)
        else:
            x = self.masked_input(node_type_one_hot, attrs, mask_nodes)
        h = self.encoder(graph, x)
        return self.decoder(h)

    def per_node_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cos = 1.0 - F.cosine_similarity(pred, target, dim=-1, eps=1e-8)
        mse = F.mse_loss(pred, target, reduction="none").mean(dim=-1)
        return self.lambda_cos * cos + self.lambda_mse * mse

    def loss(
        self,
        graph,
        node_type_one_hot: torch.Tensor,
        attrs: torch.Tensor,
        mask_nodes: torch.Tensor,
    ) -> torch.Tensor:
        pred = self.forward(graph, node_type_one_hot, attrs, mask_nodes)
        return self.per_node_loss(pred[mask_nodes], attrs[mask_nodes]).mean()

    def node_scores(
        self,
        graph,
        node_type_one_hot: torch.Tensor,
        attrs: torch.Tensor,
        mask_folds: int = 8,
    ) -> torch.Tensor:
        device = attrs.device
        num_nodes = graph.num_nodes()
        scores = torch.empty(num_nodes, device=device)
        folds = torch.arange(num_nodes, device=device).chunk(max(1, min(mask_folds, num_nodes)))
        for mask_nodes in folds:
            pred = self.forward(graph, node_type_one_hot, attrs, mask_nodes)
            scores[mask_nodes] = self.per_node_loss(pred[mask_nodes], attrs[mask_nodes])
        return scores
