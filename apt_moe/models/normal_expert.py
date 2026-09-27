from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ExpertOutput


class NormalExpert(nn.Module):
    """Graph-independent global normality expert using a training-set prototype."""

    uses_gnn = False

    def __init__(self, input_dim: int, hidden_dim: int, expert_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.expert_dim = int(expert_dim)
        self.register_buffer("prototype", torch.zeros(1, input_dim))
        self.register_buffer("prototype_initialized", torch.tensor(False))
        self.prototype_decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )
        self.residual_projector = nn.Linear(input_dim, expert_dim)

    def set_prototype(self, prototype: torch.Tensor) -> None:
        proto = prototype.detach().float().view(1, -1)
        if proto.shape[1] != self.input_dim:
            raise ValueError(f"Expected prototype dim {self.input_dim}, got {proto.shape[1]}.")
        self.prototype.copy_(proto.to(self.prototype.device))
        self.prototype_initialized.fill_(True)

    def forward(self, node_features: torch.Tensor, target_mask: Optional[torch.Tensor] = None) -> ExpertOutput:
        if bool(self.prototype_initialized):
            prototype = self.prototype
        else:
            prototype = node_features.float().mean(dim=0, keepdim=True).detach()
        decoded = self.prototype_decoder(prototype).expand_as(node_features.float())
        raw_residual = node_features.float() - decoded
        score = raw_residual.square().mean(dim=1)
        residual = self.residual_projector(raw_residual)
        if target_mask is None:
            target_mask = torch.ones(node_features.shape[0], dtype=torch.bool, device=node_features.device)
        loss = F.mse_loss(decoded[target_mask], node_features.float()[target_mask])
        return ExpertOutput(
            residual=residual,
            score=score,
            loss=loss,
            aux={"prototype": prototype, "normal_reconstruction": decoded},
        )

    def node_scores(self, node_features: torch.Tensor) -> torch.Tensor:
        return self.forward(node_features).score
