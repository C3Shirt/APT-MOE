from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ExpertOutput


class SemanticExpert(nn.Module):
    """Graph-independent semantic autoencoder over the 128D node embedding."""

    uses_gnn = False

    def __init__(self, semantic_dim: int, hidden_dim: int, expert_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        if semantic_dim <= 0:
            raise ValueError("SemanticExpert requires semantic_dim > 0.")
        self.semantic_dim = int(semantic_dim)
        self.hidden_dim = int(hidden_dim)
        self.expert_dim = int(expert_dim)
        self.encoder = nn.Sequential(
            nn.Linear(semantic_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, semantic_dim),
        )
        self.residual_projector = nn.Linear(semantic_dim, expert_dim)

    def forward(self, semantic_features: torch.Tensor, target_mask: Optional[torch.Tensor] = None) -> ExpertOutput:
        hidden = self.encoder(semantic_features.float())
        reconstruction = self.decoder(hidden)
        raw_residual = semantic_features.float() - reconstruction
        score = raw_residual.square().mean(dim=1)
        residual = self.residual_projector(raw_residual)
        if target_mask is None:
            target_mask = torch.ones(semantic_features.shape[0], dtype=torch.bool, device=semantic_features.device)
        loss = F.mse_loss(reconstruction[target_mask], semantic_features.float()[target_mask])
        return ExpertOutput(
            residual=residual,
            score=score,
            loss=loss,
            aux={"reconstruction": reconstruction, "raw_residual": raw_residual},
        )

    def node_scores(self, semantic_features: torch.Tensor) -> torch.Tensor:
        return self.forward(semantic_features).score
