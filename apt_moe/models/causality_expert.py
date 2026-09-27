from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ExpertOutput
from .gatedge_encoder import ProvFusionGATEdgeEncoder


class CausalityExpert(nn.Module):
    """ProvFusion-compatible relation expert with node-aligned max incident-edge residuals."""

    def __init__(
        self,
        node_feat_dim: int,
        edge_type_dim: int,
        hidden_dim: int,
        expert_dim: int,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.1,
        multilabel: bool = True,
        smoothmax_tau: float = 5.0,
    ) -> None:
        super().__init__()
        self.node_feat_dim = int(node_feat_dim)
        self.edge_type_dim = int(edge_type_dim)
        self.hidden_dim = int(hidden_dim)
        self.expert_dim = int(expert_dim)
        self.multilabel = bool(multilabel)
        self.smoothmax_tau = max(float(smoothmax_tau), 1e-6)
        self.encoder = ProvFusionGATEdgeEncoder(
            in_dim=node_feat_dim,
            edge_in_dim=edge_type_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, edge_type_dim),
    )
        self.residual_projector = nn.Linear(edge_type_dim, expert_dim)

    def forward(
        self,
        graph,
        node_features: torch.Tensor,
        edge_targets: torch.Tensor,
        edge_features: torch.Tensor,
        edge_indices: Optional[torch.Tensor] = None,
    ) -> ExpertOutput:
        src, dst = graph.edges()
        all_edges = torch.stack([src, dst], dim=1).to(node_features.device)
        if edge_indices is None:
            edge_indices = torch.arange(all_edges.shape[0], device=node_features.device)
        edge_indices = edge_indices.to(node_features.device)
        hidden = self.encoder(graph, node_features.float(), edge_features.float())
        selected_edges = all_edges[edge_indices]
        selected_targets = edge_targets.to(node_features.device)[edge_indices]

        src_h = hidden[selected_edges[:, 0]]
        dst_h = hidden[selected_edges[:, 1]]
        logits = self.decoder(torch.cat([src_h, dst_h], dim=-1))
        edge_loss = self.per_edge_loss(logits, selected_targets)
        edge_residual = self.edge_residual(logits, selected_targets)
        projected_edge_residual = self.residual_projector(edge_residual)
        residual, score, exact_max = self._incident_node_energies(
            num_nodes=graph.num_nodes(),
            edges=selected_edges,
            edge_residual=projected_edge_residual,
            edge_loss=edge_loss,
            device=node_features.device,
            smoothmax_tau=self.smoothmax_tau,
        )
        loss = edge_loss.mean() if edge_loss.numel() else node_features.sum() * 0.0
        return ExpertOutput(
            residual=residual,
            score=score,
            loss=loss,
            aux={
                "edge_logits": logits,
                "edge_loss": edge_loss,
                "edge_residual": edge_residual,
                "edge_indices": edge_indices,
                "hidden": hidden,
                "exact_max_incident_edge_loss": exact_max,
            },
        )

    def per_edge_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.multilabel:
            return F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none").mean(dim=-1)
        return F.cross_entropy(logits, targets.long(), reduction="none")

    def edge_residual(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.multilabel:
            return targets.float() - torch.sigmoid(logits)
        one_hot = F.one_hot(targets.long(), num_classes=self.edge_type_dim).float()
        return one_hot - torch.softmax(logits, dim=-1)

    def _incident_node_energies(
        self,
        num_nodes: int,
        edges: torch.Tensor,
        edge_residual: torch.Tensor,
        edge_loss: torch.Tensor,
        device: torch.device,
        smoothmax_tau: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = torch.zeros(num_nodes, self.expert_dim, device=device)
        smooth_score = torch.zeros(num_nodes, device=device)
        exact_max = torch.zeros(num_nodes, device=device)
        if edge_loss.numel() == 0:
            return residual, smooth_score, exact_max

        incident_nodes = torch.cat([edges[:, 0], edges[:, 1]])
        incident_losses = torch.cat([edge_loss, edge_loss])
        edge_positions = torch.arange(edge_loss.shape[0], device=device).repeat(2)
        maximum = torch.full((num_nodes,), float("-inf"), device=device)

        if hasattr(torch.Tensor, "scatter_reduce_"):
            maximum.scatter_reduce_(
                0, incident_nodes, incident_losses.detach(), reduce="amax", include_self=True
            )
            stable_sum = torch.zeros(num_nodes, device=device)
            stable_sum.scatter_add_(
                0,
                incident_nodes,
                torch.exp(float(smoothmax_tau) * (incident_losses - maximum[incident_nodes])),
            )
            valid = stable_sum > 0
            smooth_score[valid] = maximum[valid] + torch.log(stable_sum[valid]) / float(smoothmax_tau)

            candidates = torch.where(
                incident_losses.detach() == maximum[incident_nodes],
                edge_positions,
                torch.full_like(edge_positions, edge_loss.shape[0]),
            )
            best_edge = torch.full((num_nodes,), edge_loss.shape[0], dtype=torch.long, device=device)
            best_edge.scatter_reduce_(0, incident_nodes, candidates, reduce="amin", include_self=True)
            exact_max[valid] = maximum[valid]
            residual[valid] = edge_residual[best_edge[valid]]
            return residual, smooth_score, exact_max

        for node_idx in range(num_nodes):
            incident = (edges[:, 0] == node_idx) | (edges[:, 1] == node_idx)
            values = edge_loss[incident]
            if values.numel():
                smooth_score[node_idx] = torch.logsumexp(float(smoothmax_tau) * values, dim=0) / float(smoothmax_tau)
                best_pos = torch.argmax(values)
                exact_max[node_idx] = values[best_pos].detach()
                residual[node_idx] = edge_residual[torch.nonzero(incident, as_tuple=False).view(-1)[best_pos]]
        return residual, smooth_score, exact_max

    def _max_incident_node_residual(
        self,
        num_nodes: int,
        edges: torch.Tensor,
        edge_residual: torch.Tensor,
        edge_loss: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the exact max-loss incident residual for diagnostics and compatibility."""
        residual = torch.zeros(num_nodes, self.expert_dim, device=device)
        score = torch.zeros(num_nodes, device=device)
        for node_idx in range(num_nodes):
            incident = (edges[:, 0] == node_idx) | (edges[:, 1] == node_idx)
            incident_positions = torch.nonzero(incident, as_tuple=False).view(-1)
            if incident_positions.numel() == 0:
                continue
            best_position = incident_positions[torch.argmax(edge_loss[incident_positions])]
            score[node_idx] = edge_loss[best_position].detach()
            residual[node_idx] = edge_residual[best_position]
        return residual, score

    def node_scores(
        self,
        graph,
        node_features: torch.Tensor,
        edge_targets: torch.Tensor,
        edge_features: torch.Tensor,
        edge_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward(graph, node_features, edge_targets, edge_features, edge_indices).score
