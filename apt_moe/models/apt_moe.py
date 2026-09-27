from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

from .causality_expert import CausalityExpert
from .common import ExpertOutput
from .graph_expert import GraphExpert
from .normal_expert import NormalExpert
from .semantic_expert import SemanticExpert


EXPERT_NAMES = ("semantic", "graph", "normal", "causal")


@dataclass
class APTMoEOutput:
    score: torch.Tensor
    fused_energy: torch.Tensor
    weights: torch.Tensor
    raw_energies: torch.Tensor
    normalized_energies: torch.Tensor
    expert_outputs: Dict[str, ExpertOutput]
    losses: Dict[str, torch.Tensor]


class EnergyNormalizer(nn.Module):
    def __init__(self, num_experts: int, decay: float = 0.99, eps: float = 1e-6) -> None:
        super().__init__()
        self.decay = min(max(float(decay), 0.0), 0.999999)
        self.eps = max(float(eps), 1e-12)
        self.register_buffer("mean", torch.zeros(num_experts))
        self.register_buffer("std", torch.ones(num_experts))
        self.register_buffer("initialized", torch.zeros(num_experts, dtype=torch.bool))

    def forward(
        self,
        energies: torch.Tensor,
        valid_mask: torch.Tensor,
        update_stats: bool,
    ) -> torch.Tensor:
        if update_stats:
            with torch.no_grad():
                for expert_idx in range(energies.shape[1]):
                    values = energies[:, expert_idx][valid_mask[:, expert_idx]].detach()
                    if values.numel() == 0:
                        continue
                    batch_mean = values.mean()
                    batch_std = values.std(unbiased=False).clamp_min(self.eps)
                    if not bool(self.initialized[expert_idx]):
                        self.mean[expert_idx].copy_(batch_mean)
                        self.std[expert_idx].copy_(batch_std)
                        self.initialized[expert_idx] = True
                    else:
                        self.mean[expert_idx].lerp_(batch_mean, 1.0 - self.decay)
                        self.std[expert_idx].lerp_(batch_std, 1.0 - self.decay)
        normalized = (energies - self.mean.detach()) / (self.std.detach() + self.eps)
        return torch.where(valid_mask, normalized, torch.zeros_like(normalized))


class GatingNetwork(nn.Module):
    def __init__(self, node_feature_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2 * node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(EXPERT_NAMES)),
        )

    def forward(self, node_features: torch.Tensor, neighbor_context: torch.Tensor) -> torch.Tensor:
        state = torch.cat([node_features, neighbor_context], dim=-1).detach()
        return torch.softmax(self.network(state), dim=-1)


class APTMoE(nn.Module):
    """Joint four-expert anomaly detector with CAMERA-style one-class routing."""

    def __init__(
        self,
        semantic_dim: int,
        node_type_dim: int,
        edge_type_dim: int,
        input_dim: int,
        hidden_dim: int = 64,
        expert_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.1,
        multilabel_edges: bool = True,
        alpha_entropy: float = 0.01,
        beta_final: float = 1.0,
        eta_balance: float = 0.0,
        energy_ema_decay: float = 0.99,
        energy_eps: float = 1e-6,
        causal_smoothmax_tau: float = 5.0,
        graph_mask_rate: float = 0.3,
        expert_loss_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__()
        self.node_type_dim = int(node_type_dim)
        self.input_dim = int(input_dim)
        self.alpha_entropy = float(alpha_entropy)
        self.beta_final = float(beta_final)
        self.eta_balance = float(eta_balance)
        self.energy_eps = max(float(energy_eps), 1e-12)
        self.graph_mask_rate = min(max(float(graph_mask_rate), 0.0), 1.0)
        self.expert_loss_weights = {
            name: float((expert_loss_weights or {}).get(name, 1.0)) for name in EXPERT_NAMES
        }

        self.semantic_expert = SemanticExpert(semantic_dim, hidden_dim, expert_dim, dropout)
        self.graph_expert = GraphExpert(
            node_type_dim,
            edge_type_dim,
            hidden_dim,
            expert_dim,
            num_layers,
            num_heads,
            dropout,
        )
        self.normal_expert = NormalExpert(input_dim, hidden_dim, expert_dim, dropout)
        self.causality_expert = CausalityExpert(
            input_dim,
            edge_type_dim,
            hidden_dim,
            expert_dim,
            num_layers,
            num_heads,
            dropout,
            multilabel_edges,
            smoothmax_tau=causal_smoothmax_tau,
        )
        self.gating_network = GatingNetwork(input_dim, hidden_dim, dropout)
        self.energy_normalizer = EnergyNormalizer(
            len(EXPERT_NAMES), decay=energy_ema_decay, eps=energy_eps
        )

    def forward(
        self,
        graph,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_targets: torch.Tensor,
        graph_mask_nodes: Optional[torch.Tensor] = None,
        graph_inference_folds: int = 8,
        update_energy_stats: Optional[bool] = None,
    ) -> APTMoEOutput:
        x = node_features.float()
        node_type = x[:, : self.node_type_dim]
        semantic = x[:, self.node_type_dim :]
        if update_energy_stats is None:
            update_energy_stats = self.training

        semantic_output = self.semantic_expert(semantic)
        normal_output = self.normal_expert(x)
        causal_output = self.causality_expert(graph, x, edge_targets, edge_features)

        if graph_mask_nodes is not None:
            graph_output = self.graph_expert(
                graph, node_type, edge_features, mask_nodes=graph_mask_nodes
            )
            graph_valid = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
            graph_valid[graph_mask_nodes] = True
        elif not self.training:
            graph_output, graph_valid = self._graph_output_by_folds(
                graph, node_type, edge_features, graph_inference_folds
            )
        else:
            graph_mask_nodes = self._sample_graph_mask(x.shape[0], self.graph_mask_rate, x.device)
            graph_output = self.graph_expert(
                graph, node_type, edge_features, mask_nodes=graph_mask_nodes
            )
            graph_valid = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
            graph_valid[graph_mask_nodes] = True

        outputs = {
            "semantic": semantic_output,
            "graph": graph_output,
            "normal": normal_output,
            "causal": causal_output,
        }
        raw_energies = torch.stack([outputs[name].score for name in EXPERT_NAMES], dim=1)
        valid_mask = torch.ones_like(raw_energies, dtype=torch.bool)
        valid_mask[:, 1] = graph_valid
        normalized = self.energy_normalizer(raw_energies, valid_mask, bool(update_energy_stats))
        activated = F.softplus(normalized)
        activated = torch.where(valid_mask, activated, torch.zeros_like(activated))

        neighbor_context = self._mean_neighbor_features(graph, x)
        weights = self.gating_network(x, neighbor_context)
        fused_energy = torch.sum(weights * activated, dim=1)
        score = torch.sigmoid(fused_energy)

        entropy = -(weights * torch.log(weights.clamp_min(self.energy_eps))).sum(dim=1).mean()
        mean_weights = weights.mean(dim=0).clamp_min(self.energy_eps)
        balance = torch.sum(mean_weights * torch.log(len(EXPERT_NAMES) * mean_weights))
        final_nodes = valid_mask.all(dim=1)
        final_loss = F.binary_cross_entropy(
            score[final_nodes], torch.zeros_like(score[final_nodes])
        )
        losses = {
            "semantic": semantic_output.loss,
            "graph": graph_output.loss,
            "normal": normal_output.loss,
            "causal": causal_output.loss,
            "entropy": entropy,
            "balance": balance,
            "final": final_loss,
        }
        total = sum(
            self.expert_loss_weights[name] * losses[name] for name in EXPERT_NAMES
        )
        total = total + self.alpha_entropy * entropy + self.eta_balance * balance + self.beta_final * final_loss
        losses["total"] = total

        return APTMoEOutput(
            score=score,
            fused_energy=fused_energy,
            weights=weights,
            raw_energies=raw_energies,
            normalized_energies=normalized,
            expert_outputs=outputs,
            losses=losses,
        )

    def set_normal_prototype(self, prototype: torch.Tensor) -> None:
        self.normal_expert.set_prototype(prototype)

    def _graph_output_by_folds(self, graph, node_type, edge_features, num_folds):
        num_nodes = graph.num_nodes()
        folds = max(1, min(int(num_folds), num_nodes))
        node_folds = torch.tensor_split(torch.arange(num_nodes, device=node_type.device), folds)
        score = torch.zeros(num_nodes, device=node_type.device)
        residual = torch.zeros(
            num_nodes, self.graph_expert.expert_dim, device=node_type.device
        )
        losses = []
        for mask_nodes in node_folds:
            output = self.graph_expert(graph, node_type, edge_features, mask_nodes=mask_nodes)
            score[mask_nodes] = output.score[mask_nodes]
            residual[mask_nodes] = output.residual[mask_nodes]
            losses.append(output.loss)
        valid = torch.ones(num_nodes, dtype=torch.bool, device=node_type.device)
        return ExpertOutput(
            residual=residual,
            score=score,
            loss=torch.stack(losses).mean(),
            aux={"folds": folds},
        ), valid

    @staticmethod
    def _sample_graph_mask(num_nodes: int, mask_rate: float, device: torch.device) -> torch.Tensor:
        count = max(1, min(int(num_nodes), int(round(num_nodes * mask_rate))))
        return torch.randperm(num_nodes, device=device)[:count]

    @staticmethod
    def _mean_neighbor_features(graph, node_features: torch.Tensor) -> torch.Tensor:
        with graph.local_scope():
            graph.ndata["_apt_moe_x"] = node_features.detach()
            graph.update_all(fn.copy_u("_apt_moe_x", "_apt_moe_message"), fn.sum("_apt_moe_message", "_apt_moe_sum"))
            degree = graph.in_degrees().to(node_features.device)
            context = graph.ndata["_apt_moe_sum"] / degree.clamp_min(1).unsqueeze(1)
            context = context.masked_fill((degree == 0).unsqueeze(1), 0.0)
        return context


def build_apt_moe(dataset, cfg: Dict[str, object]) -> APTMoE:
    model_cfg = cfg["model"]
    moe_cfg = cfg.get("moe", {})
    training_cfg = cfg.get("training", {})
    return APTMoE(
        semantic_dim=dataset.attr_dim,
        node_type_dim=dataset.node_type_dim,
        edge_type_dim=dataset.edge_type_dim,
        input_dim=dataset.num_features,
        hidden_dim=int(model_cfg["hidden_dim"]),
        expert_dim=int(model_cfg["expert_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        num_heads=int(model_cfg["num_heads"]),
        dropout=float(model_cfg["dropout"]),
        multilabel_edges=dataset.edge_is_multilabel,
        alpha_entropy=float(moe_cfg.get("alpha_entropy", 0.01)),
        beta_final=float(moe_cfg.get("beta_final", 1.0)),
        eta_balance=float(moe_cfg.get("eta_balance", 0.0)),
        energy_ema_decay=float(moe_cfg.get("energy_ema_decay", 0.99)),
        energy_eps=float(moe_cfg.get("energy_eps", 1e-6)),
        causal_smoothmax_tau=float(moe_cfg.get("causal_smoothmax_tau", 5.0)),
        graph_mask_rate=float(training_cfg.get("mask_rate", 0.3)),
        expert_loss_weights=cfg.get("loss_weights", {}),
    )
