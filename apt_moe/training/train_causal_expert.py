from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch

from apt_moe.data.provfusion_adapter import edge_targets, edge_type_features, node_full_features, non_self_edge_mask
from apt_moe.models import CausalityExpert
from .common import mean_or_zero, run_training_loop, sample_edges


def train_causal_expert(dataset, cfg: Dict[str, object], device: torch.device) -> Dict[str, object]:
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    out_cfg = cfg["output"]
    model = CausalityExpert(
        node_feat_dim=dataset.num_features,
        edge_type_dim=dataset.edge_type_dim,
        hidden_dim=int(model_cfg["hidden_dim"]),
        expert_dim=int(model_cfg["expert_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        num_heads=int(model_cfg["num_heads"]),
        dropout=float(model_cfg["dropout"]),
        multilabel=dataset.edge_is_multilabel,
    ).to(device)

    def step(current_model, split: str) -> torch.Tensor:
        losses = []
        for batch in dataset.iter_split(split):
            graph = batch.graph.to(device)
            full = node_full_features(graph).to(device)
            edge_features = edge_type_features(graph, dataset.edge_type_dim).to(device)
            targets = edge_targets(graph, dataset.edge_is_multilabel).to(device)
            edge_idx = torch.nonzero(non_self_edge_mask(graph).to(device), as_tuple=False).view(-1)
            edge_idx = sample_edges(edge_idx, train_cfg.get("edge_train_sample_size"))
            if edge_idx.numel() == 0:
                continue
            losses.append(current_model(graph, full, targets, edge_features, edge_indices=edge_idx).loss)
        return mean_or_zero(losses, device)

    checkpoint = Path(out_cfg["checkpoint_dir"]) / "causality_expert.pt"
    return run_training_loop(
        model=model,
        train_step=step,
        val_step=step,
        epochs=int(train_cfg["epochs"]),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
        patience=int(train_cfg["patience"]),
        checkpoint_path=checkpoint,
        model_meta={
            "expert": "causal",
            "node_feat_dim": dataset.num_features,
            "edge_type_dim": dataset.edge_type_dim,
            "hidden_dim": int(model_cfg["hidden_dim"]),
            "expert_dim": int(model_cfg["expert_dim"]),
            "num_layers": int(model_cfg["num_layers"]),
            "num_heads": int(model_cfg["num_heads"]),
            "dropout": float(model_cfg["dropout"]),
            "multilabel": dataset.edge_is_multilabel,
            "uses_edge_type": True,
            "target_edge_labels_masked_for_attention": False,
        },
    )
