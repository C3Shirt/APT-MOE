from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch

from apt_moe.data.provfusion_adapter import edge_targets, node_full_features, non_self_edge_mask
from apt_moe.models import EdgeTypeExpert
from .common import mean_or_zero, run_training_loop, sample_edges


def train_edge_expert(dataset, cfg: Dict[str, object], device: torch.device) -> Dict[str, object]:
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    out_cfg = cfg["output"]
    model = EdgeTypeExpert(
        node_feat_dim=dataset.num_features,
        edge_type_dim=dataset.edge_type_dim,
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        dropout=float(model_cfg["dropout"]),
        multilabel=dataset.edge_is_multilabel,
    ).to(device)

    def step(current_model, split: str) -> torch.Tensor:
        losses = []
        for batch in dataset.iter_split(split):
            graph = batch.graph.to(device)
            node_features = node_full_features(graph).to(device)
            src, dst = graph.edges()
            all_edges = torch.stack([src, dst], dim=1).to(device)
            mask = non_self_edge_mask(graph).to(device)
            edge_idx = torch.nonzero(mask, as_tuple=False).view(-1)
            edge_idx = sample_edges(edge_idx, train_cfg.get("edge_train_sample_size"))
            if edge_idx.numel() == 0:
                continue
            targets = edge_targets(graph, dataset.edge_is_multilabel).to(device)
            losses.append(current_model.loss(graph, node_features, all_edges[edge_idx], targets[edge_idx]))
        return mean_or_zero(losses, device)

    checkpoint = Path(out_cfg["checkpoint_dir"]) / "edge_expert.pt"
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
            "expert": "edge",
            "node_feat_dim": dataset.num_features,
            "edge_type_dim": dataset.edge_type_dim,
            "hidden_dim": int(model_cfg["hidden_dim"]),
            "num_layers": int(model_cfg["num_layers"]),
            "dropout": float(model_cfg["dropout"]),
            "multilabel": dataset.edge_is_multilabel,
            "uses_edge_type": False,
        },
    )
