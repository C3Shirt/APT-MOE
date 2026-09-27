from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch

from apt_moe.data.provfusion_adapter import edge_type_features, node_type_one_hot
from apt_moe.models import GraphExpert
from .common import mean_or_zero, random_node_mask, run_training_loop


def train_graph_expert(dataset, cfg: Dict[str, object], device: torch.device) -> Dict[str, object]:
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    out_cfg = cfg["output"]
    model = GraphExpert(
        node_type_dim=dataset.node_type_dim,
        edge_type_dim=dataset.edge_type_dim,
        hidden_dim=int(model_cfg["hidden_dim"]),
        expert_dim=int(model_cfg["expert_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        num_heads=int(model_cfg["num_heads"]),
        dropout=float(model_cfg["dropout"]),
    ).to(device)

    def step(current_model, split: str) -> torch.Tensor:
        losses = []
        for batch in dataset.iter_split(split):
            graph = batch.graph.to(device)
            node_type = node_type_one_hot(graph).to(device)
            edge_features = edge_type_features(graph, dataset.edge_type_dim).to(device)
            mask_nodes = random_node_mask(graph.num_nodes(), float(train_cfg["mask_rate"]), device)
            losses.append(current_model(graph, node_type, edge_features, mask_nodes=mask_nodes).loss)
        return mean_or_zero(losses, device)

    checkpoint = Path(out_cfg["checkpoint_dir"]) / "graph_expert.pt"
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
            "expert": "graph",
            "node_type_dim": dataset.node_type_dim,
            "edge_type_dim": dataset.edge_type_dim,
            "hidden_dim": int(model_cfg["hidden_dim"]),
            "expert_dim": int(model_cfg["expert_dim"]),
            "num_layers": int(model_cfg["num_layers"]),
            "num_heads": int(model_cfg["num_heads"]),
            "dropout": float(model_cfg["dropout"]),
            "uses_edge_type": True,
        },
    )
