from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch

from apt_moe.data.provfusion_adapter import node_type_one_hot
from apt_moe.models import NodeTypeExpert
from .common import mean_or_zero, random_node_mask, run_training_loop


def train_node_expert(dataset, cfg: Dict[str, object], device: torch.device) -> Dict[str, object]:
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    out_cfg = cfg["output"]
    model = NodeTypeExpert(
        num_node_types=dataset.node_type_dim,
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        dropout=float(model_cfg["dropout"]),
    ).to(device)

    def step(current_model, split: str) -> torch.Tensor:
        losses = []
        for batch in dataset.iter_split(split):
            graph = batch.graph.to(device)
            x = node_type_one_hot(graph).to(device)
            target = x.argmax(dim=1).long()
            mask_nodes = random_node_mask(graph.num_nodes(), float(train_cfg["mask_rate"]), device)
            losses.append(current_model.loss(graph, x, target, mask_nodes))
        return mean_or_zero(losses, device)

    checkpoint = Path(out_cfg["checkpoint_dir"]) / "node_expert.pt"
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
            "expert": "node",
            "num_node_types": dataset.node_type_dim,
            "hidden_dim": int(model_cfg["hidden_dim"]),
            "num_layers": int(model_cfg["num_layers"]),
            "dropout": float(model_cfg["dropout"]),
        },
    )
