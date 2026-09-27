from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch

from apt_moe.data.provfusion_adapter import node_full_features
from apt_moe.models import NormalExpert
from .common import mean_or_zero, run_training_loop


def train_normal_expert(dataset, cfg: Dict[str, object], device: torch.device) -> Dict[str, object]:
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    out_cfg = cfg["output"]
    model = NormalExpert(
        input_dim=dataset.num_features,
        hidden_dim=int(model_cfg["hidden_dim"]),
        expert_dim=int(model_cfg["expert_dim"]),
        dropout=float(model_cfg["dropout"]),
    ).to(device)
    prototype = _training_prototype(dataset, device)
    model.set_prototype(prototype)

    def step(current_model, split: str) -> torch.Tensor:
        losses = []
        for batch in dataset.iter_split(split):
            graph = batch.graph.to(device)
            full = node_full_features(graph).to(device)
            losses.append(current_model(full).loss)
        return mean_or_zero(losses, device)

    checkpoint = Path(out_cfg["checkpoint_dir"]) / "normal_expert.pt"
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
            "expert": "normal",
            "input_dim": dataset.num_features,
            "hidden_dim": int(model_cfg["hidden_dim"]),
            "expert_dim": int(model_cfg["expert_dim"]),
            "dropout": float(model_cfg["dropout"]),
            "uses_gnn": False,
            "uses_ground_truth_for_training": False,
            "prototype_estimator": "coordinatewise_median_of_unlabeled_train_nodes",
        },
    )


def _training_prototype(dataset, device: torch.device) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    for batch in dataset.iter_split("train"):
        graph = batch.graph.to(device)
        full = node_full_features(graph).to(device)
        chunks.append(full.detach())
    if not chunks:
        first = dataset.first_graph().to(device)
        chunks.append(node_full_features(first).to(device).detach())
    return torch.cat(chunks, dim=0).median(dim=0).values
