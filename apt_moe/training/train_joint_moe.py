from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from apt_moe.data.provfusion_adapter import edge_targets, edge_type_features
from apt_moe.models.apt_moe import EXPERT_NAMES, build_apt_moe
from .common import random_node_mask


LOSS_NAMES = ("semantic", "graph", "normal", "causal", "entropy", "balance", "final", "total")


def train_joint_moe(dataset, cfg: Dict[str, object], device: torch.device) -> Dict[str, object]:
    model = build_apt_moe(dataset, cfg).to(device)
    train_cfg = cfg["training"]
    moe_cfg = cfg.get("moe", {})
    out_cfg = cfg["output"]
    all_train_batches = list(dataset.iter_split("train"))
    validation_train_indices = sorted(
        {int(index) for index in train_cfg.get("validation_train_graph_indices", [])}
    )
    train_batches = [
        batch for batch in all_train_batches if batch.graph_index not in validation_train_indices
    ]
    if len(train_batches) == 0:
        raise ValueError("At least one training graph must remain after reserving validation graphs.")
    missing_indices = set(validation_train_indices) - {batch.graph_index for batch in all_train_batches}
    if missing_indices:
        raise ValueError(f"Validation graph indices are not present in the train split: {sorted(missing_indices)}")
    heldout_batches = [
        batch for batch in all_train_batches if batch.graph_index in validation_train_indices
    ]
    validation_batches = list(dataset.iter_split("val")) + heldout_batches
    if not validation_batches:
        validation_batches = train_batches
    model.set_normal_prototype(_training_normal_prototype(train_batches).to(device))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    epochs = int(train_cfg["epochs"])
    patience = int(train_cfg["patience"])
    graph_mask_rate = float(train_cfg.get("mask_rate", 0.3))
    graph_folds = int(cfg.get("evaluation", {}).get("mask_folds", 8))
    best_state = deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = -1
    wait = 0
    history: List[Dict[str, float]] = []

    for epoch in range(epochs):
        started = time.time()
        model.train()
        train_metrics = {name: [] for name in LOSS_NAMES}
        for batch in train_batches:
            graph = batch.graph.to(device)
            features, edge_features, targets = _model_inputs(graph, dataset)
            mask = random_node_mask(graph.num_nodes(), graph_mask_rate, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                graph,
                features,
                edge_features,
                targets,
                graph_mask_nodes=mask,
                update_energy_stats=True,
            )
            output.losses["total"].backward()
            optimizer.step()
            _record_losses(train_metrics, output.losses)

        model.eval()
        val_metrics = {name: [] for name in LOSS_NAMES}
        with torch.no_grad():
            for batch in validation_batches:
                graph = batch.graph.to(device)
                features, edge_features, targets = _model_inputs(graph, dataset)
                mask = _fixed_node_mask(graph.num_nodes(), graph_mask_rate, device, batch.graph_index)
                output = model(
                    graph,
                    features,
                    edge_features,
                    targets,
                    graph_mask_nodes=mask,
                    update_energy_stats=False,
                )
                _record_losses(val_metrics, output.losses)

        train_values = _means(train_metrics)
        val_values = _means(val_metrics)
        val_loss = val_values["total"] if val_values["total"] is not None else train_values["total"]
        if val_loss is None:
            raise RuntimeError("Joint MoE training produced no batches in the train or validation split.")

        epoch_record = {f"train_{name}": value for name, value in train_values.items() if value is not None}
        epoch_record.update({f"val_{name}": value for name, value in val_values.items() if value is not None})
        epoch_record["epoch"] = float(epoch)
        history.append(epoch_record)
        print(
            f"[joint-moe] epoch={epoch} "
            + " ".join(f"{name}={train_values[name]:.5f}" for name in LOSS_NAMES)
            + f" val_total={float(val_loss):.5f} seconds={time.time() - started:.2f}"
        )

        if float(val_loss) < best_val:
            best_val = float(val_loss)
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    checkpoint_path = Path(out_cfg["checkpoint_dir"]) / "apt_moe_joint.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model_meta = {
        "expert_order": list(EXPERT_NAMES),
        "semantic_dim": dataset.attr_dim,
        "node_type_dim": dataset.node_type_dim,
        "edge_type_dim": dataset.edge_type_dim,
        "input_dim": dataset.num_features,
        "edge_is_multilabel": dataset.edge_is_multilabel,
        "training_mode": "end_to_end_joint_single_optimizer",
        "ground_truth_used_for_training": False,
        "normal_prototype": "unlabeled_mean_of_optimizer_training_nodes",
        "training_graph_indices": [batch.graph_index for batch in train_batches],
        "validation_train_graph_indices": validation_train_indices,
        "validation_graph_count": len(validation_batches),
    }
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_meta": model_meta,
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
            "history": history,
            "moe_config": dict(moe_cfg),
            "loss_weights": dict(cfg.get("loss_weights", {})),
        },
        checkpoint_path,
    )
    return {
        "checkpoint": str(checkpoint_path),
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "history": history,
        "ground_truth_used_for_training": False,
        "training_graph_indices": [batch.graph_index for batch in train_batches],
        "validation_train_graph_indices": validation_train_indices,
        "validation_graph_count": len(validation_batches),
    }


def _model_inputs(graph, dataset):
    features = graph.ndata["feat"].float()
    edge_features = edge_type_features(graph, dataset.edge_type_dim).to(features.device)
    targets = edge_targets(graph, dataset.edge_is_multilabel).to(features.device)
    return features, edge_features, targets


def _training_normal_prototype(train_batches) -> torch.Tensor:
    feature_sum = None
    num_nodes = 0
    for batch in train_batches:
        features = batch.graph.ndata["feat"].float()
        batch_sum = features.sum(dim=0, dtype=torch.float64)
        feature_sum = batch_sum if feature_sum is None else feature_sum + batch_sum
        num_nodes += features.shape[0]
    if feature_sum is None or num_nodes == 0:
        raise RuntimeError("Cannot compute the normal prototype: the train split is empty.")
    return (feature_sum / num_nodes).float().unsqueeze(0)


def _fixed_node_mask(num_nodes: int, mask_rate: float, device: torch.device, offset: int) -> torch.Tensor:
    count = max(1, min(int(num_nodes), int(round(num_nodes * mask_rate))))
    start = int(offset) % max(num_nodes, 1)
    return (torch.arange(count, device=device) + start) % num_nodes


def _record_losses(store, losses) -> None:
    for name in LOSS_NAMES:
        if name in losses:
            store[name].append(float(losses[name].detach().cpu()))


def _means(values) -> Dict[str, float | None]:
    return {name: float(np.mean(items)) if items else None for name, items in values.items()}
