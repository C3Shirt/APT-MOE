from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

from apt_moe.data.provfusion_adapter import (
    edge_targets,
    node_full_features,
    node_type_one_hot,
    non_self_edge_mask,
)
from apt_moe.evaluation.calibration import EmpiricalCDFCalibrator
from apt_moe.models import AttributeExpert, EdgeTypeExpert, NodeTypeExpert


CSV_COLUMNS = [
    "graph_id",
    "sample_id",
    "edge_id",
    "src_id",
    "dst_id",
    "timestamp",
    "split",
    "label",
    "node_src_raw",
    "node_dst_raw",
    "node_event_raw",
    "edge_raw",
    "attr_src_raw",
    "attr_dst_raw",
    "attr_event_raw",
    "node_calibrated",
    "edge_calibrated",
    "attr_calibrated",
    "detection_granularity",
]


def load_experts(dataset, cfg: Dict[str, object], device: torch.device) -> Dict[str, torch.nn.Module]:
    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    models: Dict[str, torch.nn.Module] = {}

    node_ckpt = _torch_load(ckpt_dir / "node_expert.pt")
    meta = node_ckpt["model_meta"]
    node_model = NodeTypeExpert(
        num_node_types=int(meta["num_node_types"]),
        hidden_dim=int(meta["hidden_dim"]),
        num_layers=int(meta["num_layers"]),
        dropout=float(meta["dropout"]),
    )
    node_model.load_state_dict(node_ckpt["state_dict"])
    models["node"] = node_model.to(device).eval()

    edge_ckpt = _torch_load(ckpt_dir / "edge_expert.pt")
    meta = edge_ckpt["model_meta"]
    edge_model = EdgeTypeExpert(
        node_feat_dim=int(meta["node_feat_dim"]),
        edge_type_dim=int(meta["edge_type_dim"]),
        hidden_dim=int(meta["hidden_dim"]),
        num_layers=int(meta["num_layers"]),
        dropout=float(meta["dropout"]),
        multilabel=bool(meta["multilabel"]),
    )
    edge_model.load_state_dict(edge_ckpt["state_dict"])
    models["edge"] = edge_model.to(device).eval()

    attr_ckpt = _torch_load(ckpt_dir / "attribute_expert.pt")
    meta = attr_ckpt["model_meta"]
    attr_model = AttributeExpert(
        node_type_dim=int(meta["node_type_dim"]),
        attr_dim=int(meta["attr_dim"]),
        hidden_dim=int(meta["hidden_dim"]),
        num_layers=int(meta["num_layers"]),
        dropout=float(meta["dropout"]),
        lambda_cos=float(meta["lambda_cos"]),
        lambda_mse=float(meta["lambda_mse"]),
    )
    attr_model.load_state_dict(attr_ckpt["state_dict"])
    models["attr"] = attr_model.to(device).eval()
    return models


def score_dataset_node_level(dataset, models: Dict[str, torch.nn.Module], cfg: Dict[str, object], device: torch.device):
    split_scores = {split: [] for split in dataset.all_splits()}
    mask_folds = int(cfg["evaluation"]["mask_folds"])
    with torch.no_grad():
        for split in dataset.all_splits():
            for batch in dataset.iter_split(split):
                graph = batch.graph.to(device)
                full = node_full_features(graph).to(device)
                node_type = node_type_one_hot(graph).to(device)
                attrs = full[:, dataset.node_type_dim :]
                node_target = node_type.argmax(dim=1).long()

                node_raw = models["node"].node_scores(graph, node_type, node_target, mask_folds=mask_folds)
                attr_raw = models["attr"].node_scores(graph, node_type, attrs, mask_folds=mask_folds)
                edge_raw = _edge_scores_to_nodes(graph, full, dataset, models["edge"], device)

                labels = (
                    batch.malicious_labels.numpy().astype(int)
                    if batch.malicious_labels is not None
                    else np.zeros(graph.num_nodes(), dtype=int)
                )
                rows = []
                node_ids = batch.node_ids.numpy()
                for i, node_id in enumerate(node_ids):
                    rows.append(
                        {
                            "graph_id": batch.graph_id,
                            "sample_id": int(node_id),
                            "edge_id": "",
                            "src_id": int(node_id),
                            "dst_id": int(node_id),
                            "timestamp": "",
                            "split": split,
                            "label": int(labels[i]),
                            "node_raw": float(node_raw[i].detach().cpu()),
                            "edge_raw": float(edge_raw[i].detach().cpu()),
                            "attr_raw": float(attr_raw[i].detach().cpu()),
                            "detection_granularity": "node",
                        }
                    )
                split_scores[split].extend(rows)
    return split_scores


def fit_calibrators(split_scores: Dict[str, List[Dict[str, object]]], cfg: Dict[str, object]) -> Dict[str, EmpiricalCDFCalibrator]:
    split = str(cfg["calibration"]["split"])
    candidates = split_scores.get(split, [])
    if bool(cfg["calibration"].get("use_only_benign", True)):
        candidates = [row for row in candidates if int(row["label"]) == 0]
    if not candidates:
        candidates = split_scores.get("train", [])
        if bool(cfg["calibration"].get("use_only_benign", True)):
            candidates = [row for row in candidates if int(row["label"]) == 0]

    return {
        "node": EmpiricalCDFCalibrator.fit(row["node_raw"] for row in candidates),
        "edge": EmpiricalCDFCalibrator.fit(row["edge_raw"] for row in candidates),
        "attr": EmpiricalCDFCalibrator.fit(row["attr_raw"] for row in candidates),
    }


def apply_calibrators(rows: List[Dict[str, object]], calibrators: Dict[str, EmpiricalCDFCalibrator]) -> List[Dict[str, object]]:
    node_cal = calibrators["node"].transform(row["node_raw"] for row in rows)
    edge_cal = calibrators["edge"].transform(row["edge_raw"] for row in rows)
    attr_cal = calibrators["attr"].transform(row["attr_raw"] for row in rows)
    for i, row in enumerate(rows):
        row["node_calibrated"] = float(node_cal[i])
        row["edge_calibrated"] = float(edge_cal[i])
        row["attr_calibrated"] = float(attr_cal[i])
    return rows


def export_scores_csv(rows: List[Dict[str, object]], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "graph_id": row["graph_id"],
                    "sample_id": row["sample_id"],
                    "edge_id": row["edge_id"],
                    "src_id": row["src_id"],
                    "dst_id": row["dst_id"],
                    "timestamp": row["timestamp"],
                    "split": row["split"],
                    "label": row["label"],
                    "node_src_raw": row["node_raw"],
                    "node_dst_raw": row["node_raw"],
                    "node_event_raw": row["node_raw"],
                    "edge_raw": row["edge_raw"],
                    "attr_src_raw": row["attr_raw"],
                    "attr_dst_raw": row["attr_raw"],
                    "attr_event_raw": row["attr_raw"],
                    "node_calibrated": row["node_calibrated"],
                    "edge_calibrated": row["edge_calibrated"],
                    "attr_calibrated": row["attr_calibrated"],
                    "detection_granularity": row["detection_granularity"],
                }
            )


def _edge_scores_to_nodes(graph, node_features, dataset, model, device: torch.device) -> torch.Tensor:
    src, dst = graph.edges()
    all_edges = torch.stack([src, dst], dim=1).to(device)
    mask = non_self_edge_mask(graph).to(device)
    edge_idx = torch.nonzero(mask, as_tuple=False).view(-1)
    node_scores = torch.full((graph.num_nodes(),), float("-inf"), device=device)
    if edge_idx.numel() == 0:
        return torch.zeros(graph.num_nodes(), device=device)

    targets = edge_targets(graph, dataset.edge_is_multilabel).to(device)
    scores = model.edge_scores(graph, node_features, all_edges[edge_idx], targets[edge_idx])
    endpoints = torch.cat([all_edges[edge_idx, 0], all_edges[edge_idx, 1]])
    endpoint_scores = torch.cat([scores, scores])
    try:
        node_scores.scatter_reduce_(0, endpoints, endpoint_scores, reduce="amax", include_self=True)
    except AttributeError:  # pragma: no cover - for very old torch.
        for idx, value in zip(endpoints.tolist(), endpoint_scores.tolist()):
            node_scores[idx] = torch.maximum(node_scores[idx], torch.tensor(value, device=device))
    node_scores = torch.where(torch.isinf(node_scores), torch.zeros_like(node_scores), node_scores)
    return node_scores


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")
