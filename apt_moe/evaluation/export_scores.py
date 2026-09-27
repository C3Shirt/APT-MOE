from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from apt_moe.data.provfusion_adapter import edge_targets, edge_type_features
from apt_moe.evaluation.calibration import EmpiricalCDFCalibrator
from apt_moe.models.apt_moe import EXPERT_NAMES, APTMoE, build_apt_moe


CSV_COLUMNS = [
    "graph_id",
    "sample_id",
    "edge_id",
    "src_id",
    "dst_id",
    "timestamp",
    "split",
    "threshold_role",
    "label",
    "semantic_raw",
    "graph_raw",
    "normal_raw",
    "causal_raw",
    "semantic_calibrated",
    "graph_calibrated",
    "normal_calibrated",
    "causal_calibrated",
    "semantic_norm_energy",
    "graph_norm_energy",
    "normal_norm_energy",
    "causal_norm_energy",
    "causal_max_incident",
    "gate_semantic",
    "gate_graph",
    "gate_normal",
    "gate_causal",
    "fused_energy",
    "score",
    "threshold",
    "predicted_anomaly",
    "detection_granularity",
]


def load_joint_model(dataset, cfg: Dict[str, object], device: torch.device) -> APTMoE:
    checkpoint_path = Path(cfg["output"]["checkpoint_dir"]) / "apt_moe_joint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Joint checkpoint not found: {checkpoint_path}. "
            "Run python -m apt_moe.train_experts with the same config first."
        )
    checkpoint = _torch_load(checkpoint_path)
    model = build_apt_moe(dataset, cfg).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def score_dataset_node_level(
    dataset,
    model: APTMoE,
    cfg: Dict[str, object],
    device: torch.device,
) -> Dict[str, List[Dict[str, object]]]:
    split_scores = {split: [] for split in dataset.all_splits()}
    folds = int(cfg.get("evaluation", {}).get("mask_folds", 8))
    validation_train_indices = {
        int(index) for index in cfg.get("training", {}).get("validation_train_graph_indices", [])
    }
    with torch.no_grad():
        for split in dataset.all_splits():
            for batch in dataset.iter_split(split):
                graph = batch.graph.to(device)
                features = graph.ndata["feat"].float()
                edge_features = edge_type_features(graph, dataset.edge_type_dim).to(device)
                targets = edge_targets(graph, dataset.edge_is_multilabel).to(device)
                output = model(
                    graph,
                    features,
                    edge_features,
                    targets,
                    graph_mask_nodes=None,
                    graph_inference_folds=folds,
                    update_energy_stats=False,
                )
                labels = batch.malicious_labels.numpy().astype(int) if batch.malicious_labels is not None else None
                node_ids = batch.node_ids.numpy()
                causal_max = output.expert_outputs["causal"].aux["exact_max_incident_edge_loss"].cpu().numpy()
                rows = []
                for idx, node_id in enumerate(node_ids):
                    row = {
                        "graph_id": batch.graph_id,
                        "sample_id": int(node_id),
                        "edge_id": "",
                        "src_id": int(node_id),
                        "dst_id": int(node_id),
                        "timestamp": "",
                        "split": split,
                        "threshold_role": (
                            "validation"
                            if split == "val"
                            or (split == "train" and batch.graph_index in validation_train_indices)
                            else split
                        ),
                        "label": int(labels[idx]) if labels is not None else "",
                        "semantic_raw": float(output.raw_energies[idx, 0].cpu()),
                        "graph_raw": float(output.raw_energies[idx, 1].cpu()),
                        "normal_raw": float(output.raw_energies[idx, 2].cpu()),
                        "causal_raw": float(output.raw_energies[idx, 3].cpu()),
                        "semantic_norm_energy": float(output.normalized_energies[idx, 0].cpu()),
                        "graph_norm_energy": float(output.normalized_energies[idx, 1].cpu()),
                        "normal_norm_energy": float(output.normalized_energies[idx, 2].cpu()),
                        "causal_norm_energy": float(output.normalized_energies[idx, 3].cpu()),
                        "causal_max_incident": float(causal_max[idx]),
                        "gate_semantic": float(output.weights[idx, 0].cpu()),
                        "gate_graph": float(output.weights[idx, 1].cpu()),
                        "gate_normal": float(output.weights[idx, 2].cpu()),
                        "gate_causal": float(output.weights[idx, 3].cpu()),
                        "fused_energy": float(output.fused_energy[idx].cpu()),
                        "score": float(output.score[idx].cpu()),
                        "detection_granularity": "node",
                    }
                    rows.append(row)
                split_scores[split].extend(rows)
    return split_scores


def calibrate_threshold(
    rows: List[Dict[str, object]],
    quantile: float,
) -> float:
    candidates, _ = threshold_calibration_source(rows)
    if not candidates:
        raise ValueError("Cannot calibrate anomaly threshold without validation or training scores.")
    q = min(max(float(quantile), 0.0), 1.0)
    return float(np.quantile([float(row["fused_energy"]) for row in candidates], q))


def threshold_calibration_source(rows: List[Dict[str, object]]) -> tuple[List[Dict[str, object]], str]:
    candidates = [
        row for row in rows
        if row.get("threshold_role") == "validation" or row.get("split") == "val"
    ]
    source = "validation"
    if not candidates:
        candidates = [row for row in rows if row.get("split") == "train"]
        source = "train_fallback"
    if not candidates:
        raise ValueError("No validation or training scores are available for threshold calibration.")
    labeled = [row for row in candidates if row.get("label") not in (None, "")]
    graph_count = len({row.get("graph_id", row.get("sample_id")) for row in candidates})
    if labeled:
        candidates = [row for row in labeled if int(row["label"]) == 0]
        if not candidates:
            raise ValueError(f"No benign nodes are available in the labeled {source} split for thresholding.")
        return candidates, f"{source}_benign_nodes_{graph_count}_graphs"
    return candidates, f"{source}_unlabeled_assumed_benign_{graph_count}_graphs"


def apply_threshold(rows: List[Dict[str, object]], threshold: float) -> List[Dict[str, object]]:
    for row in rows:
        row["threshold"] = float(threshold)
        row["predicted_anomaly"] = int(float(row["fused_energy"]) > threshold)
    return rows


def fit_calibrators(
    split_scores: Dict[str, List[Dict[str, object]]], cfg: Dict[str, object]
) -> Dict[str, EmpiricalCDFCalibrator]:
    candidates = split_scores.get("train", [])
    return {
        expert: EmpiricalCDFCalibrator.fit(row[f"{expert}_raw"] for row in candidates)
        for expert in EXPERT_NAMES
    }


def apply_calibrators(
    rows: List[Dict[str, object]], calibrators: Dict[str, EmpiricalCDFCalibrator]
) -> List[Dict[str, object]]:
    """Legacy score export helper; joint MoE fusion uses EMA z-score energies."""
    for expert in EXPERT_NAMES:
        calibrated = calibrators[expert].transform(
            np.asarray([row[f"{expert}_raw"] for row in rows], dtype=float)
        )
        for idx, row in enumerate(rows):
            row[f"{expert}_calibrated"] = float(calibrated[idx])
    return rows


def export_scores_csv(rows: List[Dict[str, object]], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")
