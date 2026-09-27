from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import numpy as np

from ..models.apt_moe import EXPERT_NAMES
from .metrics import binary_auc_metrics, jaccard, spearman, topk_ids, topk_metrics


def _expert_score_key(rows: List[Dict[str, object]], expert: str) -> str:
    for suffix in ("norm_energy", "calibrated", "raw"):
        key = f"{expert}_{suffix}"
        if rows and all(row.get(key) not in (None, "") for row in rows):
            return key
    raise KeyError(f"Rows do not contain a complete score column for expert {expert!r}.")


def _binary_threshold_metrics(labels: List[int], predictions: List[int]) -> Dict[str, object]:
    y = np.asarray(labels, dtype=int)
    pred = np.asarray(predictions, dtype=int)
    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }


def complementarity_report(rows: List[Dict[str, object]], k_values: List[int]) -> Dict[str, object]:
    ground_truth_available = bool(rows) and all(row.get("label") not in (None, "") for row in rows)
    labels = [int(row["label"]) for row in rows] if ground_truth_available else []
    malicious_ids = (
        {int(row["sample_id"]) for row in rows if int(row["label"]) == 1}
        if ground_truth_available
        else set()
    )
    score_keys = {expert: _expert_score_key(rows, expert) for expert in EXPERT_NAMES} if rows else {}
    predicted = (
        [int(row["predicted_anomaly"]) for row in rows]
        if rows and all(row.get("predicted_anomaly") not in (None, "") for row in rows)
        else None
    )
    report: Dict[str, object] = {
        "num_samples": len(rows),
        "num_malicious": len(malicious_ids),
        "num_detected": int(sum(predicted)) if predicted is not None else None,
        "ground_truth_available": ground_truth_available,
        "experts": {},
        "moe": {},
        "threshold_metrics": _binary_threshold_metrics(labels, predicted)
        if ground_truth_available and predicted is not None
        else None,
        "gate": {},
        "topk_overlap": {},
        "spearman": {},
        "union_hits": {},
        "single_expert_examples": {},
    }

    for expert, key in score_keys.items():
        scores = [float(row[key]) for row in rows]
        expert_report = binary_auc_metrics(labels, scores) if ground_truth_available else {
            "roc_auc": None,
            "pr_auc": None,
        }
        expert_report["score_basis"] = key
        expert_report["topk"] = topk_metrics(labels, scores, k_values) if ground_truth_available else {}
        report["experts"][expert] = expert_report

    moe_key = next(
        (key for key in ("fused_energy", "score", "moe_calibrated")
         if rows and all(row.get(key) not in (None, "") for row in rows)),
        None,
    )
    if moe_key is not None:
        fused_scores = [float(row[moe_key]) for row in rows]
        report["moe"] = binary_auc_metrics(labels, fused_scores) if ground_truth_available else {
            "roc_auc": None,
            "pr_auc": None,
        }
        report["moe"]["score_basis"] = moe_key
        report["moe"]["topk"] = topk_metrics(labels, fused_scores, k_values) if ground_truth_available else {}

    gate_keys = {expert: f"gate_{expert}" for expert in EXPERT_NAMES}
    if rows and all(all(row.get(key) not in (None, "") for key in gate_keys.values()) for row in rows):
        gate_means = {
            expert: float(np.mean([float(row[key]) for row in rows]))
            for expert, key in gate_keys.items()
        }
        report["gate"] = {
            "mean_weights": gate_means,
            "mean_entropy": float(np.mean([
                -sum(float(row[key]) * np.log(max(float(row[key]), 1e-12)) for key in gate_keys.values())
                for row in rows
            ])),
        }

    if score_keys:
        for k in k_values:
            kk = min(int(k), len(rows))
            sets = {expert: topk_ids(rows, key, kk) for expert, key in score_keys.items()}
            report["topk_overlap"][str(k)] = {
                f"{left}_{right}": jaccard(sets[left], sets[right])
                for left, right in combinations(score_keys, 2)
            }
            if ground_truth_available:
                hits = {expert: sets[expert] & malicious_ids for expert in sets}
                union_hits = {expert: len(values) for expert, values in hits.items()}
                union_hits.update(
                    {
                        f"{left}_{right}_union": len(hits[left] | hits[right])
                        for left, right in combinations(EXPERT_NAMES, 2)
                    }
                )
                union_hits["oracle_union"] = len(set().union(*hits.values())) if hits else 0
                report["union_hits"][str(k)] = union_hits
                report["single_expert_examples"][str(k)] = {
                    expert: sorted(
                        list(hits[expert] - set().union(*(hits[e] for e in hits if e != expert)))
                    )[:20]
                    for expert in hits
                }

        for left, right in combinations(score_keys, 2):
            report["spearman"][f"{left}_{right}"] = spearman(
                [float(row[score_keys[left]]) for row in rows],
                [float(row[score_keys[right]]) for row in rows],
            )

    return report


def write_complementarity_reports(rows: List[Dict[str, object]], k_values: List[int], output_dir: str | Path) -> Dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = complementarity_report(rows, k_values)
    json_path = output_dir / "complementarity_report.json"
    md_path = output_dir / "complementarity_report.md"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Four-Expert Complementarity Report\n\n")
        f.write(f"- Samples: {report['num_samples']}\n")
        f.write(f"- Malicious samples: {report['num_malicious']}\n")
        f.write(f"- Detected samples: {report['num_detected']}\n")
        f.write(f"- Ground truth available: {report['ground_truth_available']}\n\n")
        f.write("## Expert Metrics\n\n")
        for expert, metrics in report["experts"].items():
            f.write(f"### {expert}\n")
            f.write(f"- Score basis: {metrics['score_basis']}\n")
            f.write(f"- ROC-AUC: {metrics['roc_auc']}\n")
            f.write(f"- PR-AUC: {metrics['pr_auc']}\n")
            for k, values in metrics["topk"].items():
                f.write(
                    f"- K={k}: precision={values['precision']:.6f}, "
                    f"recall={values['recall']:.6f}, effective_k={values['k']}\n"
                )
            f.write("\n")
        if report["threshold_metrics"] is not None:
            f.write("## Threshold Metrics\n\n")
            for key, value in report["threshold_metrics"].items():
                f.write(f"- {key}: {value}\n")
        if report["gate"]:
            f.write("\n## Gate Routing\n\n")
            f.write(f"- Mean weights: {report['gate']['mean_weights']}\n")
            f.write(f"- Mean entropy: {report['gate']['mean_entropy']}\n")
        f.write("\n## Top-K Union Hits\n\n")
        for k, values in report["union_hits"].items():
            f.write(f"- K={k}: {values}\n")
        f.write("\n## Spearman Correlation\n\n")
        for name, value in report["spearman"].items():
            f.write(f"- {name}: {value}\n")
        if report["moe"]:
            f.write("\n## Fused MoE Result\n\n")
            f.write(f"- Score basis: {report['moe']['score_basis']}\n")
            f.write(f"- ROC-AUC: {report['moe']['roc_auc']}\n")
            f.write(f"- PR-AUC: {report['moe']['pr_auc']}\n")
            for k, values in report["moe"]["topk"].items():
                f.write(
                    f"- K={k}: precision={values['precision']:.6f}, "
                    f"recall={values['recall']:.6f}, effective_k={values['k']}\n"
                )
    return report
