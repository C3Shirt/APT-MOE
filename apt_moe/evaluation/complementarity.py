from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .metrics import binary_auc_metrics, jaccard, spearman, topk_ids, topk_metrics


EXPERT_SCORE_KEYS = {
    "node": "node_calibrated",
    "edge": "edge_calibrated",
    "attr": "attr_calibrated",
}


def complementarity_report(rows: List[Dict[str, object]], k_values: List[int]) -> Dict[str, object]:
    labels = [int(row["label"]) for row in rows]
    malicious_ids = {int(row["sample_id"]) for row in rows if int(row["label"]) == 1}
    report: Dict[str, object] = {
        "num_samples": len(rows),
        "num_malicious": len(malicious_ids),
        "experts": {},
        "topk_overlap": {},
        "spearman": {},
        "union_hits": {},
        "single_expert_examples": {},
    }

    for expert, key in EXPERT_SCORE_KEYS.items():
        scores = [float(row[key]) for row in rows]
        expert_report = binary_auc_metrics(labels, scores)
        expert_report["topk"] = topk_metrics(labels, scores, k_values)
        report["experts"][expert] = expert_report

    for k in k_values:
        kk = min(int(k), len(rows))
        sets = {expert: topk_ids(rows, key, kk) for expert, key in EXPERT_SCORE_KEYS.items()}
        report["topk_overlap"][str(k)] = {
            "node_edge": jaccard(sets["node"], sets["edge"]),
            "node_attr": jaccard(sets["node"], sets["attr"]),
            "edge_attr": jaccard(sets["edge"], sets["attr"]),
        }
        hits = {expert: sets[expert] & malicious_ids for expert in sets}
        report["union_hits"][str(k)] = {
            "node": len(hits["node"]),
            "edge": len(hits["edge"]),
            "attr": len(hits["attr"]),
            "node_edge_union": len(hits["node"] | hits["edge"]),
            "node_attr_union": len(hits["node"] | hits["attr"]),
            "edge_attr_union": len(hits["edge"] | hits["attr"]),
            "oracle_union": len(hits["node"] | hits["edge"] | hits["attr"]),
        }
        report["single_expert_examples"][str(k)] = {
            expert: sorted(list(hits[expert] - set().union(*(hits[e] for e in hits if e != expert))))[:20]
            for expert in hits
        }

    for left, right in (("node", "edge"), ("node", "attr"), ("edge", "attr")):
        report["spearman"][f"{left}_{right}"] = spearman(
            [float(row[EXPERT_SCORE_KEYS[left]]) for row in rows],
            [float(row[EXPERT_SCORE_KEYS[right]]) for row in rows],
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
        f.write("# Three-Expert Complementarity Report\n\n")
        f.write(f"- Samples: {report['num_samples']}\n")
        f.write(f"- Malicious samples: {report['num_malicious']}\n\n")
        f.write("## Independent Expert Metrics\n\n")
        for expert, metrics in report["experts"].items():
            f.write(f"### {expert}\n")
            f.write(f"- ROC-AUC: {metrics['roc_auc']}\n")
            f.write(f"- PR-AUC: {metrics['pr_auc']}\n")
            for k, values in metrics["topk"].items():
                f.write(
                    f"- K={k}: precision={values['precision']:.6f}, "
                    f"recall={values['recall']:.6f}, effective_k={values['k']}\n"
                )
            f.write("\n")
        f.write("## Top-K Union Hits\n\n")
        for k, values in report["union_hits"].items():
            f.write(f"- K={k}: {values}\n")
        f.write("\n## Spearman Correlation\n\n")
        for name, value in report["spearman"].items():
            f.write(f"- {name}: {value}\n")
    return report
