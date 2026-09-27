import csv

from apt_moe.evaluation.calibration import EmpiricalCDFCalibrator
from apt_moe.evaluation.complementarity import complementarity_report
from apt_moe.evaluation.export_scores import CSV_COLUMNS, apply_calibrators, export_scores_csv


def test_score_export_schema(tmp_path):
    rows = [
        {
            "graph_id": "test:0",
            "sample_id": 1,
            "edge_id": "",
            "src_id": 1,
            "dst_id": 1,
            "timestamp": "",
            "split": "test",
            "label": 0,
            "semantic_raw": 0.1,
            "graph_raw": 0.2,
            "normal_raw": 0.3,
            "causal_raw": 0.4,
            "detection_granularity": "node",
        },
        {
            "graph_id": "test:0",
            "sample_id": 2,
            "edge_id": "",
            "src_id": 2,
            "dst_id": 2,
            "timestamp": "",
            "split": "test",
            "label": 1,
            "semantic_raw": 0.9,
            "graph_raw": 0.4,
            "normal_raw": 0.8,
            "causal_raw": 0.7,
            "detection_granularity": "node",
        },
    ]
    calibrators = {
        "semantic": EmpiricalCDFCalibrator.fit([0.0, 0.5]),
        "graph": EmpiricalCDFCalibrator.fit([0.1, 0.3]),
        "normal": EmpiricalCDFCalibrator.fit([0.2, 0.4]),
        "causal": EmpiricalCDFCalibrator.fit([0.3, 0.5]),
    }
    rows = apply_calibrators(rows, calibrators)
    out = tmp_path / "scores.csv"
    export_scores_csv(rows, out)
    with out.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == CSV_COLUMNS
        exported = list(reader)
    assert len(exported) == 2
    assert exported[0]["semantic_raw"] == "0.1"
    assert exported[1]["causal_raw"] == "0.7"


def test_complementarity_report_contains_required_sections():
    rows = []
    for i in range(6):
        rows.append(
            {
                "sample_id": i,
                "label": int(i in {1, 4}),
                "semantic_calibrated": float(i),
                "graph_calibrated": float(5 - i),
                "normal_calibrated": float(i % 3),
                "causal_calibrated": float((i + 1) % 4),
            }
        )
    report = complementarity_report(rows, [2, 100])
    assert "experts" in report
    assert "topk_overlap" in report
    assert "spearman" in report
    assert "union_hits" in report
    assert "single_expert_examples" in report
    assert set(report["experts"]) == {"semantic", "graph", "normal", "causal"}
