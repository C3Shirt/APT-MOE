from __future__ import annotations

import argparse
from pathlib import Path

from apt_moe.config import ensure_output_dirs, load_config, save_json, select_device, set_seed
from apt_moe.data import load_provfusion_dataset
from apt_moe.evaluation import (
    export_scores_csv,
    load_experts,
    score_dataset_node_level,
    write_complementarity_reports,
)
from apt_moe.evaluation.export_scores import apply_calibrators, fit_calibrators


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate three independent ProvFusion experts.")
    parser.add_argument("--config", required=True, help="YAML/JSON config path.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_output_dirs(cfg)
    set_seed(int(cfg["seed"]))
    device = select_device(cfg["device"])
    dataset = load_provfusion_dataset(cfg)
    models = load_experts(dataset, cfg, device)

    split_scores = score_dataset_node_level(dataset, models, cfg, device)
    calibrators = fit_calibrators(split_scores, cfg)
    rows = []
    for split_rows in split_scores.values():
        rows.extend(split_rows)
    rows = apply_calibrators(rows, calibrators)

    scores_path = cfg["output"]["scores_csv"]
    export_scores_csv(rows, scores_path)
    test_rows = [row for row in rows if row["split"] == "test"]
    if not test_rows:
        test_rows = rows
    report = write_complementarity_reports(
        test_rows,
        [int(k) for k in cfg["evaluation"]["k_values"]],
        cfg["output"]["output_dir"],
    )
    save_json(
        {
            "scores_csv": scores_path,
            "calibrators": {name: calibrator.to_dict() for name, calibrator in calibrators.items()},
            "dataset": dataset.metadata(),
            "complementarity_summary": {
                "num_samples": report["num_samples"],
                "num_malicious": report["num_malicious"],
            },
        },
        Path(cfg["output"]["output_dir"]) / "evaluation_run_metadata.json",
    )
    print(f"[three-expert] wrote {scores_path}")


if __name__ == "__main__":
    main()
