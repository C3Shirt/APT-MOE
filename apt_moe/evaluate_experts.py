from __future__ import annotations

import argparse
from pathlib import Path

from apt_moe.config import ensure_output_dirs, load_config, save_json, select_device, set_seed
from apt_moe.data import load_provfusion_dataset
from apt_moe.evaluation import (
    apply_threshold,
    calibrate_threshold,
    export_scores_csv,
    load_joint_model,
    score_dataset_node_level,
    threshold_calibration_source,
    write_complementarity_reports,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the jointly trained four-expert APT-MoE model.")
    parser.add_argument("--config", required=True, help="YAML/JSON config path.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_output_dirs(cfg)
    set_seed(int(cfg["seed"]))
    device = select_device(cfg["device"])
    dataset = load_provfusion_dataset(cfg)
    model = load_joint_model(dataset, cfg, device)

    split_scores = score_dataset_node_level(dataset, model, cfg, device)
    rows = [row for split_rows in split_scores.values() for row in split_rows]
    threshold_quantile = float(cfg.get("moe", {}).get("threshold_quantile", 0.999))
    threshold = calibrate_threshold(rows, threshold_quantile)
    _, threshold_source = threshold_calibration_source(rows)
    apply_threshold(rows, threshold)

    scores_path = cfg["output"]["scores_csv"]
    export_scores_csv(rows, scores_path)
    test_rows = [row for row in rows if row["split"] == "test"] or rows
    report = write_complementarity_reports(
        test_rows,
        [int(k) for k in cfg["evaluation"]["k_values"]],
        cfg["output"]["output_dir"],
    )
    save_json(
        {
            "scores_csv": scores_path,
            "threshold": threshold,
            "threshold_quantile": threshold_quantile,
            "threshold_source": threshold_source,
            "dataset": dataset.metadata(),
            "evaluation_summary": report,
        },
        Path(cfg["output"]["output_dir"]) / "evaluation_run_metadata.json",
    )
    print(
        f"[joint-moe] wrote {scores_path} threshold={threshold:.6f} "
        f"detected={report['num_detected']}/{report['num_samples']}"
    )


if __name__ == "__main__":
    main()
