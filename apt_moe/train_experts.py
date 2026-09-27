from __future__ import annotations

import argparse
from pathlib import Path

from apt_moe.config import ensure_output_dirs, load_config, save_json, select_device, set_seed
from apt_moe.data import load_provfusion_dataset
from apt_moe.training import train_joint_moe


def main() -> None:
    parser = argparse.ArgumentParser(description="Jointly train the four-expert APT-MoE detector.")
    parser.add_argument("--config", required=True, help="YAML/JSON config path.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_output_dirs(cfg)
    set_seed(int(cfg["seed"]))
    device = select_device(cfg["device"])
    dataset = load_provfusion_dataset(cfg)

    print(f"[joint-moe] device={device}")
    print(f"[joint-moe] dataset_metadata={dataset.metadata()}")
    result = train_joint_moe(dataset, cfg, device)

    run_meta = {
        "config_path": args.config,
        "device": str(device),
        "dataset": dataset.metadata(),
        "checkpoint": result,
        "ground_truth_used_for_training": False,
        "training_mode": "end_to_end_joint_single_optimizer",
        "note": "Synthetic fixture is for code smoke testing only when data.allow_synthetic=true.",
    }
    save_json(run_meta, Path(cfg["output"]["output_dir"]) / "train_run_metadata.json")


if __name__ == "__main__":
    main()
