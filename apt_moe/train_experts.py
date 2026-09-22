from __future__ import annotations

import argparse
from pathlib import Path

from apt_moe.config import ensure_output_dirs, load_config, save_json, select_device, set_seed
from apt_moe.data import load_provfusion_dataset
from apt_moe.training import train_attribute_expert, train_edge_expert, train_node_expert


TRAINERS = {
    "node": train_node_expert,
    "edge": train_edge_expert,
    "attr": train_attribute_expert,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train three independent self-supervised ProvFusion experts.")
    parser.add_argument("--config", required=True, help="YAML/JSON config path.")
    parser.add_argument(
        "--experts",
        nargs="*",
        choices=sorted(TRAINERS),
        default=None,
        help="Optional subset of experts to train.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_output_dirs(cfg)
    set_seed(int(cfg["seed"]))
    device = select_device(cfg["device"])
    dataset = load_provfusion_dataset(cfg)
    experts = args.experts or cfg["training"]["experts"]

    print(f"[three-expert] device={device}")
    print(f"[three-expert] dataset_metadata={dataset.metadata()}")
    results = {}
    for expert in experts:
        results[expert] = TRAINERS[expert](dataset, cfg, device)

    run_meta = {
        "config_path": args.config,
        "device": str(device),
        "dataset": dataset.metadata(),
        "checkpoints": results,
        "note": "Synthetic fixture is for code smoke testing only when data.allow_synthetic=true.",
    }
    save_json(run_meta, Path(cfg["output"]["output_dir"]) / "train_run_metadata.json")


if __name__ == "__main__":
    main()
