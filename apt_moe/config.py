from __future__ import annotations

import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

try:
    import yaml
except ImportError:  # pragma: no cover - ProvFusion requirements include PyYAML.
    yaml = None


DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 1,
    "device": "auto",
    "data": {
        "dataset_name": "synthetic",
        "data_path": "",
        "raw_data_dir": "../raw_data",
        "ground_truth_path": "",
        "allow_synthetic": False,
        "max_graphs_per_split": None,
    },
    "model": {
        "hidden_dim": 64,
        "expert_dim": 64,
        "num_layers": 2,
        "num_heads": 2,
        "dropout": 0.1,
    },
    "training": {
        "epochs": 20,
        "lr": 0.001,
        "weight_decay": 0.0,
        "patience": 5,
        "mask_rate": 0.3,
        "lambda_cos": 0.5,
        "lambda_mse": 0.5,
    },
    "evaluation": {
        "k_values": [100, 500, 1000],
        "mask_folds": 8,
    },
    "moe": {
        "alpha_entropy": 0.01,
        "beta_final": 1.0,
        "eta_balance": 0.0,
        "energy_ema_decay": 0.99,
        "energy_eps": 1e-6,
        "causal_smoothmax_tau": 5.0,
        "threshold_quantile": 0.999,
    },
    "output": {
        "output_dir": "outputs",
        "checkpoint_dir": "outputs/checkpoints",
        "scores_csv": "outputs/four_expert_scores.csv",
    },
}


def _deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    if yaml is None:
        with config_path.open("r", encoding="utf-8") as f:
            user_config = json.load(f)
    else:
        with config_path.open("r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}

    cfg = _deep_update(deepcopy(DEFAULT_CONFIG), user_config)
    cfg["_config_path"] = str(config_path)
    cfg["_config_dir"] = str(config_path.parent)
    return cfg


def select_device(device_cfg: str | int) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device_cfg, int):
        return torch.device(f"cuda:{device_cfg}" if device_cfg >= 0 else "cpu")
    if isinstance(device_cfg, str) and device_cfg.isdigit():
        return torch.device(f"cuda:{device_cfg}")
    return torch.device(str(device_cfg))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_output_dirs(cfg: Dict[str, Any]) -> None:
    Path(cfg["output"]["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["output"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
