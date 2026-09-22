from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import torch


def random_node_mask(num_nodes: int, mask_rate: float, device: torch.device) -> torch.Tensor:
    count = max(1, int(round(num_nodes * mask_rate)))
    count = min(count, num_nodes)
    return torch.randperm(num_nodes, device=device)[:count]


def sample_edges(edge_indices: torch.Tensor, max_edges: Optional[int]) -> torch.Tensor:
    if max_edges is None or edge_indices.numel() <= max_edges:
        return edge_indices
    perm = torch.randperm(edge_indices.numel(), device=edge_indices.device)[: int(max_edges)]
    return edge_indices[perm]


def run_training_loop(
    model: torch.nn.Module,
    train_step: Callable[[torch.nn.Module, str], torch.Tensor],
    val_step: Callable[[torch.nn.Module, str], torch.Tensor],
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    checkpoint_path: str | Path,
    model_meta: Dict[str, object],
) -> Dict[str, object]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = -1
    wait = 0
    history: List[Dict[str, float]] = []

    for epoch in range(int(epochs)):
        start = time.time()
        model.train()
        train_loss = train_step(model, "train")
        optimizer.zero_grad()
        if train_loss.requires_grad:
            train_loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = val_step(model, "val")

        train_value = float(train_loss.detach().cpu())
        val_value = float(val_loss.detach().cpu())
        elapsed = time.time() - start
        history.append({"epoch": epoch, "train_loss": train_value, "val_loss": val_value, "seconds": elapsed})
        print(
            f"[{model_meta['expert']}] epoch={epoch} train_loss={train_value:.6f} "
            f"val_loss={val_value:.6f} seconds={elapsed:.2f}"
        )

        if val_value < best_val:
            best_val = val_value
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= int(patience):
                break

    model.load_state_dict(best_state)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_meta": model_meta,
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
            "history": history,
        },
        checkpoint_path,
    )
    return {
        "checkpoint": str(checkpoint_path),
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "history": history,
    }


def mean_or_zero(losses: List[torch.Tensor], device: torch.device) -> torch.Tensor:
    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()
