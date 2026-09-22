from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np


def binary_auc_metrics(labels: Iterable[int], scores: Iterable[float]) -> Dict[str, float | None]:
    y = np.asarray(list(labels), dtype=int)
    s = np.asarray(list(scores), dtype=float)
    if len(np.unique(y)) < 2:
        return {"roc_auc": None, "pr_auc": None}
    return {"roc_auc": _roc_auc(y, s), "pr_auc": _average_precision(y, s)}


def topk_metrics(labels: Iterable[int], scores: Iterable[float], k_values: Iterable[int]) -> Dict[str, Dict[str, float]]:
    y = np.asarray(list(labels), dtype=int)
    s = np.asarray(list(scores), dtype=float)
    order = np.argsort(-s)
    total_pos = max(int(y.sum()), 1)
    out: Dict[str, Dict[str, float]] = {}
    for k in k_values:
        kk = min(int(k), len(y))
        if kk <= 0:
            out[str(k)] = {"k": 0, "precision": 0.0, "recall": 0.0}
            continue
        hit = int(y[order[:kk]].sum())
        out[str(k)] = {"k": kk, "precision": hit / kk, "recall": hit / total_pos}
    return out


def jaccard(a: set, b: set) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def spearman(x: Iterable[float], y: Iterable[float]) -> float | None:
    xa = np.asarray(list(x), dtype=float)
    ya = np.asarray(list(y), dtype=float)
    if xa.size < 2 or np.std(xa) == 0 or np.std(ya) == 0:
        return None
    xr = _average_ranks(xa)
    yr = _average_ranks(ya)
    value = np.corrcoef(xr, yr)[0, 1]
    return None if np.isnan(value) else float(value)


def topk_ids(rows: List[Dict[str, object]], score_key: str, k: int) -> set:
    ordered = sorted(rows, key=lambda row: float(row[score_key]), reverse=True)
    return {int(row["sample_id"]) for row in ordered[: min(k, len(ordered))]}


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    ranks = _average_ranks(scores)
    pos = labels == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    pos_rank_sum = ranks[pos].sum()
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / max(n_pos * n_neg, 1)
    return float(auc)


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    total_pos = int(sorted_labels.sum())
    if total_pos == 0:
        return 0.0
    tp = np.cumsum(sorted_labels)
    precision = tp / (np.arange(len(sorted_labels)) + 1)
    return float((precision * sorted_labels).sum() / total_pos)
