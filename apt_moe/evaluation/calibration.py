from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np


@dataclass
class EmpiricalCDFCalibrator:
    sorted_scores: np.ndarray

    @classmethod
    def fit(cls, scores: Iterable[float]) -> "EmpiricalCDFCalibrator":
        arr = np.asarray(list(scores), dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            arr = np.asarray([0.0], dtype=float)
        return cls(sorted_scores=np.sort(arr))

    def transform(self, scores: Iterable[float]) -> np.ndarray:
        arr = np.asarray(list(scores), dtype=float)
        ranks = np.searchsorted(self.sorted_scores, arr, side="right")
        return ranks / max(len(self.sorted_scores), 1)

    def to_dict(self) -> Dict[str, object]:
        return {
            "count": int(len(self.sorted_scores)),
            "min": float(self.sorted_scores[0]),
            "max": float(self.sorted_scores[-1]),
        }
