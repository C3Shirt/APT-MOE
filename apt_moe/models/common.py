from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class ExpertOutput:
    residual: torch.Tensor
    score: torch.Tensor
    loss: torch.Tensor
    aux: Dict[str, object]
