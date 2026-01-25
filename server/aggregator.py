from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from clients.local_train import LRParams


@dataclass
class ClientUpdate:
    params: LRParams
    n_samples: int


def fedavg(updates: List[ClientUpdate]) -> LRParams:
    """
    Weighted average of client parameters by number of samples.
    """
    total = sum(u.n_samples for u in updates)
    if total == 0:
        raise ValueError("No samples in updates.")

    w_sum = None
    b_sum = 0.0

    for u in updates:
        weight = u.n_samples / total
        if w_sum is None:
            w_sum = weight * u.params.w
        else:
            w_sum += weight * u.params.w
        b_sum += weight * u.params.b

    return LRParams(w=w_sum, b=float(b_sum))
