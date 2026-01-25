from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np
from metrics.utility import binary_classification_metrics

@dataclass(frozen=True)
class ClientFairness:
    worst_f1: float
    best_f1: float
    mean_f1: float

def per_client_f1(client_to_ytrue: Dict[int, np.ndarray], client_to_ypred: Dict[int, np.ndarray]) -> ClientFairness:
    f1s = []
    for cid in client_to_ytrue:
        m = binary_classification_metrics(client_to_ytrue[cid], client_to_ypred[cid])
        f1s.append(m.f1)
    if not f1s:
        return ClientFairness(worst_f1=0.0, best_f1=0.0, mean_f1=0.0)
    return ClientFairness(worst_f1=float(min(f1s)), best_f1=float(max(f1s)), mean_f1=float(np.mean(f1s)))
