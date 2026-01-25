from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass(frozen=True)
class ScheduleResult:
    selected: List[int]
    survivors: List[int]  # clients that actually return updates

def sample_clients(rng: np.random.Generator, client_ids: List[int], frac: float) -> List[int]:
    m = max(1, int(len(client_ids) * frac))
    return list(rng.choice(client_ids, size=m, replace=False))

def apply_dropout(rng: np.random.Generator, selected: List[int], dropout_rate: float) -> List[int]:
    if dropout_rate <= 0:
        return selected
    survivors = [cid for cid in selected if rng.uniform() >= dropout_rate]
    # Ensure at least 1 update returns
    return survivors if survivors else [selected[int(rng.integers(0, len(selected)))]]

def schedule_round(
    rng: np.random.Generator,
    client_ids: List[int],
    frac: float,
    dropout_rate: float
) -> ScheduleResult:
    selected = sample_clients(rng, client_ids, frac)
    survivors = apply_dropout(rng, selected, dropout_rate)
    return ScheduleResult(selected=selected, survivors=survivors)
