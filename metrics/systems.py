"""System performance metrics for federated learning experiments.

Tracks per-round and aggregate metrics:
  - Wall-clock time per round
  - Communication cost (bytes uploaded / downloaded)
  - Client participation and dropout rates
  - Rounds needed to reach a convergence threshold
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Per-round system metrics
# ---------------------------------------------------------------------------

@dataclass
class RoundSystemMetrics:
    round_num: int
    wall_time_s: float
    bytes_broadcast: int    # server -> selected clients
    bytes_uploaded: int     # surviving clients -> server
    n_selected: int
    n_returned: int

    @property
    def total_bytes(self) -> int:
        return self.bytes_broadcast + self.bytes_uploaded

    @property
    def dropout_rate(self) -> float:
        return 1.0 - (self.n_returned / self.n_selected) if self.n_selected > 0 else 0.0

    @property
    def participation_rate(self) -> float:
        return self.n_returned / self.n_selected if self.n_selected > 0 else 0.0


# ---------------------------------------------------------------------------
# Aggregate summary
# ---------------------------------------------------------------------------

@dataclass
class SystemSummary:
    total_rounds: int
    total_wall_time_s: float
    total_comm_bytes: int
    cumulative_bytes_per_round: List[int]   # running total
    avg_participation_rate: float
    avg_dropout_rate: float
    rounds_to_convergence: int              # -1 if never converged


def compute_system_summary(
    round_metrics: List[RoundSystemMetrics],
    f1_per_round: Optional[List[float]] = None,
    convergence_f1: float = 0.70,
) -> SystemSummary:
    """
    Aggregate per-round system metrics into a summary.

    Parameters
    ----------
    round_metrics     : list of per-round system metrics.
    f1_per_round      : global F1 score at the end of each round.
    convergence_f1    : F1 threshold to declare convergence.
    """
    if not round_metrics:
        return SystemSummary(
            total_rounds=0,
            total_wall_time_s=0.0,
            total_comm_bytes=0,
            cumulative_bytes_per_round=[],
            avg_participation_rate=0.0,
            avg_dropout_rate=0.0,
            rounds_to_convergence=-1,
        )

    total_time = sum(m.wall_time_s for m in round_metrics)
    cumulative_bytes: List[int] = []
    running = 0
    for m in round_metrics:
        running += m.total_bytes
        cumulative_bytes.append(running)

    participation_rates = [m.participation_rate for m in round_metrics]
    dropout_rates = [m.dropout_rate for m in round_metrics]

    rounds_to_conv = -1
    if f1_per_round is not None:
        for i, f1 in enumerate(f1_per_round):
            if f1 >= convergence_f1:
                rounds_to_conv = i + 1
                break

    return SystemSummary(
        total_rounds=len(round_metrics),
        total_wall_time_s=float(total_time),
        total_comm_bytes=int(running),
        cumulative_bytes_per_round=cumulative_bytes,
        avg_participation_rate=float(np.mean(participation_rates)),
        avg_dropout_rate=float(np.mean(dropout_rates)),
        rounds_to_convergence=rounds_to_conv,
    )
