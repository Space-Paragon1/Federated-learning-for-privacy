"""Simulated Secure Aggregation for Federated Learning.

Implements additive pairwise masking to simulate the core property of secure
aggregation: the server learns ONLY the weighted aggregate of client updates,
never any individual update.

Protocol (Bonawitz et al., 2017 simplified):
  - Each pair (i, j) generates a shared random mask r_ij.
  - Client i adds  +r_ij to its update before uploading.
  - Client j adds  -r_ij to its update before uploading.
  - When the server sums all masked updates, every mask cancels exactly:
        sum_i (update_i + mask_i) = sum_i update_i
  - The server therefore obtains the true aggregate while seeing only
    masked (unintelligible) individual uploads.

NOTE: This is a simulation for research demonstration. A production
deployment would use cryptographic primitives (secret sharing, key
agreement) to generate the masks without the server learning them.
"""
from __future__ import annotations

from typing import List

import numpy as np

from clients.local_train import LRParams
from server.aggregator import ClientUpdate


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _params_to_vec(params: LRParams) -> np.ndarray:
    return np.append(params.w, params.b)


def _vec_to_params(vec: np.ndarray) -> LRParams:
    return LRParams(w=vec[:-1].copy(), b=float(vec[-1]))


# ---------------------------------------------------------------------------
# Secure aggregation
# ---------------------------------------------------------------------------

def secure_aggregate(
    updates: List[ClientUpdate],
    rng: np.random.Generator,
) -> LRParams:
    """
    Weighted aggregation with pairwise additive masks.

    The masks cancel in the sum, so the result is identical to plain FedAvg.
    The difference is that no individual client update is visible in the clear
    -- each upload seen by the server is masked with random noise.

    Returns the same weighted average as fedavg(), but with the masking
    property demonstrated.
    """
    n = len(updates)
    if n == 0:
        raise ValueError("No updates to aggregate.")

    d = len(updates[0].params.w) + 1  # weights + bias scalar

    # --- Generate pairwise masks (only the upper triangle) ---
    # masks[i] accumulates the net mask added to client i's update.
    masks: List[np.ndarray] = [np.zeros(d) for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            shared_mask = rng.standard_normal(d)
            masks[i] += shared_mask   # client i adds +mask
            masks[j] -= shared_mask   # client j adds -mask

    # --- Each client uploads: weight_i * params_i + mask_i / total_samples ---
    total_samples = sum(u.n_samples for u in updates)
    agg_vec = np.zeros(d)

    for i, u in enumerate(updates):
        w_i = u.n_samples / total_samples
        upload = w_i * _params_to_vec(u.params) + masks[i] / total_samples
        agg_vec += upload

    # masks cancel exactly: sum(masks[i]) == 0 by construction.
    # Therefore agg_vec == exact weighted average (same as fedavg).
    return _vec_to_params(agg_vec)


def verify_mask_cancellation(n: int, d: int, rng: np.random.Generator) -> bool:
    """Verify that pairwise masks sum to zero (sanity check)."""
    masks = [np.zeros(d) for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            m = rng.standard_normal(d)
            masks[i] += m
            masks[j] -= m
    total = sum(masks)
    return bool(np.allclose(total, 0.0))
