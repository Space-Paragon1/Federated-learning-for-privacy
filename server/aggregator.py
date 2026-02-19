"""Server-side aggregation methods for federated learning.

Implemented aggregators:
  - fedavg          : Weighted average by number of samples (McMahan et al., 2017).
  - coord_median    : Coordinate-wise median -- Byzantine-robust.
  - trimmed_mean    : Coordinate-wise trimmed mean -- Byzantine-robust.
  - krum            : Krum selection -- Byzantine-robust (Blanchard et al., 2017).

All functions accept List[ClientUpdate] and return LRParams.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from clients.local_train import LRParams


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ClientUpdate:
    params: LRParams
    n_samples: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_matrix(updates: List[ClientUpdate]) -> np.ndarray:
    """Stack client parameter vectors into a (n_clients, d+1) matrix."""
    vecs = [np.append(u.params.w, u.params.b) for u in updates]
    return np.stack(vecs, axis=0)   # shape (n, d+1)


def _from_vec(vec: np.ndarray) -> LRParams:
    return LRParams(w=vec[:-1].copy(), b=float(vec[-1]))


# ---------------------------------------------------------------------------
# Aggregators
# ---------------------------------------------------------------------------

def fedavg(updates: List[ClientUpdate]) -> LRParams:
    """
    Federated Averaging: weighted mean by number of local training samples.

    Reference: McMahan et al. (2017) "Communication-Efficient Learning of
    Deep Networks from Decentralized Data".
    """
    total = sum(u.n_samples for u in updates)
    if total == 0:
        raise ValueError("No samples in updates.")

    w_agg = None
    b_agg = 0.0
    for u in updates:
        weight = u.n_samples / total
        if w_agg is None:
            w_agg = weight * u.params.w
        else:
            w_agg += weight * u.params.w
        b_agg += weight * u.params.b

    return LRParams(w=w_agg, b=float(b_agg))


def coord_median(updates: List[ClientUpdate]) -> LRParams:
    """
    Coordinate-wise median aggregation.

    For each parameter dimension, take the median over client updates.
    Robust to up to floor((n-1)/2) Byzantine clients (Yin et al., 2018).
    """
    if not updates:
        raise ValueError("No updates.")
    mat = _to_matrix(updates)          # (n, d+1)
    median_vec = np.median(mat, axis=0)
    return _from_vec(median_vec)


def trimmed_mean(updates: List[ClientUpdate], trim_frac: float = 0.1) -> LRParams:
    """
    Coordinate-wise trimmed mean.

    Removes the top and bottom `trim_frac` fraction of values in each
    coordinate before averaging.  Robust to bounded Byzantine fraction.

    Parameters
    ----------
    trim_frac : fraction of clients to trim from each tail (e.g., 0.1 = 10%).
    """
    if not updates:
        raise ValueError("No updates.")
    mat = _to_matrix(updates)          # (n, d+1)
    n = mat.shape[0]
    k = max(1, int(np.floor(n * trim_frac)))

    sorted_mat = np.sort(mat, axis=0)
    # Keep rows from index k to n-k
    trimmed = sorted_mat[k: n - k, :]
    if len(trimmed) == 0:
        trimmed = sorted_mat   # fall back if too few clients
    return _from_vec(trimmed.mean(axis=0))


def krum(updates: List[ClientUpdate], n_byzantine: int = 1) -> LRParams:
    """
    Krum aggregation: selects the single client update closest to its
    (n - n_byzantine - 2) nearest neighbors.

    The selected update is used as the new global model.
    Robust to n_byzantine malicious clients (Blanchard et al., 2017).

    Parameters
    ----------
    n_byzantine : assumed upper bound on number of malicious clients (f).
                  Must satisfy: 2*f + 2 < n_clients.
    """
    n = len(updates)
    if n == 0:
        raise ValueError("No updates.")
    if n == 1:
        return updates[0].params

    mat = _to_matrix(updates)   # (n, d+1)
    k = n - n_byzantine - 2     # number of neighbors to consider
    k = max(1, min(k, n - 1))

    # Pairwise squared distances
    scores = np.zeros(n)
    for i in range(n):
        dists = np.sum((mat - mat[i]) ** 2, axis=1)
        dists[i] = np.inf                           # exclude self
        nearest = np.sort(dists)[:k]
        scores[i] = nearest.sum()

    # Select client with minimum Krum score
    best = int(np.argmin(scores))
    return updates[best].params


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def aggregate(
    updates: List[ClientUpdate],
    method: str = "fedavg",
    **kwargs,
) -> LRParams:
    """
    Dispatch to the chosen aggregation method.

    Parameters
    ----------
    method : one of 'fedavg', 'coord_median', 'trimmed_mean', 'krum'.
    kwargs : extra arguments forwarded to the selected method.
    """
    if method == "fedavg":
        return fedavg(updates)
    elif method == "coord_median":
        return coord_median(updates)
    elif method == "trimmed_mean":
        return trimmed_mean(updates, **kwargs)
    elif method == "krum":
        return krum(updates, **kwargs)
    else:
        raise ValueError(f"Unknown aggregation method: '{method}'. "
                         f"Choose from: fedavg, coord_median, trimmed_mean, krum.")
