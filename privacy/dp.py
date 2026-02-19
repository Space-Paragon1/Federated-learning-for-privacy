"""Differential Privacy for Federated Learning.

Implements:
- Gradient/parameter clipping  (L2 sensitivity bounding)
- Gaussian noise mechanism
- Privacy accounting via Renyi Differential Privacy (RDP) -> (epsilon, delta)-DP

Reference: Abadi et al. (2016) "Deep Learning with Differential Privacy"
           Mironov (2017) "Renyi Differential Privacy of the Gaussian Mechanism"
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import numpy as np

from clients.local_train import LRParams


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DPConfig:
    """Differential privacy hyperparameters."""
    enabled: bool = True
    noise_multiplier: float = 1.1   # sigma = noise_multiplier * max_grad_norm
    max_grad_norm: float = 1.0      # L2 clip threshold (C)
    target_delta: float = 1e-5      # delta for (epsilon, delta)-DP


# ---------------------------------------------------------------------------
# Privacy Accountant  (Renyi DP moments accountant)
# ---------------------------------------------------------------------------

@dataclass
class PrivacyAccountant:
    """
    Tracks cumulative privacy cost using Renyi DP moments accountant.

    At each round, a random subset of clients (fraction q = client_frac) is
    sampled and their updates are perturbed with Gaussian noise (std = sigma).
    This is the 'Sampled Gaussian Mechanism' (SGM).

    RDP budget per step (Theorem 3, Wang et al. 2019, tight bound for alpha >= 2):
        RDP(alpha) <= alpha * q^2 / (2 * sigma^2)    [simplified upper bound]

    Final (epsilon, delta)-DP is obtained via the optimal RDP->DP conversion
    (Canonne et al. 2020):
        epsilon = min_alpha [ RDP(alpha) + log(1/delta) / (alpha - 1) ]
    """
    noise_multiplier: float
    target_delta: float
    client_frac: float      # fraction of clients sampled per round

    _orders: List[float] = field(default_factory=list, init=False)
    _rdp: np.ndarray = field(default=None, init=False)
    _steps: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        # Standard Renyi orders used in the literature
        self._orders = list(range(2, 64)) + [128, 256, 512]
        self._rdp = np.zeros(len(self._orders))

    def _rdp_per_step(self, alpha: float) -> float:
        """
        Upper bound on RDP(alpha) for one step of the Sampled Gaussian Mechanism.

        Uses the simple bound: RDP(alpha) <= alpha * q^2 / (2 * sigma^2)
        which holds for sigma >= 1 and is tight for small q.
        """
        q = self.client_frac
        sigma = self.noise_multiplier
        if sigma == 0:
            return float("inf")
        return alpha * (q ** 2) / (2.0 * sigma ** 2)

    def _rdp_to_dp(self) -> float:
        """
        Convert accumulated RDP to (epsilon, delta)-DP via optimal conversion.

        epsilon = min_alpha [ RDP(alpha) + log(1/delta) / (alpha - 1) ]
        """
        delta = self.target_delta
        best_eps = float("inf")
        for i, alpha in enumerate(self._orders):
            if alpha <= 1:
                continue
            rdp = self._rdp[i]
            if rdp == float("inf"):
                continue
            eps = rdp + math.log(1.0 / delta) / (alpha - 1.0)
            if eps < best_eps:
                best_eps = eps
        return best_eps

    def step(self) -> None:
        """Record one training round (adds per-round RDP to running total)."""
        for i, alpha in enumerate(self._orders):
            self._rdp[i] += self._rdp_per_step(alpha)
        self._steps += 1

    @property
    def epsilon(self) -> float:
        """Current cumulative privacy budget epsilon."""
        if self._steps == 0:
            return 0.0
        return self._rdp_to_dp()

    @property
    def rounds_completed(self) -> int:
        return self._steps


# ---------------------------------------------------------------------------
# Clipping and noise
# ---------------------------------------------------------------------------

def _params_to_vec(params: LRParams) -> np.ndarray:
    return np.append(params.w, params.b)


def _vec_to_params(vec: np.ndarray) -> LRParams:
    return LRParams(w=vec[:-1].copy(), b=float(vec[-1]))


def clip_update(delta: LRParams, max_norm: float) -> LRParams:
    """
    Clip a model update (delta) to have L2 norm at most max_norm.
    This bounds the sensitivity of each client contribution.
    """
    vec = _params_to_vec(delta)
    norm = float(np.linalg.norm(vec))
    if norm > max_norm:
        vec = vec * (max_norm / norm)
    return _vec_to_params(vec)


def add_gaussian_noise(
    params: LRParams,
    noise_multiplier: float,
    max_grad_norm: float,
    rng: np.random.Generator,
) -> LRParams:
    """
    Add calibrated Gaussian noise to model parameters.
    Noise std = noise_multiplier * max_grad_norm  (equals sigma * C).
    """
    std = noise_multiplier * max_grad_norm
    vec = _params_to_vec(params)
    vec = vec + rng.normal(0.0, std, size=vec.shape)
    return _vec_to_params(vec)


def apply_dp_noise(
    params: LRParams,
    dp_config: DPConfig,
    rng: np.random.Generator,
) -> LRParams:
    """Add Gaussian noise to already-aggregated params (server-side DP)."""
    if not dp_config.enabled:
        return params
    return add_gaussian_noise(
        params, dp_config.noise_multiplier, dp_config.max_grad_norm, rng
    )
