"""Gradient Compression for Communication-Efficient Federated Learning.

Implements stochastic quantization (QSGD) to reduce the bit-width of
transmitted model updates.  This trades a small accuracy penalty for a
large reduction in communication cost — important when bandwidth is limited
(e.g., mobile devices, hospitals on slow connections).

Key insight: combining compression WITH differential privacy is safe because
the quantisation noise is orthogonal to the privacy guarantee (the DP noise
is added on top of the already-quantised update).

Reference: Alistarh et al. (2017) "QSGD: Communication-Efficient SGD via
           Gradient Quantization and Encoding" (NeurIPS 2017).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from clients.local_train import LRParams


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CompressionConfig:
    """Hyperparameters for stochastic gradient quantisation."""
    enabled: bool = True
    num_levels: int = 16   # quantisation levels; 16 ≈ 4-bit, 256 ≈ 8-bit

    @property
    def bits(self) -> int:
        """Effective bit-width (log2 of levels)."""
        return max(1, int(np.floor(np.log2(self.num_levels))))

    @property
    def compression_ratio(self) -> float:
        """Approximate ratio of original bytes saved (float64 baseline)."""
        return 64.0 / max(1, self.bits)

    def __str__(self) -> str:
        return f"QSGD-{self.bits}bit ({self.num_levels} levels)"


# ---------------------------------------------------------------------------
# Core quantisation
# ---------------------------------------------------------------------------

def stochastic_quantize(
    vec: np.ndarray,
    num_levels: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    QSGD unbiased stochastic quantisation.

    Each element v_i is mapped to a discrete value in
    [-||v||_2, +||v||_2] using stochastic rounding so that
    E[quantize(v)] = v  (unbiasedness).

    The resulting vector uses only `log2(num_levels)` bits per element
    instead of 64 bits (float64), giving a compression_ratio of
    64 / log2(num_levels) in the number of bits transmitted.

    Parameters
    ----------
    vec        : flat parameter or gradient vector to quantise.
    num_levels : number of quantisation levels (s).  s=2 = 1-bit sign,
                 s=16 = 4-bit, s=256 = 8-bit.
    rng        : PRNG for stochastic rounding (reproducible).

    Returns
    -------
    Quantised vector with the same shape and dtype as `vec`,
    but with values restricted to `num_levels` discrete magnitudes.
    """
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12 or num_levels <= 1:
        return vec.copy()

    s = float(num_levels - 1)           # number of intervals
    scaled = np.abs(vec) * s / norm     # scale to [0, s]

    floor_v = np.floor(scaled)          # lower quantisation level
    prob_up = scaled - floor_v          # P(round up)

    # Stochastic rounding: each element independently rounded up/down
    round_up = rng.random(size=vec.shape) < prob_up
    quantised_levels = floor_v + round_up.astype(float)

    # Reconstruct: scale back and restore sign
    return norm * np.sign(vec) * (quantised_levels / s)


# ---------------------------------------------------------------------------
# Top-k sparsification (alternative compression)
# ---------------------------------------------------------------------------

def top_k_sparsify(
    vec: np.ndarray,
    k_frac: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Top-k sparsification: keep only the k largest-magnitude elements.

    Transmit only the indices and values of the top-k elements.
    The rest are zeroed out.  Reduces communication by ~(1 - k_frac) factor.

    Parameters
    ----------
    k_frac : fraction of elements to keep (e.g., 0.1 = top 10%).

    Returns
    -------
    (sparse_vec, top_indices)
    """
    k = max(1, int(np.floor(len(vec) * k_frac)))
    top_idx = np.argsort(np.abs(vec))[-k:]   # indices of top-k
    sparse = np.zeros_like(vec)
    sparse[top_idx] = vec[top_idx]
    return sparse, top_idx


# ---------------------------------------------------------------------------
# Convenience wrappers for LRParams
# ---------------------------------------------------------------------------

def compress_params(
    params: LRParams,
    cfg: CompressionConfig,
    rng: np.random.Generator,
) -> LRParams:
    """
    Apply stochastic quantisation to a full set of LR parameters.

    Used client-side: the client quantises its update before uploading,
    reducing the bytes sent to the server.
    """
    if not cfg.enabled:
        return params

    vec = np.append(params.w, params.b)
    q_vec = stochastic_quantize(vec, cfg.num_levels, rng)
    return LRParams(w=q_vec[:-1].copy(), b=float(q_vec[-1]))


def compressed_bytes(original_bytes: int, cfg: CompressionConfig) -> int:
    """Estimated bytes after compression (approximate)."""
    if not cfg.enabled:
        return original_bytes
    return max(1, int(round(original_bytes / cfg.compression_ratio)))


# ---------------------------------------------------------------------------
# Compression statistics helper
# ---------------------------------------------------------------------------

def quantisation_snr(original: np.ndarray, quantised: np.ndarray) -> float:
    """
    Signal-to-noise ratio of quantisation (dB).

    Higher SNR = less information lost in compression.
    SNR ≥ 20 dB is generally acceptable for model updates.
    """
    signal_power = float(np.mean(original ** 2))
    noise_power = float(np.mean((original - quantised) ** 2))
    if noise_power < 1e-15:
        return float("inf")
    return float(10.0 * np.log10(signal_power / noise_power))
