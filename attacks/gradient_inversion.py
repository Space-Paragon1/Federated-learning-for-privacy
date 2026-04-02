"""Gradient Inversion Attack for Federated Learning.

Demonstrates that raw gradient sharing (without differential privacy) leaks
private training data.  The server — or a network eavesdropper — can
analytically reconstruct individual training samples directly from the
gradients a client uploads.

Attack Description
------------------
For logistic regression on a single training sample (x, y), the gradient is:

    ∂L/∂w  =  x · (σ(w·x + b) − y)   →  g_w
    ∂L/∂b  =      σ(w·x + b) − y      →  g_b

Dividing:   x  =  g_w / g_b   (exact, when g_b ≠ 0)

This means ANY single-sample gradient broadcast by a client directly
reveals the raw feature vector x — no ML attack needed.

Impact of Differential Privacy
-------------------------------
DP adds Gaussian noise N(0, σ²) to every gradient before transmission.
The reconstructed x̂ = (g_w + ε_w) / (g_b + ε_b) becomes noisy:
• Low DP noise → high reconstruction fidelity (privacy breach)
• High DP noise → reconstruction collapses to noise (private)

This module quantifies the reconstruction quality both with and without DP,
providing empirical support for using DP-FedAvg.

Reference: Zhu et al. (2019) "Deep Leakage from Gradients" (NeurIPS 2019)
           Zhao et al. (2019) "iDLG: Improved Deep Leakage from Gradients"
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from clients.local_train import LRParams


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InversionResult:
    """Outcome of one gradient inversion attempt."""
    # Cosine similarity between recovered and true feature vector [−1, 1]
    cosine_similarity: float
    # Normalised MSE: ||x̂ − x||² / ||x||²
    normalised_mse: float
    # Fraction of feature signs correctly recovered (directional accuracy)
    sign_accuracy: float
    # Whether the attack was analytically exact (single-sample gradient)
    is_exact: bool
    # Number of features
    n_features: int


@dataclass
class LeakageReport:
    """Aggregate leakage statistics across many samples."""
    mean_cosine_similarity: float
    std_cosine_similarity: float
    mean_normalised_mse: float
    mean_sign_accuracy: float
    n_evaluated: int
    dp_noise_std: float       # 0.0 = no DP
    reconstruction_risk: str  # 'HIGH' / 'MEDIUM' / 'LOW'

    def summary(self) -> str:
        return (
            f"Gradient Leakage Report  (DP noise σ={self.dp_noise_std:.3f})\n"
            f"  Samples evaluated   : {self.n_evaluated}\n"
            f"  Mean cosine sim     : {self.mean_cosine_similarity:.3f}  "
            f"(1.0 = perfect reconstruction)\n"
            f"  Mean normalised MSE : {self.mean_normalised_mse:.4f}  "
            f"(0 = perfect reconstruction)\n"
            f"  Sign accuracy       : {self.mean_sign_accuracy:.3f}  "
            f"(feature directions)\n"
            f"  Reconstruction risk : {self.reconstruction_risk}\n"
        )


# ---------------------------------------------------------------------------
# Core analytical attack
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _compute_single_gradient(
    params: LRParams,
    x: np.ndarray,     # shape (n_features,)
    y: float,
) -> np.ndarray:
    """Gradient of binary cross-entropy on a single sample."""
    residual = float(_sigmoid(np.dot(params.w, x) + params.b) - y)
    g_w = x * residual
    g_b = residual
    return np.append(g_w, g_b)    # shape (n_features + 1,)


def invert_single_gradient(
    gradient: np.ndarray,
    n_features: int,
) -> InversionResult:
    """
    Analytically recover a training sample from its single-sample gradient.

    For logistic regression:
        x̂  =  g_w / g_b   when g_b ≠ 0

    This is an EXACT reconstruction (up to floating-point precision) when
    the gradient comes from a single un-modified sample.

    Parameters
    ----------
    gradient  : gradient vector (g_w concatenated with g_b), length n+1.
    n_features: number of input features (n).

    Returns
    -------
    InversionResult with cosine_similarity, normalised_mse, sign_accuracy.
    """
    g_w = gradient[:n_features]
    g_b = float(gradient[n_features])

    is_exact = abs(g_b) > 1e-10
    if is_exact:
        x_hat = g_w / g_b
    else:
        # Fall back: use gradient magnitude as a proxy
        x_hat = g_w.copy()

    return x_hat, is_exact


def evaluate_inversion(
    true_x: np.ndarray,
    recovered_x: np.ndarray,
) -> InversionResult:
    """Compute reconstruction quality metrics between true and recovered x."""
    n_features = len(true_x)

    # Cosine similarity
    norm_true = float(np.linalg.norm(true_x))
    norm_rec = float(np.linalg.norm(recovered_x))
    if norm_true < 1e-10 or norm_rec < 1e-10:
        cos_sim = 0.0
    else:
        cos_sim = float(np.dot(true_x, recovered_x) / (norm_true * norm_rec))

    # Normalised MSE
    if norm_true < 1e-10:
        nmse = 1.0
    else:
        nmse = float(np.mean((true_x - recovered_x) ** 2) / (norm_true ** 2 / n_features))

    # Sign accuracy
    sign_match = (np.sign(true_x) == np.sign(recovered_x)).mean()

    return InversionResult(
        cosine_similarity=cos_sim,
        normalised_mse=nmse,
        sign_accuracy=float(sign_match),
        is_exact=True,
        n_features=n_features,
    )


# ---------------------------------------------------------------------------
# Leakage evaluation across many samples (with/without DP)
# ---------------------------------------------------------------------------

def evaluate_gradient_leakage(
    params: LRParams,
    x_train: np.ndarray,
    y_train: np.ndarray,
    rng: np.random.Generator,
    n_samples: int = 50,
    dp_noise_std: float = 0.0,
) -> LeakageReport:
    """
    Measure gradient inversion leakage across many training samples.

    For each sample, compute the gradient, optionally add DP noise,
    run the analytical inversion, and compare to the true sample.

    Parameters
    ----------
    dp_noise_std : Gaussian noise std added to simulate DP (0 = no DP).
                   Set to  noise_multiplier × max_grad_norm  to match
                   the DP-FedAvg configuration you want to evaluate.

    Returns
    -------
    LeakageReport with aggregate statistics and risk classification.
    """
    n = min(n_samples, len(x_train))
    idx = rng.choice(len(x_train), size=n, replace=False)
    n_features = x_train.shape[1]

    cos_sims: List[float] = []
    nmses: List[float] = []
    sign_accs: List[float] = []

    for i in idx:
        x_i = x_train[i]
        y_i = float(y_train[i])

        # Compute true gradient for this single sample
        grad = _compute_single_gradient(params, x_i, y_i)

        # Optionally corrupt with DP noise
        if dp_noise_std > 0:
            grad = grad + rng.normal(0.0, dp_noise_std, size=grad.shape)

        # Analytical inversion
        x_hat, _ = invert_single_gradient(grad, n_features)

        # Evaluate reconstruction quality
        result = evaluate_inversion(x_i, x_hat)
        cos_sims.append(result.cosine_similarity)
        nmses.append(result.normalised_mse)
        sign_accs.append(result.sign_accuracy)

    mean_cos = float(np.mean(cos_sims))

    # Risk classification based on cosine similarity
    if mean_cos > 0.90:
        risk = "HIGH"      # attacker can closely reconstruct features
    elif mean_cos > 0.50:
        risk = "MEDIUM"    # partial leakage
    else:
        risk = "LOW"       # DP noise dominates, reconstruction fails

    return LeakageReport(
        mean_cosine_similarity=mean_cos,
        std_cosine_similarity=float(np.std(cos_sims)),
        mean_normalised_mse=float(np.mean(nmses)),
        mean_sign_accuracy=float(np.mean(sign_accs)),
        n_evaluated=n,
        dp_noise_std=dp_noise_std,
        reconstruction_risk=risk,
    )


# ---------------------------------------------------------------------------
# Comparison: no DP vs multiple DP noise levels
# ---------------------------------------------------------------------------

def leakage_vs_dp_noise(
    params: LRParams,
    x_train: np.ndarray,
    y_train: np.ndarray,
    rng: np.random.Generator,
    noise_levels: Optional[List[float]] = None,
    n_samples: int = 50,
) -> List[LeakageReport]:
    """
    Evaluate gradient leakage across a range of DP noise levels.

    Useful for showing how increasing DP noise progressively degrades
    the attacker's ability to reconstruct training data.

    Parameters
    ----------
    noise_levels : list of DP noise std values to test.
                   Defaults to [0, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0].

    Returns
    -------
    List of LeakageReport, one per noise level.
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0]

    reports = []
    for sigma in noise_levels:
        report = evaluate_gradient_leakage(
            params, x_train, y_train, rng,
            n_samples=n_samples,
            dp_noise_std=sigma,
        )
        reports.append(report)
    return reports
