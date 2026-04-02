from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


@dataclass
class LRParams:
    w: np.ndarray
    b: float


def local_train_logreg(
    x: np.ndarray,
    y: np.ndarray,
    params: LRParams,
    lr: float = 0.05,
    local_steps: int = 200,
    l2: float = 1e-3,
) -> LRParams:
    """
    One client trains locally starting from global params (FedAvg style).
    """
    w = params.w.copy()
    b = float(params.b)
    n = x.shape[0]

    for _ in range(local_steps):
        p = sigmoid(x @ w + b)
        dw = (x.T @ (p - y)) / n + l2 * w
        db = float((p - y).mean())
        w -= lr * dw
        b -= lr * db

    return LRParams(w=w, b=b)


def local_train_fedprox(
    x: np.ndarray,
    y: np.ndarray,
    params: LRParams,
    global_params: LRParams,
    mu: float = 0.01,
    lr: float = 0.05,
    local_steps: int = 200,
    l2: float = 1e-3,
) -> LRParams:
    """
    FedProx local training: standard SGD + proximal regularisation term.

    The proximal term  (mu/2) * ||w - w_global||^2  prevents local models
    from drifting too far from the global model during multiple local steps.
    This improves convergence and fairness under non-IID data distributions.

    Reference: Li et al. (2020) "Federated Optimization in Heterogeneous
               Networks" (MLSys 2020).

    Parameters
    ----------
    global_params : global model parameters broadcast from the server.
    mu            : proximal coefficient.  0 reduces to plain FedAvg.
                    Typical values: 0.001, 0.01, 0.1.
    """
    w = params.w.copy()
    b = float(params.b)
    w0 = global_params.w.copy()
    b0 = float(global_params.b)
    n = x.shape[0]

    for _ in range(local_steps):
        p = sigmoid(x @ w + b)
        # Standard cross-entropy gradient + L2 weight decay + proximal term
        dw = (x.T @ (p - y)) / n + l2 * w + mu * (w - w0)
        db = float((p - y).mean()) + mu * (b - b0)
        w -= lr * dw
        b -= lr * db

    return LRParams(w=w, b=b)
