from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


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
    One client trains locally starting from global params.
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
