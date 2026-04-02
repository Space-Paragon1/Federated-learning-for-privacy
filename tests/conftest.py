"""pytest fixtures shared across all test modules."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path so all imports resolve
sys.path.insert(0, str(Path(__file__).parent.parent))

from clients.local_train import LRParams


# ── random seed fixture ───────────────────────────────────────────────────────

@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


# ── tiny dataset fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def small_dataset(rng) -> tuple:
    """100 samples, 8 features, binary labels — deterministic."""
    n, d = 100, 8
    x = rng.normal(0, 1, (n, d))
    w_true = rng.normal(0, 1, d)
    logits = x @ w_true
    y = (1.0 / (1.0 + np.exp(-logits)) > 0.5).astype(float)
    return x, y


@pytest.fixture
def multi_client_dataset(rng) -> dict:
    """5 clients, 50 samples each, 8 features."""
    clients = {}
    for cid in range(5):
        x = rng.normal(cid * 0.5, 1, (50, 8))   # slight distribution shift
        y = (rng.random(50) > 0.6).astype(float)
        clients[cid] = (x, y)
    return clients


# ── model fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def zero_params() -> LRParams:
    return LRParams(w=np.zeros(8), b=0.0)


@pytest.fixture
def small_params(rng) -> LRParams:
    return LRParams(w=rng.normal(0, 0.1, 8), b=float(rng.normal()))
