from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from metrics.utility import binary_classification_metrics


FEATURES = [
    "attendance_rate",
    "avg_quiz_score",
    "assignment_completion",
    "lms_activity",
    "study_hours",
    "prior_gpa",
    "late_submissions",
    "support_requests",
]
LABEL = "at_risk"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def standardize_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = x.mean(axis=0)
    sigma = x.std(axis=0) + 1e-8
    return mu, sigma


def standardize_apply(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (x - mu) / sigma


@dataclass
class LRModel:
    w: np.ndarray  # shape (d,)
    b: float

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return sigmoid(x @ self.w + self.b)

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(x) >= threshold).astype(int)


def train_logreg(
    x: np.ndarray,
    y: np.ndarray,
    lr: float = 0.05,
    steps: int = 2000,
    l2: float = 1e-3,
    seed: int = 0,
) -> LRModel:
    rng = np.random.default_rng(seed)
    n, d = x.shape
    w = rng.normal(0, 0.01, size=d)
    b = 0.0

    for _ in range(steps):
        p = sigmoid(x @ w + b)
        # gradients
        dw = (x.T @ (p - y)) / n + l2 * w
        db = float((p - y).mean())
        w -= lr * dw
        b -= lr * db

    return LRModel(w=w, b=b)


def train_test_split(df: pd.DataFrame, test_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    cut = int(len(df) * (1 - test_ratio))
    train_idx, test_idx = idx[:cut], idx[cut:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def main(csv_path: Path, test_ratio: float, seed: int) -> None:
    df = pd.read_csv(csv_path)
    train_df, test_df = train_test_split(df, test_ratio=test_ratio, seed=seed)

    x_train = train_df[FEATURES].to_numpy(dtype=float)
    y_train = train_df[LABEL].to_numpy(dtype=float)

    x_test = test_df[FEATURES].to_numpy(dtype=float)
    y_test = test_df[LABEL].to_numpy(dtype=int)

    mu, sigma = standardize_fit(x_train)
    x_train_s = standardize_apply(x_train, mu, sigma)
    x_test_s = standardize_apply(x_test, mu, sigma)

    model = train_logreg(x_train_s, y_train, lr=0.05, steps=2500, l2=1e-3, seed=seed)

    y_pred = model.predict(x_test_s)
    metrics = binary_classification_metrics(y_test, y_pred)

    print("Centralized Logistic Regression Baseline")
    print(f"Test size: {len(test_df)}")
    print(f"Accuracy:  {metrics.accuracy:.3f}")
    print(f"Precision: {metrics.precision:.3f}")
    print(f"Recall:    {metrics.recall:.3f}")
    print(f"F1:        {metrics.f1:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="data/synthetic/students.csv")
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    main(Path(args.csv), test_ratio=args.test_ratio, seed=args.seed)
