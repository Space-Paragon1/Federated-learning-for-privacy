from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def binary_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> BinaryMetrics:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    acc = _safe_div(tp + tn, tp + tn + fp + fn)
    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * prec * rec, (prec + rec)) if (prec + rec) != 0 else 0.0

    return BinaryMetrics(accuracy=acc, precision=prec, recall=rec, f1=f1)
