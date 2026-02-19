"""Membership Inference Attack (MIA) for Federated Learning.

Implements a confidence-based membership inference attack to quantify privacy
leakage of a trained federated model.

Attack idea (Shokri et al., 2017):
  - Members:     training samples the model has seen.
  - Non-members: held-out test samples the model has NOT seen.
  - Score:       model's confidence P(true_label | x, theta).
                 Members tend to receive higher confidence than non-members
                 because the model has overfit to them.
  - Classifier:  threshold on the score to predict membership.

Metrics reported:
  - AUC-ROC: overall discriminability (0.5 = random, 1.0 = perfect attack).
  - Advantage: max(TPR - FPR) across thresholds.
  - Accuracy:  best attack accuracy.
  - TPR @ FPR=0.1: true positive rate when false positive rate is at most 10%.
                   This is a commonly used privacy auditing metric.

A model with no privacy protection will have AUC well above 0.5.
DP-FedAvg with sufficient noise should push AUC back toward 0.5.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from clients.local_train import LRParams


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _predict_proba(params: LRParams, x: np.ndarray) -> np.ndarray:
    return _sigmoid(x @ params.w + params.b)


def _confidence_scores(
    params: LRParams, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """
    Per-sample confidence: probability assigned to the TRUE label.

    Higher score -> model is more certain -> more likely to be a member.
    """
    probs = _predict_proba(params, x)
    return np.where(y == 1, probs, 1.0 - probs)


def _trapz_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Trapezoidal AUC-ROC (pure numpy, no sklearn dependency)."""
    pos = int(y_true.sum())
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        return 0.5

    order = np.argsort(-scores)   # descending
    tprs = [0.0]
    fprs = [0.0]
    tp = fp = 0
    for idx in order:
        if y_true[idx] == 1:
            tp += 1
        else:
            fp += 1
        tprs.append(tp / pos)
        fprs.append(fp / neg)
    tprs.append(1.0)
    fprs.append(1.0)
    return float(abs(np.trapz(tprs, fprs)))


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MIAResult:
    """Results of a single membership inference attack evaluation."""
    auc: float            # AUC-ROC  (0.5 = random, 1.0 = perfect leak)
    advantage: float      # max(TPR - FPR)  (0 = no leak)
    accuracy: float       # best attack accuracy
    tpr_at_fpr10: float   # TPR when FPR <= 0.10  (privacy audit metric)
    n_members: int
    n_nonmembers: int


# ---------------------------------------------------------------------------
# Main attack function
# ---------------------------------------------------------------------------

def membership_inference_attack(
    params: LRParams,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    rng: np.random.Generator,
    n_members: int = 500,
    n_nonmembers: int = 500,
) -> MIAResult:
    """
    Run a confidence-based membership inference attack.

    Parameters
    ----------
    params       : trained model parameters to attack.
    x_train / y_train : training set (members).
    x_test  / y_test  : held-out set (non-members).
    rng          : random generator for reproducible sampling.
    n_members    : how many training samples to use as member examples.
    n_nonmembers : how many test samples to use as non-member examples.

    Returns
    -------
    MIAResult with AUC, advantage, accuracy, and TPR@FPR=0.1.
    """
    n_m = min(n_members, len(x_train))
    n_nm = min(n_nonmembers, len(x_test))

    m_idx = rng.choice(len(x_train), size=n_m, replace=False)
    nm_idx = rng.choice(len(x_test), size=n_nm, replace=False)

    x_m, y_m = x_train[m_idx], y_train[m_idx]
    x_nm, y_nm = x_test[nm_idx], y_test[nm_idx]

    scores_m = _confidence_scores(params, x_m, y_m.astype(int))
    scores_nm = _confidence_scores(params, x_nm, y_nm.astype(int))

    all_scores = np.concatenate([scores_m, scores_nm])
    # Label: 1 = member, 0 = non-member
    all_labels = np.concatenate([np.ones(n_m), np.zeros(n_nm)])

    auc = _trapz_auc(all_labels, all_scores)

    # Sweep thresholds to compute advantage and accuracy
    pos, neg = n_m, n_nm
    best_advantage = 0.0
    best_accuracy = 0.0
    tpr_at_fpr10 = 0.0

    thresholds = np.unique(all_scores)
    for t in thresholds:
        pred = (all_scores >= t).astype(int)
        tp = int(((pred == 1) & (all_labels == 1)).sum())
        fp = int(((pred == 1) & (all_labels == 0)).sum())
        tpr = tp / pos if pos > 0 else 0.0
        fpr = fp / neg if neg > 0 else 0.0
        advantage = tpr - fpr
        acc = float((pred == all_labels).mean())
        if advantage > best_advantage:
            best_advantage = advantage
        if acc > best_accuracy:
            best_accuracy = acc
        if fpr <= 0.10:
            tpr_at_fpr10 = max(tpr_at_fpr10, tpr)

    return MIAResult(
        auc=auc,
        advantage=float(best_advantage),
        accuracy=float(best_accuracy),
        tpr_at_fpr10=float(tpr_at_fpr10),
        n_members=n_m,
        n_nonmembers=n_nm,
    )
