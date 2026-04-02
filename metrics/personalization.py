"""Personalized Federated Learning Evaluation.

After global aggregation, each client fine-tunes the global model on its own
local data for a small number of steps.  This 'local fine-tuning' (LFT)
strategy is the simplest form of personalized FL and often delivers large
per-client accuracy gains on non-IID datasets.

Why personalization matters
---------------------------
In a standard FL setup (FedAvg), the global model is a compromise across all
clients.  Clients with unusual data distributions (e.g., a school with very
low-attendance students) may get a model that under-serves them.  Local
fine-tuning adapts the global model to each client's distribution at the cost
of only a handful of additional local SGD steps — with NO extra communication.

Comparison
----------
  Global model    : single shared model, no local adaptation
  Fine-tuned model: per-client model obtained by running k local steps on
                    top of the global model  (k = fine_tune_steps)

Reference: Yu et al. (2020) "Salvaging Federated Learning by Local
           Adaptation" — shows that even 1–5 fine-tuning steps can close
           most of the gap between federated and local-only training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from clients.local_train import LRParams, local_train_logreg
from metrics.utility import binary_classification_metrics


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClientPersonalizationResult:
    client_id: int
    global_f1: float       # F1 with the shared global model
    personal_f1: float     # F1 after local fine-tuning
    global_acc: float
    personal_acc: float
    improvement: float     # personal_f1 - global_f1

    @property
    def improved(self) -> bool:
        return self.improvement > 0.0


@dataclass
class PersonalizationReport:
    """Aggregate personalization statistics across all clients."""
    clients: List[ClientPersonalizationResult]
    fine_tune_steps: int

    @property
    def mean_global_f1(self) -> float:
        return float(np.mean([c.global_f1 for c in self.clients]))

    @property
    def mean_personal_f1(self) -> float:
        return float(np.mean([c.personal_f1 for c in self.clients]))

    @property
    def mean_improvement(self) -> float:
        return float(np.mean([c.improvement for c in self.clients]))

    @property
    def worst_global_f1(self) -> float:
        return float(min(c.global_f1 for c in self.clients))

    @property
    def worst_personal_f1(self) -> float:
        return float(min(c.personal_f1 for c in self.clients))

    @property
    def pct_clients_improved(self) -> float:
        return float(sum(c.improved for c in self.clients) / len(self.clients))

    def summary(self) -> str:
        return (
            f"Personalization Report  (fine-tune steps={self.fine_tune_steps})\n"
            f"  Clients              : {len(self.clients)}\n"
            f"  Mean Global F1       : {self.mean_global_f1:.3f}\n"
            f"  Mean Personal F1     : {self.mean_personal_f1:.3f}  "
            f"(D = {self.mean_improvement:+.3f})\n"
            f"  Worst-client Global  : {self.worst_global_f1:.3f}\n"
            f"  Worst-client Personal: {self.worst_personal_f1:.3f}\n"
            f"  % clients improved   : {self.pct_clients_improved*100:.1f}%\n"
        )

    def to_dict(self) -> dict:
        return {
            "fine_tune_steps": self.fine_tune_steps,
            "mean_global_f1": self.mean_global_f1,
            "mean_personal_f1": self.mean_personal_f1,
            "mean_improvement": self.mean_improvement,
            "worst_global_f1": self.worst_global_f1,
            "worst_personal_f1": self.worst_personal_f1,
            "pct_clients_improved": self.pct_clients_improved,
            "per_client": [
                {
                    "client_id": c.client_id,
                    "global_f1": c.global_f1,
                    "personal_f1": c.personal_f1,
                    "improvement": c.improvement,
                }
                for c in self.clients
            ],
        }


# ---------------------------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------------------------

def evaluate_personalization(
    global_params: LRParams,
    train_clients: Dict[int, Tuple[np.ndarray, np.ndarray]],
    test_clients: Dict[int, Tuple[np.ndarray, np.ndarray]],
    fine_tune_steps: int = 10,
    lr: float = 0.05,
    l2: float = 1e-3,
) -> PersonalizationReport:
    """
    Compare per-client performance of the global model vs. fine-tuned model.

    For each client:
      1. Evaluate the global model on its test split → global_f1
      2. Fine-tune the global model on the client's training data for
         `fine_tune_steps` local SGD steps
      3. Evaluate the fine-tuned model on the same test split → personal_f1

    Parameters
    ----------
    global_params   : the current global federated model.
    train_clients   : {client_id: (x_train, y_train)}
    test_clients    : {client_id: (x_test,  y_test)}
    fine_tune_steps : number of local SGD steps for personalisation.
    lr              : learning rate for fine-tuning.
    l2              : weight decay for fine-tuning.

    Returns
    -------
    PersonalizationReport with per-client and aggregate statistics.
    """
    results: List[ClientPersonalizationResult] = []

    for cid in sorted(train_clients.keys()):
        if cid not in test_clients:
            continue

        x_tr, y_tr = train_clients[cid]
        x_te, y_te = test_clients[cid]

        if len(x_te) == 0:
            continue

        # 1. Global model performance
        def _predict(params: LRParams, x: np.ndarray) -> np.ndarray:
            p = 1.0 / (1.0 + np.exp(-np.clip(x @ params.w + params.b, -500, 500)))
            return (p >= 0.5).astype(int)

        y_pred_global = _predict(global_params, x_te)
        global_metrics = binary_classification_metrics(y_te.astype(int), y_pred_global)

        # 2. Fine-tune on local training data
        personal_params = local_train_logreg(
            x_tr, y_tr, global_params,
            lr=lr,
            local_steps=fine_tune_steps,
            l2=l2,
        )

        # 3. Personalised model performance
        y_pred_personal = _predict(personal_params, x_te)
        personal_metrics = binary_classification_metrics(y_te.astype(int), y_pred_personal)

        results.append(ClientPersonalizationResult(
            client_id=int(cid),
            global_f1=global_metrics.f1,
            personal_f1=personal_metrics.f1,
            global_acc=global_metrics.accuracy,
            personal_acc=personal_metrics.accuracy,
            improvement=personal_metrics.f1 - global_metrics.f1,
        ))

    return PersonalizationReport(clients=results, fine_tune_steps=fine_tune_steps)


def personalization_sweep(
    global_params: LRParams,
    train_clients: Dict[int, Tuple[np.ndarray, np.ndarray]],
    test_clients: Dict[int, Tuple[np.ndarray, np.ndarray]],
    step_counts: List[int] = None,
    lr: float = 0.05,
    l2: float = 1e-3,
) -> List[PersonalizationReport]:
    """
    Evaluate personalization at multiple fine-tuning step counts.

    Useful for a 'fine-tune step sweep' plot showing how quickly local
    adaptation improves per-client performance.

    Parameters
    ----------
    step_counts : list of fine-tuning step counts to evaluate.
                  Defaults to [0, 5, 10, 20, 50, 100].
    """
    if step_counts is None:
        step_counts = [0, 5, 10, 20, 50, 100]

    reports = []
    for steps in step_counts:
        report = evaluate_personalization(
            global_params, train_clients, test_clients,
            fine_tune_steps=steps, lr=lr, l2=l2,
        )
        reports.append(report)
    return reports
