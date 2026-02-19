"""Evaluation utilities for federated learning experiments.

Provides a unified interface for evaluating a trained model against:
  - Global test metrics (accuracy, precision, recall, F1)
  - Per-client fairness metrics (worst / mean / best F1)
  - Membership inference attack (privacy leakage)
  - Privacy budget (epsilon if DP was used)

Can be run standalone to evaluate a saved experiment result, or
imported as a module by run_experiment.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from clients.local_train import LRParams
from metrics.utility import binary_classification_metrics, BinaryMetrics
from metrics.fairness import per_client_f1, ClientFairness
from attacks.membership_inference import membership_inference_attack, MIAResult


# ---------------------------------------------------------------------------
# Full evaluation result
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    global_metrics: BinaryMetrics
    fairness: ClientFairness
    mia: Optional[MIAResult] = None
    epsilon: Optional[float] = None
    delta: Optional[float] = None

    def print_report(self, label: str = "Model") -> None:
        print(f"\n{'='*55}")
        print(f"Evaluation Report: {label}")
        print(f"{'='*55}")
        print(f"  Accuracy:   {self.global_metrics.accuracy:.4f}")
        print(f"  Precision:  {self.global_metrics.precision:.4f}")
        print(f"  Recall:     {self.global_metrics.recall:.4f}")
        print(f"  F1-Score:   {self.global_metrics.f1:.4f}")
        print(f"  --- Fairness (per-client F1) ---")
        print(f"  Worst:      {self.fairness.worst_f1:.4f}")
        print(f"  Mean:       {self.fairness.mean_f1:.4f}")
        print(f"  Best:       {self.fairness.best_f1:.4f}")
        print(f"  Disparity:  {self.fairness.best_f1 - self.fairness.worst_f1:.4f}")
        if self.mia is not None:
            print(f"  --- Privacy (Membership Inference Attack) ---")
            print(f"  MIA AUC:        {self.mia.auc:.4f}  (0.5=random, 1.0=total leak)")
            print(f"  MIA Advantage:  {self.mia.advantage:.4f}  (0=no leak)")
            print(f"  MIA Accuracy:   {self.mia.accuracy:.4f}")
            print(f"  TPR@FPR=0.1:    {self.mia.tpr_at_fpr10:.4f}")
        if self.epsilon is not None:
            print(f"  --- Differential Privacy ---")
            print(f"  (epsilon, delta)-DP:  ({self.epsilon:.4f}, {self.delta})")
        print(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    params: LRParams,
    x_test: np.ndarray,
    y_test: np.ndarray,
    test_clients: Dict[int, Tuple[np.ndarray, np.ndarray]],
    x_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    run_mia: bool = True,
    epsilon: Optional[float] = None,
    delta: Optional[float] = None,
) -> EvalResult:
    """
    Full evaluation of a trained model.

    Parameters
    ----------
    params       : trained model (LRParams).
    x_test/y_test: global held-out test set.
    test_clients : per-client test sets for fairness evaluation.
                   Dict mapping client_id -> (x, y).
    x_train/y_train : training set needed for MIA (optional).
    rng          : random generator for MIA sampling.
    run_mia      : whether to run the membership inference attack.
    epsilon/delta: privacy budget if DP was used.
    """
    # Global metrics
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    y_prob = _sigmoid(x_test @ params.w + params.b)
    y_pred = (y_prob >= 0.5).astype(int)
    global_m = binary_classification_metrics(y_test, y_pred)

    # Per-client fairness
    client_ytrue: Dict[int, np.ndarray] = {}
    client_ypred: Dict[int, np.ndarray] = {}
    for cid, (x_c, y_c) in test_clients.items():
        client_ytrue[cid] = y_c
        p_c = _sigmoid(x_c @ params.w + params.b)
        client_ypred[cid] = (p_c >= 0.5).astype(int)
    fair = per_client_f1(client_ytrue, client_ypred)

    # MIA
    mia_result = None
    if run_mia and x_train is not None and y_train is not None:
        if rng is None:
            rng = np.random.default_rng(0)
        mia_result = membership_inference_attack(
            params, x_train, y_train, x_test, y_test, rng
        )

    return EvalResult(
        global_metrics=global_m,
        fairness=fair,
        mia=mia_result,
        epsilon=epsilon,
        delta=delta,
    )


# ---------------------------------------------------------------------------
# CLI — evaluate from a saved experiment JSON
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate a federated learning experiment")
    p.add_argument(
        "--results",
        type=str,
        default="results/experiment_results.json",
        help="Path to experiment_results.json produced by run_experiment.py",
    )
    p.add_argument(
        "--mode",
        choices=["fedavg", "dp_best", "centralized"],
        default="fedavg",
        help="Which experiment to evaluate",
    )
    args = p.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        print("Run 'python scripts/run_experiment.py' first to generate results.")
        sys.exit(1)

    with open(results_path) as f:
        data = json.load(f)

    if args.mode == "fedavg":
        res = data.get("fedavg", {})
        final = res.get("final", {})
        priv = res.get("privacy", {})
        mia = res.get("mia", {})

        print("\nFedAvg (No DP) — Final Evaluation")
        print(f"  Accuracy:   {final.get('accuracy', 0):.4f}")
        print(f"  F1-Score:   {final.get('f1', 0):.4f}")
        print(f"  Worst F1:   {final.get('worst_f1', 0):.4f}")
        if mia:
            print(f"  MIA AUC:    {mia.get('auc', 0):.4f}")
            print(f"  MIA Adv:    {mia.get('advantage', 0):.4f}")

    elif args.mode == "dp_best":
        sweep = data.get("dp_fedavg_sweep", [])
        if not sweep:
            print("No DP results found.")
            sys.exit(1)
        # Find entry with best F1 (i.e. lowest noise that still provides coverage)
        best = max(sweep, key=lambda e: e.get("final", {}).get("f1", 0))
        eps = best.get("privacy", {}).get("epsilon", "?")
        final = best.get("final", {})
        mia = best.get("mia", {})

        print(f"\nDP-FedAvg (Best F1, ε≈{eps:.2f if isinstance(eps, float) else eps}) — Final Evaluation")
        print(f"  Accuracy:   {final.get('accuracy', 0):.4f}")
        print(f"  F1-Score:   {final.get('f1', 0):.4f}")
        print(f"  Worst F1:   {final.get('worst_f1', 0):.4f}")
        print(f"  Epsilon:    {eps}")
        if mia:
            print(f"  MIA AUC:    {mia.get('auc', 0):.4f}")
            print(f"  MIA Adv:    {mia.get('advantage', 0):.4f}")

    elif args.mode == "centralized":
        c = data.get("centralized", {})
        print("\nCentralized Baseline — Final Evaluation")
        print(f"  Accuracy:   {c.get('accuracy', 0):.4f}")
        print(f"  F1-Score:   {c.get('f1', 0):.4f}")
        print(f"  MIA AUC:    {c.get('mia_auc', 0):.4f}")
        print(f"  MIA Adv:    {c.get('mia_advantage', 0):.4f}")
