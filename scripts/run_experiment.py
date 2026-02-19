"""Full Experiment Runner for Federated Learning Privacy Study.

Runs a complete experimental suite comparing:
  1. Centralized logistic regression baseline
  2. FedAvg (no DP)
  3. DP-FedAvg at multiple privacy budgets (noise_multiplier sweep)
  4. Byzantine-robust aggregation variants
  5. Membership Inference Attack evaluation on each trained model

Results are saved to results/experiment_results.json and
figures are generated in results/figures/.

Usage (from federated-student-privacy/ directory):
    python scripts/run_experiment.py
    python scripts/run_experiment.py --quick     # fewer rounds / DP levels
    python scripts/run_experiment.py --no-plots  # skip visualization
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent dir to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from federated_train import FedConfig, run_federated
from privacy.dp import DPConfig
from attacks.membership_inference import membership_inference_attack
from metrics.utility import binary_classification_metrics
from clients.local_train import LRParams


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def run_centralized(csv_path: Path, test_ratio: float, seed: int) -> dict:
    """Train a centralized logistic regression model and return metrics."""
    from centralized_baseline import (
        train_test_split, standardize_fit, standardize_apply,
        train_logreg, FEATURES, LABEL,
    )
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
    m = binary_classification_metrics(y_test, y_pred)

    # MIA on centralized model
    rng = np.random.default_rng(seed + 100)
    params = LRParams(w=model.w, b=model.b)
    mia = membership_inference_attack(
        params, x_train_s, y_train.astype(int), x_test_s, y_test, rng
    )

    print(f"  [Centralized] acc={m.accuracy:.3f}  f1={m.f1:.3f}  MIA_AUC={mia.auc:.3f}")
    return {
        "accuracy": m.accuracy,
        "precision": m.precision,
        "recall": m.recall,
        "f1": m.f1,
        "mia_auc": mia.auc,
        "mia_advantage": mia.advantage,
        "mia_accuracy": mia.accuracy,
        "mia_tpr_at_fpr10": mia.tpr_at_fpr10,
    }


def run_mia(result: dict, seed: int) -> dict:
    """Run membership inference attack on a federated experiment result."""
    rng = np.random.default_rng(seed + 999)
    params: LRParams = result["_final_params"]
    mia = membership_inference_attack(
        params,
        result["_x_train"],
        result["_y_train"],
        result["_x_test"],
        result["_y_test"],
        rng,
    )
    return {
        "auc": mia.auc,
        "advantage": mia.advantage,
        "accuracy": mia.accuracy,
        "tpr_at_fpr10": mia.tpr_at_fpr10,
    }


def serialize_result(result: dict) -> dict:
    """Remove numpy arrays from result dict for JSON serialisation."""
    out = {}
    for k, v in result.items():
        if k.startswith("_"):          # skip raw arrays
            continue
        if isinstance(v, np.integer):
            out[k] = int(v)
        elif isinstance(v, np.floating):
            out[k] = float(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, list):
            out[k] = [
                float(x) if isinstance(x, (np.floating, float)) else
                int(x) if isinstance(x, (np.integer, int)) else x
                for x in v
            ]
        elif isinstance(v, dict):
            out[k] = serialize_result(v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Main experiment suite
# ---------------------------------------------------------------------------

def run_all(
    csv_path: Path,
    test_ratio: float,
    seed: int,
    rounds: int,
    quick: bool,
    no_plots: bool,
) -> None:
    results_dir = Path("results")
    figures_dir = results_dir / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    all_results: dict = {}
    t_start = time.perf_counter()

    # ------------------------------------------------------------------ 1. Centralized
    print("\n" + "="*60)
    print("STEP 1: Centralized Baseline")
    print("="*60)
    all_results["centralized"] = run_centralized(csv_path, test_ratio, seed)

    # ------------------------------------------------------------------ 2. FedAvg (no DP)
    print("\n" + "="*60)
    print("STEP 2: FedAvg (No Differential Privacy)")
    print("="*60)
    cfg_fedavg = FedConfig(
        rounds=rounds,
        client_frac=0.5,
        dropout=0.0,
        local_steps=200,
        lr=0.05,
        l2=1e-3,
        seed=seed,
        aggregator="fedavg",
        dp=None,
    )
    res_fedavg = run_federated(csv_path, test_ratio, cfg_fedavg, verbose=True)
    mia_fedavg = run_mia(res_fedavg, seed)
    res_fedavg["mia"] = mia_fedavg
    print(f"  MIA on FedAvg: AUC={mia_fedavg['auc']:.3f}  advantage={mia_fedavg['advantage']:.3f}")
    all_results["fedavg"] = serialize_result(res_fedavg)

    # ------------------------------------------------------------------ 3. DP-FedAvg sweep
    print("\n" + "="*60)
    print("STEP 3: DP-FedAvg (Privacy Budget Sweep)")
    print("="*60)

    # noise_multiplier controls epsilon: higher sigma -> smaller epsilon
    if quick:
        noise_levels = [0.5, 1.1, 2.0]
    else:
        noise_levels = [0.3, 0.5, 0.7, 1.0, 1.1, 1.5, 2.0, 3.0]

    dp_results = []
    for nm in noise_levels:
        dp_cfg = DPConfig(
            enabled=True,
            noise_multiplier=nm,
            max_grad_norm=1.0,
            target_delta=1e-5,
        )
        cfg_dp = FedConfig(
            rounds=rounds,
            client_frac=0.5,
            dropout=0.0,
            local_steps=200,
            lr=0.05,
            l2=1e-3,
            seed=seed,
            aggregator="fedavg",
            dp=dp_cfg,
        )
        print(f"\n  >> noise_multiplier={nm}")
        res_dp = run_federated(csv_path, test_ratio, cfg_dp, verbose=False)
        mia_dp = run_mia(res_dp, seed)
        eps = res_dp["privacy"]["epsilon"]
        f1 = res_dp["final"]["f1"]
        print(
            f"     epsilon={eps:.3f}  f1={f1:.3f}  "
            f"MIA_AUC={mia_dp['auc']:.3f}"
        )
        entry = serialize_result(res_dp)
        entry["mia"] = mia_dp
        dp_results.append(entry)

    all_results["dp_fedavg_sweep"] = dp_results

    # ------------------------------------------------------------------ 4. Byzantine-robust
    print("\n" + "="*60)
    print("STEP 4: Byzantine-Robust Aggregation Comparison")
    print("="*60)
    robust_methods = ["fedavg", "coord_median", "trimmed_mean", "krum"]
    robust_results = {}
    for method in robust_methods:
        cfg_robust = FedConfig(
            rounds=rounds,
            client_frac=0.5,
            dropout=0.1,
            local_steps=200,
            lr=0.05,
            l2=1e-3,
            seed=seed,
            aggregator=method,
            dp=None,
        )
        print(f"\n  >> aggregator={method}  (dropout=10%)")
        res_robust = run_federated(csv_path, test_ratio, cfg_robust, verbose=False)
        f1 = res_robust["final"]["f1"]
        worst_f1 = res_robust["final"]["worst_f1"]
        print(f"     f1={f1:.3f}  worst_f1={worst_f1:.3f}")
        robust_results[method] = serialize_result(res_robust)

    all_results["robust_aggregation"] = robust_results

    # ------------------------------------------------------------------ 5. Save
    out_path = results_dir / "experiment_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = time.perf_counter() - t_start
    print(f"\n{'='*60}")
    print(f"All experiments complete in {elapsed:.1f}s")
    print(f"Results saved to: {out_path}")

    # ------------------------------------------------------------------ 6. Plots
    if not no_plots:
        print("\nGenerating figures...")
        try:
            from scripts.plot_results import generate_all_plots
            generate_all_plots(all_results, figures_dir)
        except Exception as e:
            print(f"  Warning: plotting failed: {e}")
            print("  Run 'python scripts/plot_results.py' manually.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run full federated learning experiment suite")
    p.add_argument("--csv", type=str, default="data/synthetic/students.csv")
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rounds", type=int, default=30,
                   help="Number of federated rounds (use fewer for quick runs)")
    p.add_argument("--quick", action="store_true",
                   help="Run a reduced sweep (faster, for testing)")
    p.add_argument("--no-plots", action="store_true", dest="no_plots",
                   help="Skip figure generation")
    args = p.parse_args()

    run_all(
        csv_path=Path(args.csv),
        test_ratio=args.test_ratio,
        seed=args.seed,
        rounds=args.rounds,
        quick=args.quick,
        no_plots=args.no_plots,
    )
