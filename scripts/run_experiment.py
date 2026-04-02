"""Full Experiment Runner for Federated Learning Privacy Study.

Runs a complete experimental suite comparing:
  1. Centralized logistic regression baseline
  2. FedAvg (no DP)
  3. DP-FedAvg at multiple privacy budgets (noise_multiplier sweep)
  4. Byzantine-robust aggregation variants
  5. FedProx sweep (proximal coefficient μ)          [NEW]
  6. Gradient compression sweep (QSGD bit-widths)   [NEW]
  7. Gradient inversion leakage evaluation          [NEW]
  8. Personalization (local fine-tuning) evaluation [NEW]
  9. Membership Inference Attack on all trained models

Results are saved to results/experiment_results.json and
figures are generated in results/figures/.

Usage (from federated-student-privacy/ directory):
    python scripts/run_experiment.py
    python scripts/run_experiment.py --quick     # fewer rounds / DP levels
    python scripts/run_experiment.py --no-plots  # skip visualization
    python scripts/run_experiment.py --skip-new  # skip new experiments (fast)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Add parent dir to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from federated_train import FedConfig, run_federated
from privacy.dp import DPConfig
from privacy.compression import CompressionConfig
from attacks.membership_inference import membership_inference_attack
from attacks.gradient_inversion import evaluate_gradient_leakage, leakage_vs_dp_noise
from metrics.utility import binary_classification_metrics
from metrics.personalization import evaluate_personalization, personalization_sweep
from clients.local_train import LRParams


FEATURES = [
    "attendance_rate", "avg_quiz_score", "assignment_completion",
    "lms_activity", "study_hours", "prior_gpa",
    "late_submissions", "support_requests",
]
LABEL = "at_risk"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_centralized(csv_path: Path, test_ratio: float, seed: int) -> dict:
    """Train a centralized logistic regression model and return metrics."""
    from centralized_baseline import (
        train_test_split, standardize_fit, standardize_apply,
        train_logreg, FEATURES as FEAT, LABEL as LBL,
    )
    df = pd.read_csv(csv_path)
    train_df, test_df = train_test_split(df, test_ratio=test_ratio, seed=seed)

    x_train = train_df[FEAT].to_numpy(dtype=float)
    y_train = train_df[LBL].to_numpy(dtype=float)
    x_test = test_df[FEAT].to_numpy(dtype=float)
    y_test = test_df[LBL].to_numpy(dtype=int)

    mu, sigma = standardize_fit(x_train)
    x_train_s = standardize_apply(x_train, mu, sigma)
    x_test_s = standardize_apply(x_test, mu, sigma)

    model = train_logreg(x_train_s, y_train, lr=0.05, steps=2500, l2=1e-3, seed=seed)
    y_pred = model.predict(x_test_s)
    m = binary_classification_metrics(y_test, y_pred)

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


def _build_client_dicts(result: dict, csv_path: Path, test_ratio: float, seed: int):
    """Reconstruct per-client train/test dicts for personalization evaluation."""
    df = pd.read_csv(csv_path)
    from federated_train import train_test_split, standardize_fit, standardize_apply

    train_df, test_df = train_test_split(df, test_ratio=test_ratio, seed=seed)
    x_tr_raw = train_df[FEATURES].to_numpy(dtype=float)
    mu, sigma = standardize_fit(x_tr_raw)

    def _std(x):
        return (x - mu) / (sigma + 1e-8)

    train_clients = {}
    for cid, g in train_df.groupby("client_id"):
        train_clients[int(cid)] = (
            _std(g[FEATURES].to_numpy(dtype=float)),
            g[LABEL].to_numpy(dtype=float),
        )

    test_clients = {}
    for cid, g in test_df.groupby("client_id"):
        test_clients[int(cid)] = (
            _std(g[FEATURES].to_numpy(dtype=float)),
            g[LABEL].to_numpy(dtype=int),
        )

    return train_clients, test_clients


def serialize_result(result: dict) -> dict:
    """Remove numpy arrays from result dict for JSON serialisation."""
    out = {}
    for k, v in result.items():
        if k.startswith("_"):
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
    skip_new: bool,
) -> None:
    results_dir = Path("results")
    figures_dir = results_dir / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    all_results: dict = {}
    t_start = time.perf_counter()

    # ------------------------------------------------------------------ 1. Centralized
    print("\n" + "=" * 60)
    print("STEP 1: Centralized Baseline")
    print("=" * 60)
    all_results["centralized"] = run_centralized(csv_path, test_ratio, seed)

    # ------------------------------------------------------------------ 2. FedAvg (no DP)
    print("\n" + "=" * 60)
    print("STEP 2: FedAvg (No Differential Privacy)")
    print("=" * 60)
    cfg_fedavg = FedConfig(
        rounds=rounds, client_frac=0.5, dropout=0.0,
        local_steps=200, lr=0.05, l2=1e-3, seed=seed,
        aggregator="fedavg", dp=None,
    )
    res_fedavg = run_federated(csv_path, test_ratio, cfg_fedavg, verbose=True)
    mia_fedavg = run_mia(res_fedavg, seed)
    res_fedavg["mia"] = mia_fedavg
    print(f"  MIA on FedAvg: AUC={mia_fedavg['auc']:.3f}  advantage={mia_fedavg['advantage']:.3f}")
    all_results["fedavg"] = serialize_result(res_fedavg)

    # ------------------------------------------------------------------ 3. DP-FedAvg sweep
    print("\n" + "=" * 60)
    print("STEP 3: DP-FedAvg (Privacy Budget Sweep)")
    print("=" * 60)
    noise_levels = [0.5, 1.1, 2.0] if quick else [0.3, 0.5, 0.7, 1.0, 1.1, 1.5, 2.0, 3.0]

    dp_results = []
    for nm in noise_levels:
        dp_cfg = DPConfig(enabled=True, noise_multiplier=nm, max_grad_norm=1.0, target_delta=1e-5)
        cfg_dp = FedConfig(
            rounds=rounds, client_frac=0.5, dropout=0.0,
            local_steps=200, lr=0.05, l2=1e-3, seed=seed,
            aggregator="fedavg", dp=dp_cfg,
        )
        print(f"\n  >> noise_multiplier={nm}")
        res_dp = run_federated(csv_path, test_ratio, cfg_dp, verbose=False)
        mia_dp = run_mia(res_dp, seed)
        eps = res_dp["privacy"]["epsilon"]
        f1 = res_dp["final"]["f1"]
        print(f"     epsilon={eps:.3f}  f1={f1:.3f}  MIA_AUC={mia_dp['auc']:.3f}")
        entry = serialize_result(res_dp)
        entry["mia"] = mia_dp
        dp_results.append(entry)

    all_results["dp_fedavg_sweep"] = dp_results

    # ------------------------------------------------------------------ 4. Byzantine-robust
    print("\n" + "=" * 60)
    print("STEP 4: Byzantine-Robust Aggregation Comparison")
    print("=" * 60)
    robust_methods = ["fedavg", "coord_median", "trimmed_mean", "krum"]
    robust_results = {}
    for method in robust_methods:
        cfg_robust = FedConfig(
            rounds=rounds, client_frac=0.5, dropout=0.1,
            local_steps=200, lr=0.05, l2=1e-3, seed=seed,
            aggregator=method, dp=None,
        )
        print(f"\n  >> aggregator={method}  (dropout=10%)")
        res_robust = run_federated(csv_path, test_ratio, cfg_robust, verbose=False)
        f1 = res_robust["final"]["f1"]
        worst_f1 = res_robust["final"]["worst_f1"]
        print(f"     f1={f1:.3f}  worst_f1={worst_f1:.3f}")
        robust_results[method] = serialize_result(res_robust)

    all_results["robust_aggregation"] = robust_results

    # ------------------------------------------------------------------ 5. FedProx sweep [NEW]
    if not skip_new:
        print("\n" + "=" * 60)
        print("STEP 5: FedProx — Proximal Coefficient Sweep [NEW]")
        print("=" * 60)
        mu_values = [0.001, 0.01, 0.1] if quick else [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5]
        fedprox_results = []
        for mu in mu_values:
            cfg_fp = FedConfig(
                rounds=rounds, client_frac=0.5, dropout=0.0,
                local_steps=200, lr=0.05, l2=1e-3, seed=seed,
                aggregator="fedavg", dp=None, fedprox_mu=mu,
            )
            print(f"\n  >> FedProx mu={mu}")
            res_fp = run_federated(csv_path, test_ratio, cfg_fp, verbose=False)
            f1 = res_fp["final"]["f1"]
            worst_f1 = res_fp["final"]["worst_f1"]
            print(f"     f1={f1:.3f}  worst_f1={worst_f1:.3f}")
            fedprox_results.append(serialize_result(res_fp))

        all_results["fedprox_sweep"] = fedprox_results

    # ------------------------------------------------------------------ 6. Compression sweep [NEW]
    if not skip_new:
        print("\n" + "=" * 60)
        print("STEP 6: Gradient Compression — QSGD Bit-Width Sweep [NEW]")
        print("=" * 60)
        bit_widths = [4, 8] if quick else [2, 4, 8, 16]
        compression_results = []
        for bits in bit_widths:
            comp_cfg = CompressionConfig(enabled=True, num_levels=2 ** bits)
            cfg_comp = FedConfig(
                rounds=rounds, client_frac=0.5, dropout=0.0,
                local_steps=200, lr=0.05, l2=1e-3, seed=seed,
                aggregator="fedavg", dp=None, compression=comp_cfg,
            )
            print(f"\n  >> QSGD {bits}-bit  ({comp_cfg.num_levels} levels)")
            res_comp = run_federated(csv_path, test_ratio, cfg_comp, verbose=False)
            f1 = res_comp["final"]["f1"]
            comm = res_comp["system"]["total_comm_bytes"]
            baseline_comm = all_results["fedavg"]["system"]["total_comm_bytes"]
            savings = 1.0 - comm / baseline_comm if baseline_comm > 0 else 0.0
            print(f"     f1={f1:.3f}  comm_savings={savings*100:.1f}%")
            compression_results.append(serialize_result(res_comp))

        all_results["compression_sweep"] = compression_results

    # ------------------------------------------------------------------ 7. Gradient Inversion [NEW]
    if not skip_new:
        print("\n" + "=" * 60)
        print("STEP 7: Gradient Inversion Leakage Evaluation [NEW]")
        print("=" * 60)
        params = res_fedavg["_final_params"]
        x_train = res_fedavg["_x_train"]
        y_train = res_fedavg["_y_train"]
        rng_inv = np.random.default_rng(seed + 77)

        # Evaluate leakage at several DP noise levels
        noise_levels_inv = [0.0, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0]
        leakage_reports = []
        print(f"  {'Noise s':>10} {'Cos Sim':>10} {'Sign Acc':>10} {'Risk':>8}")
        for sigma in noise_levels_inv:
            report = evaluate_gradient_leakage(
                params, x_train, y_train.astype(float),
                rng_inv, n_samples=50, dp_noise_std=sigma,
            )
            print(
                f"  {sigma:>10.2f} {report.mean_cosine_similarity:>10.3f} "
                f"{report.mean_sign_accuracy:>10.3f} {report.reconstruction_risk:>8}"
            )
            leakage_reports.append({
                "dp_noise_std": sigma,
                "mean_cosine_similarity": report.mean_cosine_similarity,
                "std_cosine_similarity": report.std_cosine_similarity,
                "mean_normalised_mse": report.mean_normalised_mse,
                "mean_sign_accuracy": report.mean_sign_accuracy,
                "reconstruction_risk": report.reconstruction_risk,
                "n_evaluated": report.n_evaluated,
            })

        all_results["gradient_leakage"] = leakage_reports

    # ------------------------------------------------------------------ 8. Personalization [NEW]
    if not skip_new:
        print("\n" + "=" * 60)
        print("STEP 8: Personalized FL — Local Fine-Tuning Evaluation [NEW]")
        print("=" * 60)
        train_clients, test_clients = _build_client_dicts(
            res_fedavg, csv_path, test_ratio, seed
        )
        global_params = res_fedavg["_final_params"]

        # Fine-tune sweep
        step_counts = [0, 5, 10, 20] if quick else [0, 5, 10, 20, 50, 100]
        sweep = personalization_sweep(
            global_params, train_clients, test_clients,
            step_counts=step_counts,
        )
        print(f"\n  {'Steps':>6} {'Global F1':>11} {'Personal F1':>13} {'D F1':>8}")
        for report in sweep:
            print(
                f"  {report.fine_tune_steps:>6} "
                f"{report.mean_global_f1:>11.3f} "
                f"{report.mean_personal_f1:>13.3f} "
                f"{report.mean_improvement:>+8.3f}"
            )

        # Save best (most steps) personalization result in full detail
        best_report = sweep[-1]
        print(f"\n{best_report.summary()}")
        all_results["personalization"] = best_report.to_dict()
        all_results["personalization_sweep"] = [r.to_dict() for r in sweep]

    # ------------------------------------------------------------------ 9. Save
    out_path = results_dir / "experiment_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = time.perf_counter() - t_start
    print(f"\n{'=' * 60}")
    print(f"All experiments complete in {elapsed:.1f}s")
    print(f"Results saved to: {out_path}")

    # ------------------------------------------------------------------ 10. Plots
    if not no_plots:
        print("\nGenerating static figures...")
        try:
            from scripts.plot_results import generate_all_plots
            generate_all_plots(all_results, figures_dir)
        except Exception as e:
            print(f"  Warning: plotting failed: {e}")
            print("  Run 'python scripts/plot_results.py' manually.")

        if not skip_new:
            print("\nGenerating Pareto frontier...")
            try:
                from scripts.pareto_analysis import extract_configs, compute_pareto_front
                from scripts.pareto_analysis import plot_pareto_2d, print_pareto_report
                configs = extract_configs(all_results)
                frontier, dominated = compute_pareto_front(configs)
                print_pareto_report(frontier, dominated)
                plot_pareto_2d(frontier, dominated, figures_dir / "fig7_pareto_frontier.png")
            except Exception as e:
                print(f"  Pareto plot failed: {e}")

        try:
            print("\nGenerating interactive HTML dashboard...")
            from dashboard.generate_html import build_dashboard
            import plotly.io as pio
            fig = build_dashboard(all_results)
            dash_path = results_dir / "dashboard.html"
            fig.write_html(
                str(dash_path), include_plotlyjs="cdn", full_html=True,
                config={"displayModeBar": True, "scrollZoom": True},
            )
            print(f"  Dashboard saved: {dash_path}")
        except ImportError:
            print("  plotly not installed — skipping dashboard.  pip install plotly")
        except Exception as e:
            print(f"  Dashboard failed: {e}")

    print(f"\nDone!  Open results/dashboard.html in a browser for interactive exploration.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run full federated learning experiment suite")
    p.add_argument("--csv", type=str, default="data/synthetic/students.csv")
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--quick", action="store_true",
                   help="Reduced sweep (faster, for testing)")
    p.add_argument("--no-plots", action="store_true", dest="no_plots",
                   help="Skip figure generation")
    p.add_argument("--skip-new", action="store_true", dest="skip_new",
                   help="Skip new experiments (FedProx, compression, inversion, personalization)")
    args = p.parse_args()

    run_all(
        csv_path=Path(args.csv),
        test_ratio=args.test_ratio,
        seed=args.seed,
        rounds=args.rounds,
        quick=args.quick,
        no_plots=args.no_plots,
        skip_new=args.skip_new,
    )
