"""Federated Learning Training Pipeline.

Supports:
  - FedAvg baseline (no privacy)
  - DP-FedAvg with Gaussian mechanism and RDP privacy accounting
  - Byzantine-robust aggregation (coord_median, trimmed_mean, krum)
  - Client dropout simulation
  - Per-round fairness tracking
  - System metrics (communication cost, wall-clock time, convergence speed)
  - Structured results dict for use by the experiment runner
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from clients.local_train import LRParams, local_train_logreg
from server.aggregator import ClientUpdate, aggregate
from server.scheduler import schedule_round
from metrics.utility import binary_classification_metrics
from metrics.fairness import per_client_f1
from metrics.systems import RoundSystemMetrics, compute_system_summary
from privacy.dp import DPConfig, PrivacyAccountant, clip_update, apply_dp_noise


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def standardize_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = x.mean(axis=0)
    sigma = x.std(axis=0) + 1e-8
    return mu, sigma


def standardize_apply(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (x - mu) / sigma


def train_test_split(
    df: pd.DataFrame, test_ratio: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    cut = int(len(df) * (1 - test_ratio))
    return (
        df.iloc[idx[:cut]].reset_index(drop=True),
        df.iloc[idx[cut:]].reset_index(drop=True),
    )


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def predict(params: LRParams, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    p = sigmoid(x @ params.w + params.b)
    return (p >= threshold).astype(int)


def predict_probs(params: LRParams, x: np.ndarray) -> np.ndarray:
    return sigmoid(x @ params.w + params.b)


def params_bytes(params: LRParams) -> int:
    """Estimate bytes for one set of model parameters (float64)."""
    return int(params.w.size * 8 + 8)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FedConfig:
    rounds: int = 30
    client_frac: float = 0.5
    dropout: float = 0.0
    local_steps: int = 200
    lr: float = 0.05
    l2: float = 1e-3
    seed: int = 42
    aggregator: str = "fedavg"      # fedavg | coord_median | trimmed_mean | krum
    dp: Optional[DPConfig] = None   # None = no differential privacy


# ---------------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------------

def run_federated(
    csv_path: Path,
    test_ratio: float,
    cfg: FedConfig,
    verbose: bool = True,
) -> dict:
    """
    Run one full federated learning experiment.

    Returns a results dict with per-round and final metrics, suitable for
    downstream analysis and plotting.
    """
    # ------------------------------------------------------------------ data
    df = pd.read_csv(csv_path)
    train_df, test_df = train_test_split(df, test_ratio=test_ratio, seed=cfg.seed)

    x_train_raw = train_df[FEATURES].to_numpy(dtype=float)
    y_train = train_df[LABEL].to_numpy(dtype=float)
    x_test_raw = test_df[FEATURES].to_numpy(dtype=float)
    y_test = test_df[LABEL].to_numpy(dtype=int)

    # Fit standardisation on training data only
    mu, sigma = standardize_fit(x_train_raw)
    x_train_s = standardize_apply(x_train_raw, mu, sigma)
    x_test_s = standardize_apply(x_test_raw, mu, sigma)

    train_df_s = train_df.copy()
    test_df_s = test_df.copy()
    train_df_s[FEATURES] = x_train_s
    test_df_s[FEATURES] = x_test_s

    # Per-client splits
    clients: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for cid, g in train_df_s.groupby("client_id"):
        clients[int(cid)] = (
            g[FEATURES].to_numpy(dtype=float),
            g[LABEL].to_numpy(dtype=float),
        )

    test_clients: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for cid, g in test_df_s.groupby("client_id"):
        test_clients[int(cid)] = (
            g[FEATURES].to_numpy(dtype=float),
            g[LABEL].to_numpy(dtype=int),
        )

    # --------------------------------------------------- initialise
    rng = np.random.default_rng(cfg.seed)
    d = len(FEATURES)
    global_params = LRParams(w=rng.normal(0, 0.01, size=d), b=0.0)
    client_ids = sorted(clients.keys())

    # Privacy accountant (only used when DP is on)
    dp_cfg = cfg.dp if cfg.dp is not None else DPConfig(enabled=False)
    accountant: Optional[PrivacyAccountant] = None
    if dp_cfg.enabled:
        accountant = PrivacyAccountant(
            noise_multiplier=dp_cfg.noise_multiplier,
            target_delta=dp_cfg.target_delta,
            client_frac=cfg.client_frac,
        )

    # --------------------------------------------------- per-round storage
    per_round: List[dict] = []
    system_metrics: List[RoundSystemMetrics] = []
    f1_per_round: List[float] = []

    # --------------------------------------------------- training loop
    for r in range(cfg.rounds):
        t0 = time.perf_counter()
        sched = schedule_round(rng, client_ids, cfg.client_frac, cfg.dropout)

        updates: List[ClientUpdate] = []
        for cid in sched.survivors:
            x_c, y_c = clients[cid]
            new_params = local_train_logreg(
                x_c, y_c, global_params,
                lr=cfg.lr, local_steps=cfg.local_steps, l2=cfg.l2,
            )

            # DP: clip each client's update delta before aggregation
            if dp_cfg.enabled:
                delta_w = new_params.w - global_params.w
                delta_b = new_params.b - global_params.b
                delta = LRParams(w=delta_w, b=delta_b)
                delta = clip_update(delta, dp_cfg.max_grad_norm)
                new_params = LRParams(
                    w=global_params.w + delta.w,
                    b=global_params.b + delta.b,
                )

            updates.append(ClientUpdate(params=new_params, n_samples=len(x_c)))

        # Server aggregation
        global_params = aggregate(updates, method=cfg.aggregator)

        # DP: add Gaussian noise to the aggregate
        if dp_cfg.enabled and accountant is not None:
            global_params = apply_dp_noise(global_params, dp_cfg, rng)
            accountant.step()

        elapsed = time.perf_counter() - t0

        # ------------------------------------------- evaluation
        x_test_np = test_df_s[FEATURES].to_numpy(dtype=float)
        y_pred = predict(global_params, x_test_np)
        metrics = binary_classification_metrics(y_test, y_pred)
        f1_per_round.append(metrics.f1)

        # Per-client fairness
        client_ytrue = {cid: test_clients[cid][1] for cid in test_clients}
        client_ypred = {
            cid: (predict_probs(global_params, test_clients[cid][0]) >= 0.5).astype(int)
            for cid in test_clients
        }
        fair = per_client_f1(client_ytrue, client_ypred)

        # Communication cost
        model_bytes = params_bytes(global_params)
        bytes_bc = model_bytes * len(sched.selected)
        bytes_up = model_bytes * len(sched.survivors)

        system_metrics.append(RoundSystemMetrics(
            round_num=r + 1,
            wall_time_s=elapsed,
            bytes_broadcast=bytes_bc,
            bytes_uploaded=bytes_up,
            n_selected=len(sched.selected),
            n_returned=len(sched.survivors),
        ))

        current_eps = accountant.epsilon if accountant else None

        round_result = {
            "round": r + 1,
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "worst_f1": fair.worst_f1,
            "mean_f1": fair.mean_f1,
            "best_f1": fair.best_f1,
            "bytes_broadcast": bytes_bc,
            "bytes_uploaded": bytes_up,
            "total_bytes": bytes_bc + bytes_up,
            "n_selected": len(sched.selected),
            "n_returned": len(sched.survivors),
            "wall_time_s": elapsed,
            "epsilon": current_eps,
        }
        per_round.append(round_result)

        if verbose and ((r + 1) % 5 == 0 or r == 0):
            eps_str = f"  eps={current_eps:.2f}" if current_eps is not None else ""
            print(
                f"[Round {r+1:02d}] acc={metrics.accuracy:.3f}  f1={metrics.f1:.3f}"
                f"  worst_f1={fair.worst_f1:.3f}{eps_str}"
                f"  comm={( bytes_bc + bytes_up)/1024:.1f}KB"
            )

    # --------------------------------------------------- final summary
    sys_summary = compute_system_summary(
        system_metrics, f1_per_round=f1_per_round, convergence_f1=0.70
    )

    final_metrics = per_round[-1]
    if verbose:
        dp_label = ""
        if dp_cfg.enabled and accountant:
            dp_label = (
                f"\nFinal Privacy Budget:  epsilon={accountant.epsilon:.3f}  "
                f"delta={dp_cfg.target_delta}"
            )
        print(
            f"\n{'='*50}"
            f"\nFinal Results ({cfg.aggregator.upper()}"
            f"{' + DP' if dp_cfg.enabled else ''})"
            f"\nAccuracy:   {final_metrics['accuracy']:.3f}"
            f"\nPrecision:  {final_metrics['precision']:.3f}"
            f"\nRecall:     {final_metrics['recall']:.3f}"
            f"\nF1-Score:   {final_metrics['f1']:.3f}"
            f"\nWorst F1:   {final_metrics['worst_f1']:.3f}"
            f"\nTotal Comm: {sys_summary.total_comm_bytes/1024:.1f} KB"
            f"\nRounds to Convergence: {sys_summary.rounds_to_convergence}"
            f"{dp_label}"
            f"\n{'='*50}"
        )

    return {
        "config": {
            "rounds": cfg.rounds,
            "client_frac": cfg.client_frac,
            "dropout": cfg.dropout,
            "local_steps": cfg.local_steps,
            "lr": cfg.lr,
            "l2": cfg.l2,
            "seed": cfg.seed,
            "aggregator": cfg.aggregator,
            "dp_enabled": dp_cfg.enabled,
            "noise_multiplier": dp_cfg.noise_multiplier if dp_cfg.enabled else None,
            "max_grad_norm": dp_cfg.max_grad_norm if dp_cfg.enabled else None,
            "target_delta": dp_cfg.target_delta if dp_cfg.enabled else None,
        },
        "per_round": per_round,
        "final": {
            "accuracy": final_metrics["accuracy"],
            "precision": final_metrics["precision"],
            "recall": final_metrics["recall"],
            "f1": final_metrics["f1"],
            "worst_f1": final_metrics["worst_f1"],
            "mean_f1": final_metrics["mean_f1"],
            "best_f1": final_metrics["best_f1"],
        },
        "privacy": {
            "epsilon": accountant.epsilon if accountant else None,
            "delta": dp_cfg.target_delta if dp_cfg.enabled else None,
        },
        "system": {
            "total_wall_time_s": sys_summary.total_wall_time_s,
            "total_comm_bytes": sys_summary.total_comm_bytes,
            "cumulative_bytes_per_round": sys_summary.cumulative_bytes_per_round,
            "avg_participation_rate": sys_summary.avg_participation_rate,
            "rounds_to_convergence": sys_summary.rounds_to_convergence,
        },
        # Raw arrays exposed for MIA and plotting
        "_x_train": x_train_s,
        "_y_train": y_train.astype(int),
        "_x_test": x_test_s,
        "_y_test": y_test,
        "_final_params": global_params,
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Federated Learning Training")
    p.add_argument("--csv", type=str, default="data/synthetic/students.csv")
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--client_frac", type=float, default=0.5)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--local_steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--l2", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--aggregator",
        type=str,
        default="fedavg",
        choices=["fedavg", "coord_median", "trimmed_mean", "krum"],
    )
    p.add_argument("--dp", action="store_true", help="Enable differential privacy")
    p.add_argument("--noise_multiplier", type=float, default=1.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--target_delta", type=float, default=1e-5)
    args = p.parse_args()

    dp_cfg = None
    if args.dp:
        dp_cfg = DPConfig(
            enabled=True,
            noise_multiplier=args.noise_multiplier,
            max_grad_norm=args.max_grad_norm,
            target_delta=args.target_delta,
        )

    cfg = FedConfig(
        rounds=args.rounds,
        client_frac=args.client_frac,
        dropout=args.dropout,
        local_steps=args.local_steps,
        lr=args.lr,
        l2=args.l2,
        seed=args.seed,
        aggregator=args.aggregator,
        dp=dp_cfg,
    )

    run_federated(Path(args.csv), test_ratio=args.test_ratio, cfg=cfg, verbose=True)


if __name__ == "__main__":
    main()
