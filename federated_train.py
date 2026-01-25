from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import sched
from typing import Dict, List, Tuple
from server.scheduler import schedule_round
from metrics.fairness import per_client_f1



import numpy as np
import pandas as pd

from clients.local_train import LRParams, local_train_logreg
from server.aggregator import ClientUpdate, fedavg
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

def estimate_bytes_for_params(params: LRParams) -> int:
    # float64: 8 bytes each
    return int(params.w.size * 8 + 8)  # weights + bias

def standardize_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = x.mean(axis=0)
    sigma = x.std(axis=0) + 1e-8
    return mu, sigma


def standardize_apply(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (x - mu) / sigma


def train_test_split(df: pd.DataFrame, test_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    cut = int(len(df) * (1 - test_ratio))
    train_idx, test_idx = idx[:cut], idx[cut:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def predict(params: LRParams, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    p = sigmoid(x @ params.w + params.b)
    return (p >= threshold).astype(int)

def predict_probs(params: LRParams, x: np.ndarray) -> np.ndarray:
    return sigmoid(x @ params.w + params.b)


@dataclass
class FedConfig:
    rounds: int = 30
    dropout: float = 0.0
    client_frac: float = 0.5
    local_steps: int = 200
    lr: float = 0.05
    l2: float = 1e-3
    seed: int = 42


def main(csv_path: Path, test_ratio: float, cfg: FedConfig) -> None:
    df = pd.read_csv(csv_path)
    train_df, test_df = train_test_split(df, test_ratio=test_ratio, seed=cfg.seed)

    # Standardize using TRAIN ONLY (important!)
    x_train = train_df[FEATURES].to_numpy(dtype=float)
    y_train = train_df[LABEL].to_numpy(dtype=float)

    x_test = test_df[FEATURES].to_numpy(dtype=float)
    y_test = test_df[LABEL].to_numpy(dtype=int)

    mu, sigma = standardize_fit(x_train)
    train_df_s = train_df.copy()
    test_df_s = test_df.copy()

    train_df_s[FEATURES] = standardize_apply(x_train, mu, sigma)
    test_df_s[FEATURES] = standardize_apply(x_test, mu, sigma)
    # ---------------------------------------------------------
# Build per-client test sets (for fairness evaluation)
# ---------------------------------------------------------
    test_clients = {}
    for cid, g in test_df_s.groupby("client_id"):
        test_clients[int(cid)] = (
            g[FEATURES].to_numpy(dtype=float),
            g[LABEL].to_numpy(dtype=int),
        )

    # Group train data by client
    clients = {}
    for cid, g in train_df_s.groupby("client_id"):
        clients[int(cid)] = (
            g[FEATURES].to_numpy(dtype=float),
            g[LABEL].to_numpy(dtype=float),
        )

    rng = np.random.default_rng(cfg.seed)
    d = len(FEATURES)
    global_params = LRParams(w=rng.normal(0, 0.01, size=d), b=0.0)

    client_ids = sorted(clients.keys())

    for r in range(cfg.rounds):
        sched = schedule_round(rng, client_ids, cfg.client_frac, cfg.dropout)
        selected = sched.selected
        survivors = sched.survivors


        updates: List[ClientUpdate] = []
        for cid in survivors:
            x_c, y_c = clients[int(cid)]
            new_params = local_train_logreg(
                x_c, y_c, global_params,
                lr=cfg.lr, local_steps=cfg.local_steps, l2=cfg.l2
            )
            updates.append(ClientUpdate(params=new_params, n_samples=len(x_c)))

        global_params = fedavg(updates)

        bytes_per_model = estimate_bytes_for_params(global_params)

        broadcast_bytes = bytes_per_model * len(selected)     # server -> selected clients
        upload_bytes = bytes_per_model * len(survivors)       # survivors -> server
        total_bytes = broadcast_bytes + upload_bytes

        if (r + 1) % 5 == 0 or r == 0:
            print(
            f"         comm: broadcast={broadcast_bytes/1024:.2f}KB "
            f"upload={upload_bytes/1024:.2f}KB total={total_bytes/1024:.2f}KB "
            f"selected={len(selected)} returned={len(survivors)}"
            )


        # quick eval each round
        # quick eval each round
        y_pred = predict(global_params, test_df_s[FEATURES].to_numpy(dtype=float))
        metrics = binary_classification_metrics(y_test, y_pred)

# ---------- FAIRNESS COMPUTATION (THIS IS THE PLACE) ----------
        client_to_ytrue = {}
        client_to_ypred = {}
        for cid, (x_c, y_c) in test_clients.items():
            client_to_ytrue[cid] = y_c
            client_to_ypred[cid] = (predict_probs(global_params, x_c) >= 0.5).astype(int)
        fair = per_client_f1(client_to_ytrue, client_to_ypred)
# -------------------------------------------------------------

        if (r + 1) % 5 == 0 or r == 0:
            print(f"[Round {r+1:02d}] acc={metrics.accuracy:.3f} f1={metrics.f1:.3f}")
            print(
                f"         fairness: worst_f1={fair.worst_f1:.3f} "
                f"mean_f1={fair.mean_f1:.3f} best_f1={fair.best_f1:.3f}"
            )

    print("\nFedAvg Baseline Complete")
    y_pred = predict(global_params, test_df_s[FEATURES].to_numpy(dtype=float))
    metrics = binary_classification_metrics(y_test, y_pred)
    print(f"Final Accuracy:  {metrics.accuracy:.3f}")
    print(f"Final Precision: {metrics.precision:.3f}")
    print(f"Final Recall:    {metrics.recall:.3f}")
    print(f"Final F1:        {metrics.f1:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dropout", type=float, default=0.0)  # 0.0 to 0.5
    p.add_argument("--csv", type=str, default="data/synthetic/students.csv")
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--client_frac", type=float, default=0.5)
    p.add_argument("--local_steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--l2", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = FedConfig(
        rounds=args.rounds,
        dropout=args.dropout,
        client_frac=args.client_frac,
        local_steps=args.local_steps,
        lr=args.lr,
        l2=args.l2,
        seed=args.seed,
    )

    main(Path(args.csv), test_ratio=args.test_ratio, cfg=cfg)
