from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GenConfig:
    n_clients: int = 20
    students_per_client: int = 300
    seed: int = 42


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def logit(p: float) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(np.log(p / (1 - p)))



def generate_client_students(rng: np.random.Generator, client_id: int, n: int) -> pd.DataFrame:
    """
    Generate synthetic student records for one client (e.g., school/classroom).
    We introduce client-level shifts to simulate non-identical distributions.
    """
    # Client-level "environment" shifts (non-IID)
    env_attendance_shift = rng.normal(loc=0.0, scale=0.08)
    env_quiz_shift = rng.normal(loc=0.0, scale=7.0)
    env_lms_shift = rng.normal(loc=0.0, scale=0.25)

    attendance_rate = np.clip(rng.beta(7, 2, size=n) + env_attendance_shift, 0, 1)
    avg_quiz_score = np.clip(rng.normal(75, 12, size=n) + env_quiz_shift, 0, 100)
    assignment_completion = np.clip(rng.beta(6, 2.2, size=n) + rng.normal(0, 0.05, size=n), 0, 1)
    lms_activity = np.clip(rng.lognormal(mean=1.2 + env_lms_shift, sigma=0.5, size=n), 0, 50)
    study_hours = np.clip(rng.normal(10, 4, size=n), 0, 25)
    prior_gpa = np.clip(rng.normal(3.0, 0.45, size=n), 0, 4.0)
    late_submissions = np.clip(rng.poisson(lam=1.3, size=n), 0, 12)
    support_requests = np.clip(rng.poisson(lam=0.6, size=n), 0, 10)

    # Risk score: higher risk when engagement/performance is low, lateness high, etc.
    # (You can adjust coefficients later as part of research tuning.)
    linear = (
        -2.2
        - 2.5 * attendance_rate
        - 0.035 * avg_quiz_score
        - 2.0 * assignment_completion
        - 0.05 * study_hours
        - 0.7 * prior_gpa
        + 0.25 * late_submissions
        + 0.18 * support_requests
        - 0.03 * np.sqrt(lms_activity + 1.0)
        + rng.normal(0, 0.35, size=n)  # noise
    )

    # Calibrate prevalence so we don't get a degenerate dataset.
    target_prevalence = 0.25  # 25% at-risk (change to 0.15–0.35 if you want)
    linear = linear - linear.mean() + logit(target_prevalence)
    p_risk = sigmoid(linear)
    at_risk = (rng.uniform(size=n) < p_risk).astype(int)

    df = pd.DataFrame({
        "client_id": client_id,
        "attendance_rate": attendance_rate,
        "avg_quiz_score": avg_quiz_score,
        "assignment_completion": assignment_completion,
        "lms_activity": lms_activity,
        "study_hours": study_hours,
        "prior_gpa": prior_gpa,
        "late_submissions": late_submissions,
        "support_requests": support_requests,
        "at_risk": at_risk,
    })
    return df


def main(cfg: GenConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    frames = []
    for cid in range(cfg.n_clients):
        frames.append(generate_client_students(rng, cid, cfg.students_per_client))

    df = pd.concat(frames, ignore_index=True)

    out_csv = out_dir / "students.csv"
    df.to_csv(out_csv, index=False)

    meta = {
        "n_clients": cfg.n_clients,
        "students_per_client": cfg.students_per_client,
        "n_rows": int(df.shape[0]),
        "seed": cfg.seed,
        "label_name": "at_risk",
    }
    (out_dir / "meta.json").write_text(pd.Series(meta).to_json(indent=2))

    print(f"Saved: {out_csv} ({df.shape[0]} rows)")
    print(f"Class balance (at_risk=1): {df['at_risk'].mean():.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_clients", type=int, default=20)
    p.add_argument("--students_per_client", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default=str(Path("data/synthetic")))
    args = p.parse_args()

    cfg = GenConfig(
        n_clients=args.n_clients,
        students_per_client=args.students_per_client,
        seed=args.seed,
    )
    main(cfg, Path(args.out_dir))
