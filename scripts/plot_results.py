"""Visualization Suite for Federated Learning Experiments.

Generates publication-quality figures from experiment_results.json:

  Figure 1 — Convergence Curves
      F1-score and Accuracy per round: FedAvg vs DP-FedAvg
      (shows cost of privacy on learning speed)

  Figure 2 — Privacy-Utility Tradeoff
      Final F1 vs epsilon (lower epsilon = stronger privacy, lower utility)
      Also overlays MIA AUC to show privacy actually improves with DP

  Figure 3 — Fairness Disparity Over Rounds
      Worst / mean / best client F1 over rounds for FedAvg vs best DP config

  Figure 4 — Communication Cost
      Cumulative bytes over rounds for all methods

  Figure 5 — MIA Results Comparison
      Bar chart: AUC and advantage for centralized / FedAvg / DP configs

  Figure 6 — Byzantine Robustness
      Final F1 and worst-client F1 across aggregation methods

Usage:
    python scripts/plot_results.py                          # uses results/experiment_results.json
    python scripts/plot_results.py --results path/to/file.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend (safe for all environments)
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

PALETTE = {
    "fedavg":       "#2196F3",   # blue
    "dp_strong":    "#F44336",   # red  (low eps)
    "dp_medium":    "#FF9800",   # orange
    "dp_weak":      "#8BC34A",   # light green
    "centralized":  "#9C27B0",   # purple
    "krum":         "#009688",   # teal
    "trimmed_mean": "#FF5722",   # deep orange
    "coord_median": "#607D8B",   # blue-grey
    "worst":        "#E53935",
    "mean":         "#43A047",
    "best":         "#1E88E5",
}

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "font.size":        11,
}


def _apply_style() -> None:
    plt.rcParams.update(STYLE)


def _save(fig: "plt.Figure", path: Path, title: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ---------------------------------------------------------------------------
# Figure 1: Convergence Curves
# ---------------------------------------------------------------------------

def plot_convergence(
    fedavg_result: dict,
    dp_results: List[dict],
    save_dir: Path,
) -> None:
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Convergence Curves: FedAvg vs DP-FedAvg", fontsize=14, fontweight="bold")

    rounds_fa = [r["round"] for r in fedavg_result["per_round"]]
    f1_fa = [r["f1"] for r in fedavg_result["per_round"]]
    acc_fa = [r["accuracy"] for r in fedavg_result["per_round"]]

    axes[0].plot(rounds_fa, f1_fa, color=PALETTE["fedavg"], lw=2.0, label="FedAvg (no DP)")
    axes[1].plot(rounds_fa, acc_fa, color=PALETTE["fedavg"], lw=2.0, label="FedAvg (no DP)")

    # Pick a few representative DP configs to show
    shown = 0
    for entry in dp_results:
        eps = entry.get("privacy", {}).get("epsilon")
        if eps is None:
            continue
        rounds_dp = [r["round"] for r in entry["per_round"]]
        f1_dp = [r["f1"] for r in entry["per_round"]]
        acc_dp = [r["accuracy"] for r in entry["per_round"]]
        nm = entry["config"].get("noise_multiplier", "?")
        label = f"DP-FedAvg  ε≈{eps:.1f}"
        color = PALETTE["dp_strong"] if eps < 5 else (PALETTE["dp_medium"] if eps < 15 else PALETTE["dp_weak"])
        axes[0].plot(rounds_dp, f1_dp, lw=1.5, ls="--", color=color, label=label, alpha=0.85)
        axes[1].plot(rounds_dp, acc_dp, lw=1.5, ls="--", color=color, label=label, alpha=0.85)
        shown += 1
        if shown >= 3:
            break

    for ax, ylabel in zip(axes, ["F1-Score", "Accuracy"]):
        ax.set_xlabel("Round")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)

    axes[0].set_title("F1-Score per Round")
    axes[1].set_title("Accuracy per Round")
    fig.tight_layout()
    _save(fig, save_dir / "fig1_convergence.png", "convergence")


# ---------------------------------------------------------------------------
# Figure 2: Privacy-Utility Tradeoff
# ---------------------------------------------------------------------------

def plot_privacy_utility(
    fedavg_result: dict,
    dp_results: List[dict],
    centralized: Optional[dict],
    save_dir: Path,
) -> None:
    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title("Privacy–Utility Tradeoff", fontsize=14, fontweight="bold")

    epsilons, f1s, mia_aucs = [], [], []
    for entry in dp_results:
        eps = entry.get("privacy", {}).get("epsilon")
        f1 = entry.get("final", {}).get("f1")
        mia = entry.get("mia", {}).get("auc")
        if eps is not None and f1 is not None:
            epsilons.append(eps)
            f1s.append(f1)
            mia_aucs.append(mia if mia is not None else float("nan"))

    if epsilons:
        order = sorted(range(len(epsilons)), key=lambda i: epsilons[i])
        epsilons = [epsilons[i] for i in order]
        f1s = [f1s[i] for i in order]
        mia_aucs = [mia_aucs[i] for i in order]

        ax.plot(epsilons, f1s, "o-", color=PALETTE["fedavg"], lw=2, ms=7, label="F1-Score (left axis)")

        ax2 = ax.twinx()
        ax2.plot(epsilons, mia_aucs, "s--", color=PALETTE["dp_strong"], lw=1.5, ms=6, label="MIA AUC (right axis)")
        ax2.axhline(0.5, color="grey", lw=1, ls=":", label="Random MIA (AUC=0.5)")
        ax2.set_ylabel("MIA AUC", color=PALETTE["dp_strong"])
        ax2.set_ylim(0.45, 1.05)
        ax2.tick_params(axis="y", colors=PALETTE["dp_strong"])
        ax2.legend(loc="upper right", fontsize=9)

    # Reference lines
    fa_f1 = fedavg_result["final"]["f1"]
    ax.axhline(fa_f1, color=PALETTE["fedavg"], lw=1.2, ls=":", label=f"FedAvg (no DP) F1={fa_f1:.3f}")
    if centralized:
        c_f1 = centralized["f1"]
        ax.axhline(c_f1, color=PALETTE["centralized"], lw=1.2, ls="--", label=f"Centralized F1={c_f1:.3f}")

    ax.set_xlabel("Privacy Budget ε  (lower = stronger privacy)")
    ax.set_ylabel("F1-Score", color=PALETTE["fedavg"])
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    _save(fig, save_dir / "fig2_privacy_utility.png", "privacy-utility")


# ---------------------------------------------------------------------------
# Figure 3: Fairness Disparity
# ---------------------------------------------------------------------------

def plot_fairness(
    fedavg_result: dict,
    dp_results: List[dict],
    save_dir: Path,
) -> None:
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Fairness: Per-Client F1 Disparity Over Rounds", fontsize=14, fontweight="bold")

    def _plot_fairness_single(ax: "plt.Axes", result: dict, title: str) -> None:
        rounds = [r["round"] for r in result["per_round"]]
        worst = [r["worst_f1"] for r in result["per_round"]]
        mean = [r["mean_f1"] for r in result["per_round"]]
        best = [r["best_f1"] for r in result["per_round"]]
        ax.fill_between(rounds, worst, best, alpha=0.15, color=PALETTE["mean"], label="Worst–Best range")
        ax.plot(rounds, worst, color=PALETTE["worst"], lw=1.5, ls="--", label="Worst client F1")
        ax.plot(rounds, mean, color=PALETTE["mean"], lw=2.0, label="Mean client F1")
        ax.plot(rounds, best, color=PALETTE["best"], lw=1.5, ls="--", label="Best client F1")
        ax.set_xlabel("Round")
        ax.set_ylabel("F1-Score")
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.legend(fontsize=9)

    _plot_fairness_single(axes[0], fedavg_result, "FedAvg (no DP)")

    # Find the DP config with moderate epsilon for comparison
    dp_moderate = None
    for entry in dp_results:
        eps = (entry.get("privacy") or {}).get("epsilon")
        if eps is not None and 3 < eps < 20:
            dp_moderate = entry
            break
    if dp_moderate is None and dp_results:
        dp_moderate = dp_results[len(dp_results) // 2]

    if dp_moderate:
        eps = (dp_moderate.get("privacy") or {}).get("epsilon", "?")
        _plot_fairness_single(axes[1], dp_moderate, f"DP-FedAvg  ε≈{eps:.1f}")
    else:
        axes[1].text(0.5, 0.5, "No DP results available", ha="center", va="center", transform=axes[1].transAxes)

    fig.tight_layout()
    _save(fig, save_dir / "fig3_fairness.png", "fairness")


# ---------------------------------------------------------------------------
# Figure 4: Communication Cost
# ---------------------------------------------------------------------------

def plot_communication(
    fedavg_result: dict,
    robust_results: Optional[Dict[str, dict]],
    save_dir: Path,
) -> None:
    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title("Cumulative Communication Cost Over Rounds", fontsize=14, fontweight="bold")

    cum = fedavg_result["system"].get("cumulative_bytes_per_round", [])
    if cum:
        rounds = list(range(1, len(cum) + 1))
        cum_kb = [b / 1024 for b in cum]
        ax.plot(rounds, cum_kb, color=PALETTE["fedavg"], lw=2, label="FedAvg")

    if robust_results:
        method_colors = {
            "coord_median": PALETTE["coord_median"],
            "trimmed_mean": PALETTE["trimmed_mean"],
            "krum":         PALETTE["krum"],
        }
        for method, res in robust_results.items():
            if method == "fedavg":
                continue
            cum_m = res.get("system", {}).get("cumulative_bytes_per_round", [])
            if cum_m:
                rounds_m = list(range(1, len(cum_m) + 1))
                cum_kb_m = [b / 1024 for b in cum_m]
                ax.plot(rounds_m, cum_kb_m, lw=1.5, ls="--",
                        color=method_colors.get(method, "grey"), label=method)

    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative Bytes (KB)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, save_dir / "fig4_communication.png", "communication")


# ---------------------------------------------------------------------------
# Figure 5: MIA AUC Comparison
# ---------------------------------------------------------------------------

def plot_mia_comparison(
    fedavg_result: dict,
    dp_results: List[dict],
    centralized: Optional[dict],
    save_dir: Path,
) -> None:
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Membership Inference Attack: Privacy Leakage", fontsize=14, fontweight="bold")

    labels, aucs, advantages = [], [], []

    if centralized:
        labels.append("Centralized")
        aucs.append(centralized.get("mia_auc", float("nan")))
        advantages.append(centralized.get("mia_advantage", float("nan")))

    fedavg_mia = fedavg_result.get("mia", {})
    labels.append("FedAvg\n(no DP)")
    aucs.append(fedavg_mia.get("auc", float("nan")))
    advantages.append(fedavg_mia.get("advantage", float("nan")))

    # Add a few DP points sorted by epsilon
    dp_sorted = sorted(
        [(e.get("privacy", {}).get("epsilon", float("inf")), e) for e in dp_results],
        key=lambda x: x[0],
    )
    for eps, entry in dp_sorted[:4]:
        mia = entry.get("mia", {})
        labels.append(f"DP-FedAvg\nε≈{eps:.1f}")
        aucs.append(mia.get("auc", float("nan")))
        advantages.append(mia.get("advantage", float("nan")))

    x = np.arange(len(labels))
    colors = [PALETTE["centralized"] if "Central" in l else
              PALETTE["fedavg"] if "no DP" in l else PALETTE["dp_medium"]
              for l in labels]

    axes[0].bar(x, aucs, color=colors, alpha=0.85, edgecolor="white", linewidth=1.2)
    axes[0].axhline(0.5, color="black", lw=1.2, ls="--", label="Random (AUC=0.5)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel("AUC-ROC")
    axes[0].set_title("MIA AUC (lower = more private)")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=9)

    axes[1].bar(x, advantages, color=colors, alpha=0.85, edgecolor="white", linewidth=1.2)
    axes[1].axhline(0.0, color="black", lw=1.2, ls="--", label="No advantage = 0")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel("Attack Advantage (TPR − FPR)")
    axes[1].set_title("MIA Advantage (lower = more private)")
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    _save(fig, save_dir / "fig5_mia_comparison.png", "mia")


# ---------------------------------------------------------------------------
# Figure 6: Byzantine Robustness
# ---------------------------------------------------------------------------

def plot_robust_aggregation(
    robust_results: Dict[str, dict],
    save_dir: Path,
) -> None:
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Aggregation Method Comparison (10% Client Dropout)",
                 fontsize=14, fontweight="bold")

    method_order = ["fedavg", "coord_median", "trimmed_mean", "krum"]
    method_labels = ["FedAvg", "Coord-Median", "Trimmed Mean", "Krum"]
    method_colors = [
        PALETTE["fedavg"], PALETTE["coord_median"],
        PALETTE["trimmed_mean"], PALETTE["krum"],
    ]

    final_f1s, worst_f1s = [], []
    for method in method_order:
        res = robust_results.get(method, {})
        final_f1s.append(res.get("final", {}).get("f1", 0.0))
        worst_f1s.append(res.get("final", {}).get("worst_f1", 0.0))

    x = np.arange(len(method_order))
    axes[0].bar(x, final_f1s, color=method_colors, alpha=0.85, edgecolor="white", lw=1.2)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(method_labels)
    axes[0].set_ylabel("Global F1-Score")
    axes[0].set_title("Global F1 by Aggregation Method")
    axes[0].set_ylim(0, 1.0)

    axes[1].bar(x, worst_f1s, color=method_colors, alpha=0.85, edgecolor="white", lw=1.2)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(method_labels)
    axes[1].set_ylabel("Worst-Client F1-Score")
    axes[1].set_title("Worst-Client F1 (Fairness)")
    axes[1].set_ylim(0, 1.0)

    # Convergence round comparison
    for i, method in enumerate(method_order):
        res = robust_results.get(method, {})
        rounds = res.get("system", {}).get("rounds_to_convergence", -1)
        label = f"{rounds}r" if rounds > 0 else "—"
        for ax in axes:
            ax.text(i, ax.get_ylim()[1] * 0.96, label,
                    ha="center", va="top", fontsize=9, color="black")

    fig.text(0.5, 0.01, "Numbers above bars = rounds to F1≥0.70 convergence",
             ha="center", fontsize=9, color="grey")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, save_dir / "fig6_robust_aggregation.png", "robustness")


# ---------------------------------------------------------------------------
# Master entry point
# ---------------------------------------------------------------------------

def generate_all_plots(all_results: dict, figures_dir: Path) -> None:
    if not HAS_MPL:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    figures_dir.mkdir(parents=True, exist_ok=True)

    fedavg = all_results.get("fedavg", {})
    dp_sweep = all_results.get("dp_fedavg_sweep", [])
    centralized = all_results.get("centralized")
    robust = all_results.get("robust_aggregation", {})

    print(f"  Generating figures in {figures_dir}/")

    try:
        plot_convergence(fedavg, dp_sweep, figures_dir)
    except Exception as e:
        print(f"  fig1 error: {e}")

    try:
        plot_privacy_utility(fedavg, dp_sweep, centralized, figures_dir)
    except Exception as e:
        print(f"  fig2 error: {e}")

    try:
        plot_fairness(fedavg, dp_sweep, figures_dir)
    except Exception as e:
        print(f"  fig3 error: {e}")

    try:
        plot_communication(fedavg, robust, figures_dir)
    except Exception as e:
        print(f"  fig4 error: {e}")

    try:
        plot_mia_comparison(fedavg, dp_sweep, centralized, figures_dir)
    except Exception as e:
        print(f"  fig5 error: {e}")

    try:
        if robust:
            plot_robust_aggregation(robust, figures_dir)
    except Exception as e:
        print(f"  fig6 error: {e}")

    print("  All figures generated.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Plot federated learning experiment results")
    p.add_argument(
        "--results",
        type=str,
        default="results/experiment_results.json",
        help="Path to experiment_results.json",
    )
    p.add_argument(
        "--out",
        type=str,
        default="results/figures",
        help="Output directory for figures",
    )
    args = p.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        print("Run 'python scripts/run_experiment.py' first.")
        sys.exit(1)

    with open(results_path) as f:
        data = json.load(f)

    generate_all_plots(data, Path(args.out))
