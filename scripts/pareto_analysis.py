"""Pareto Frontier Analysis: Privacy–Utility–Fairness Tradeoffs.

Computes the Pareto-optimal set of federated learning configurations in the
three-dimensional objective space:

    Maximize:  Utility    (F1-score on test set)
    Minimize:  Privacy    (epsilon — smaller ε = stronger privacy)
    Minimize:  Unfairness (worst-client F1 disparity)

A configuration is Pareto-optimal if no other configuration simultaneously
achieves better utility, stronger privacy, AND less unfairness.

Why this matters for research
------------------------------
The Pareto frontier makes the fundamental tradeoffs explicit and quantified:
  - Every point on the frontier represents a "rational choice" — moving to
    any other frontier point helps one objective at the expense of another.
  - Points BELOW the frontier are dominated (strictly worse in every way).
  - The frontier itself is the decision space for the system designer.

Usage
-----
    python scripts/pareto_analysis.py
    python scripts/pareto_analysis.py --results path/to/experiment_results.json
    python scripts/pareto_analysis.py --3d          # 3D scatter plot
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

class Config(NamedTuple):
    label: str
    epsilon: float          # lower = stronger privacy  (minimise)
    utility: float          # F1-score (maximise)
    fairness: float         # worst-client F1 (maximise)
    disparity: float        # best - worst F1 (minimise)
    mia_auc: Optional[float]
    aggregator: str


# ─────────────────────────────────────────────────────────────────────────────
# Pareto dominance
# ─────────────────────────────────────────────────────────────────────────────

def _dominates(a: Config, b: Config) -> bool:
    """
    Return True if config `a` dominates config `b`.

    `a` dominates `b` iff `a` is at least as good as `b` on ALL objectives
    and strictly better on at least one.

    Objectives (all to be maximised after negating ε and disparity):
        1.  F1 utility   (maximise)
        2.  −epsilon     (maximise = minimise epsilon)
        3.  worst F1     (maximise = minimise unfairness)
    """
    # Convert to maximisation objectives
    a_obj = (a.utility, -a.epsilon, a.fairness)
    b_obj = (b.utility, -b.epsilon, b.fairness)

    at_least_as_good = all(ao >= bo for ao, bo in zip(a_obj, b_obj))
    strictly_better = any(ao > bo for ao, bo in zip(a_obj, b_obj))
    return at_least_as_good and strictly_better


def compute_pareto_front(configs: List[Config]) -> Tuple[List[Config], List[Config]]:
    """
    Partition configs into Pareto-optimal (frontier) and dominated sets.

    Returns
    -------
    (frontier, dominated)
    """
    frontier: List[Config] = []
    dominated: List[Config] = []

    for i, cand in enumerate(configs):
        is_dominated = any(
            _dominates(other, cand)
            for j, other in enumerate(configs)
            if j != i
        )
        if is_dominated:
            dominated.append(cand)
        else:
            frontier.append(cand)

    # Sort frontier by epsilon (ascending) for clean plotting
    frontier.sort(key=lambda c: c.epsilon)
    return frontier, dominated


# ─────────────────────────────────────────────────────────────────────────────
# Extract configs from experiment results
# ─────────────────────────────────────────────────────────────────────────────

def extract_configs(results: dict) -> List[Config]:
    """Parse all experiment results into a flat list of Config objects."""
    configs: List[Config] = []

    # FedAvg (no DP): treat epsilon as inf (no privacy)
    fedavg = results.get("fedavg", {})
    if fedavg.get("final"):
        final = fedavg["final"]
        pr = fedavg.get("per_round", [])
        worst = final.get("worst_f1", 0.0)
        best = final.get("best_f1", worst)
        mia = fedavg.get("mia", {}).get("auc")
        configs.append(Config(
            label="FedAvg (no DP)",
            epsilon=float("inf"),
            utility=final.get("f1", 0.0),
            fairness=worst,
            disparity=best - worst,
            mia_auc=mia,
            aggregator="fedavg",
        ))

    # DP-FedAvg sweep
    for entry in results.get("dp_fedavg_sweep", []):
        priv = entry.get("privacy", {})
        final = entry.get("final", {})
        eps = priv.get("epsilon")
        f1 = final.get("f1")
        if eps is None or f1 is None:
            continue
        worst = final.get("worst_f1", 0.0)
        best = final.get("best_f1", worst)
        nm = entry.get("config", {}).get("noise_multiplier", "?")
        mia = entry.get("mia", {}).get("auc")
        configs.append(Config(
            label=f"DP-FedAvg σ={nm}",
            epsilon=float(eps),
            utility=float(f1),
            fairness=float(worst),
            disparity=float(best - worst),
            mia_auc=mia,
            aggregator="dp_fedavg",
        ))

    # Byzantine-robust (no DP)
    for method, entry in results.get("robust_aggregation", {}).items():
        final = entry.get("final", {})
        f1 = final.get("f1")
        if f1 is None:
            continue
        worst = final.get("worst_f1", 0.0)
        best = final.get("best_f1", worst)
        configs.append(Config(
            label=f"{method} (dropout)",
            epsilon=float("inf"),
            utility=float(f1),
            fairness=float(worst),
            disparity=float(best - worst),
            mia_auc=None,
            aggregator=method,
        ))

    # FedProx sweep (if present)
    for entry in results.get("fedprox_sweep", []):
        final = entry.get("final", {})
        f1 = final.get("f1")
        if f1 is None:
            continue
        worst = final.get("worst_f1", 0.0)
        best = final.get("best_f1", worst)
        mu = entry.get("config", {}).get("fedprox_mu", "?")
        configs.append(Config(
            label=f"FedProx μ={mu}",
            epsilon=float("inf"),
            utility=float(f1),
            fairness=float(worst),
            disparity=float(best - worst),
            mia_auc=None,
            aggregator="fedprox",
        ))

    return configs


# ─────────────────────────────────────────────────────────────────────────────
# 2D Pareto plot (epsilon vs utility, coloured by fairness)
# ─────────────────────────────────────────────────────────────────────────────

def plot_pareto_2d(
    frontier: List[Config],
    dominated: List[Config],
    save_path: Path,
) -> None:
    if not HAS_MPL:
        print("matplotlib not available for Pareto 2D plot.")
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 7))

    # Separate finite epsilon from infinite
    def _finite(configs: List[Config]) -> List[Config]:
        return [c for c in configs if c.epsilon < 1e8]

    def _infinite(configs: List[Config]) -> List[Config]:
        return [c for c in configs if c.epsilon >= 1e8]

    all_eps = [c.epsilon for c in frontier + dominated if c.epsilon < 1e8]
    all_f1 = [c.utility for c in frontier + dominated]

    # Plot dominated (grey)
    for c in _finite(dominated):
        ax.scatter(c.epsilon, c.utility, color="#BDBDBD", s=70, zorder=2, alpha=0.6)
    for c in _infinite(dominated):
        ax.scatter([], [], color="#BDBDBD", s=70, alpha=0.6, label="Dominated")

    # Plot frontier (coloured by worst-client F1)
    all_worst = [c.fairness for c in frontier]
    worst_min = min(all_worst) if all_worst else 0
    worst_max = max(all_worst) if all_worst else 1

    cmap = plt.get_cmap("RdYlGn")
    for c in _finite(frontier):
        norm = (c.fairness - worst_min) / (worst_max - worst_min + 1e-8)
        color = cmap(norm)
        sc = ax.scatter(c.epsilon, c.utility, color=color,
                        s=180, zorder=4, edgecolors="white", linewidths=1.5)
        ax.annotate(
            c.label, (c.epsilon, c.utility),
            textcoords="offset points", xytext=(6, 4),
            fontsize=7, color="#333",
        )

    # Connect frontier points with a line
    fe = _finite(frontier)
    if fe:
        fe_sorted = sorted(fe, key=lambda c: c.epsilon)
        ax.plot(
            [c.epsilon for c in fe_sorted],
            [c.utility for c in fe_sorted],
            "k--", lw=1.2, alpha=0.4, zorder=3, label="Pareto Frontier",
        )

    # No-DP points (ε = ∞) — shown as vertical label on right
    for c in _infinite(frontier):
        y_jitter = c.utility + np.random.uniform(-0.002, 0.002)
        ax.axhline(c.utility, color="#9E9E9E", lw=0.8, ls=":", alpha=0.5)
        ax.annotate(
            f"{c.label}\n(no DP, ε=∞)",
            xy=(max(all_eps) * 1.05 if all_eps else 50, c.utility),
            fontsize=7, color="#555", va="center",
        )

    # Colourbar for fairness
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=worst_min, vmax=worst_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Worst-client F1 (fairness ↑)", fontsize=10)

    ax.set_xlabel("Privacy Budget ε  (← stronger privacy)", fontsize=12)
    ax.set_ylabel("Global F1-Score (utility ↑)", fontsize=12)
    ax.set_title(
        "Pareto Frontier: Privacy–Utility–Fairness Tradeoff\n"
        "(Pareto-optimal configs are not dominated; dominated configs in grey)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(max(0, min(all_f1) - 0.05), min(1.0, max(all_f1) + 0.08))

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Pareto 2D saved: {save_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 3D Pareto plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_pareto_3d_plotly(
    frontier: List[Config],
    dominated: List[Config],
    save_path: Path,
) -> None:
    if not HAS_PLOTLY:
        print("plotly not available for 3D Pareto plot.")
        return

    def _make_trace(configs: List[Config], name: str, color: str, symbol: str, size: int):
        eps = [min(c.epsilon, 100) for c in configs]  # cap inf for display
        f1 = [c.utility for c in configs]
        worst = [c.fairness for c in configs]
        labels = [c.label for c in configs]
        return go.Scatter3d(
            x=eps, y=f1, z=worst,
            mode="markers",
            name=name,
            marker=dict(color=color, size=size, symbol=symbol,
                        line=dict(width=0.5, color="white")),
            text=labels,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "ε = %{x:.2f}<br>F1 = %{y:.3f}<br>Worst-F1 = %{z:.3f}"
                "<extra></extra>"
            ),
        )

    fig = go.Figure()
    fig.add_trace(_make_trace(dominated, "Dominated", "#BDBDBD", "circle", 5))
    fig.add_trace(_make_trace(frontier, "Pareto Frontier", "#1976D2", "diamond", 10))

    fig.update_layout(
        title=dict(
            text="<b>3D Pareto Frontier: Privacy–Utility–Fairness</b>",
            x=0.5,
        ),
        scene=dict(
            xaxis_title="Privacy Budget ε (← smaller = more private)",
            yaxis_title="Global F1 (utility ↑)",
            zaxis_title="Worst-client F1 (fairness ↑)",
            bgcolor="#F5F5F5",
        ),
        height=650,
        paper_bgcolor="white",
    )
    html_path = save_path.with_suffix(".html")
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    print(f"  Pareto 3D HTML saved: {html_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Text report
# ─────────────────────────────────────────────────────────────────────────────

def print_pareto_report(frontier: List[Config], dominated: List[Config]) -> None:
    print("\n" + "=" * 65)
    print("PARETO FRONTIER  (Privacy–Utility–Fairness)")
    print("=" * 65)
    print(f"{'Config':<30} {'ε':>8} {'F1':>7} {'WorstF1':>9} {'MIA AUC':>9}")
    print("-" * 65)
    for c in frontier:
        eps_str = f"{c.epsilon:.2f}" if c.epsilon < 1e8 else "∞"
        mia_str = f"{c.mia_auc:.3f}" if c.mia_auc is not None else "  N/A"
        print(f"{c.label:<30} {eps_str:>8} {c.utility:>7.3f} {c.fairness:>9.3f} {mia_str:>9}")

    print(f"\n  {len(frontier)} Pareto-optimal configs, {len(dominated)} dominated.\n")

    if frontier:
        best_utility = max(frontier, key=lambda c: c.utility)
        best_privacy = min(frontier, key=lambda c: c.epsilon)
        best_fairness = max(frontier, key=lambda c: c.fairness)
        print("  Best utility   :", best_utility.label,
              f"  F1={best_utility.utility:.3f}")
        print("  Best privacy   :", best_privacy.label,
              f"  ε={best_privacy.epsilon:.2f}")
        print("  Best fairness  :", best_fairness.label,
              f"  WorstF1={best_fairness.fairness:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Pareto frontier analysis of FL experiments")
    p.add_argument("--results", default="results/experiment_results.json")
    p.add_argument("--out", default="results/figures",
                   help="Output directory for Pareto figures")
    p.add_argument("--3d", dest="three_d", action="store_true",
                   help="Also generate interactive 3D Pareto plot (requires plotly)")
    args = p.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Results not found: {results_path}")
        sys.exit(1)

    with open(results_path) as f:
        data = json.load(f)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = extract_configs(data)
    print(f"Extracted {len(configs)} configurations from results.")

    frontier, dominated = compute_pareto_front(configs)
    print_pareto_report(frontier, dominated)

    if HAS_MPL:
        plot_pareto_2d(frontier, dominated, out_dir / "fig7_pareto_frontier.png")

    if args.three_d and HAS_PLOTLY:
        plot_pareto_3d_plotly(frontier, dominated, out_dir / "fig7_pareto_3d")
    elif args.three_d and not HAS_PLOTLY:
        print("plotly not installed — skipping 3D plot.  pip install plotly")


if __name__ == "__main__":
    main()
