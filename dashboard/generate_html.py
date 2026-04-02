"""Interactive HTML Dashboard for Federated Learning Experiments.

Generates a self-contained, interactive HTML report from experiment_results.json.
No server required — open the output file in any modern browser.

Features
--------
  • Convergence curves (F1 / Accuracy vs rounds) with hover tooltips
  • Privacy-Utility tradeoff scatter plot with MIA AUC overlay
  • Per-client fairness bands (worst / mean / best F1)
  • Cumulative communication cost comparison
  • Membership Inference Attack bar chart
  • Byzantine robustness comparison
  • Gradient Inversion Leakage plot (if present in results)
  • Personalization gain plot (if present in results)

Usage
-----
    python dashboard/generate_html.py
    python dashboard/generate_html.py --results path/to/experiment_results.json
    python dashboard/generate_html.py --out results/dashboard.html
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── dependency check ─────────────────────────────────────────────────────────
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ── colour palette (matches static matplotlib palette) ───────────────────────
C = {
    "fedavg":        "#2196F3",
    "dp_strong":     "#F44336",
    "dp_medium":     "#FF9800",
    "dp_weak":       "#8BC34A",
    "centralized":   "#9C27B0",
    "krum":          "#009688",
    "trimmed_mean":  "#FF5722",
    "coord_median":  "#607D8B",
    "worst":         "#E53935",
    "mean":          "#43A047",
    "best":          "#1E88E5",
    "compression":   "#795548",
    "personal":      "#00BCD4",
}

_DP_COLORS = [
    "#EF5350", "#EC407A", "#AB47BC",
    "#7E57C2", "#42A5F5", "#26A69A", "#66BB6A", "#FFA726",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helper: dp colour by epsilon value
# ─────────────────────────────────────────────────────────────────────────────

def _dp_color(eps: float) -> str:
    if eps < 3:
        return C["dp_strong"]
    if eps < 10:
        return C["dp_medium"]
    return C["dp_weak"]


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert '#RRGGBB' into a Plotly-compatible rgba(...) string."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return hex_color
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ─────────────────────────────────────────────────────────────────────────────
# Individual plot builders
# ─────────────────────────────────────────────────────────────────────────────

def _trace_convergence(fig: go.Figure, results: dict, row: int, col: int) -> None:
    """Add convergence curves (F1 + Accuracy) to subplots."""
    fedavg = results.get("fedavg", {})
    dp_sweep = results.get("dp_fedavg_sweep", [])

    pr = fedavg.get("per_round", [])
    if pr:
        rounds = [r["round"] for r in pr]
        fig.add_trace(go.Scatter(
            x=rounds, y=[r["f1"] for r in pr],
            name="FedAvg (no DP) F1", line=dict(color=C["fedavg"], width=2.5),
            legendgroup="fedavg", showlegend=True,
            hovertemplate="Round %{x}<br>F1 = %{y:.3f}<extra>FedAvg</extra>",
        ), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=rounds, y=[r["accuracy"] for r in pr],
            name="FedAvg Acc", line=dict(color=C["fedavg"], width=1.5, dash="dot"),
            legendgroup="fedavg", showlegend=True,
            hovertemplate="Round %{x}<br>Acc = %{y:.3f}<extra>FedAvg</extra>",
        ), row=row, col=col)

    shown = 0
    for entry in dp_sweep:
        eps = (entry.get("privacy") or {}).get("epsilon")
        if eps is None:
            continue
        pr_dp = entry.get("per_round", [])
        if not pr_dp:
            continue
        color = _dp_color(eps)
        name = f"DP-FedAvg ε≈{eps:.1f}"
        fig.add_trace(go.Scatter(
            x=[r["round"] for r in pr_dp],
            y=[r["f1"] for r in pr_dp],
            name=name, line=dict(color=color, width=1.5, dash="dash"),
            legendgroup=name, showlegend=True,
            hovertemplate=f"Round %{{x}}<br>F1=%{{y:.3f}}<extra>{name}</extra>",
        ), row=row, col=col)
        shown += 1
        if shown >= 4:
            break


def _trace_privacy_utility(fig: go.Figure, results: dict, row: int, col: int) -> None:
    """Add privacy-utility scatter with dual y-axis."""
    dp_sweep = results.get("dp_fedavg_sweep", [])
    fedavg = results.get("fedavg", {})
    centralized = results.get("centralized", {})

    eps_list, f1_list, mia_list = [], [], []
    for entry in dp_sweep:
        eps = (entry.get("privacy") or {}).get("epsilon")
        f1 = (entry.get("final") or {}).get("f1")
        mia = (entry.get("mia") or {}).get("auc")
        if eps is not None and f1 is not None:
            eps_list.append(eps)
            f1_list.append(f1)
            mia_list.append(mia if mia is not None else None)

    if eps_list:
        order = sorted(range(len(eps_list)), key=lambda i: eps_list[i])
        eps_s = [eps_list[i] for i in order]
        f1_s = [f1_list[i] for i in order]
        mia_s = [mia_list[i] for i in order]

        fig.add_trace(go.Scatter(
            x=eps_s, y=f1_s,
            name="F1-Score vs ε", mode="lines+markers",
            line=dict(color=C["fedavg"], width=2),
            marker=dict(size=8),
            hovertemplate="ε=%{x:.2f}<br>F1=%{y:.3f}<extra>DP-FedAvg</extra>",
        ), row=row, col=col)

        if any(m is not None for m in mia_s):
            fig.add_trace(go.Scatter(
                x=eps_s, y=[m if m is not None else None for m in mia_s],
                name="MIA AUC vs ε", mode="lines+markers",
                line=dict(color=C["dp_strong"], width=1.5, dash="dash"),
                marker=dict(symbol="square", size=7),
                yaxis="y2",
                hovertemplate="ε=%{x:.2f}<br>MIA AUC=%{y:.3f}<extra>MIA</extra>",
            ), row=row, col=col)

    # Reference lines
    fa_f1 = (fedavg.get("final") or {}).get("f1")
    if fa_f1:
        fig.add_hline(y=fa_f1, line_dash="dot", line_color=C["fedavg"],
                      annotation_text=f"FedAvg F1={fa_f1:.3f}", row=row, col=col)

    c_f1 = centralized.get("f1")
    if c_f1:
        fig.add_hline(y=c_f1, line_dash="dot", line_color=C["centralized"],
                      annotation_text=f"Centralized F1={c_f1:.3f}", row=row, col=col)


def _trace_fairness(fig: go.Figure, results: dict, row: int, col: int) -> None:
    """Add worst/mean/best F1 fairness bands."""
    fedavg = results.get("fedavg", {})
    pr = fedavg.get("per_round", [])
    if not pr:
        return
    rounds = [r["round"] for r in pr]
    worst = [r.get("worst_f1", 0) for r in pr]
    mean = [r.get("mean_f1", 0) for r in pr]
    best = [r.get("best_f1", 0) for r in pr]

    # Shaded band between worst and best
    fig.add_trace(go.Scatter(
        x=rounds + rounds[::-1],
        y=best + worst[::-1],
        fill="toself", fillcolor="rgba(67,160,71,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Worst–Best range", showlegend=True,
        hoverinfo="skip",
    ), row=row, col=col)
    fig.add_trace(go.Scatter(
        x=rounds, y=worst,
        name="Worst-client F1", line=dict(color=C["worst"], width=1.5, dash="dash"),
        hovertemplate="Round %{x}<br>Worst F1=%{y:.3f}<extra></extra>",
    ), row=row, col=col)
    fig.add_trace(go.Scatter(
        x=rounds, y=mean,
        name="Mean-client F1", line=dict(color=C["mean"], width=2.0),
        hovertemplate="Round %{x}<br>Mean F1=%{y:.3f}<extra></extra>",
    ), row=row, col=col)
    fig.add_trace(go.Scatter(
        x=rounds, y=best,
        name="Best-client F1", line=dict(color=C["best"], width=1.5, dash="dash"),
        hovertemplate="Round %{x}<br>Best F1=%{y:.3f}<extra></extra>",
    ), row=row, col=col)


def _trace_communication(fig: go.Figure, results: dict, row: int, col: int) -> None:
    """Add cumulative communication cost curves."""
    fedavg = results.get("fedavg", {})
    robust = results.get("robust_aggregation", {})

    cum = (fedavg.get("system") or {}).get("cumulative_bytes_per_round", [])
    if cum:
        rounds = list(range(1, len(cum) + 1))
        fig.add_trace(go.Scatter(
            x=rounds, y=[b / 1024 for b in cum],
            name="FedAvg", line=dict(color=C["fedavg"], width=2),
            hovertemplate="Round %{x}<br>%{y:.1f} KB<extra>FedAvg</extra>",
        ), row=row, col=col)

    method_colors = {"coord_median": C["coord_median"],
                     "trimmed_mean": C["trimmed_mean"],
                     "krum": C["krum"]}
    for method, res in robust.items():
        if method == "fedavg":
            continue
        cum_m = (res.get("system") or {}).get("cumulative_bytes_per_round", [])
        if cum_m:
            fig.add_trace(go.Scatter(
                x=list(range(1, len(cum_m) + 1)),
                y=[b / 1024 for b in cum_m],
                name=method, line=dict(color=method_colors.get(method, "grey"),
                                       width=1.5, dash="dash"),
                hovertemplate=f"Round %{{x}}<br>%{{y:.1f}} KB<extra>{method}</extra>",
            ), row=row, col=col)

    # Compression result (if present)
    comp = results.get("compression_sweep", [])
    for entry in comp:
        bits = entry.get("config", {}).get("compression_bits")
        cum_c = (entry.get("system") or {}).get("cumulative_bytes_per_round", [])
        if cum_c and bits:
            fig.add_trace(go.Scatter(
                x=list(range(1, len(cum_c) + 1)),
                y=[b / 1024 for b in cum_c],
                name=f"FedAvg+QSGD-{bits}bit",
                line=dict(color=C["compression"], width=1.5, dash="dot"),
                hovertemplate=f"Round %{{x}}<br>%{{y:.1f}} KB<extra>QSGD-{bits}bit</extra>",
            ), row=row, col=col)


def _trace_mia(fig: go.Figure, results: dict, row: int, col: int) -> None:
    """Bar chart: MIA AUC across methods."""
    labels, aucs, advs = [], [], []
    colors = []

    cent = results.get("centralized", {})
    if cent:
        labels.append("Centralized")
        aucs.append(cent.get("mia_auc", 0))
        advs.append(cent.get("mia_advantage", 0))
        colors.append(C["centralized"])

    fedavg = results.get("fedavg", {})
    mia_fa = fedavg.get("mia", {})
    labels.append("FedAvg (no DP)")
    aucs.append(mia_fa.get("auc", 0))
    advs.append(mia_fa.get("advantage", 0))
    colors.append(C["fedavg"])

    dp_sweep = results.get("dp_fedavg_sweep", [])
    dp_sorted = sorted(
        [(e.get("privacy", {}).get("epsilon", 1e9), e) for e in dp_sweep],
        key=lambda x: x[0],
    )
    for eps, entry in dp_sorted[:5]:
        mia = entry.get("mia", {})
        labels.append(f"DP ε≈{eps:.1f}")
        aucs.append(mia.get("auc", 0))
        advs.append(mia.get("advantage", 0))
        colors.append(_dp_color(eps))

    fig.add_trace(go.Bar(
        x=labels, y=aucs,
        name="MIA AUC", marker_color=colors,
        hovertemplate="%{x}<br>AUC=%{y:.3f}<extra></extra>",
    ), row=row, col=col)
    fig.add_hline(y=0.5, line_dash="dash", line_color="black",
                  annotation_text="Random (AUC=0.5)", row=row, col=col)


def _trace_robustness(fig: go.Figure, results: dict, row: int, col: int) -> None:
    """Bar chart: Byzantine-robust aggregation final F1."""
    robust = results.get("robust_aggregation", {})
    method_order = ["fedavg", "coord_median", "trimmed_mean", "krum"]
    method_labels = ["FedAvg", "Coord-Median", "Trimmed Mean", "Krum"]
    method_colors = [C["fedavg"], C["coord_median"], C["trimmed_mean"], C["krum"]]

    f1s = [robust.get(m, {}).get("final", {}).get("f1", 0) for m in method_order]
    worst = [robust.get(m, {}).get("final", {}).get("worst_f1", 0) for m in method_order]

    fig.add_trace(go.Bar(
        x=method_labels, y=f1s,
        name="Global F1", marker_color=method_colors,
        hovertemplate="%{x}<br>F1=%{y:.3f}<extra>Global</extra>",
    ), row=row, col=col)
    fig.add_trace(go.Bar(
        x=method_labels, y=worst,
        name="Worst-client F1",
        marker_color=[_hex_to_rgba(c, 0.53) for c in method_colors],
        hovertemplate="%{x}<br>Worst F1=%{y:.3f}<extra>Worst client</extra>",
    ), row=row, col=col)


def _trace_gradient_leakage(fig: go.Figure, results: dict, row: int, col: int) -> None:
    """Gradient inversion leakage vs DP noise level."""
    leakage = results.get("gradient_leakage", [])
    if not leakage:
        fig.add_annotation(
            text="Gradient leakage data not in results.<br>"
                 "Run run_experiment.py to generate.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=13, color="grey"),
            row=row, col=col,
        )
        return
    noise = [e.get("dp_noise_std", 0) for e in leakage]
    cos_sim = [e.get("mean_cosine_similarity", 0) for e in leakage]
    risk_colors = {
        "HIGH": C["dp_strong"],
        "MEDIUM": C["dp_medium"],
        "LOW": C["dp_weak"],
    }
    marker_colors = [risk_colors.get(e.get("reconstruction_risk", "HIGH"), "grey")
                     for e in leakage]

    fig.add_trace(go.Scatter(
        x=noise, y=cos_sim,
        mode="lines+markers",
        name="Feature cosine similarity",
        line=dict(color=C["fedavg"], width=2),
        marker=dict(size=10, color=marker_colors, line=dict(width=1, color="white")),
        hovertemplate="σ=%{x:.2f}<br>Cos sim=%{y:.3f}<extra>Gradient Inversion</extra>",
    ), row=row, col=col)
    fig.add_hline(y=0.0, line_dash="dot", line_color="grey",
                  annotation_text="No correlation baseline", row=row, col=col)


def _trace_personalization(fig: go.Figure, results: dict, row: int, col: int) -> None:
    """Personalization gain: global vs fine-tuned F1."""
    personal = results.get("personalization", {})
    if not personal:
        fig.add_annotation(
            text="Personalization data not in results.<br>"
                 "Run run_experiment.py to generate.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=13, color="grey"),
            row=row, col=col,
        )
        return

    per_client = personal.get("per_client", [])
    if per_client:
        cids = [c["client_id"] for c in per_client]
        global_f1s = [c["global_f1"] for c in per_client]
        personal_f1s = [c["personal_f1"] for c in per_client]

        fig.add_trace(go.Bar(
            x=cids, y=global_f1s,
            name="Global model F1",
            marker_color=C["fedavg"],
            hovertemplate="Client %{x}<br>F1=%{y:.3f}<extra>Global</extra>",
        ), row=row, col=col)
        fig.add_trace(go.Bar(
            x=cids, y=personal_f1s,
            name=f"Fine-tuned F1 (k={personal.get('fine_tune_steps', '?')} steps)",
            marker_color=C["personal"],
            hovertemplate="Client %{x}<br>F1=%{y:.3f}<extra>Personalised</extra>",
        ), row=row, col=col)


# ─────────────────────────────────────────────────────────────────────────────
# Master dashboard builder
# ─────────────────────────────────────────────────────────────────────────────

def build_dashboard(results: dict) -> go.Figure:
    """Assemble all plots into a single interactive figure."""
    specs = [
        [{"type": "xy"}, {"type": "xy"}],
        [{"type": "xy"}, {"type": "xy"}],
        [{"type": "xy"}, {"type": "xy"}],
        [{"type": "xy"}, {"type": "xy"}],
    ]
    subplot_titles = [
        "Convergence Curves (F1 per Round)",
        "Privacy–Utility Tradeoff",
        "Per-Client Fairness (FedAvg)",
        "Cumulative Communication Cost",
        "Membership Inference Attack (AUC)",
        "Byzantine-Robust Aggregation",
        "Gradient Inversion Leakage vs DP Noise",
        "Personalization: Global vs Fine-Tuned F1",
    ]

    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=subplot_titles,
        specs=specs,
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
    )

    _trace_convergence(fig, results, row=1, col=1)
    _trace_privacy_utility(fig, results, row=1, col=2)
    _trace_fairness(fig, results, row=2, col=1)
    _trace_communication(fig, results, row=2, col=2)
    _trace_mia(fig, results, row=3, col=1)
    _trace_robustness(fig, results, row=3, col=2)
    _trace_gradient_leakage(fig, results, row=4, col=1)
    _trace_personalization(fig, results, row=4, col=2)

    # ── axis labels ──────────────────────────────────────────────────────────
    axis_labels = {
        "xaxis":  "Round", "yaxis":  "Score",
        "xaxis2": "Privacy Budget ε", "yaxis2": "F1-Score",
        "xaxis3": "Round", "yaxis3": "F1-Score",
        "xaxis4": "Round", "yaxis4": "Cumulative Bytes (KB)",
        "xaxis5": "Method", "yaxis5": "MIA AUC",
        "xaxis6": "Aggregation Method", "yaxis6": "F1-Score",
        "xaxis7": "DP Noise σ", "yaxis7": "Feature Cosine Similarity",
        "xaxis8": "Client ID", "yaxis8": "F1-Score",
    }
    for k, v in axis_labels.items():
        fig.update_layout(**{k: dict(title_text=v)})

    # ── layout ───────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=(
                "<b>Federated Learning for Student Data Privacy</b><br>"
                "<sup>Privacy · Utility · Fairness · Robustness · Communication</sup>"
            ),
            x=0.5, xanchor="center", font=dict(size=20),
        ),
        height=1600,
        paper_bgcolor="white",
        plot_bgcolor="#FAFAFA",
        legend=dict(
            orientation="h", y=-0.02, x=0, font=dict(size=10),
            bgcolor="rgba(255,255,255,0.8)",
        ),
        barmode="group",
        hovermode="x unified",
        font=dict(family="Inter, Arial, sans-serif", size=12),
        margin=dict(t=120, b=80, l=60, r=60),
    )

    # Grid styling
    for i in range(1, 9):
        fig.update_xaxes(showgrid=True, gridcolor="#E0E0E0", row=(i - 1) // 2 + 1,
                         col=(i - 1) % 2 + 1)
        fig.update_yaxes(showgrid=True, gridcolor="#E0E0E0", row=(i - 1) // 2 + 1,
                         col=(i - 1) % 2 + 1)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not HAS_PLOTLY:
        print("plotly is required.  Install with:  pip install plotly")
        sys.exit(1)

    p = argparse.ArgumentParser(description="Generate interactive FL dashboard")
    p.add_argument("--results", default="results/experiment_results.json")
    p.add_argument("--out", default="results/dashboard.html")
    args = p.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        print("Run 'python scripts/run_experiment.py' first.")
        sys.exit(1)

    with open(results_path) as f:
        data = json.load(f)

    print(f"Building dashboard from {results_path} ...")
    fig = build_dashboard(data)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(out_path),
        include_plotlyjs="cdn",   # links to CDN — keep file small
        full_html=True,
        config={"displayModeBar": True, "scrollZoom": True},
    )
    print(f"Dashboard saved: {out_path}")
    print("Open in a browser to explore interactively.")


if __name__ == "__main__":
    main()
