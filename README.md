# Federated Learning for Student Data Privacy
**AI + Education + Privacy-Preserving Distributed Systems**

A research-grade federated learning system for early-warning student retention
prediction. Models are trained across decentralized clients (schools/classrooms)
— raw student data never leaves each institution. This project studies the
**privacy–utility–fairness tradeoff** under realistic distributed settings.

---

## Research Questions

1. Can federated learning match centralized accuracy on student retention prediction?
2. How does differential privacy affect model utility (accuracy/F1) and fairness across clients?
3. Do privacy mechanisms actually reduce membership inference attack success?
4. How do Byzantine-robust aggregation methods compare under client dropout?
5. What is the communication cost at each privacy level?
6. Can raw gradient sharing leak private training data — and how does DP prevent it?
7. Does local fine-tuning (personalized FL) improve per-client equity?

---

## Architecture

```
Clients (Schools / Classrooms)           Server
 ┌─────────────────────────────┐         ┌──────────────────────────────┐
 │  Local student data         │         │  1. Sample clients (50%)     │
 │  (never leaves institution) │         │  2. Broadcast global model   │
 │                             │─update─▶│  3. Clip updates (if DP)     │
 │  Local SGD training         │         │  4. Aggregate (FedAvg /      │
 │  (FedAvg or FedProx)        │◀─model──│     FedProx / Krum / ...)    │
 │  Optional: QSGD compress    │         │  5. Add Gaussian noise (DP)  │
 └─────────────────────────────┘         │  6. Track privacy budget ε   │
                                         └──────────────────────────────┘
```

---

## Features

| Component | Description |
|---|---|
| Synthetic dataset | 6,000 students, 20 clients, non-IID distribution |
| Centralized baseline | Logistic regression upper bound |
| FedAvg | Weighted federated averaging (McMahan et al., 2017) |
| **FedProx** | Proximal regularisation to reduce client drift (Li et al., 2020) |
| DP-FedAvg | Gaussian mechanism + RDP privacy accounting |
| **QSGD compression** | Stochastic gradient quantisation (4/8/16-bit) |
| Secure aggregation | Pairwise additive masking simulation |
| Membership inference attack | Confidence-based MIA evaluation |
| **Gradient inversion attack** | Analytical data reconstruction from raw gradients |
| Byzantine robustness | Krum, trimmed mean, coordinate median |
| Client dropout | Configurable per-round dropout |
| Fairness metrics | Per-client worst/mean/best F1 |
| **Personalized FL** | Local fine-tuning gain over the global model |
| **Pareto frontier** | 3-objective (privacy, utility, fairness) analysis |
| **Interactive dashboard** | Self-contained HTML report (no server needed) |
| System metrics | Wall time, communication cost, convergence speed |
| Privacy accounting | Renyi DP → (epsilon, delta)-DP composition |
| Visualization | 11 publication-quality figures |
| Test suite | 39 pytest unit tests covering every module |
| YAML config files | Reproducible experiment configs |

---

## Dataset

Synthetic privacy-safe student records with realistic non-IID distribution:

| Feature | Description |
|---|---|
| `attendance_rate` | Fraction of classes attended |
| `avg_quiz_score` | Mean quiz performance (0–100) |
| `assignment_completion` | Assignment submission rate |
| `lms_activity` | Learning management system engagement |
| `study_hours` | Weekly self-reported study hours |
| `prior_gpa` | Previous semester GPA |
| `late_submissions` | Count of late submissions |
| `support_requests` | Help-desk requests |
| **`at_risk`** | **Label: 1 = at risk of dropout (~25% prevalence)** |

- **6,000 records** across **20 clients** (300 per client)
- **Non-IID**: each client has unique environmental shifts (simulating different schools)
- **No real student data** — safe for open research

---

## Quick Start

```bash
# 1. Install dependencies
python -m pip install -r requirements.txt

# 2. Run the full experiment suite (all comparisons + figures + dashboard)
python scripts/run_experiment.py

# 3. Open the interactive dashboard
#    → results/dashboard.html  (open in any browser)

# --- Individual experiments ---

# FedAvg baseline
python federated_train.py

# DP-FedAvg (epsilon ~5)
python federated_train.py --dp --noise_multiplier 1.1

# FedProx (proximal coefficient mu=0.01)
python federated_train.py --fedprox_mu 0.01

# 4-bit QSGD compression (~47% bandwidth savings)
python federated_train.py --compression_bits 4

# Byzantine-robust aggregation with dropout
python federated_train.py --aggregator krum --dropout 0.1

# --- Analysis tools ---

# Interactive HTML dashboard
python dashboard/generate_html.py

# Pareto frontier analysis (privacy vs utility vs fairness)
python scripts/pareto_analysis.py

# Static plots from saved results
python scripts/plot_results.py

# Run test suite
python -m pytest tests/ -v
```

---

## CLI Reference

### `federated_train.py`

```
--csv                path to dataset CSV                [data/synthetic/students.csv]
--rounds             number of FL rounds                [30]
--client_frac        fraction of clients per round      [0.5]
--dropout            per-client dropout probability     [0.0]
--local_steps        local SGD steps per client         [200]
--lr                 learning rate                      [0.05]
--l2                 L2 regularization                  [0.001]
--seed               random seed                        [42]
--aggregator         fedavg|krum|trimmed_mean|coord_median  [fedavg]
--dp                 enable differential privacy        (flag)
--noise_multiplier   noise std / clip norm (sigma)      [1.1]
--max_grad_norm      L2 clip threshold C                [1.0]
--target_delta       delta for (eps,delta)-DP           [1e-5]
--fedprox_mu         FedProx proximal coefficient       [0.0 = disabled]
--compression_bits   QSGD bit-width (4 or 8)           [0 = disabled]
```

### `scripts/run_experiment.py`

```
--rounds      FL rounds per experiment          [30]
--quick       smaller sweep (for testing)
--no-plots    skip figure generation
--skip-new    skip FedProx/compression/inversion/personalization steps
```

### `scripts/pareto_analysis.py`

```
--results     path to experiment_results.json
--out         output directory for figures
--3d          also generate interactive 3D Pareto HTML (requires plotly)
```

---

## Privacy Implementation

### Differential Privacy (DP-FedAvg)

Implements the **Gaussian Mechanism** with server-side noise injection:

1. **Clipping** — each client's update is clipped to L2 norm ≤ C (bounds sensitivity)
2. **Aggregation** — clipped updates averaged via FedAvg
3. **Noise injection** — Gaussian noise N(0, σ²) added, where σ = `noise_multiplier × C`
4. **Accounting** — cumulative budget tracked via **Renyi DP**, converted to (ε, δ)-DP

**Privacy budget by noise level (30 rounds, q = 0.5):**

| noise_multiplier | Approx. ε | Utility impact |
|---|---|---|
| 0.3 | ~95 | Very weak privacy |
| 0.5 | ~42 | Weak |
| 1.0 | ~17 | Moderate |
| 1.1 | ~15 | Good |
| 2.0 | ~8  | Strong |
| 3.0 | ~5  | Very strong |

### FedProx

Adds a **proximal regularisation term** to the local objective:

```
min_w  F(w) + (μ/2) · ‖w − w_global‖²
```

Prevents local models from drifting too far from the global model during
multiple SGD steps — crucial for convergence under non-IID data.
Setting μ = 0 recovers standard FedAvg.

**Recommended μ values:**
- Low heterogeneity (near-IID): μ = 0.0001–0.001
- Realistic non-IID: μ = 0.01–0.05
- High heterogeneity: μ = 0.1–0.5

### Gradient Compression (QSGD)

Stochastic quantisation reduces gradient bit-width before upload:

- **4-bit** (16 levels) → ~47% bandwidth savings, negligible accuracy drop
- **8-bit** (256 levels) → ~44% savings, near-lossless
- **Unbiased**: E[quantize(v)] = v — no systematic error introduced
- Compatible with DP (quantisation is applied before noise injection)

### Gradient Inversion Attack

Demonstrates that **raw gradient sharing is not private**. For logistic
regression with a single training sample (x, y):

```
∂L/∂w = x · residual   →  g_w
∂L/∂b = residual        →  g_b

Reconstruction:  x̂ = g_w / g_b   (exact when g_b ≠ 0)
```

A curious server can analytically recover individual feature vectors
from unprotected client gradients. DP noise corrupts this reconstruction:

| DP noise σ | Cosine similarity | Risk |
|---|---|---|
| 0.0 | 1.000 | HIGH |
| 0.1 | ~0.77 | MEDIUM |
| 0.3 | ~0.42 | LOW |
| ≥ 0.5 | < 0.40 | LOW |

### Secure Aggregation

Simulates **pairwise additive masking** (Bonawitz et al., 2017):
- Each client pair (i, j) shares a random mask r_ij
- Client i uploads `update_i + r_ij`, client j uploads `update_j − r_ij`
- Masks cancel in the sum → server sees only the aggregate, never individual updates

### Membership Inference Attack

Evaluates **empirical privacy leakage** (Shokri et al., 2017):
- **Score**: model confidence on the true label
- **Metrics**: AUC-ROC, advantage (max TPR−FPR), TPR@FPR=0.1
- AUC → 0.5 as DP noise increases (random = no leakage)

---

## Byzantine Robustness

Four aggregation methods compared under 10% client dropout:

| Method | Mechanism | Byzantine Tolerance |
|---|---|---|
| `fedavg` | Weighted mean | None |
| `coord_median` | Coordinate-wise median | Up to ⌊(n−1)/2⌋ clients |
| `trimmed_mean` | Trim 10% tails, then mean | Bounded fraction |
| `krum` | Nearest-neighbor selection | Up to f clients (f < n/2 − 1) |

---

## Personalized Federated Learning

After global aggregation, each client fine-tunes the global model locally
for k SGD steps — no extra communication required.

```
global_model  →  local_finetune(k steps)  →  personal_model
```

Fine-tuning adapts the global compromise to each client's local data
distribution. Even 5–10 steps can recover significant per-client accuracy.

---

## Pareto Frontier Analysis

Computes the **Pareto-optimal set** in 3-objective space:

- Maximize **utility** (F1-score)
- Minimize **privacy cost** (ε — lower = more private)
- Maximize **fairness** (worst-client F1)

A configuration on the Pareto frontier cannot be improved on any one
objective without sacrificing at least one other.

```bash
python scripts/pareto_analysis.py --3d   # interactive 3D plot
```

---

## Output Files

After running `scripts/run_experiment.py`:

```
results/
├── experiment_results.json          # All metrics (JSON, not committed)
├── dashboard.html                   # Interactive Plotly dashboard
└── figures/
    ├── fig1_convergence.png         # F1/Accuracy: FedAvg vs DP-FedAvg
    ├── fig2_privacy_utility.png     # F1 vs epsilon + MIA AUC overlay
    ├── fig3_fairness.png            # Worst/mean/best client F1 over rounds
    ├── fig4_communication.png       # Cumulative bytes over rounds
    ├── fig5_mia_comparison.png      # MIA AUC and advantage bar charts
    ├── fig6_robust_aggregation.png  # F1 by aggregation method
    ├── fig7_pareto_frontier.png     # Privacy-utility-fairness Pareto front
    ├── fig8_fedprox.png             # FedProx vs FedAvg (mu sweep)
    ├── fig9_compression.png         # QSGD: accuracy vs bandwidth savings
    ├── fig10_gradient_leakage.png   # Gradient inversion risk vs DP noise
    └── fig11_personalization.png    # Per-client fine-tuning gain
```

---

## Repository Structure

```
federated-student-privacy/
├── data/synthetic/
│   ├── generate.py              # Non-IID dataset generator
│   ├── students.csv             # 6,000 synthetic student records
│   └── meta.json                # Dataset metadata
│
├── clients/
│   └── local_train.py           # FedAvg + FedProx local SGD
│
├── server/
│   ├── aggregator.py            # FedAvg, Krum, Trimmed Mean, Coord-Median
│   └── scheduler.py             # Client sampling + dropout simulation
│
├── privacy/
│   ├── dp.py                    # Gaussian mechanism + RDP accountant
│   ├── compression.py           # QSGD stochastic gradient quantisation
│   └── secure_aggregation.py    # Pairwise masking simulation
│
├── attacks/
│   ├── membership_inference.py  # Confidence-based MIA
│   └── gradient_inversion.py   # Analytical gradient reconstruction attack
│
├── metrics/
│   ├── utility.py               # Accuracy, precision, recall, F1
│   ├── fairness.py              # Per-client F1 (worst/mean/best)
│   ├── personalization.py       # Local fine-tuning gain evaluation
│   └── systems.py               # Communication cost, wall time, convergence
│
├── dashboard/
│   └── generate_html.py         # Interactive Plotly HTML dashboard
│
├── scripts/
│   ├── run_experiment.py        # Full experiment suite runner (8 steps)
│   ├── plot_results.py          # 11-figure static visualization suite
│   └── pareto_analysis.py       # Pareto frontier analysis + plots
│
├── tests/
│   ├── conftest.py              # pytest fixtures
│   └── test_core.py             # 39 unit tests (all modules)
│
├── configs/
│   ├── fedavg.yaml              # FedAvg baseline config
│   ├── dp_fedavg.yaml           # DP-FedAvg config
│   └── fedprox.yaml             # FedProx config with tuning guidance
│
├── centralized_baseline.py      # Centralized LR for comparison
├── federated_train.py           # Main FL training loop
├── eval.py                      # Standalone evaluation utility
├── requirements.txt             # Python dependencies
└── README.md
```

---

## Running Tests

```bash
python -m pytest tests/ -v
# 39 tests covering: local training, FedProx, aggregators, DP,
# compression, gradient inversion, MIA, fairness, personalization
```

## Troubleshooting

### `No module named pytest`

If tests fail with `No module named pytest`, install dependencies in the active
environment and re-run:

```bash
python -m pip install -r requirements.txt
python -m pytest tests/ -v
```

### Plotly color error when generating dashboard

If `python dashboard/generate_html.py` raises:
`Invalid element(s) received for the 'color' property of bar.marker`,
upgrade to the latest code in this repository. The dashboard now uses
Plotly-compatible `rgba(...)` colors for semi-transparent bars.

---

## References

- McMahan et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. *(FedAvg)*
- Abadi et al. (2016). Deep Learning with Differential Privacy. *(DP-SGD)*
- Mironov (2017). Renyi Differential Privacy of the Gaussian Mechanism.
- Canonne et al. (2020). The Discrete Gaussian for Differential Privacy.
- Li et al. (2020). Federated Optimization in Heterogeneous Networks. *(FedProx)*
- Alistarh et al. (2017). QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding.
- Zhu et al. (2019). Deep Leakage from Gradients. *(Gradient inversion)*
- Shokri et al. (2017). Membership Inference Attacks Against Machine Learning Models.
- Bonawitz et al. (2017). Practical Secure Aggregation for Privacy-Preserving Machine Learning.
- Blanchard et al. (2017). Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent. *(Krum)*
- Yin et al. (2018). Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates. *(Coord-Median)*
- Yu et al. (2020). Salvaging Federated Learning by Local Adaptation. *(Personalization)*

---

## Why This Project

Demonstrates applied mastery of:
- **Federated learning systems** — distributed training without centralizing data
- **Differential privacy** — formal (ε, δ)-DP guarantees with RDP composition
- **Privacy attack evaluation** — gradient inversion and membership inference
- **Byzantine robustness** — defense against malicious or faulty clients
- **Communication efficiency** — gradient quantisation with formal compression bounds
- **Fairness analysis** — per-institution equity and personalization under non-IID data
- **End-to-end research engineering** — reproducible experiments, structured results, interactive visualizations, 39-test suite

Directly applicable to: **AI safety research**, **privacy-preserving ML**, **distributed systems**, **healthcare/education AI**.
