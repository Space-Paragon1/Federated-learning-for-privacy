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

---

## Architecture

```
Clients (Schools / Classrooms)           Server
 ┌─────────────────────────────┐         ┌──────────────────────────────┐
 │  Local student data         │         │  1. Sample clients (50%)     │
 │  (never leaves institution) │         │  2. Broadcast global model   │
 │                             │─update─▶│  3. Clip updates (if DP)     │
 │  Local SGD training         │         │  4. Aggregate (FedAvg /      │
 │  (logistic regression)      │◀─model──│     Krum / Trimmed Mean /    │
 └─────────────────────────────┘         │     Coord-Median)            │
                                         │  5. Add Gaussian noise (DP)  │
                                         │  6. Track privacy budget ε   │
                                         └──────────────────────────────┘
```

---

## Features

| Component | Status | Description |
|---|---|---|
| Synthetic dataset | Done | 6,000 students, 20 clients, non-IID |
| Centralized baseline | Done | Logistic regression upper bound |
| FedAvg | Done | Weighted federated averaging |
| DP-FedAvg | Done | Gaussian mechanism + RDP accounting |
| Secure aggregation | Done | Pairwise additive masking simulation |
| Membership inference attack | Done | Confidence-based MIA evaluation |
| Byzantine robustness | Done | Krum, trimmed mean, coordinate median |
| Client dropout | Done | Configurable per-round dropout |
| Fairness metrics | Done | Per-client worst/mean/best F1 |
| System metrics | Done | Wall time, communication cost, convergence |
| Privacy accounting | Done | Renyi DP → (epsilon, delta)-DP composition |
| Visualization | Done | 6 publication-quality figures |
| Experiment runner | Done | Full sweep with JSON results |
| YAML config files | Done | Reproducible experiment configs |

---

## Dataset

Synthetic privacy-safe student records with realistic non-IID distribution:

| Feature | Description |
|---|---|
| `attendance_rate` | Fraction of classes attended |
| `avg_quiz_score` | Mean quiz performance (0-1) |
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
pip install -r requirements.txt

# 2. Generate synthetic dataset (already present in repo)
python data/synthetic/generate.py

# 3. Run centralized baseline
python centralized_baseline.py

# 4. Run FedAvg (no privacy)
python federated_train.py

# 5. Run DP-FedAvg
python federated_train.py --dp --noise_multiplier 1.1

# 6. Run with Byzantine-robust aggregation
python federated_train.py --aggregator krum --dropout 0.2

# 7. Run the full experiment suite (all comparisons + figures)
python scripts/run_experiment.py

# 8. Generate plots from saved results
python scripts/plot_results.py
```

---

## CLI Reference

### `federated_train.py`

```
--csv            path to dataset CSV            [data/synthetic/students.csv]
--rounds         number of FL rounds            [30]
--client_frac    fraction of clients per round  [0.5]
--dropout        per-client dropout probability [0.0]
--local_steps    local SGD steps per client     [200]
--lr             learning rate                  [0.05]
--l2             L2 regularization              [0.001]
--seed           random seed                    [42]
--aggregator     fedavg|krum|trimmed_mean|coord_median  [fedavg]
--dp             enable differential privacy    (flag)
--noise_multiplier   sigma / C                  [1.1]
--max_grad_norm      L2 clip threshold C        [1.0]
--target_delta       delta for (eps,delta)-DP   [1e-5]
```

### `scripts/run_experiment.py`

```
--rounds    FL rounds per experiment    [30]
--quick     smaller sweep (for testing)
--no-plots  skip figure generation
```

### `eval.py`

```
--results   path to experiment_results.json
--mode      fedavg | dp_best | centralized
```

---

## Privacy Implementation

### Differential Privacy (DP-FedAvg)

Implements the **Gaussian Mechanism** with server-side noise injection:

1. **Clipping**: Each client's parameter delta is clipped to L2 norm <= C
   (bounds the sensitivity of each update)
2. **Aggregation**: Clipped deltas are averaged via FedAvg
3. **Noise injection**: Gaussian noise N(0, sigma^2) added to the aggregate,
   where sigma = `noise_multiplier x max_grad_norm`
4. **Accounting**: Cumulative privacy budget tracked via **Renyi DP** moments accountant,
   converted to final (epsilon, delta)-DP guarantee

**Privacy budget by noise level (30 rounds, q=0.5):**

| noise_multiplier | Approx. epsilon | Utility impact |
|---|---|---|
| 0.3 | ~60 | Very weak privacy |
| 0.5 | ~20 | Weak privacy |
| 1.0 | ~7  | Moderate |
| 1.1 | ~5  | Good |
| 2.0 | ~2  | Strong |
| 3.0 | ~1  | Very strong |

### Secure Aggregation

Simulates **pairwise additive masking** (Bonawitz et al., 2017):
- Each client pair (i, j) shares a random mask r_ij
- Client i uploads `update_i + r_ij`, client j uploads `update_j - r_ij`
- Masks cancel in the server's sum → exact weighted average, zero individual leakage
- Server only ever sees masked (unintelligible) individual uploads

### Membership Inference Attack

Evaluates **empirical privacy leakage** via confidence-based MIA (Shokri et al., 2017):
- **Members**: samples from training set (model has seen them)
- **Non-members**: held-out test samples
- **Score**: model confidence assigned to the true label
- **Metrics**: AUC-ROC, advantage (max TPR-FPR), accuracy, TPR@FPR=0.1

A model with no privacy protection will have AUC > 0.5.
DP-FedAvg with sufficient noise pushes AUC back toward 0.5 (random = no leakage).

---

## Byzantine Robustness

Four aggregation methods compared under 10% client dropout:

| Method | Mechanism | Byzantine Tolerance |
|---|---|---|
| `fedavg` | Weighted mean | None |
| `coord_median` | Coordinate-wise median | Up to floor((n-1)/2) clients |
| `trimmed_mean` | Trim 10% tails then mean | Bounded fraction |
| `krum` | Nearest-neighbor selection | Up to f clients (f < n/2 - 1) |

---

## Output Files

After running `scripts/run_experiment.py`:

```
results/
├── experiment_results.json          # All metrics for all experiments
└── figures/
    ├── fig1_convergence.png         # F1/Accuracy per round: FedAvg vs DP
    ├── fig2_privacy_utility.png     # F1 vs epsilon + MIA AUC overlay
    ├── fig3_fairness.png            # Worst/mean/best client F1 over rounds
    ├── fig4_communication.png       # Cumulative bytes over rounds
    ├── fig5_mia_comparison.png      # MIA AUC and advantage bar charts
    └── fig6_robust_aggregation.png  # F1 by aggregation method
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
│   └── local_train.py           # Client-side SGD (logistic regression)
│
├── server/
│   ├── aggregator.py            # FedAvg, Krum, Trimmed Mean, Coord-Median
│   └── scheduler.py             # Client sampling + dropout simulation
│
├── privacy/
│   ├── dp.py                    # Gaussian mechanism + RDP accountant
│   └── secure_aggregation.py    # Pairwise masking simulation
│
├── attacks/
│   └── membership_inference.py  # Confidence-based MIA
│
├── metrics/
│   ├── utility.py               # Accuracy, precision, recall, F1
│   ├── fairness.py              # Per-client F1 (worst/mean/best)
│   └── systems.py               # Communication cost, wall time, convergence
│
├── scripts/
│   ├── run_experiment.py        # Full experiment suite runner
│   └── plot_results.py          # 6-figure visualization suite
│
├── configs/
│   ├── fedavg.yaml              # FedAvg baseline config
│   └── dp_fedavg.yaml           # DP-FedAvg config
│
├── centralized_baseline.py      # Centralized LR for comparison
├── federated_train.py           # Main FL training loop (DP-aware)
├── eval.py                      # Standalone evaluation utility
├── requirements.txt             # Python dependencies
└── README.md
```

---

## References

- McMahan et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. (FedAvg)
- Abadi et al. (2016). Deep Learning with Differential Privacy. (DP-SGD)
- Mironov (2017). Renyi Differential Privacy of the Gaussian Mechanism.
- Canonne et al. (2020). The Discrete Gaussian for Differential Privacy.
- Shokri et al. (2017). Membership Inference Attacks Against Machine Learning Models.
- Bonawitz et al. (2017). Practical Secure Aggregation for Privacy-Preserving Machine Learning.
- Blanchard et al. (2017). Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent. (Krum)
- Yin et al. (2018). Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates. (Coord-Median)

---

## Why This Project

Demonstrates applied mastery of:
- **Federated learning systems** — distributed training without centralizing data
- **Differential privacy** — formal (epsilon, delta)-DP guarantees with RDP composition
- **Privacy attack evaluation** — empirical measurement of actual privacy risk
- **Byzantine robustness** — defense against malicious or faulty clients
- **Fairness analysis** — per-institution equity under non-IID data
- **End-to-end research engineering** — reproducible experiments, structured results, publication-quality figures

Directly applicable to: **AI safety research**, **privacy-preserving ML**, **distributed systems**, **healthcare/education AI**.
