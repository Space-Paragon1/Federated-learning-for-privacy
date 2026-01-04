    Federated Learning for Student Data Privacy
AI + Education + Distributed Systems Research Project

Overview
This project explores privacy-preserving machine learning for education using federated learning (FL). Instead of collecting and centralizing sensitive student data (grades, engagement logs, learning behaviors), models are trained across decentralized clients (e.g., schools or classrooms), sharing only model updates rather than raw data.
The goal is to study how model utility, privacy guarantees, system efficiency, and fairness interact in realistic educational settings with heterogeneous (non-IID) data distributions.
This repository is part of a broader AI + Education + Systems research portfolio, alongside work on offline AI tutoring, explainable learning analytics, and distributed AI systems.

        Research Motivation
Educational data is:
-Highly sensitive (academic performance, learning behavior)
-Distributed across institutions
-Subject to privacy, ethical, and regulatory constraints

Federated learning provides a promising alternative by enabling collaborative model training without exposing raw student data.
This project asks:
-Can federated learning achieve performance close to centralized models?
-What privacy–utility trade-offs arise when privacy mechanisms are added?
-How do system constraints (communication, dropouts) affect learning?
-Does federated learning introduce or mitigate fairness gaps across institutions?

Core Use Case (Phase 1)
Early-Warning Student Retention Prediction
-Task: Binary classification (at-risk vs not at-risk)
-Clients: Schools or classrooms
-Data: Synthetic student engagement and performance features
-Motivation: Retention prediction is useful, sensitive, and realistic

The framework is designed to later support:
-Knowledge tracing
-Personalized practice recommendation
-On-device AI tutor personalization

System Architecture
Clients (Schools / Classrooms)
 ├─ Local student data (never shared)
 ├─ Local model training
 └─ Model updates only
        ↓
Federated Server
 ├─ Client sampling
 ├─ Secure aggregation (planned)
 ├─ Differential privacy (planned)
 └─ Global model update
        ↓
Updated model broadcast to clients

Repository Structure
federated-student-privacy/
│
├── data/
│   └── synthetic/
│       ├── generate.py        # Synthetic student dataset generator
│       └── students.csv       # Generated dataset
│
├── clients/
│   └── local_train.py         # Client-side local training
│
├── server/
│   └── aggregator.py          # FedAvg aggregation logic
│
├── metrics/
│   └── utility.py             # Accuracy, precision, recall, F1
│
├── scripts/
│   └── run_experiment.py      # Experiment runner (planned)
│
├── centralized_baseline.py    # Centralized logistic regression
├── federated_train.py         # Federated (FedAvg) training loop
├── eval.py                    # Evaluation utilities (planned)
└── README.md


Phase 1: Implemented Features
-Synthetic student dataset generation (privacy-safe)
-Centralized logistic regression baseline
-Federated Averaging (FedAvg) implementation
-Non-IID data simulation via client-level shifts
-End-to-end training and evaluation pipeline

Dataset (Synthetic)
Each student record includes:
-Attendance rate
-Average quiz score
-Assignment completion rate
-LMS activity level
-Study hours
-Prior GPA
-Late submissions count
-Support requests count

Label:
-at_risk ∈ {0, 1}

-Synthetic data is calibrated to produce a realistic at-risk prevalence and avoid privacy or ethical concerns associated with real student data.

Running the Project
1. Generate synthetic data
python data/synthetic/generate.py

2. Train centralized baseline
python centralized_baseline.py

3. Train federated baseline (FedAvg)
python federated_train.py

Evaluation Metrics
-Accuracy
-Precision
-Recall
-F1-score

Future phases will include:
-Privacy leakage metrics
-Communication cost
-Fairness across clients
-Robustness under client dropout

Roadmap
Phase 1 — Foundations ✅
Synthetic data
Centralized baseline
FedAvg baseline

Phase 2 — Systems Realism
Client dropout simulation
Communication cost tracking
Stronger non-IID scenarios

Phase 3 — Privacy
Differentially private FedAvg
Secure aggregation
Privacy–utility trade-off analysis

Phase 4 — Fairness & Robustness
Per-client performance analysis
Membership inference attacks
Robust aggregation methods

Phase 5 — Research Polish
Experiment sweeps
Figures and plots
Research-style documentation

Why This Project Matters
This work demonstrates:
Applied federated learning in a high-impact domain
Strong understanding of ML systems trade-offs
Ethical and privacy-aware AI design
Research readiness for advanced CS programs and industry labs