<div align="center">

#  Decentralized IoT Botnet Detection

**A multi-phase, research-grade pipeline for detecting IoT botnet attacks using Graph-Spectral Feature Engineering and Privacy-Preserving Federated Learning.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.1-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Flower](https://img.shields.io/badge/Flower-FL_Framework-9146FF?style=flat-square)](https://flower.ai/)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.4.2-brightgreen?style=flat-square)](https://networkx.org/)
[![Dataset](https://img.shields.io/badge/Dataset-CICIoT2023-blue?style=flat-square)](https://www.kaggle.com/datasets/himadri07/ciciot2023)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

</div>

---

##  Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Key Results](#-key-results)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Execution Pipeline](#-execution-pipeline)
- [Module Reference](#-module-reference)
- [Testing](#-testing)
- [Reports & Artifacts](#-reports--artifacts)
- [Dashboard](#-dashboard)
- [Technical Deep Dive](#-technical-deep-dive)

---

##  Overview

This project implements a complete, end-to-end pipeline for **decentralized botnet detection** in IoT network traffic. It is designed around the premise that edge devices in an IoT network cannot share raw traffic data due to privacy and bandwidth constraints — requiring detection to happen locally, with only model intelligence shared globally.

The pipeline is structured in four major phases:

| Phase | Name | Key Innovation |
|-------|------|----------------|
| **A** | Baseline ML Pipeline | Feature selection via Pearson/MI intersection; RF + LR classifiers |
| **B** | Attack-Specific Models | True One-vs-Rest binary detectors per attack family |
| **C/D** | Graph-Spectral Features | Temporal device graphs + Laplacian eigen-decomposition |
| **E** | Federated Learning | Tree-pooling aggregation with XOR-masked secure aggregation |

The project uses the **CICIoT2023** dataset — a large-scale, realistic IoT attack benchmark with **34 distinct attack types** spanning DDoS, DoS, Mirai, brute-force, and more, across over **5 million flow records**.

---

##  Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     CICIoT2023 Raw Dataset                      │
│              (~5.5M flow records, 46 raw features)              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  Phase A    │  Preprocessing & Baseline
                    │             │  • Pearson + MI feature selection
                    │             │  • StandardScaler normalization
                    │             │  • RF (binary & family) classifiers
                    └──────┬──────┘
                           │
          ┌────────────────┼────────────────────┐
          │                │                    │
   ┌──────▼──────┐  ┌──────▼──────┐    ┌───────▼──────┐
   │  Phase B    │  │  Phase C/D  │    │   Phase E    │
   │             │  │             │    │              │
   │Attack-Spec. │  │Graph Build  │    │  Federated   │
   │ Models (OvR)│  │+ Spectral   │    │  Learning    │
   │             │  │  Features   │    │              │
   └──────┬──────┘  └──────┬──────┘    └───────┬──────┘
          │                │                    │
          │         ┌──────▼──────┐    ┌───────▼──────┐
          │         │ Augmented   │    │ Global Model │
          │         │  Feature    │    │  (Secure     │
          │         │  Matrix     │    │  Aggregated) │
          │         │(17+18 feats)│    │              │
          └─────────┴──────┬──────┘    └───────┬──────┘
                           │                    │
                    ┌──────▼────────────────────▼──────┐
                    │         Interactive Dashboard     │
                    │    (Unified performance view)     │
                    └──────────────────────────────────┘
```

### Graph-Spectral Pipeline (Phase C/D)

```
Network Flows
     │
     ▼
[MiniBatch KMeans]  →  Virtual Device Nodes (K=20 clusters)
     │
     ▼
[Temporal Windows]  →  W = ⌈N / 500⌉ windows
     │
     ▼
[Edge Construction] →  Weighted graph G = (V, E) per window
     │
     ▼
[Graph Partition]   →  P=4 spectral/METIS partitions
     │
     ▼
[Laplacian Eigen]   →  Lₙ = I − D⁻¹ᐟ² A D⁻¹ᐟ²  →  λ₁..λₖ, v₁..vₖ
     │
     ▼
[Feature Augment]   →  spectral_eigen_i, spectral_proj_i, fiedler_value
```

### Federated Learning Architecture (Phase E)

```
┌──────────────┐    ┌──────────────┐
│   Client 0   │    │   Client 1   │
│  (Train set) │    │  (Val. set)  │
│              │    │              │
│  Local RF    │    │  Local RF    │
│  Training    │    │  Training    │
└──────┬───────┘    └───────┬──────┘
       │  masked update      │  masked update
       │  (XOR + PRG seed)   │  (XOR + PRG seed)
       └──────────┬──────────┘
                  ▼
          ┌───────────────┐
          │  FL Server    │
          │  Tree-Pool    │   ← Combines estimator lists
          │  Aggregation  │
          └───────┬───────┘
                  ▼
          Global RF Model
          (federated_artifacts/)
```

---

##  Key Results

### Phase A — Baseline (17 Features)

| Model | Task | Accuracy | F1 (Binary) | Macro F1 | ROC-AUC |
|-------|------|----------|-------------|----------|---------|
| Random Forest | Binary Detection | 0.9928 | 0.9963 | 0.9208 | 0.9980 |
| Random Forest | Family Classification | — | — | — | — |

### Phase B — Attack-Specific Models (True One-vs-Rest)

| Attack | Model | Features | Accuracy | Recall | F1 |
|--------|-------|----------|----------|--------|-----|
| DDoS-ICMP Flood | Random Forest | 22 (ICMP-tailored) | **0.9998** | 0.9993 | **0.9995** |
| DDoS-SYN Flood | Random Forest | 23 (SYN-tailored) | 0.9773 | **0.9959** | 0.9197 |
| Mirai-Greeth Flood | LightGBM | 22 (GRE-tailored) | **0.9989** | **0.9955** | **0.9752** |

> **Key Finding:** Attack-specific models trained under a rigorous OvR schema (negative class = benign + 33 other attacks) learn true structural signatures — not just bandwidth anomalies. Cross-attack evaluation confirms zero false-positive overlap between detectors.

### Phase C/D — Graph-Spectral Augmentation

| Model | Features | Accuracy | Macro F1 | PR-AUC |
|-------|----------|----------|----------|--------|
| Baseline RF | 17 | 0.9928 | 0.9208 | 0.9999 |
| Spectral RF | 35 (+18 graph features) | 0.9925 | 0.9181 | **1.0000** |
| Matrix RF | 34 (+17 statistical features) | **0.9947** | **0.9472** | 0.9992 |
| Combined RF | 51 (all features) | 0.9946 | 0.9465 | 0.9992 |

> Top spectral features (`spectral_proj_6`, `spectral_proj_7`) ranked 5th and 8th by importance, capturing coordinated botnet topology patterns invisible to flow-level features alone.

### Phase E — Federated Learning

| Round | Global Accuracy | Global F1 | Clients |
|-------|----------------|-----------|---------|
| 1 | 0.9882 | 0.9939 | 2 |
| 2 | **0.9979** | **0.9989** | 2 |

**Final Global Model (Frozen Test Set — 1.17M samples):**

| Metric | Score |
|--------|-------|
| Accuracy | **0.9931** |
| Binary F1 | **0.9965** |
| Macro F1 | **0.9247** |
| ROC-AUC | **0.9983** |
| PR-AUC | **1.0000** |

---

##  Project Structure

```
.
├── src/
│   ├── data/
│   │   ├── preprocess_ciciot23.py      # Feature selection, scaling, train/val/test split
│   │   ├── preprocess_attack_specific.py # Per-attack OvR preprocessing
│   │   ├── packet_features.py          # Packet-level statistical feature extraction
│   │   ├── packet_ingest.py            # Raw PCAP ingestion utilities
│   │   └── EDA.py                      # Automated dataset profiling (ydata-profiling)
│   │
│   ├── graph/
│   │   ├── graph_builder.py            # Virtual device graph construction (KMeans + windows)
│   │   ├── graph_partition.py          # METIS / spectral graph partitioning
│   │   ├── spectral_features.py        # Laplacian eigen-decomposition & feature augmentation
│   │   └── train_spectral_rf.py        # RF training on spectral-augmented features
│   │
│   ├── models/
│   │   ├── train_binary_baseline.py    # Binary RF/LR baseline classifier
│   │   ├── train_family_baseline.py    # Multi-class attack family classifier
│   │   ├── train_attack_models.py      # Attack-specific OvR model training
│   │   ├── utils_hyperopt.py           # Hyperparameter tuning utilities
│   │   └── regenerate_scaler.py        # Scaler regeneration for new splits
│   │
│   └── federated/
│       ├── fl_server.py                # Flower FL server + tree-pool aggregation
│       ├── fl_client.py                # Flower FL client (local RF training)
│       ├── matrix_features.py          # Higher-order statistical feature extraction
│       ├── train_matrix_rf.py          # RF training on matrix-augmented features
│       ├── secure_aggregation.py       # XOR-masked secure model aggregation
│       └── federated_evaluation.py     # Final global model evaluation on test set
│
├── tests/
│   ├── test_section_a.py               # Preprocessing & baseline model tests
│   ├── test_graph_builder.py           # Graph construction & edge logic tests
│   ├── test_graph_partition.py         # Partitioning algorithm tests
│   ├── test_spectral_features.py       # Laplacian & spectral feature tests
│   └── test_section_e.py               # Federated learning component tests
│
├── notebooks/
│   ├── baseline_evaluation.ipynb       # End-to-end baseline analysis notebook
│   └── attack_model_evaluation.ipynb   # Attack-specific model comparison notebook
│
├── reports/
│   ├── complexity_report.md            # Big-O analysis for all pipeline stages
│   ├── spectral_experiment_report.md   # Spectral feature experiment results
│   ├── matrix_experiment_report.md     # Matrix feature augmentation results
│   ├── attack_comparison_report.md     # OvR vs. general baseline comparison
│   ├── secure_aggregation.md           # Secure aggregation protocol documentation
│   └── attack_specific/                # Per-attack detailed reports
│
├── dashboard/
│   └── index.html                      # Interactive results dashboard (standalone HTML)
│
├── federated_artifacts/
│   ├── global_model_final.pkl          # Final federated global model
│   ├── federated_round_metrics.json    # Per-round accuracy & F1 scores
│   └── federated_evaluation_report.md  # Full evaluation on frozen test set
│
├── graph_artifacts/                    # Built graphs, spectral-augmented CSVs
├── matrix_artifacts/                   # Matrix-augmented feature CSVs
├── models/                             # Saved model checkpoints (.pkl)
├── processed_ciciot23/                 # Preprocessed splits (train/val/test CSVs)
├── CICIOT23/                           # Raw dataset (not included — see below)
├── conftest.py                         # Pytest configuration
└── requirements.txt
```

---

##  Setup & Installation

### Prerequisites

- Python **3.10+**
- ~8GB RAM (for full dataset preprocessing)
- ~30GB free disk (raw + processed datasets)

### 1. Clone the Repository

```bash
git clone https://github.com/SuhasRaghavendra/Decentralized-IOT-Botnet-Detection.git
cd Decentralized-IOT-Botnet-Detection
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Core dependencies:**

| Package | Version | Purpose |
|---------|---------|---------|
| `scikit-learn` | 1.5.1 | RF, LR, KMeans, metrics |
| `numpy` | 1.26.4 | Numerical computing |
| `pandas` | 2.3.3 | Data manipulation |
| `networkx` | 3.4.2 | Graph construction & analysis |
| `scipy` | 1.15.3 | Sparse matrices, eigen-decomposition |
| `imbalanced-learn` | 0.12.3 | SMOTE / class balancing |
| `flwr` | ≥1.5.0 | Federated learning framework |
| `ydata-profiling` | ≥4.6.0 | Automated EDA reporting |

### 4. Data Acquisition

This project uses the **CICIoT2023** dataset. Due to its size, raw data is not included.

1. Download from [Kaggle — CICIoT2023](https://www.kaggle.com/datasets/himadri07/ciciot2023).
2. Organize into the following structure in the project root:

```
CICIOT23/
├── train/
│   └── train.csv
├── validation/
│   └── validation.csv
└── test/
    └── test.csv
```

---

##  Execution Pipeline

Run each step in order. Each phase produces artifacts consumed by the next.

### Phase A — Preprocessing & Baseline

**1. Preprocess the raw dataset**
```bash
python -m src.data.preprocess_ciciot23
```
*Performs: null removal, Pearson + Mutual Information feature intersection, StandardScaler fit, train/val/test CSV export.*
*Output: `processed_ciciot23/`*

**2. Train binary baseline classifier**
```bash
python -m src.models.train_binary_baseline
```
*Output: `models/binary_rf_best.pkl`, `models/binary_lr_best.pkl`*

**3. Train attack family classifier**
```bash
python -m src.models.train_family_baseline
```
*Output: `models/family_rf_best.pkl`*

---

### Phase B — Attack-Specific Models

**4. Preprocess per-attack OvR splits & train**
```bash
python -m src.data.preprocess_attack_specific
python -m src.models.train_attack_models
```
*Output: Per-attack model checkpoints and evaluation reports in `reports/attack_specific/`*

---

### Phase C/D — Graph Construction & Spectral Features

**5. Build virtual device graphs**
```bash
python -m src.graph.graph_builder --split validation
```
*Output: `graph_artifacts/device_graph_validation.pkl`*

**6. Partition the graph**
```bash
python -m src.graph.graph_partition --split validation
```
*Output: `graph_artifacts/partitioned_graph_validation.pkl`*

**7. Extract spectral Laplacian features**
```bash
python -m src.graph.spectral_features --split validation
```
*Output: `graph_artifacts/spectral_augmented_validation.csv` (17 + 18 = 35 features)*

**8. Train spectral-augmented RF**
```bash
python -m src.graph.train_spectral_rf --split validation
```
*Output: `models/spectral_rf_best.pkl`, `reports/spectral_experiment_report.md`*

---

### Phase E — Federated Learning

**9. Extract higher-order matrix features**
```bash
python -m src.federated.matrix_features --split validation
```
*Output: `matrix_artifacts/matrix_augmented_validation.csv` (17 + 17 = 34 features)*

**10. Train matrix-augmented RF**
```bash
python -m src.federated.train_matrix_rf --split validation
```
*Output: `models/matrix_rf_best.pkl`, `reports/matrix_experiment_report.md`*

**11. Run Federated Learning Simulation**

```bash
# Simulation mode (single process, no network required):
python -m src.federated.fl_server --simulate --n-clients 2 --rounds 3

# Network mode (requires separate terminals):
# Terminal 1 — Server
python -m src.federated.fl_server --rounds 3 --min-clients 2

# Terminal 2 — Client 0
python -m src.federated.fl_client --client-id 0 --split train

# Terminal 3 — Client 1
python -m src.federated.fl_client --client-id 1 --split validation
```
*Output: `federated_artifacts/global_model_final.pkl`, `federated_artifacts/federated_round_metrics.json`*

**12. Enable Secure Aggregation (optional)**
```bash
# Test the secure aggregation module:
python -m src.federated.secure_aggregation --test

# Run FL simulation with XOR masking enabled:
python -m src.federated.fl_server --simulate --use-encryption
```

**13. Evaluate the global federated model**
```bash
python -m src.federated.federated_evaluation
```
*Output: `federated_artifacts/federated_evaluation_report.md`*

---

##  Module Reference

### `src/data/`

| Module | Description |
|--------|-------------|
| `preprocess_ciciot23.py` | Full preprocessing pipeline: null handling, Pearson/MI feature intersection, `StandardScaler` fit & transform, stratified train/val/test CSV export. |
| `preprocess_attack_specific.py` | Constructs per-attack One-vs-Rest datasets. Negative class includes both benign traffic *and* all other 33 attack types to prevent shortcut learning. |
| `packet_features.py` | Derives higher-order packet-level statistical features (mean, variance, covariance, skewness, etc.) from raw flows. |
| `packet_ingest.py` | PCAP/raw packet ingestion utilities for online inference mode. |
| `EDA.py` | Generates a comprehensive HTML profiling report via `ydata-profiling`. |

### `src/graph/`

| Module | Description |
|--------|-------------|
| `graph_builder.py` | Assigns each flow to a virtual device node (via MiniBatch KMeans, K=20), then constructs weighted temporal device-to-device graphs using configurable time windows (default: 500 rows). |
| `graph_partition.py` | Partitions the graph into P=4 subgraphs using METIS (primary) with spectral clustering as a fallback. |
| `spectral_features.py` | Builds the normalized Graph Laplacian `Lₙ = I − D⁻¹ᐟ² A D⁻¹ᐟ²` for each partition, performs eigen-decomposition, and augments the dataset with `spectral_eigen_i`, `spectral_proj_i`, `fiedler_value`, and `partition_id`. |
| `train_spectral_rf.py` | Trains a Random Forest on the spectral-augmented feature matrix and generates the spectral experiment report. |

### `src/models/`

| Module | Description |
|--------|-------------|
| `train_binary_baseline.py` | Trains RF and Logistic Regression for binary (Benign vs. Attack) classification. |
| `train_family_baseline.py` | Trains a multi-class RF for attack family (e.g., DDoS, Mirai, Brute-Force) classification. |
| `train_attack_models.py` | Trains attack-specific binary classifiers (RF + LightGBM) for DDoS-ICMP, DDoS-SYN, and Mirai-Greeth under a rigorous OvR schema. |
| `utils_hyperopt.py` | Grid search and cross-validation utilities for hyperparameter tuning. |

### `src/federated/`

| Module | Description |
|--------|-------------|
| `fl_server.py` | Flower-based FL server implementing **tree-pool aggregation** — combines `estimators_` lists from all client RF models into a single global forest. Supports both simulation and network modes. |
| `fl_client.py` | Flower FL client that trains a local RF on its private data partition and returns serialized model bytes to the server. |
| `matrix_features.py` | Extracts 17 higher-order statistical features from packet-size and inter-arrival timing distributions (mean, std, skewness, kurtosis, percentiles, etc.). |
| `train_matrix_rf.py` | Trains and compares Baseline RF, Matrix-augmented RF, and Combined RF (all 51 features). |
| `secure_aggregation.py` | Implements **additive XOR masking** over serialized model bytes using a 256-bit PRG-seeded shared secret. Ensures the server cannot reconstruct individual client models. |
| `federated_evaluation.py` | Loads the final global model and evaluates it on the frozen test set, generating a full classification report. |

---

##  Testing

The project has a comprehensive test suite covering all pipeline stages.

```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run a specific test module
pytest tests/test_section_a.py
pytest tests/test_section_e.py
```

| Test File | Coverage Area |
|-----------|--------------|
| `test_section_a.py` | Preprocessing logic, feature selection, scaler behavior, baseline model training |
| `test_graph_builder.py` | Virtual node assignment, edge construction, window logic |
| `test_graph_partition.py` | METIS and spectral partitioning correctness |
| `test_spectral_features.py` | Laplacian construction, eigen-decomposition, feature augmentation |
| `test_section_e.py` | Matrix feature extraction, FL server/client, secure aggregation |

---

##  Reports & Artifacts

| Report | Location | Description |
|--------|----------|-------------|
| EDA Report | `reports/eda_report.html` | Full ydata-profiling analysis of raw dataset |
| Complexity Analysis | `reports/complexity_report.md` | Big-O analysis for all pipeline stages |
| Spectral Experiment | `reports/spectral_experiment_report.md` | Baseline vs. Spectral RF comparison |
| Matrix Experiment | `reports/matrix_experiment_report.md` | Baseline vs. Matrix vs. Combined RF |
| Attack Comparison | `reports/attack_comparison_report.md` | General baseline vs. OvR attack-specific models |
| Secure Aggregation | `reports/secure_aggregation.md` | XOR masking protocol documentation |
| Federated Evaluation | `federated_artifacts/federated_evaluation_report.md` | Final global model metrics on frozen test set |

---

##  Dashboard

An interactive, standalone HTML dashboard is provided to visualize all experimental results in one place.

```bash
# Open directly in any modern browser — no server required
open dashboard/index.html
```

The dashboard aggregates:
- Baseline binary & family model metrics
- Graph-spectral feature experiment comparisons
- Federated learning round progression
- Attack-specific OvR model results
- Side-by-side feature importance rankings

---

##  Technical Deep Dive

### Feature Selection Strategy

Features are selected via the **intersection of two criteria**:
1. **Pearson Correlation**: Top F features correlated with the binary label (Benign/Attack).
2. **Mutual Information**: Top F features with highest MI score against the label.

Only features appearing in **both** ranked lists are retained. This dual-filter approach eliminates features that are linearly correlated with the label but carry no non-linear information (and vice versa), resulting in a robust 17-feature set.

### Graph-Spectral Intuition

Standard ML treats each network flow as an independent sample, ignoring the **relational structure** of IoT traffic. This pipeline:

1. **Clusters flows into virtual device nodes** — flows with similar statistical profiles are grouped via KMeans, approximating device identities without requiring raw IP addresses (preserving privacy).
2. **Builds temporal graphs** — edges between devices represent co-occurrence within time windows, with edge weights counting interaction frequency.
3. **Applies spectral analysis** — the Graph Laplacian's eigenvalues characterize network topology:
   - **Small Fiedler value (λ₁)**: Weakly connected graph → typical of isolated attack clusters (C&C isolated botnets).
   - **Large spectral gap**: Dense subgraph → typical of coordinated DDoS swarms.

### Secure Aggregation Protocol

```
Client side:   masked_bytes = model_bytes ⊕ PRG(shared_seed)
Server side:   model_bytes  = masked_bytes ⊕ PRG(shared_seed)
```

The server only ever sees `masked_bytes`. The shared seed is exchanged once (out-of-band). This guarantees that even a compromised server cannot reconstruct any individual client's model, only the pooled result. For production deployments, the seed exchange should use TLS + Diffie-Hellman key agreement, and stronger primitives such as **Paillier HE** or **SPDZ** MPC should replace the XOR scheme.

### Algorithmic Complexity Summary

| Stage | Time Complexity | Bottleneck? |
|-------|----------------|-------------|
| Feature Selection | O(N log N · F) | Memory / Disk I/O |
| RF Training | O(M · N · F log N) | **Yes** |
| Virtual Node Assignment | O(N · K · I) | Moderate |
| Graph Construction | O(W · n²_w) | No |
| Graph Partitioning | O(K log K) | No |
| Spectral Decomposition | O(P · V³_p) = O(K³/P²) | No (≈500 flops) |
| Feature Augmentation | O(N) | Moderate |

> With K=20, P=4: spectral decomposition is effectively free (~500 floating point operations), making the graph-spectral pipeline a near-zero overhead augmentation over the baseline.

---

<div align="center">

**Built with Python · scikit-learn · NetworkX · Flower FL**

*CICIoT2023 Dataset — Canadian Institute for Cybersecurity*

</div>
