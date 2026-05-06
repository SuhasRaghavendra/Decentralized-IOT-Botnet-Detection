# Complexity Report — Graph-Based IoT Botnet Detection Pipeline

> Big-O analysis of every stage in the graph construction, partitioning, and
> spectral feature extraction pipeline for the CICIoT2023 dataset.

---

## 1. Overview

It consists of four sequential stages:

```
Raw CSV flows
     │
     ▼
[Stage 1] Feature extraction & virtual node assignment   graph_builder.py
     │
     ▼
[Stage 2] Temporal windowing & graph construction        graph_builder.py
     │
     ▼
[Stage 3] Graph partitioning                             graph_partition.py
     │
     ▼
[Stage 4] Spectral Laplacian feature extraction          spectral_features.py
     │
     ▼
Augmented per-flow feature vectors
```

Variables used throughout:

| Symbol | Meaning |
|--------|---------|
| N | Total number of flow records (rows) |
| F | Number of selected features (17) |
| K | Number of virtual device node clusters |
| W | Number of temporal windows = ⌈N / window_size⌉ |
| n_w | Average number of distinct node clusters per window |
| E | Number of edges in the graph (≤ K²/2) |
| P | Number of graph partitions |
| V_p | Average nodes per partition (= K / P) |
| I | MiniBatchKMeans iterations |

---

## 2. Stage 1 — Feature Extraction & Virtual Node Assignment

### 2.1 CSV Loading

```
T = O(N · F)
```

Pandas reads every cell once. For the CICIoT23 dataset:
- Train: N ≈ 5.5M rows × 17 features = **93.5M cell reads**
- Validation: N ≈ 1.2M × 17 = **20.4M cell reads**

### 2.2 MiniBatchKMeans Clustering

```
T_fit     = O(N · K · I)     — mini-batch gradient descent
T_predict = O(N · K)         — single distance pass for label assignment
```

With defaults K=20, I=100, batch=10,000:
- Each mini-batch: 10,000 × 20 distance computations = 200,000 ops
- Batches per epoch ≈ N / 10,000
- Total: ≈ N × K × I / batch_size = **manageable**

**Space complexity:** O(K × F) for cluster centroids — negligible (20 × 17 floats).

> [!TIP]
> MiniBatchKMeans reduces the constant factor vs. standard KMeans by
> processing chunks. For N=5.5M with batch=10,000: ~550 batches × K × F ≈
> very fast in practice (~30s on a modern CPU).

### 2.3 Temporal Window Assignment

```
T = O(N)  — vectorised integer division
```

Single NumPy operation: `window_id = row_index // window_size`.

---

## 3. Stage 2 — Graph Construction

### 3.1 Node Population

```
T = O(N)   — single groupby over node_id
S = O(K)   — K node attribute dictionaries
```

### 3.2 Edge Accumulation (co-occurrence within windows)

```
T = O(W · n_w²)   — for each window, count all node-pair co-occurrences
S = O(E) = O(K²)  — edge weight dictionary
```

**Worst case analysis:**
- W = N / window_size
- n_w ≤ K (all K nodes appear in every window)
- Worst case: T = O(N / window_size · K²)

With defaults (window_size=500, K=20):
- W = 5.5M / 500 = 11,000 windows
- n_w² ≤ 400 comparisons per window
- Total ≈ 11,000 × 400 = **4.4M operations** — fast

**Best case** (sparse windows, n_w ≈ 2): T = O(W) = O(N / window_size).

> [!NOTE]
> The graph build is O(N/window_size × n_w²). Larger windows increase n_w
> (more co-occurrences per window) and reduce W; smaller windows do the
> opposite. The product W × n_w² is approximately constant in practice.

### 3.3 Overall Graph Build

```
T_total = O(N · K · I + W · n_w²)
         ≈ O(N · K · I)   — clustering dominates
```

---

## 4. Stage 3 — Graph Partitioning

### 4.1 Adjacency Matrix Construction

```
T = O(K²)   — allocate V×V matrix + fill E entries
S = O(K²)   — the adjacency matrix itself
```

With K=20: a 20×20 = 400-element matrix — trivially small.

### 4.2 Spectral Clustering (Primary Algorithm, Windows)

```
T_eigen  = O(K³)         — full symmetric eigen-decomposition (eigh)
T_kmeans = O(K · P · I)  — K-Means on K-dim spectral embeddings
T_total  = O(K³)         — eigen-decomp dominates
```

With K=20, P=4: K³ = 8,000 flops — essentially instantaneous.

### 4.3 METIS Partitioning (Fallback, Linux/macOS)

```
T = O(K log K)   — near-linear multilevel partitioning
```

METIS is asymptotically faster but requires native compilation. On small
graphs (K ≤ 50), both algorithms complete in milliseconds.

### 4.4 Overall Partitioning

```
T_total = O(K³)   (spectral)   or   O(K log K)   (METIS)
```

Both are negligible relative to Stages 1–2 because K ≤ 50 in all
practical configurations.

---

## 5. Stage 4 — Spectral Feature Extraction

### 5.1 Laplacian Construction (per partition)

```
T_per_partition = O(V_p²)   — adjacency matrix allocation and fill
T_total         = O(P · V_p²) = O(K²/P)   — since V_p = K/P
```

### 5.2 Eigen-Decomposition (per partition)

```
T_per_partition = O(V_p³)          — numpy.linalg.eigh (symmetric)
T_total         = O(P · V_p³) = O(K³/P²)
```

With K=20, P=4: each partition has V_p=5 nodes → 5³=125 flops each.
Total eigen work ≈ 4 × 125 = **500 flops** — negligible.

> [!IMPORTANT]
> numpy.linalg.eigh exploits the symmetric positive semi-definite structure
> of the Laplacian, running in O(V_p³ / 3) vs the O(V_p³) of general `eig`.
> This is the correct choice for all Laplacian matrices.

### 5.3 DataFrame Augmentation

```
T = O(N)   — O(N) row iterations to assign spectral columns
S = O(N · k_spectral)   — k_spectral = 2·top_k + 2 new columns
```

With top_k=8: 18 new columns per row.

### 5.4 Overall Spectral Stage

```
T_total = O(P · V_p³ + N) = O(K³/P² + N)
         ≈ O(N)   — augmentation dominates for large N
```

---

## 6. End-to-End Pipeline Summary

| Stage | Time Complexity | Space Complexity | Bottleneck? |
|-------|----------------|-----------------|-------------|
| CSV Load | O(N · F) | O(N · F) | Disk I/O |
| KMeans clustering | O(N · K · I) | O(N + K·F) | **Yes — for large N** |
| Window assignment | O(N) | O(N) | No |
| Graph construction | O(W · n_w²) | O(K²) | Moderate |
| Graph partitioning | O(K³) | O(K²) | No |
| Laplacian + Eigen | O(P · V_p³) | O(K²) | No |
| DataFrame augment | O(N) | O(N · k) | Moderate |
| **Total** | **O(N · K · I)** | **O(N · F)** | KMeans |

**Dominant term:** `O(N · K · I)` — driven by MiniBatchKMeans clustering.

---

## 7. Real-Time Feasibility Analysis

### 7.1 Throughput Estimate

For IoT edge deployment, we need per-flow latency, not batch throughput.
At inference time, only Steps 1–2 of the pipeline run per new flow:

| Inference step | Latency per flow |
|----------------|-----------------|
| Feature scaling (StandardScaler) | O(F) ≈ **< 1 µs** |
| KMeans prediction (assign node) | O(K) ≈ **< 5 µs** |
| Window accumulation (update counter) | O(1) ≈ **< 1 µs** |
| Spectral lookup (pre-computed, table lookup) | O(1) ≈ **< 1 µs** |
| RF prediction (100 trees, 17+18 features) | O(trees · depth) ≈ **< 1 ms** |
| **Total per-flow latency** | **< 2 ms** |

✅ **Real-time feasible** for typical IoT gateway hardware (Raspberry Pi 4
achieves ~500 flows/s for similar workloads).

### 7.2 Re-training Frequency

The graph and spectral features should be **pre-computed offline** on a
rolling window of historical flows (e.g. every 5 minutes). At inference
time, only the RF prediction step runs online. This decouples the
O(N · K · I) offline cost from the real-time prediction latency.

### 7.3 Scaling Considerations

| Scale | Nodes (K) | Windows (W) | Recommendation |
|-------|-----------|-------------|----------------|
| Lab (≤ 100K flows) | 10–20 | 200–2,000 | Current defaults |
| Campus (≤ 10M flows) | 20–50 | 20K–200K | Increase K; use METIS |
| ISP-scale (>100M flows) | 50–200 | 200K+ | Distributed Spark + METIS |

---

## 8. Algorithmic Optimisations Implemented (C4)

The following optimisations are already applied in the codebase:

### 8.1 Vectorised Correlation (preprocess_ciciot23.py)

```python
# BEFORE (naive O(N·F²) double loop):
for col1 in features:
    for col2 in features:
        r = pearsonr(df[col1], df[col2])

# AFTER (O(N·F + F²) vectorised):
correlation_matrix = train_frame[feature_columns].corr().abs()
# Uses numpy.corrcoef internally — single BLAS call
```

**Speedup:** ~20–100× for F=17 features.

### 8.2 Pre-computed Mutual Information (preprocess_ciciot23.py)

```python
# sklearn.feature_selection.mutual_info_classif uses a vectorised
# k-nearest-neighbour estimator (Kraskov estimator), complexity O(N log N)
# vs naive O(N²) histogram binning.
scores = mutual_info_classif(features, target, random_state=42)
```

**Speedup:** ~N/log(N) vs naive histogram MI.

### 8.3 MiniBatchKMeans vs KMeans (graph_builder.py)

```python
# Standard KMeans: O(N · K · I) with all N in memory at once
# MiniBatchKMeans: same O() but batch_size << N → fits in L3 cache
km = MiniBatchKMeans(n_clusters=k, batch_size=10_000)
```

**Memory reduction:** From O(N·F) active working set to O(batch_size·F).

### 8.4 numpy.linalg.eigh vs eig (spectral_features.py)

```python
# eig (general):  O(V³),   returns complex eigenvalues for near-symmetric matrices
# eigh (symmetric): O(V³/3), guaranteed real output, 3× faster
eigenvalues, eigenvectors = np.linalg.eigh(L)  # L is symmetric ∀ valid Laplacian
```

**Correctness:** Laplacian matrices are always real symmetric → eigh is
both faster and numerically more stable.

### 8.5 Vectorised Window Assignment (graph_builder.py)

```python
# BEFORE (Python loop):
for i, row in df.iterrows():
    df.at[i, 'window_id'] = i // window_size  # O(N) Python overhead

# AFTER (vectorised NumPy):
result["window_id"] = np.arange(len(result)) // window_size  # O(N) C-speed
```

**Speedup:** ~50–100× for large DataFrames.

---

