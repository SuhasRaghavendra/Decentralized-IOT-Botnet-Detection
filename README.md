# Decentralized IoT Botnet Detection

To reproduce the results from scratch, follow these steps:

### 1. Environment Setup
Create a Python environment and install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Data Acquisition
This project uses the **CICIoT23** dataset. Due to its size, the raw data is not included in the repository.
1. Download the dataset from the kaggle (https://www.kaggle.com/datasets/himadri07/ciciot2023?hl=en-IN).
2. Organize the raw CSV files into the following directory structure in the project root:
   ```text
   CICIOT23/
   ├── train/
   │   └── train.csv
   ├── validation/
   │   └── validation.csv
   └── test/
       └── test.csv
   ```

### 3. Execution Pipeline
Run the scripts in the following order:

1. **Preprocessing**: Cleans the raw data, selects top features, and scales the values.
   ```bash
   python preprocess_ciciot23.py
   ```
   *Output: `processed_ciciot23/`*

2. **Graph Construction**: Builds temporal device-flow graphs via clustering and windowing.
   ```bash
   python graph_builder.py --split validation
   ```
   *Output: `graph_artifacts/`*

3. **Spectral Feature Extraction**: Computes Laplacian eigenvalues and spectral partitions.
   ```bash
   python spectral_features.py --split validation
   ```
   *Output: `graph_artifacts/spectral_augmented_validation.csv`*

4. **Model Training**: Trains a Random Forest classifier on the spectral features.
   ```bash
   python train_spectral_rf.py --split validation
   ```

---

##  Project Structure

- `preprocess_ciciot23.py`: Data cleaning, feature selection (Pearson/MI intersection), and scaling.
- `graph_builder.py`: Virtual device node assignment (KMeans) and temporal edge construction.
- `spectral_features.py`: Laplacian matrix construction and spectral feature engineering.
- `train_spectral_rf.py`: Random Forest training and evaluation pipeline.
- `EDA.py`: Automated profiling of the raw dataset.
- `tests/`: Comprehensive unit tests for all core modules.

##  Testing
To run the test suite and verify the implementation:
```bash
pytest
```

##  Reports
- `complexity_report.md`: Detailed analysis of algorithmic complexity and optimizations.
- `spectral_experiment_report.md`: Results and performance metrics for the spectral modeling approach.
