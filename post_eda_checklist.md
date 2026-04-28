# Post-EDA Checklist for CICIoT2023

Use this as the execution checklist after reviewing the EDA report.

## Data Preparation
- [x] Load the existing train, validation, and test CSV splits without reshuffling.
- [x] Detect the label column and preserve the original label text.
- [x] Create a binary target: 0 for benign traffic, 1 for attack traffic.
- [x] Create a family-level multiclass target from the original labels.
- [x] Replace infinite values in numeric features using train-set statistics.
- [x] Fill remaining missing numeric values using train-set statistics.
- [x] Drop constant columns (train-set nunique ≤ 1).
- [x] Drop highly correlated duplicate features using a Pearson threshold (0.95).
- [x] Save cleaned train, validation, and test CSV files.

## Feature Engineering
- [x] Rank features by Pearson correlation with the binary target (top 20).
- [x] Rank features by mutual information with the binary target (top 20).
- [x] Keep the intersection of the two top-20 sets as the final feature set (17 features).
- [x] Save `selected_features.json` for use by all downstream scripts.
- [x] Fit the StandardScaler **only** on the training split.
- [x] Transform validation and test splits with the training scaler.

## Artefacts Saved
- [x] `train_clean.csv` / `validation_clean.csv` / `test_clean.csv` — scaled, feature-selected CSVs.
- [x] `binary_pearson_top20.csv` / `binary_mutual_info_top20.csv` — feature rankings.
- [x] `family_pearson_top20.csv` / `family_mutual_info_top20.csv` — reference rankings.
- [x] `selected_features.json` — canonical feature list for downstream scripts.
- [x] `label_family_map.json` — integer ID → attack family name mapping.
- [x] `class_weights.json` — balanced class weights for binary and family tasks.
- [x] `numeric_fill_stats.json` — train-set fill statistics for reproducibility.
- [x] `scaler.pkl` — fitted StandardScaler for inference pipelines.
- [x] `preprocess_metadata.json` — full pipeline provenance record.

## Imbalance Handling
- [x] Compute class weights for binary classification.
- [x] Compute class weights for family-level multiclass classification.
- [x] Optionally generate SMOTE-resampled training data (`--use-smote` flag).

## Test Split — LOCKED
- [x] **Test split is frozen for final reporting only.**
- [x] `preprocess_metadata.json` records `"test_split_locked_for_final_reporting": true`.
- [x] Do NOT evaluate on `test_clean.csv` during model development or hyperparameter tuning.

## Next Modeling Step
- [ ] Train centralized baselines on `train_clean.csv`.
- [ ] Evaluate all model iterations on `validation_clean.csv` only.
- [ ] Select best model configuration using validation metrics.
- [ ] Run a single final evaluation on `test_clean.csv` for reporting.
