# Diabetes Screening from Plantar Thermograms — Project Report

_Generated: 2026-09-30_

## 1. Objective

Build a machine-learning model that classifies a person as **Control (0)** or **Diabetes Mellitus, DM (1)** from
**plantar (sole-of-foot) thermograms**: 2D temperature matrices of the left and right foot. The aim is a
non-invasive **screening aid**, not a diagnosis. The project also tests whether the signal is truly thermal or is
explained by confounders (age, gender, image size/batch effects).

## 2. Dataset

- Source: Plantar Thermogram Database (Control Group and DM Group folders + an Excel workbook with subject ID and age).
- Each subject has a left-foot and a right-foot CSV (temperature in °C, background = 0) and optional angiosome CSVs
  (four foot regions per side: LCA, LPA, MCA, MPA).
- **Subjects:** 167 (**Controls:** 45, **DM:** 122)
- **Subject-level features extracted:** 36
- Folder labels were cross-checked against the Excel labels; duplicate IDs and missing files raise errors.

**Age by group**

- **Control**: n=45, age 27.8 ± 8.2 (range 21–52)
- **DM**: n=122, age 56.0 ± 10.6 (range 23–84)

## 3. Pipeline (what each stage does)

| Stage | Cell | Purpose |
|---|---|---|
| Setup | 1 | Imports, random seed 42, Drive paths |
| Master table | 2 | Match folders to Excel metadata; verify labels and files |
| Quality control | 3 | Flag empty, split, holed, flat or implausible-temperature images |
| Conservative repair | 4 | Fill only small enclosed holes (≤10 px); keep unusual temperatures |
| Batch-effect check | 5 | Can image size or raw temperature alone predict the label? |
| Geometry standardization | 6 | Flip left foot, crop, resize to 128×64 keeping aspect ratio, mask-aware interpolation, within-foot z-score |
| Feature extraction | 7 | Thermal statistics, texture (entropy, GLCM), left/right asymmetry, angiosome means |
| Baselines | 8 | Age only vs thermal only vs thermal + age |
| Model comparison | 9 | Logistic regression (L1), SVM, Random Forest, Gradient Boosting |
| Confound ablation | 10, 10A | Does the result survive without age/gender, and in a 35–60 age band? |
| Threshold calibration | 11 | Pick a threshold giving ≥90% sensitivity (screening setting) |
| Feature importance | 12 | Which thermal features drive predictions |
| Stress test | 13 | Robustness to +0.3 °C drift and 0.1 °C sensor noise |
| CNN | 14 | Six-channel deep-learning comparison |
| Matched pairs | 20 | Does the model rank the DM subject above an age/gender-matched control? |
| Export | 21 | Train final model and save `thermal_dm_model.joblib` |

**Key methodological safeguards:** every cross-validation uses `StratifiedGroupKFold` (patient-level splits, 5 folds,
repeated with 5 seeds where noted); imputation, scaling and feature selection are fitted **inside the training fold only**;
metrics are reported as mean ± SD over folds.

## 4. Results

### 4.1 Data quality
334 feet examined, 316 passed all checks, 18 flagged for review. Flag counts: TEMPERATURE_REVIEW=16, SPLIT_MASK=2.

### 4.2 Batch-effect diagnostic (Cell 5)
If image *dimensions* alone predict the label well, the model could be learning acquisition artefacts.

| Feature set | Mean AUC | SD |
|---|---|---|
| Dimensions only | 0.676 | 0.074 |
| Temperature only | 0.889 | 0.030 |
| Dimensions + temperature | 0.898 | 0.033 |


### 4.3 Baseline benchmarks (Cell 8, repeated grouped 5-fold CV, logistic regression)

| Model | AUC mean | AUC SD | Bal. Acc mean | Bal. Acc SD | Folds |
|---|---|---|---|---|---|
| Age Only | 0.969 | 0.031 | 0.919 | 0.062 | 25 |
| Thermal Only | 0.942 | 0.040 | 0.889 | 0.070 | 25 |
| Thermal + Age | 0.979 | 0.025 | 0.912 | 0.055 | 25 |


### 4.4 Classical model comparison (Cell 9, thermal features only)

| Model | AUC Mean | AUC Std | Balanced Acc | Sensitivity | Specificity | Valid Folds |
|---|---|---|---|---|---|---|
| SVM (RBF Kernel) | 0.910 | 0.051 | 0.791 | 0.875 | 0.707 | 25 |
| Random Forest | 0.896 | 0.053 | 0.826 | 0.848 | 0.804 | 25 |
| Logistic Regression (L1) | 0.893 | 0.047 | 0.843 | 0.788 | 0.898 | 25 |
| Gradient Boosting | 0.864 | 0.058 | 0.764 | 0.857 | 0.671 | 25 |


### 4.5 Confound ablation (Cell 10)

| Condition | Subjects | AUC | Balanced Accuracy |
|---|---|---|---|
| (1) Thermal Features Only | 167 | 0.942 ± 0.040 | 0.889 |
| (2) Thermal + Age + Gender | 167 | 0.983 ± 0.020 | 0.926 |
| (3) Age + Gender Only | 167 | 0.968 ± 0.031 | 0.919 |
| (4) Thermal Only, Age 35-60 | 85 | 0.771 ± 0.110 | 0.727 |

A large gap between "Thermal only" and "Age + Gender only" suggests the thermal features carry information beyond demographics.
If "Age + Gender only" scores high, age is a strong confounder.

### 4.6 Screening threshold calibration (Cell 11)

| Metric | Value |
|---|---|
| Out-of-fold ROC AUC | 0.945 |
| Sensitivity at threshold 0.50 | 0.852 |
| **Selected threshold (target sensitivity ≥ 0.90)** | **0.3780** |
| Sensitivity at selected threshold | 0.926 |
| Specificity at selected threshold | 0.844 |

### 4.7 Top thermal features (Cell 12, Random Forest importance across folds)

| Feature | Mean Importance | Std Across Folds | Top 10 Frequency |
|---|---|---|---|
| R_foot_min | 0.0764 | 0.0140 | 5 |
| R_foot_range | 0.0763 | 0.0218 | 5 |
| L_foot_min | 0.0577 | 0.0149 | 5 |
| L_LPA_mean | 0.0523 | 0.0185 | 4 |
| asym_mean_diff | 0.0408 | 0.0101 | 3 |
| L_MPA_mean | 0.0405 | 0.0197 | 2 |
| L_glcm_contrast | 0.0404 | 0.0114 | 4 |
| R_glcm_contrast | 0.0376 | 0.0154 | 2 |
| R_LPA_mean | 0.0372 | 0.0055 | 4 |
| L_foot_range | 0.0312 | 0.0115 | 2 |


### 4.8 Stress test (Cell 13, out-of-fold AUC)

| Condition | AUC |
|---|---|
| Baseline | 0.945 |
| Temperature drift +0.3 °C | 0.946 (Δ 0.001) |
| Gaussian noise σ=0.1 °C (20 repeats) | 0.946 ± 0.000 |

### 4.9 Six-channel CNN (Cell 14, patient-level 5-fold CV)

| Metric | Value |
|---|---|
| Mean fold AUC | 0.935 |
| Fold AUC SD | 0.042 |
| Overall out-of-fold AUC | 0.924 |
| Balanced accuracy | 0.851 |
| Sensitivity | 0.746 |
| Specificity | 0.956 |

### 4.10 Matched-pair concordance (Cell 20)

- Matched pairs: **8**
- Pair concordance: **1.000** (0.5 = chance, 1.0 = every DM subject ranked above their matched control)

## 5. Final model delivered

- **Model:** median imputer → RobustScaler → Logistic Regression (L2, C=0.5, balanced class weights), trained on all subjects.
- **File:** `thermal_dm_model.joblib` (pipeline + feature order + threshold + sklearn version).
- **Backend module:** `thermal_inference.py` (feature extraction + `ThermalDMPredictor`).
- **Input:** left and right foot temperature CSVs (optionally the 8 angiosome CSVs). **Output:** DM probability, class, threshold, risk flag.

## 6. Limitations (state these honestly)

1. **Small dataset** — performance estimates have wide variance; see the SD columns.
2. **Age/gender confounding** — groups may differ in age; see sections 4.5 and 2.
3. **Threshold chosen on the same out-of-fold predictions it is reported on**, so specificity at the chosen threshold is optimistic.
4. **Assumptions to verify visually:** 0 °C pixels are background; left feet are mirrored to match right feet.
5. **No external validation** — the final model was trained on all data, so there is no untouched test set. Results come from cross-validation only.
6. **Screening aid only** — not a clinical diagnostic tool.

## 7. How to run

1. Run Cells 1–14 in order, then the fixed Cell 20, then Cell 21 (export)
2. Give the backend team `thermal_dm_model.joblib` and `thermal_inference.py`.
3. Install: `scikit-learn` (same version as printed at export), `pandas`, `numpy`, `scipy`, `opencv-python`, `joblib`.
