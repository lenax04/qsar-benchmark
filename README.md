# qsar-benchmark

[![CI](https://github.com/dawidx1233/qsar-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/dawidx1233/qsar-benchmark/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-pending-lightgrey)](CITATION.cff)

**qsar-benchmark** is a systematic, fully reproducible benchmark of machine learning models for predicting EGFR kinase inhibitory activity (pIC50) from molecular structure. It compares four ML algorithms (Random Forest, XGBoost, SVM, MLP) across five molecular descriptor sets using 8,453 unique compounds from ChEMBL.

## Key Features

- **Standardized data pipeline**: automated retrieval and curation from ChEMBL via REST API
- **Five descriptor sets**: Morgan fingerprints (ECFP4), RDKit topological FP, MACCS keys, physicochemical descriptors, combined
- **Four ML algorithms**: RF, XGBoost, SVM, MLP — both regression (pIC50) and classification (active/inactive)
- **Rigorous evaluation**: 5-fold cross-validation + external test set (80/20 split)
- **SHAP interpretation**: feature importance linked to known EGFR pharmacophore elements
- **Full reproducibility**: conda environment, Docker container, fixed random seeds, MLflow tracking

## Results Summary

Benchmark on 8,453 EGFR inhibitors from ChEMBL (80/20 train/test split, 5-fold CV). Best values per metric in **bold**.

| Model | Descriptor | Test R² | Test RMSE | Test ROC-AUC |
|---|---|---|---|---|
| **XGBoost** | **RDKit FP** | **0.703** | **0.713** | 0.935 |
| XGBoost | Morgan | 0.644 | 0.776 | 0.922 |
| XGBoost | MACCS | 0.606 | 0.817 | 0.914 |
| XGBoost | Combined | 0.649 | 0.772 | 0.927 |
| XGBoost | PhysChem | 0.438 | 0.975 | 0.850 |
| **Random Forest** | **Combined** | 0.689 | 0.727 | **0.936** |
| Random Forest | Morgan | 0.690 | 0.726 | 0.933 |
| Random Forest | RDKit FP | 0.689 | 0.727 | 0.927 |
| Random Forest | MACCS | 0.617 | 0.808 | 0.919 |
| Random Forest | PhysChem | 0.460 | 0.957 | 0.861 |
| MLP | Morgan | 0.612 | 0.812 | 0.908 |
| MLP | RDKit FP | 0.613 | 0.811 | — |
| MLP | MACCS | 0.525 | 0.898 | 0.894 |
| MLP | Combined | 0.602 | 0.822 | 0.904 |
| MLP | PhysChem | 0.308 | 1.085 | 0.819 |
| LinearSVM | RDKit FP | 0.424 | 0.990 | 0.848 |
| LinearSVM | Morgan | 0.370 | 1.035 | 0.818 |
| LinearSVM | MACCS | 0.367 | 1.038 | 0.860 |
| LinearSVM | Combined | 0.353 | 1.049 | 0.815 |
| LinearSVM | PhysChem | 0.156 | 1.198 | 0.742 |

> **Key finding:** XGBoost + RDKit FP achieves the best regression performance (R² = 0.703, RMSE = 0.713 pIC50 units). Random Forest + Combined descriptors achieves the highest classification ROC-AUC (0.936). Physicochemical descriptors alone (12 features) reach R² = 0.460, demonstrating strong interpretable baselines.

## Repository Structure

```
qsar-benchmark/
├── README.md               # This file
├── CITATION.cff            # Citation metadata (CFF standard)
├── LICENSE                 # MIT License
├── environment.yml         # Conda environment for reproducibility
├── .github/workflows/      # GitHub Actions CI/CD
├── src/
│   ├── features.py         # Molecular descriptor calculation (RDKit)
│   └── models.py           # ML model definitions and evaluation utilities
├── tests/
│   ├── test_features.py    # Unit tests for descriptor calculation
│   └── test_models.py      # Unit tests for model training/evaluation
├── notebooks/
│   ├── 01_data_collection.ipynb      # ChEMBL data retrieval and curation
│   ├── 02_feature_engineering.ipynb  # Descriptor calculation and analysis
│   ├── 03_model_training.ipynb       # ML training, CV, and evaluation
│   └── 04_model_interpretation.ipynb # SHAP analysis and visualization
├── data/
│   ├── raw/                # Raw ChEMBL data (not tracked by git)
│   └── processed/          # Descriptor matrices (.npz, not tracked by git)
├── models/                 # Trained models (.joblib, not tracked by git)
├── results/
│   ├── plots/              # Publication-quality figures (300 DPI)
│   └── metrics/            # Benchmark tables (CSV)
└── paper/
    └── paper.md            # Manuscript draft (CSBJ/J. Cheminformatics format)
```

## Installation

### Option 1: Conda (recommended)

```bash
git clone https://github.com/lenax04/qsar-benchmark.git
cd qsar-benchmark
conda env create -f environment.yml
conda activate qsar-benchmark
```

### Option 2: pip

```bash
git clone https://github.com/lenax04/qsar-benchmark.git
cd qsar-benchmark
pip install rdkit chembl-webresource-client xgboost shap scikit-learn \
            joblib mlflow seaborn matplotlib pandas numpy scipy
```

## Quick Start

### Step 1: Download ChEMBL data

```bash
python3 qsar_01_data_collection.py
```

This downloads IC50 data for EGFR (CHEMBL203) and saves a cleaned dataset with pIC50 values.

### Step 2: Compute molecular descriptors

```bash
python3 qsar_02_feature_engineering.py
```

Computes all five descriptor sets and saves them as compressed NumPy arrays.

### Step 3: Train and evaluate models

```bash
python3 qsar_03_model_training.py
```

Runs the full benchmark (20 model-descriptor combinations, 5-fold CV + test set).

### Step 4: SHAP interpretation

```bash
python3 qsar_04_shap_interpretation.py
```

Computes SHAP values for the best model and generates interpretation plots.

### Using Jupyter Notebooks

```bash
jupyter lab notebooks/
```

Interactive notebooks with step-by-step analysis and visualizations.

## Reproducing Results

All experiments use fixed random seed (42). To reproduce exactly:

```bash
# Using conda environment
conda activate qsar-benchmark
python3 qsar_01_data_collection.py
python3 qsar_02_feature_engineering.py
python3 qsar_03_model_training.py
python3 qsar_04_shap_interpretation.py
```

## Data

Data are retrieved from [ChEMBL](https://www.ebi.ac.uk/chembl/) (release 34), a publicly available database of bioactive molecules. Target: **EGFR kinase** (CHEMBL203), IC50 measurements in nM.

- Raw data: 15,001 IC50 records
- After curation: **8,453 unique molecules**
- pIC50 range: 1.60 – 11.52 (mean ± SD: 6.77 ± 1.35)
- Active (pIC50 ≥ 6.0): 6,002 (71.0%)

## Citation

If you use this work, please cite:

```bibtex
@software{qsar_benchmark_2025,
  title  = {qsar-benchmark: ML-based QSAR models for EGFR inhibitor activity prediction},
  author = {[Author Name]},
  year   = {2025},
  url    = {https://github.com/dawidx1233/qsar-benchmark},
  version = {0.1.0}
}
```

See also [CITATION.cff](CITATION.cff) for full citation metadata.

## License

MIT License — see [LICENSE](LICENSE) for details.
