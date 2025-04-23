![Build Status](https://img.shields.io/github/actions/workflow/status/an3317/BIOADAPT/ci.yml)
![Coverage Status](https://img.shields.io/codecov/c/gh/an3317/BIOADAPT)
![PyPI](https://img.shields.io/pypi/v/bioadapt)

# BIOADAPT: Biomedical Adaptation Pipeline

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [CLI Usage](#cli-usage)
  - [train](#train)
  - [evaluate](#evaluate)
  - [view-log](#view-log)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Overview

BIOADAPT is a configurable, end-to-end machine learning pipeline tailored for biomedical datasets. It automates data validation, preprocessing (including normalization, batch correction, imputation, and optional outlier detection), feature selection, model training, evaluation (cross-validation with MCC, F1, AUC/AUPR), interpretability analyses (feature gradients, ICE, permutation importance, SHAP), and report generation in PDF format.

**Key features:**
- **Schema validation**: Ensures input CSVs conform to expected columns and types (response, age, sex, biomarkers, optional batch).
- **Flexible preprocessing**: Log₂ or z-score normalization, median imputation, oversampling/undersampling for class imbalance, optional outlier removal.
- **Multiple algorithms**: Logistic regression, SVM, random forest, XGBoost, decision tree, with grid search over hyperparameters and feature counts.
- **Interpretability hooks**: Gradient variance, ICE plots, permutation importance, SHAP explanations (configurable).
- **Independent evaluation**: Separate command to test on held-out data and produce reports.
- **Command-line interface**: Simple `bioadapt` CLI with subcommands for training, evaluation, and log inspection.

## Repository Structure
```text
BIOADAPT/
├── bioadapt_pkg/             # Python package containing pipeline code
│   ├── cli.py               # CLI entrypoints
│   ├── config_loader.py     # YAML→dataclass config parser
│   ├── core.py              # Main pipeline function
│   ├── core_helpers.py      # Plotting and report helpers
│   ├── data_loading.py      # CSV loader + schema validation
│   ├── evaluation.py        # Independent-test evaluation
│   ├── explain.py           # SHAP explanation utilities
│   └── transformers.py      # Custom sklearn transformers
├── config.yaml              # Example configuration file
├── requirements.txt         # Python dependencies
├── setup.py                 # Package setup & console_scripts
├── tests/                   # Pytest unit tests
└── README.md                # Project documentation
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/an3317/BIOADAPT.git
   cd BIOADAPT
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate    # Windows
   ```

3. **Install dependencies and the package:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .
   ```

## Configuration
Most pipeline behavior is controlled via `config.yaml`. Example:
```yaml
data:
  paths:
    - data/train.csv
  response: response
  independent_test_path: data/test.csv

pipeline:
  output_folder: results
  algorithm: random_forest
  feature_selection: anova
  use_outlier_detection: false
  outlier_method: iqr
  iqr_threshold: 0.05
  zscore_threshold: 0.05
  zscore_limit: 3.0
  iso_forest_threshold: 0.05
  pca_reconstruction_error_threshold: 0.1
  n_components_pca: null
  explain: shap

cv:
  random_seeds: [42, 43, 44, 45, 46]
```

## CLI Usage

### `train`
Trains models, runs CV, generates interpretability plots, and exports PDF reports.
```bash
bioadapt train --config config.yaml [--verbose]
```
- `-c/--config`: Path to YAML config (default `config.yaml`).
- `-v/--verbose`: Enable DEBUG-level logging.

**Outputs** under `pipeline.output_folder`:
```
results/
├── seed_42/
│   ├── model.pkl
│   ├── run.log
│   ├── report.pdf
│   └── plots/...
├── seed_43/...
└── all_seeds_results.csv
```

### `evaluate`
Runs a trained model on an independent test set and generates a report.
```bash
bioadapt evaluate --config config.yaml
```
Reads model from `results/seed_42/model.pkl`, test CSV from `config.yaml`, and writes `independent_test_report.pdf` to `results/independent/`.

### `view-log`
Displays the tail of the last `run.log` without rerunning the pipeline.
```bash
bioadapt view-log --config config.yaml --lines 100
```

## Testing
Run all unit tests:
```bash
pytest -q
```

## Contributing
1. Fork this repo.
2. Create a branch: `git checkout -b feature/my-change`.
3. Install editable, add tests.
4. Commit, push, and open a PR.

## License
MIT © 2025

