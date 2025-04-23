# BIOADAPT: Biomedical Adaptation Pipeline

## Overview

BIOADAPT is a configurable, end-to-end machine learning pipeline tailored for biomedical datasets. It automates data validation, preprocessing (including normalization, batch correction, imputation, and optional outlier detection), feature selection, model training, evaluation (cross‑validation with MCC, F1, AUC/AUPR), interpretability analyses (feature gradients, ICE, permutation importance, SHAP), and report generation in PDF format.

Key features:
- **Schema validation**: ensures input CSVs conform to expected columns and types (response, age, sex, biomarkers, optional batch).
- **Flexible preprocessing**: log₂ or z‑score normalization, median imputation, oversampling/undersampling for class imbalance, optional outlier removal.
- **Multiple algorithms**: logistic regression, SVM, random forest, XGBoost, decision tree, with grid search over hyperparameters and feature counts.
- **Interpretability hooks**: gradient variance, ICE plots, permutation importance, SHAP explanations (configurable).
- **Independent evaluation**: separate command to test on held‑out data and produce reports.
- **Command‑line interface**: simple `bioadapt` CLI with subcommands for training, evaluation, and log inspection.

## Repository Structure

```
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
└── README.md                # This documentation
```

## Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/an3317/BIOADAPT.git
   cd BIOADAPT
   ```

2. Create a Python virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. Install dependencies and the package in editable mode:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .
   ```

After installation, the `bioadapt` command is available on your PATH.

## Configuration

Most pipeline behavior is controlled via `config.yaml`. Example:

```yaml
data:
  paths:
    - data/train.csv              # One or more training CSVs
  response: response              # Name of the target column
  independent_test_path: data/test.csv

pipeline:
  output_folder: results         # Where to save models & reports
  algorithm: random_forest       # Options: logistic_regression, svm, xgboost, random_forest, cart
  feature_selection: anova       # Options: anova, mutual_info, lasso, rfe
  use_outlier_detection: false
  outlier_method: iqr            # iqr, zscore, isolation_forest, pca
  iqr_threshold: 0.05
  zscore_threshold: 0.05
  zscore_limit: 3.0
  iso_forest_threshold: 0.05
  pca_reconstruction_error_threshold: 0.1
  n_components_pca: null
  explain: shap                  # Options: shap, none

cv:
  random_seeds: [42, 43, 44, 45, 46]
```

- **data.paths**: list of CSV files for training.
- **data.response**: target column name.
- **data.independent_test_path**: CSV for held‑out evaluation.
- **pipeline.algorithm**: ML algorithm to train.
- **pipeline.feature_selection**: feature‑selection method.
- **pipeline.use_outlier_detection** & thresholds: toggle and parameters for FeatureOutlierRemover.
- **pipeline.explain**: set to `shap` to generate SHAP plots and HTML, or `none` to skip.
- **cv.random_seeds**: list of random seeds for repeated CV runs.

## CLI Usage

### `train`
Trains models according to your configuration, runs cross‑validation, generates interpretability plots, and exports a PDF report per seed.

```bash
bioadapt train --config config.yaml [--verbose]
```

- `-c / --config`: path to your YAML config (default `config.yaml`).
- `-v / --verbose`: show DEBUG‑level logs during execution.

**Output**: folder structure under `pipeline.output_folder`:
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
Runs a trained model on an independent test set and generates an independent‐test report.

```bash
bioadapt evaluate --config config.yaml
```

- Loads `model.pkl` from `results/seed_42` (you can customize the code to select a different seed).
- Reads `data.independent_test_path`.
- Produces confusion matrix, metrics, and `independent_test_report.pdf` in `results/independent/`.

### `view-log`
*(Optional)* Display the tail of the last `run.log` without rerunning the pipeline.

```bash
bioadapt view-log --config config.yaml --lines 200
```

- `--lines`: number of last lines to show (default 100).

## Testing

Run all unit tests with:
```bash
pytest -q
```

Tests cover schema validation, config loading, data‐loading, and transformer behavior.

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-change`.
3. Install in editable mode and add tests for your changes.
4. Commit & push; open a Pull Request.

Please ensure all CI checks pass (`pytest`, linters, GitHub Actions workflows) before merging.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

