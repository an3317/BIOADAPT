import os
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
    matthews_corrcoef,
    f1_score,
)

from .core_helpers import (
    ensure_directory_exists,
    create_pdf_report,
)

from .utils import (
    show_feature_ranges,
)

logger = logging.getLogger(__name__)

def evaluate_on_independent_dataset(
    model,
    df: pd.DataFrame,
    response_col: str,
    output_folder: Path,
) -> dict[str, float]:
    """
    Runs the trained pipeline `model` on `df` (independent test set),
    writes a PDF report in `output_folder`, and returns metrics.
    """

    ensure_directory_exists(output_folder)
    df = df.copy()
    # lowercase columns & drop patient identifiers
    df.columns = df.columns.str.lower()
    for col in ("patient", "patient_id"):
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    response = response_col.lower()
    if response not in df.columns:
        raise ValueError(f"Response column '{response}' not found in test data")

    X_test = df.drop(columns=[response])
    y_test = df[response]

    # predictions
    y_pred = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba") else None
    )

    # metrics
    test_mcc  = matthews_corrcoef(y_test, y_pred)
    test_f1   = f1_score(y_test, y_pred, average="weighted")
    test_auc  = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    precision, recall, _ = precision_recall_curve(y_test, y_proba) if y_proba is not None else (None, None, None)
    test_aupr = auc(recall, precision) if (precision is not None) else None
    test_cm   = confusion_matrix(y_test, y_pred)

    # track which features survived selection
    selected = list(X_test.columns)
    if hasattr(model, "named_steps") and "selection" in model.named_steps:
        sel = model.named_steps["selection"]
        if hasattr(sel, "get_support"):
            mask = sel.get_support()
            selected = list(X_test.columns[mask])

    # feature ranges for the first 10
    ranges_df = show_feature_ranges(X_test[selected])
    ranges_str = ranges_df.head(10).to_string()

    # build summary text
    results_text = (
        "Independent Test Results\n"
        f"Test MCC: {test_mcc:.4f}\n"
        f"Test F1 Score: {test_f1:.4f}\n"
        f"Test AUC: {test_auc if test_auc is not None else 'N/A'}\n"
        f"Test AUPR: {test_aupr if test_aupr is not None else 'N/A'}\n"
        f"Selected Features: {selected}\n"
        f"Number of Selected Features: {len(selected)}\n"
        f"Feature Ranges (First 10 Features):\n{ranges_str}"
    )

    # confusion‐matrix plot
    cm_png = output_folder / "test_confusion_matrix.png"
    plt.figure(figsize=(8,6))
    sns.heatmap(test_cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=np.unique(y_test),
                yticklabels=np.unique(y_test))
    plt.title("Independent Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(cm_png)
    plt.close()

    # PDF report
    pdf_path = output_folder / "independent_test_report.pdf"
    create_pdf_report(
        "Independent Model",
        "N/A",
        results_text,
        [str(cm_png)],
        pdf_path,
        cms={1: test_cm},
    )
    logger.info("Independent test report written to %s", pdf_path)

    return {
        "test_mcc":  test_mcc,
        "test_f1":   test_f1,
        "test_auc":  test_auc,
        "test_aupr": test_aupr,
    }
