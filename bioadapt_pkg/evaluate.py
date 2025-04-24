import os
import joblib
import pandas as pd
import click
from pathlib import Path
from fpdf import FPDF
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, f1_score, matthews_corrcoef,
    roc_auc_score, precision_recall_curve, auc
)
from bioadapt_pkg.config_loader import load_config


# ──────────────────────── PDF helpers ───────────────────────────────────────────────────
class PDFReport(FPDF):
    def header(self):
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, "Independent Test Report", ln=True, align="C")
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def _save_pdf_report(
    title: str,
    summary_text: str,
    image_paths: list[str] | list[Path],
    outfile: str | Path
) -> None:
    pdf = PDFReport()
    pdf.set_auto_page_break(True, 15)
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 18)
    pdf.multi_cell(0, 10, title, align="C")
    pdf.ln(2)

    # Summary text
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 7, summary_text)
    pdf.ln(4)

    # Images
    for img in image_paths:
        img = Path(img)
        if not img.exists():
            continue
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, img.name, ln=True, align="C")
        pdf.ln(3)
        pdf.image(str(img), x=15, w=pdf.w - 30)

    pdf.output(str(outfile))
    print(f"✅ PDF written → {outfile}")


# ─────────────────────── Main evaluation helper ─────────────────────────────────────────
def evaluate_on_independent_dataset(
    trained_pipe,
    df_test_raw: pd.DataFrame,
    response_col: str,
    out_dir: str | Path,
):
    """
    Run a trained Pipeline on an independent test DataFrame.
    Writes a confusion-matrix PNG + PDF report, returns key metrics.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = df_test_raw.copy()
    df.columns   = df.columns.str.lower()
    response_col = response_col.lower()
    # drop optional patient columns
    for c in ("patient", "patient_id"):
        df = df.drop(columns=[c], errors="ignore")

    if response_col not in df.columns:
        raise ValueError(f"response col “{response_col}” not found.")

    X_raw = df.drop(columns=[response_col])
    y     = df[response_col]

    # align columns to training
    scaler = trained_pipe.named_steps.get("scaler")
    if hasattr(scaler, "feature_names_in_"):
        train_cols = list(scaler.feature_names_in_)
    else:
        train_cols = list(X_raw.columns)

    X_aligned = X_raw.reindex(columns=train_cols, fill_value=0.0)

    # predictions & metrics
    y_pred  = trained_pipe.predict(X_aligned)
    y_proba = None
    if hasattr(trained_pipe.named_steps["est"], "predict_proba"):
        y_proba = trained_pipe.predict_proba(X_aligned)[:, 1]

    mcc       = matthews_corrcoef(y, y_pred)
    f1_sc     = f1_score(y, y_pred, average="weighted")
    auc_score = aupr = None
    if y_proba is not None:
        auc_score = roc_auc_score(y, y_proba)
        prec, rec, _ = precision_recall_curve(y, y_proba)
        aupr = auc(rec, prec)

    cm = confusion_matrix(y, y_pred)

    # save confusion matrix heatmap
    cm_png = out_dir / "confusion_matrix.png"
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=np.unique(y), yticklabels=np.unique(y)
    )
    plt.xlabel("Predicted"); plt.ylabel("Actual");
    plt.tight_layout(); plt.savefig(cm_png); plt.close()

    # PDF report
    summary = (
        f"Samples  : {len(y)}\n"
        f"MCC      : {mcc:.4f}\n"
        f"F1 Score : {f1_sc:.4f}\n"
        f"AUC      : {auc_score if auc_score is not None else 'N/A'}\n"
        f"AUPR     : {aupr if aupr is not None else 'N/A'}\n"
        f"Confusion matrix:\n{cm}"
    )
    _save_pdf_report(
        "Independent-set evaluation",
        summary,
        [cm_png],
        out_dir / "report.pdf",
    )

    return {"mcc": mcc, "f1": f1_sc, "auc": auc_score, "aupr": aupr, "pdf": str(out_dir / "report.pdf")}


# ─────────────────────────── CLI entry point ───────────────────────────────────────────
@click.command()
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    default="config.yaml",
    help="Path to YAML config file."
)
@click.option(
    "--seed", "-s",
    type=int,
    default=None,
    help="Seed folder to load model from (e.g. 42)."
)
def evaluate(config, seed):
    """
    Evaluate trained pipeline on independent test set specified in config.
    """
    cfg = load_config(Path(config))
    out_root = Path(cfg.pipeline.output_folder)
    algo_fs  = out_root / f"{cfg.pipeline.algorithm}_{cfg.pipeline.feature_selection}"
    algo_fs.mkdir(parents=True, exist_ok=True)

    # determine model path
    if seed is not None:
        model_path = algo_fs / f"seed_{seed}" / "model.pkl"
    else:
        # pick the first found
        candidates = list(algo_fs.glob("seed_*/model.pkl"))
        if not candidates:
            raise FileNotFoundError("No model.pkl found under any seed_*/ folder.")
        model_path = candidates[0]
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = joblib.load(model_path)
    df_test = pd.read_csv(cfg.data.independent_test_path)

    seed_dir = model_path.parent                                   
    eval_dir = seed_dir / "independent"                            
    metrics  = evaluate_on_independent_dataset(
        model,
        df_test,
        cfg.data.response,
        eval_dir,
    )

    print("Independent test metrics:", metrics)

if __name__ == "__main__":
    evaluate()
