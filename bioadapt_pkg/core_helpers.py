import os
import re
import time
import functools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from fpdf import FPDF
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import matthews_corrcoef, f1_score
from joblib import Parallel, delayed

from bioadapt_pkg.utils import ensure_directory_exists


# -----------------------------
# Gradient & ICE plots
# -----------------------------

def plot_gradient_importance(features, output_folder, top_n=10):
    top_features = features.head(top_n)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_features['variance'], y=top_features['feature'], palette='viridis')
    plt.title('Gradient Importance Variance')
    plt.xlabel('Variance')
    plt.ylabel('Features')
    img_path = os.path.join(output_folder, 'gradient_importance_variance.png')
    plt.tight_layout()
    plt.savefig(img_path)
    plt.close()
    print(f"Gradient importance plot saved at: {img_path}")
    return img_path



def plot_ice_plots(model, X, selected_features, output_folder):
    ice_img_paths = []
    top_features_for_ice = selected_features[:10]
    for feature in top_features_for_ice:
        fig, ax = plt.subplots(figsize=(6, 4))
        PartialDependenceDisplay.from_estimator(model, X, [feature], kind='individual', ax=ax)
        ax.set_title(f'ICE Plot for {feature}')
        img_path = os.path.join(output_folder, f'ice_{feature}.png')
        plt.tight_layout()
        plt.savefig(img_path)
        plt.close(fig)
        ice_img_paths.append(img_path)
        print(f"ICE plot for feature '{feature}' saved at: {img_path}")
    return ice_img_paths


# -----------------------------
# Feature gradients
# -----------------------------

def calculate_feature_gradients(model, X, selected_features):
    estimator = model.named_steps.get('est')
    if not estimator:
        raise ValueError("Estimator step 'est' not found in the pipeline.")
    if hasattr(estimator, 'feature_importances_'):
        importances = estimator.feature_importances_
    elif hasattr(estimator, 'coef_'):
        coef = estimator.coef_
        if len(coef.shape) > 1:
            importances = np.mean(np.abs(coef), axis=0)
        else:
            importances = np.abs(coef)
    else:
        importances = np.zeros(len(selected_features))
    gradients = {'feature': selected_features, 'variance': importances}
    return pd.DataFrame(gradients)


def rank_features_by_gradient_variance(gradients_df):
    return gradients_df.sort_values(by='variance', ascending=False)



def run_gradient_analysis(model, X, selected_features, output_folder):
    gradients_df = calculate_feature_gradients(model, X, selected_features)
    sorted_features = rank_features_by_gradient_variance(gradients_df)
    gradient_img_path = plot_gradient_importance(sorted_features, output_folder, top_n=10)
    ice_img_paths = plot_ice_plots(model, X, selected_features, output_folder)
    return gradient_img_path, ice_img_paths


# -----------------------------
# Permutation importance
# -----------------------------

def plot_permutation_importance(best_model, X, y, selected_features, output_folder):
    try:
        steps = []
        if 'scaler' in best_model.named_steps:
            steps.append(('scaler', best_model.named_steps['scaler']))
        if 'outlier_removal' in best_model.named_steps:
            steps.append(('outlier_removal', best_model.named_steps['outlier_removal']))
        if 'selection' in best_model.named_steps:
            steps.append(('selection', best_model.named_steps['selection']))
        sub_pipeline = Pipeline(steps)
        X_transformed = sub_pipeline.transform(X)
        temp_X = X.copy()
        if 'outlier_removal' in best_model.named_steps:
            temp_X = best_model.named_steps['outlier_removal'].transform(temp_X)
        if 'selection' in best_model.named_steps and hasattr(best_model.named_steps['selection'], 'get_support'):
            mask = best_model.named_steps['selection'].get_support()
            feature_names = temp_X.columns[mask]
        else:
            feature_names = temp_X.columns
        if len(feature_names) != X_transformed.shape[1]:
            raise ValueError(f"Mismatch: feature_names length ({len(feature_names)}) != X_transformed.shape[1] ({X_transformed.shape[1]}).")
        estimator = best_model.named_steps['est']
        perm_importance = permutation_importance(estimator, X_transformed, y, n_repeats=10, random_state=42, n_jobs=-1)
        perm_sorted_idx = np.argsort(perm_importance.importances_mean)[::-1]
        top_n = min(10, X_transformed.shape[1])
        top_features = [feature_names[i] for i in perm_sorted_idx[:top_n]]
        top_importances = perm_importance.importances_mean[perm_sorted_idx[:top_n]]
        plt.figure(figsize=(10, 6))
        plot_df = pd.DataFrame({'feature': top_features, 'importance': top_importances})
        sns.barplot(data=plot_df, x='importance', y='feature', palette='plasma')
        plt.title('Permutation Feature Importance')
        plt.xlabel('Importance Score (Mean)')
        plt.ylabel('Features')
        img_path = os.path.join(output_folder, 'permutation_importance.png')
        plt.tight_layout()
        plt.savefig(img_path)
        plt.close()
        print(f"Permutation feature importance plot saved at: {img_path}")
        return img_path
    except Exception as e:
        print(f"An error occurred in plot_permutation_importance: {e}")
        raise


# -----------------------------
# PDF reporting
# -----------------------------
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Model Performance Report', ln=True, align='C')
        self.ln(5)
    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')


def add_confusion_matrix_table(pdf, cm, title):
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, title, ln=True)
    pdf.set_font("Arial", '', 12)
    col_width = pdf.w / 4.5
    row_height = pdf.font_size * 1.5
    classes = list(range(cm.shape[0]))
    pdf.cell(col_width, row_height, '', border=1)
    for cls in classes:
        pdf.cell(col_width, row_height, f'Predicted {cls}', border=1, align='C')
    pdf.ln(row_height)
    for idx, cls in enumerate(classes):
        pdf.cell(col_width, row_height, f'Actual {cls}', border=1)
        for val in cm[idx]:
            pdf.cell(col_width, row_height, str(val), border=1, align='C')
        pdf.ln(row_height)
    pdf.ln(row_height + 5)


def create_pdf_report(
    model_name: str,
    feature_selection: str,
    results_text: str,
    images: list[str] | list[Path],
    output_pdf: str | Path,
    cms: dict[int, np.ndarray] | None = None,
    extra_files: list[str] | list[Path] | None = None,
) -> None:
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 10, "Model Performance Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", '', 14)
    pdf.cell(0, 10, f"Model: {model_name}", ln=True, align='C')
    pdf.cell(0, 10, f"Feature Selection: {feature_selection}", ln=True, align='C')
    pdf.ln(20)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, 'Results Summary', ln=True)
    pdf.ln(5)
    for line in results_text.split('\n'):
        if not line.strip():
            continue
        if line.startswith(('Top Model Results', 'Ranked Models Information')):
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, line, ln=True)
            pdf.ln(5)
        elif line.startswith('Rank '):
            pdf.set_font("Arial", 'B', 12)
            pdf.ln(5)
            pdf.cell(0, 10, line, ln=True)
        else:
            pdf.set_font("Arial", '', 12)
            pdf.multi_cell(0, 8, line)
            pdf.ln(2)
    pdf.ln(5)
    if cms:
        for rank, cm in cms.items():
            add_confusion_matrix_table(pdf, cm, f"Confusion Matrix (Rank {rank})")
    if images:
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, 'Visualizations', ln=True)
        pdf.ln(5)
        for image in images:
            try:
                pdf.add_page()
                name = Path(image).stem.replace('_', ' ').title()
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, name, ln=True, align='C')
                pdf.ln(5)
                pdf.image(str(image), x=15, w=pdf.w - 30)
                pdf.ln(10)
            except Exception as e:
                print(f"Error adding image {image}: {e}")
    if extra_files:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, 'Extra files / Dashboards', ln=True)
        pdf.ln(5)
        pdf.set_font("Arial", '', 12)
        for path in extra_files:
            fname = Path(path).name
            # embed clickable link directly
            url = Path(path).absolute().as_uri()
            pdf.cell(0, 8, f"- {fname}", ln=True, link=url)
    try:
        pdf.output(str(output_pdf))
        print(f"PDF report saved to {output_pdf}")
    except Exception as e:
        print(f"Error saving PDF report to {output_pdf}: {e}")
        raise

# -----------------------------
# Performance drop analysis
# -----------------------------
def evaluate_performance_drop(selected_features, model, X, y):
    base = clone(model)
    base.fit(X[selected_features], y)
    y_base = base.predict(X[selected_features])
    base_mcc = matthews_corrcoef(y, y_base)
    base_f1  = f1_score(y, y_base, average='weighted')
    drops_mcc = {}
    drops_f1  = {}

    def _drop(feature):
        rem = [f for f in selected_features if f != feature]
        if not rem:
            return feature, 0, 0
        m = clone(model)
        m.fit(X[rem], y)
        y_pred = m.predict(X[rem])
        # Return a tuple of (feature, mcc_drop, f1_drop)
        return (
            feature,
            base_mcc - matthews_corrcoef(y, y_pred),
            base_f1  - f1_score(y, y_pred, average='weighted')
        )

    # Use all available cores for parallel evaluation
    results = Parallel(n_jobs=-1)(delayed(_drop)(f) for f in selected_features)
    for feat, mcc_d, f1_d in results:
        drops_mcc[feat] = mcc_d
        drops_f1[feat] = f1_d
    top_mcc = sorted(drops_mcc.items(), key=lambda x: x[1], reverse=True)[:10]
    top_f1  = sorted(drops_f1.items(),  key=lambda x: x[1], reverse=True)[:10]
    return top_mcc, top_f1


