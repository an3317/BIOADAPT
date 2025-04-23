from pathlib import Path
import os
import re
import json
import logging
import multiprocessing
import time
import functools

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fpdf import FPDF
from tqdm import tqdm
import psutil
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (confusion_matrix, roc_auc_score, precision_recall_curve, auc,
                             matthews_corrcoef, f1_score, make_scorer)
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

from bioadapt_pkg.config_loader import Config
from bioadapt_pkg.logging_conf import setup_logger
from bioadapt_pkg.data_loading import load_csvs
from bioadapt_pkg.transformers import Log2NormalizationAndBatchCorrectionTransformer, FeatureOutlierRemover
from bioadapt_pkg.utils import (
    ensure_directory_exists,
    show_feature_ranges,
    get_feature_selection_step,
    get_param_grid,
    get_model,
)


# Set environment variables for BLAS libraries to use 16 threads
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"

# -----------------------------
# NEW: Log2 Normalization and Batch Correction Transformer
# -----------------------------
def combat_batch_correction(data, batch_info):
    """
    A placeholder batch correction function.
    For each batch, subtract the batch-specific mean.
    Replace with a robust method (e.g., pyCombat) for production use.
    """
    corrected_data = data.copy()
    for batch in batch_info.unique():
        mask = batch_info == batch
        batch_mean = data.loc[mask].mean()
        corrected_data.loc[mask] = data.loc[mask] - batch_mean
    return corrected_data

class Log2NormalizationAndBatchCorrectionTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that applies a log₂(x+1) normalization safely and performs batch correction if a batch column exists.
    
    - If X is a list of DataFrames (e.g., multiple datasets), each DataFrame is normalized and then merged.
    - If X is a single DataFrame:
         * Without a batch column: only log₂ normalization is applied.
         * With a batch column: log₂ normalization is applied followed by batch correction.
    
    Any infinities or NaN values produced are replaced (NaNs are filled with the median of the column).
    """
    def __init__(self, batch_col='batch'):
        self.batch_col = batch_col

    def safe_log2_transform(self, df):
        # Apply log2(x+1) to all numeric columns.
        df_numeric = df.select_dtypes(include=[np.number])
        transformed = np.log2(df_numeric + 1)
        # Replace inf and -inf with NaN
        transformed = transformed.replace([np.inf, -np.inf], np.nan)
        # Fill NaNs with the median of the column
        for col in transformed.columns:
            median_val = transformed[col].median()
            transformed[col] = transformed[col].fillna(median_val)
        df_result = df.copy()
        df_result[transformed.columns] = transformed
        return df_result

    def transform(self, X, y=None):
        if isinstance(X, list):
            # Process each DataFrame individually
            processed_list = []
            for df in X:
                if not isinstance(df, pd.DataFrame):
                    df = pd.DataFrame(df)
                df_trans = self.safe_log2_transform(df)
                processed_list.append(df_trans)
            # Merge all DataFrames along rows
            merged_df = pd.concat(processed_list, ignore_index=True)
            if self.batch_col in merged_df.columns:
                batch_info = merged_df[self.batch_col]
                numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
                corrected_numeric = combat_batch_correction(merged_df[numeric_cols], batch_info)
                merged_df[corrected_numeric.columns] = corrected_numeric
            return merged_df
        elif isinstance(X, pd.DataFrame):
            X_ = self.safe_log2_transform(X.copy())
            if self.batch_col in X_.columns:
                batch_info = X_[self.batch_col]
                numeric_cols = X_.select_dtypes(include=[np.number]).columns
                corrected_numeric = combat_batch_correction(X_[numeric_cols], batch_info)
                X_[corrected_numeric.columns] = corrected_numeric
            return X_
        else:
            raise ValueError("Input X must be a DataFrame or a list of DataFrames.")

    def fit(self, X, y=None):
        # This transformer is stateless, so no fitting is required.
        return self

# -----------------------------
# (The rest of your code remains unchanged below)
# -----------------------------
def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        print(f"Function {func.__name__} took {elapsed:.4f} seconds.")
        return result
    return wrapper

def ensure_directory_exists(directory):
    os.makedirs(directory, exist_ok=True)

def show_feature_ranges(df):
    return df.agg(['min', 'max'])

class FeatureOutlierRemover(BaseEstimator, TransformerMixin):
    """
    A transformer to remove features considered noisy or outlier-prone.
    Methods supported:
      - 'iqr': Remove features with a high proportion of extreme outliers based on IQR.
      - 'zscore': Remove features that frequently exceed a specified z-score limit.
      - 'isolation_forest': For each feature individually, run IsolationForest to identify outliers.
      - 'pca': Use PCA reconstruction error. Features with high average reconstruction error are removed.
    """
    def __init__(self, method='iqr', 
                 iqr_threshold=0.05, 
                 zscore_threshold=0.05, 
                 zscore_limit=3.0,
                 iso_forest_threshold=0.05,
                 pca_reconstruction_error_threshold=0.1,
                 n_components_pca=None):
        self.method = method
        self.iqr_threshold = iqr_threshold
        self.zscore_threshold = zscore_threshold
        self.zscore_limit = zscore_limit
        self.iso_forest_threshold = iso_forest_threshold
        self.pca_reconstruction_error_threshold = pca_reconstruction_error_threshold
        self.n_components_pca = n_components_pca
        self.features_to_remove_ = []

    def fit(self, X, y=None):
        if not hasattr(X, 'columns'):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        else:
            X = X.copy()

        self.features_to_remove_ = []

        if self.method == 'iqr':
            Q1 = X.quantile(0.25)
            Q3 = X.quantile(0.75)
            IQR = Q3 - Q1
            for col in X.columns:
                lower_bound = Q1[col] - 1.5 * IQR[col]
                upper_bound = Q3[col] + 1.5 * IQR[col]
                outliers = ((X[col] < lower_bound) | (X[col] > upper_bound)).sum()
                proportion = outliers / X.shape[0]
                if proportion > self.iqr_threshold:
                    self.features_to_remove_.append(col)

        elif self.method == 'zscore':
            for col in X.columns:
                outliers = (X[col].abs() > self.zscore_limit).sum()
                proportion = outliers / X.shape[0]
                if proportion > self.zscore_threshold:
                    self.features_to_remove_.append(col)

        elif self.method == 'isolation_forest':
            for col in X.columns:
                iso = IsolationForest(random_state=42)
                vals = X[[col]].values
                iso.fit(vals)
                preds = iso.predict(vals)
                outlier_count = (preds == -1).sum()
                proportion = outlier_count / X.shape[0]
                if proportion > self.iso_forest_threshold:
                    self.features_to_remove_.append(col)

        elif self.method == 'pca':
            n_components = self.n_components_pca if self.n_components_pca is not None else min(10, X.shape[1])
            pca = PCA(n_components=n_components)
            pca.fit(X)
            X_pca = pca.transform(X)
            X_reconstructed = pca.inverse_transform(X_pca)
            errors = np.mean(np.abs(X.values - X_reconstructed), axis=0)
            for i, col in enumerate(X.columns):
                if errors[i] > self.pca_reconstruction_error_threshold:
                    self.features_to_remove_.append(col)
        else:
            raise ValueError(f"Unrecognized outlier removal method: {self.method}")

        return self

    def transform(self, X):
        if not hasattr(X, 'columns'):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        return X.drop(columns=self.features_to_remove_, errors='ignore')

    def get_removed_features(self):
        return self.features_to_remove_

@timed
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

@timed
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

@timed
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
        # warnings.warn("Estimator does not have 'feature_importances_' or 'coef_'. Setting all importances to zero.")
    gradients = {'feature': selected_features, 'variance': importances}
    return pd.DataFrame(gradients)

def rank_features_by_gradient_variance(gradients_df):
    return gradients_df.sort_values(by='variance', ascending=False)

@timed
def run_gradient_analysis(model, X, selected_features, output_folder):
    gradients_df = calculate_feature_gradients(model, X, selected_features)
    sorted_features = rank_features_by_gradient_variance(gradients_df)
    gradient_img_path = plot_gradient_importance(sorted_features, output_folder, top_n=10)
    ice_img_paths = plot_ice_plots(model, X, selected_features, output_folder)
    return gradient_img_path, ice_img_paths

@timed
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
            support_mask = best_model.named_steps['selection'].get_support()
            feature_names = temp_X.columns[support_mask]
        else:
            feature_names = temp_X.columns
        if len(feature_names) != X_transformed.shape[1]:
            raise ValueError(f"Mismatch: feature_names length ({len(feature_names)}) != X_transformed.shape[1] ({X_transformed.shape[1]}).")
        estimator = best_model.named_steps['est']
        perm_importance = permutation_importance(
            estimator, 
            X_transformed, 
            y, 
            n_repeats=10, 
            random_state=42, 
            n_jobs=-1  # Use all cores here
        )
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
    num_classes = cm.shape[0]
    classes = list(range(num_classes))
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

@timed
def create_pdf_report(model_name, feature_selection, results_text, images, output_pdf, cms=None):
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 10, "Model Performance Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", '', 14)
    pdf.cell(0, 10, f"Model: {model_name}", ln=True, align='C')
    pdf.cell(0, 10, f"Feature Selection Method: {feature_selection}", ln=True, align='C')
    pdf.ln(20)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, 'Results Summary', ln=True)
    pdf.ln(5)
    lines = results_text.split('\n')
    for line in lines:
        if line.strip() == '':
            continue
        elif line.startswith('Top Model Results') or line.startswith('Ranked Models Information'):
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
                image_name = os.path.basename(image).replace('_', ' ').split('.')[0].title()
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, image_name, ln=True, align='C')
                pdf.ln(5)
                max_width = pdf.w - 30
                pdf.image(image, x=15, y=None, w=max_width)
                pdf.ln(10)
            except Exception as e:
                print(f"Error adding image {image}: {e}")
                continue
    try:
        pdf.output(output_pdf)
        print(f"PDF report created and saved to: {output_pdf}")
    except Exception as e:
        print(f"Error saving PDF report to {output_pdf}: {e}")
        raise

@timed
def evaluate_performance_drop(selected_features, model, X, y):
    model_clone = clone(model)
    model_clone.fit(X[selected_features], y)
    y_pred = model_clone.predict(X[selected_features])
    baseline_mcc = matthews_corrcoef(y, y_pred)
    baseline_f1 = f1_score(y, y_pred, average='weighted')
    mcc_performance_drop = {}
    f1_performance_drop = {}
    def evaluate_feature_drop(feature):
        remaining_features = [f for f in selected_features if f != feature]
        if len(remaining_features) == 0:
            return (feature, 0, 0)
        model_feature = clone(model)
        model_feature.fit(X[remaining_features], y)
        y_pred_reduced = model_feature.predict(X[remaining_features])
        reduced_mcc = matthews_corrcoef(y, y_pred_reduced)
        reduced_f1 = f1_score(y, y_pred_reduced, average='weighted')
        mcc_drop = baseline_mcc - reduced_mcc
        f1_drop = baseline_f1 - reduced_f1
        return (feature, mcc_drop, f1_drop)
    # Use all available cores for parallel evaluation
    results = Parallel(n_jobs=-1)(delayed(evaluate_feature_drop)(feature) for feature in selected_features)
    for feature, mcc_drop, f1_drop in results:
        mcc_performance_drop[feature] = mcc_drop
        f1_performance_drop[feature] = f1_drop
    sorted_mcc_drop = sorted(mcc_performance_drop.items(), key=lambda x: x[1], reverse=True)[:10]
    sorted_f1_drop = sorted(f1_performance_drop.items(), key=lambda x: x[1], reverse=True)[:10]
    return sorted_mcc_drop, sorted_f1_drop

@timed
def plot_performance_drop(performance_drop, title, xlabel, ylabel, output_file):
    features = [item[0] for item in performance_drop]
    drops = [item[1] for item in performance_drop]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=drops, y=features, palette='viridis')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Performance drop plot saved at: {output_file}")

def integrate_into_pipeline(X, y, selected_features, model, output_folder):
    ensure_directory_exists(output_folder)
    top_10_mcc_drop, top_10_f1_drop = evaluate_performance_drop(selected_features, model, X, y)
    mcc_output_path = os.path.join(output_folder, 'mcc_performance_drop.png')
    plot_performance_drop(top_10_mcc_drop, 'Top 10 Features with the Highest MCC Drop',
                          'MCC Performance Drop', 'Features', mcc_output_path)
    f1_output_path = os.path.join(output_folder, 'f1_performance_drop.png')
    plot_performance_drop(top_10_f1_drop, 'Top 10 Features with the Highest F1 Score Drop',
                          'F1 Score Performance Drop', 'Features', f1_output_path)
    return [mcc_output_path, f1_output_path]

def get_feature_selection_step(fs_method):
    if fs_method == 'anova':
        return ('selection', SelectKBest(score_func=f_classif))
    elif fs_method == 'mutual_info':
        return ('selection', SelectKBest(score_func=mutual_info_classif))
    elif fs_method == 'lasso':
        return ('selection', SelectFromModel(Lasso(max_iter=10000)))
    elif fs_method == 'rfe':
        return ('selection', RFE(estimator=LogisticRegression(max_iter=1000)))
    else:
        raise ValueError(f"Invalid feature selection method '{fs_method}' chosen.")

def get_param_grid(fs_method, algorithm, k_range):
    if algorithm == 'logistic_regression':
        model_param_grid = {'est__C': [0.01, 0.1, 1, 10], 'est__penalty': ['l2']}
    elif algorithm == 'svm':
        model_param_grid = {'est__C': [0.1, 1, 10], 'est__kernel': ['linear', 'rbf']}
    elif algorithm == 'xgboost':
        model_param_grid = {'est__n_estimators': [100, 200], 'est__max_depth': [3, 6], 'est__learning_rate': [0.01, 0.1]}
    elif algorithm == 'random_forest':
        model_param_grid = {'est__n_estimators': [100, 200], 'est__max_depth': [None, 10, 20]}
    elif algorithm == 'cart':
        model_param_grid = {'est__max_depth': [None, 10, 20], 'est__min_samples_split': [2, 10]}
    else:
        raise ValueError(f"Invalid algorithm '{algorithm}' chosen.")
    if fs_method in ['anova', 'mutual_info']:
        fs_param = {'selection__k': k_range}
    elif fs_method == 'lasso':
        fs_param = {'selection__estimator__alpha': [0.001, 0.01, 0.1, 1]}
    elif fs_method == 'rfe':
        fs_param = {'selection__n_features_to_select': k_range}
    else:
        fs_param = {}
    return {**model_param_grid, **fs_param}

def get_model(algorithm):
    if algorithm == 'logistic_regression':
        return LogisticRegression(max_iter=1000)
    elif algorithm == 'svm':
        return SVC(probability=True)
    elif algorithm == 'xgboost':
        return XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    elif algorithm == 'random_forest':
        return RandomForestClassifier()
    elif algorithm == 'cart':
        return DecisionTreeClassifier()
    else:
        raise ValueError(f"Invalid algorithm '{algorithm}' chosen.")

def create_algorithm_feature_selection_folder(output_folder, algorithm, feature_selection):
    folder_name = f"{algorithm}_{feature_selection}"
    folder_name = re.sub(r'\W+', '_', folder_name)
    full_folder_path = os.path.join(output_folder, folder_name)
    ensure_directory_exists(full_folder_path)
    return full_folder_path

def print_cpu_usage(context=""):
    usage = psutil.cpu_percent(interval=1)
    print(f"[{context}] CPU usage: {usage}%")



def track_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_time = time.perf_counter() - start_time
        print(f"⏱️ {func.__name__} executed in {elapsed_time:.2f} seconds.")
        return result
    return wrapper

@track_time

def machine_learning_pipeline(cfg: Config) -> None:
    """Run the end‑to‑end ML pipeline based on a Config object."""
    # 1) Unpack runtime settings
    data_paths = [str(p) for p in cfg.data.paths]
    response = cfg.data.response.lower()
    selected_algorithm = cfg.pipeline.algorithm
    selected_feature_selection = cfg.pipeline.feature_selection
    output_folder = str(cfg.pipeline.output_folder)
    random_seeds = cfg.cv.random_seeds
    use_outlier_detection = cfg.pipeline.use_outlier_detection
    outlier_method = cfg.pipeline.outlier_method


    # 2) Set up logging
    logger = setup_logger(Path(output_folder) / "run.log")
    logger.info("Pipeline started")
    logger.debug("Config: %s", cfg)

    # 3) Load & validate data
    datasets = load_csvs(data_paths)
    if isinstance(datasets, pd.DataFrame):
        data = datasets
    else:
        data = pd.concat(datasets, ignore_index=True)
    logger.info(f"Data loaded & schema-validated. Shape: {data.shape}")


    # 5) Basic system info
    num_cores = multiprocessing.cpu_count()
    logger.info(f"Number of CPU cores: {num_cores}")

    # 6) Prepare output directory
    ensure_directory_exists(output_folder)
    logger.info(f"Output folder set to: {output_folder}")

    # 7) Preprocess DataFrame
    data.columns = list(map(str.lower, data.columns))
    if 'patient_id' in data.columns:
        data = data.drop(columns=['patient_id'])
    X = data.drop(columns=[response])
    y = data[response]
    constant_columns = [col for col in X.columns if X[col].nunique() <= 1]
    if constant_columns:
        logger.info(f"Dropping constant columns: {constant_columns}")
        X = X.drop(columns=constant_columns)
    logger.info(f"Data preprocessed. Shape of X: {X.shape}")

    # 8) Handle class imbalance
    class_counts = y.value_counts()
    sampler = None
    if min(class_counts) / max(class_counts) < 0.5:
        if min(class_counts) * 2 < max(class_counts):
            sampler = RandomOverSampler(random_state=42)
            sampling_method = 'Oversampling'
        else:
            sampler = RandomUnderSampler(random_state=42)
            sampling_method = 'Undersampling'
        logger.info(f"Sampling method set to: {sampling_method}")

    # 9) Feature outlier removal (optional)
    outlier_remover_instance = None
    use_outlier_detection = cfg.pipeline.use_outlier_detection
    outlier_method = getattr(cfg.pipeline, 'outlier_method', None)
    if use_outlier_detection and outlier_method:
        outlier_remover_instance = FeatureOutlierRemover(
            method=outlier_method,
            iqr_threshold=cfg.pipeline.iqr_threshold,
            zscore_threshold=cfg.pipeline.zscore_threshold,
            zscore_limit=cfg.pipeline.zscore_limit,
            iso_forest_threshold=cfg.pipeline.iso_forest_threshold,
            pca_reconstruction_error_threshold=cfg.pipeline.pca_reconstruction_error_threshold,
            n_components_pca=cfg.pipeline.n_components_pca,
        )
        X_temp = outlier_remover_instance.fit_transform(X)
        n_features = X_temp.shape[1]
        logger.info(f"After outlier detection, number of features: {n_features}")
    else:
        n_features = X.shape[1]
        logger.info(f"No outlier detection. Number of features: {n_features}")

    # 10) Determine k-range for feature selection
    k_values = list(range(10, 65, 5))
    k_range = [k for k in k_values if k <= n_features] or [min(10, n_features)]
    logger.info(f"Candidate k values for feature selection: {k_range}")

    # 11) Build the sklearn/imb pipeline
    fs_step = get_feature_selection_step(selected_feature_selection)
    model = get_model(selected_algorithm)
    steps = [
        ('norm_batch_corr', Log2NormalizationAndBatchCorrectionTransformer(batch_col='batch')),
        ('scaler', StandardScaler())
    ]
    if sampler:
        steps.append(('sampler', sampler))
    if outlier_remover_instance:
        steps.append(('outlier_removal', outlier_remover_instance))
        logger.info("Added outlier_removal step to pipeline")
    steps += [fs_step, ('est', model)]
    logger.info(f"Pipeline steps: {[name for name, _ in steps]}")
    pipe = ImbPipeline(steps=steps)

    # 12) Set up grid search
    param_grid = get_param_grid(selected_feature_selection, selected_algorithm, k_range)
    mcc_scorer = make_scorer(matthews_corrcoef)
    f1_scorer = make_scorer(f1_score, average='weighted')

    # 13) Cross-validation loop
    all_results = []
    seed_progress = tqdm(total=len(random_seeds), desc="Pipeline Progress", unit="seed")
    for idx, seed in enumerate(random_seeds, start=1):
        logger.info(f"Processing seed {idx}/{len(random_seeds)} (seed={seed})")
        seed_folder = os.path.join(output_folder, f"seed_{seed}")
        ensure_directory_exists(seed_folder)

        outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        grid_search = GridSearchCV(pipe, param_grid, cv=inner_cv,
                                   scoring={'MCC': mcc_scorer, 'F1': f1_scorer},
                                   refit='MCC', n_jobs=-1)

        # Out‑of‑sample MCC
        cv_scores = cross_val_score(grid_search, X, y, cv=outer_cv,
                                    scoring=mcc_scorer, n_jobs=-1)
        logger.info(f"Seed {seed} CV MCC: {np.mean(cv_scores):.4f}")

        # Fit best model
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_

        # Feature tracking
        removed = (best_model.named_steps['outlier_removal'].get_removed_features()
                   if 'outlier_removal' in best_model.named_steps else [])
        logger.info(f"Seed {seed} features removed by outlier: {removed}")
        selector = best_model.named_steps.get('selection')
        if selector and hasattr(selector, 'get_support'):
            mask = selector.get_support()
            selected_features = [f for f, m in zip(X.columns, mask) if m]
        else:
            selected_features = list(X.columns)
        logger.info(f"Seed {seed} selected features: {selected_features}")

        # Predictions & metrics
        y_cv_pred = cross_val_predict(grid_search, X, y, cv=outer_cv, n_jobs=-1)
        cv_cm = confusion_matrix(y, y_cv_pred)
        cv_auc = (roc_auc_score(y, cross_val_predict(grid_search, X, y,
                      cv=outer_cv, method='predict_proba', n_jobs=-1)[:,1])
                  if hasattr(best_model.named_steps['est'], 'predict_proba') else None)
        cv_f1 = f1_score(y, y_cv_pred, average='weighted')

        # Save model & reports
        joblib.dump(best_model, os.path.join(seed_folder, 'model.pkl'))
        gradient_img, ice_imgs = run_gradient_analysis(best_model, X, selected_features, seed_folder)
        perm_img = plot_permutation_importance(best_model, X, y, selected_features, seed_folder)
        perf_imgs = integrate_into_pipeline(X, y, selected_features, best_model.named_steps['est'], seed_folder)

        # Confusion matrices
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(cv_cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        fig.savefig(os.path.join(seed_folder, 'cv_confusion.png'))
        plt.close(fig)

        # PDF report
        images = [gradient_img, perm_img] + ice_imgs + perf_imgs + \
                 [os.path.join(seed_folder, 'cv_confusion.png')]
        create_pdf_report(selected_algorithm, selected_feature_selection,
                          f"Seed {seed} results", images,
                          os.path.join(seed_folder, 'report.pdf'),
                          cms={1: cv_cm})

        # Collect summary
        all_results.append({
            'seed': seed,
            'cv_mcc': np.mean(cv_scores),
            'cv_f1': cv_f1,
            'cv_auc': cv_auc,
            'n_features': len(selected_features),
        })

        seed_progress.update(1)

    seed_progress.close()

    # Combine across seeds
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(output_folder, 'all_seeds_results.csv'), index=False)
    logger.info(f"Combined results saved to {output_folder}/all_seeds_results.csv")

    # Finalize
    stop_cpu_monitor()
    logger.info("Machine learning pipeline completed successfully.")


def test_on_independent_dataset(model, independent_test, response, output_folder):
    # Remains unchanged from your original working code
    ensure_directory_exists(output_folder)
    independent_test.columns = map(str.lower, independent_test.columns)
    response = response.lower()
    if 'patient' in independent_test.columns:
        independent_test = independent_test.drop(columns=['patient'])
    
    if response not in independent_test.columns:
        raise ValueError(f"Response column '{response}' not found in the independent test dataset.")
    
    X_test = independent_test.drop(columns=[response])
    y_test = independent_test[response]
    
    y_test_pred = model.predict(X_test)
    if hasattr(model.named_steps['est'], "predict_proba"):
        y_test_proba = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_test_proba)
        precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
        test_aupr = auc(recall, precision)
    else:
        test_auc = None
        test_aupr = None
    
    test_mcc = matthews_corrcoef(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    test_cm = confusion_matrix(y_test, y_test_pred)
    
    selected_features = X_test.columns.tolist()
    if hasattr(model, 'named_steps') and 'selection' in model.named_steps:
        selector = model.named_steps['selection']
        if hasattr(selector, 'get_support'):
            mask = selector.get_support()
            selected_features = X_test.columns[mask].tolist()
    
    feature_ranges = show_feature_ranges(X_test[selected_features])
    feature_ranges_display = feature_ranges.head(10).to_string()
    
    results_text = (
        f"Independent Test Results\n"
        f"Test MCC: {test_mcc:.4f}\n"
        f"Test F1 Score: {test_f1:.4f}\n"
        f"Test AUC: {test_auc if test_auc is not None else 'N/A'}\n"
        f"Test AUPR: {test_aupr if test_aupr is not None else 'N/A'}\n"
        f"Selected Features: {selected_features}\n"
        f"Number of Selected Features: {len(selected_features)}\n"
        f"Feature Ranges (First 10 Features):\n{feature_ranges_display}"
    )
    
    plt.figure(figsize=(8,6))
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title('Independent Test Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    test_cm_img_path = os.path.join(output_folder, 'test_confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(test_cm_img_path)
    plt.close()
    
    output_pdf = os.path.join(output_folder, 'independent_test_report.pdf')
    create_pdf_report("Loaded Model", "N/A", results_text, [test_cm_img_path], output_pdf, cms={1: test_cm})
    print("Independent test evaluation completed and PDF created.")
    
    # Return a dictionary of test metrics so they can be aggregated later.
    return {
        "test_mcc": test_mcc,
        "test_f1": test_f1,
        "test_auc": test_auc,
        "test_aupr": test_aupr
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run ML pipeline with optional outlier detection and extra analysis.")
    parser.add_argument('--algorithm', required=True, help='Algorithm name, e.g., svm, cart, etc.')
    parser.add_argument('--feature_selection', required=True, help='Feature selection method, e.g., anova, lasso.')
    # Allow multiple dataset paths
    parser.add_argument('--data_path', required=True, nargs='+', help='Path(s) to CSV dataset(s).')
    parser.add_argument('--response', required=True, help='Name of response column in dataset.')
    parser.add_argument('--output_folder', required=True, help='Where to save results.')
    parser.add_argument('--use_outlier_detection', type=str, default='False', help='Whether to use outlier detection.')
    parser.add_argument('--outlier_method', type=str, default=None, help='Method for outlier detection.')
    parser.add_argument('--iqr_threshold', type=float, default=0.05)
    parser.add_argument('--zscore_threshold', type=float, default=0.05)
    parser.add_argument('--zscore_limit', type=float, default=3.0)
    parser.add_argument('--iso_forest_threshold', type=float, default=0.05)
    parser.add_argument('--pca_reconstruction_error_threshold', type=float, default=0.1)
    parser.add_argument('--n_components_pca', type=int, default=None)
    parser.add_argument('--run_extra_analysis', type=str, default='False',
                        help='If True, run extra analysis (gradient analysis, permutation importance, performance drop analysis).')
    args = parser.parse_args()
    
    use_outlier_bool = (args.use_outlier_detection.lower() == 'true')
    run_extra = (args.run_extra_analysis.lower() == 'true')

    # Load one or more datasets based on the provided paths
    if len(args.data_path) == 1:
        data = pd.read_csv(args.data_path[0])
    else:
        data = [pd.read_csv(path) for path in args.data_path]

    print(f">>> Loaded {len(args.data_path)} dataset(s).")
    print(">>> Starting machine_learning_pipeline...")
    machine_learning_pipeline(
        data=data,
        response=args.response,
        selected_algorithm=args.algorithm,
        selected_feature_selection=args.feature_selection,
        output_folder=args.output_folder,
        use_outlier_detection=use_outlier_bool,
        outlier_method=args.outlier_method,
        iqr_threshold=args.iqr_threshold,
        zscore_threshold=args.zscore_threshold,
        zscore_limit=args.zscore_limit,
        iso_forest_threshold=args.iso_forest_threshold,
        pca_reconstruction_error_threshold=args.pca_reconstruction_error_threshold,
        n_components_pca=args.n_components_pca,
        run_extra_analysis=run_extra
    )
    print(">>> Done with machine_learning_pipeline.")

if __name__ == "__main__":
    main()
