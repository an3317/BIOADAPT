# --------------------------------------------------------------------------- #
# bioadapt_pkg/core.py  – full, corrected machine_learning_pipeline function
# --------------------------------------------------------------------------- #

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path
import re

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import (confusion_matrix, f1_score, make_scorer,
                             matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_predict, cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from bioadapt_pkg.config_loader import Config
from bioadapt_pkg.explain import shap_explain
from bioadapt_pkg.logging_conf import setup_logger
from bioadapt_pkg.transformers import (FeatureOutlierRemover,
                                       Log2OrZscoreTransformer)
from bioadapt_pkg.utils import (ensure_directory_exists,
                                get_feature_selection_step, get_model,
                                get_param_grid, show_feature_ranges)
from bioadapt_pkg.core_helpers import (
    plot_gradient_importance,
    plot_ice_plots,
    calculate_feature_gradients,
    rank_features_by_gradient_variance,
    run_gradient_analysis,
    plot_permutation_importance,
    PDFReport,
    add_confusion_matrix_table,
    create_pdf_report,
    evaluate_performance_drop,
)

# --------------------------------------------------------------------------- #


# Set environment variables for BLAS libraries to use 16 threads
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"

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

def create_algorithm_feature_selection_folder(output_folder, algorithm, feature_selection):
    folder_name = f"{algorithm}_{feature_selection}"
    folder_name = re.sub(r'\W+', '_', folder_name)
    full_folder_path = os.path.join(output_folder, folder_name)
    ensure_directory_exists(full_folder_path)
    return full_folder_path

def plot_performance_drop(performance_drop, title, xlabel, ylabel, output_file):
    features = [item[0] for item in performance_drop]
    drops = [item[1] for item in performance_drop]
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=drops,                
        y=features,             
        hue=features,           
        palette='viridis',     
        legend=False            
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Performance drop plot saved at: {output_file}")

def machine_learning_pipeline(cfg: Config) -> None:
    """Run the end-to-end ML pipeline based on a Config object."""

    # ───────────────────────── 1) unpack runtime settings ──────────────────
    data_paths           = [str(p) for p in cfg.data.paths]
    response_col         = cfg.data.response.lower()
    algorithm            = cfg.pipeline.algorithm
    feat_sel_method      = cfg.pipeline.feature_selection
    output_root          = Path(cfg.pipeline.output_folder)
    random_seeds         = cfg.cv.random_seeds

    # ───────────────────────── 2) logging & IO ─────────────────────────────
    algo_fs  = output_root / f"{cfg.pipeline.algorithm}_{cfg.pipeline.feature_selection}"
    algo_fs.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_root / "run.log")
    logger.info("Pipeline started")
    logger.debug("Config: %s", cfg)

    # ───────────────────────── 3) load & validate data ─────────────────────
    from bioadapt_pkg.data_loading import load_csvs  # local import avoids cycle
    df_raw = load_csvs(data_paths)
    if isinstance(df_raw, list):
        df_raw = pd.concat(df_raw, ignore_index=True)
    logger.info("Data loaded & schema-validated. Shape %s", df_raw.shape)

    # ───────────────────────── 4) basic system info ────────────────────────
    logger.info("CPU cores detected: %d", multiprocessing.cpu_count())

    # ───────────────────────── 5) preprocessing ────────────────────────────
    df_raw.columns = df_raw.columns.str.lower()
    if "patient_id" in df_raw.columns:
        df_raw = df_raw.drop(columns=["patient_id"])

    if "patient" in df_raw.columns:
        df_raw = df_raw.drop(columns=["patient"])    

    X = df_raw.drop(columns=[response_col])
    y = df_raw[response_col]

    const_cols = [c for c in X.columns if X[c].nunique() <= 1]
    if const_cols:
        logger.info("Dropping constant columns: %s", const_cols)
        X = X.drop(columns=const_cols)

    # ───────────────────────── 6) class imbalance ──────────────────────────
    sampler: RandomOverSampler | RandomUnderSampler | None = None
    sampling_method = "none"
    vc = y.value_counts()
    if min(vc) / max(vc) < 0.5:
        if min(vc) * 2 < max(vc):
            sampler = RandomOverSampler(random_state=42)
            sampling_method = "oversampling"
        else:
            sampler = RandomUnderSampler(random_state=42)
            sampling_method = "undersampling"
        logger.info("Resampling strategy: %s", sampling_method)

    # ───────────────────────── 7) outlier removal (optional) ───────────────
    outlier_remover = None
    if cfg.pipeline.use_outlier_detection and cfg.pipeline.outlier_method:
        outlier_remover = FeatureOutlierRemover(
            method=cfg.pipeline.outlier_method,
            iqr_threshold=cfg.pipeline.iqr_threshold,
            zscore_threshold=cfg.pipeline.zscore_threshold,
            zscore_limit=cfg.pipeline.zscore_limit,
            iso_forest_threshold=cfg.pipeline.iso_forest_threshold,
            pca_reconstruction_error_threshold=cfg.pipeline.pca_reconstruction_error_threshold,
            n_components_pca=cfg.pipeline.n_components_pca,
        )
        X_tmp = outlier_remover.fit_transform(X)
        n_features = X_tmp.shape[1]
    else:
        n_features = X.shape[1]

    # ───────────────────────── 8) feature-selection k-range ────────────────
    k_range = [k for k in range(10, 65, 5) if k <= n_features] or [min(10, n_features)]
    logger.info("k-range for feature selection: %s", k_range)

    # ───────────────────────── 9) build pipeline steps ─────────────────────
    steps: list[tuple[str, object]] = [
        ("norm", Log2OrZscoreTransformer(batch_col="batch")),
        ("scaler", StandardScaler()),
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if sampler:
        steps.append(("sampler", sampler))
    if outlier_remover:
        steps.append(("outlier_removal", outlier_remover))

    steps.append(get_feature_selection_step(feat_sel_method))
    steps.append(("est", get_model(algorithm)))
    est_pipeline = ImbPipeline(steps)

    logger.info("Pipeline steps: %s", [name for name, _ in steps])

    # ───────────────────────── 10) grid search objects ─────────────────────
    param_grid = get_param_grid(feat_sel_method, algorithm, k_range)

    logger.debug("Param grid keys: %s", list(param_grid.keys()))
    logger.debug("Param grid sizes: %s", {k: len(v) for k, v in param_grid.items()})

    mcc_scorer = make_scorer(matthews_corrcoef)
    f1_scorer  = make_scorer(f1_score, average="weighted")

    # ───────────────────────── 11) outer CV loop (per-seed) ────────────────
    all_results: list[dict[str, object]] = []
    seed_progress = tqdm(total=len(random_seeds), desc="Pipeline progress", unit="seed")

    for seed in random_seeds:
        seed_run_folder = algo_fs / f"seed_{seed}"
        seed_run_folder.mkdir(parents=True, exist_ok=True)
        logger.info("► Seed %d — output: %s", seed, seed_run_folder)

        outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

        gs = GridSearchCV(
            est_pipeline,
            param_grid,
            cv=inner_cv,
            scoring={"MCC": mcc_scorer, "F1": f1_scorer},
            refit="MCC",
            n_jobs=-1,
        )

        cv_mcc_scores = cross_val_score(gs, X, y, cv=outer_cv, scoring=mcc_scorer, n_jobs=-1)
        logger.info("Seed %d CV-MCC: %.4f ± %.4f", seed, cv_mcc_scores.mean(), cv_mcc_scores.std())

        gs.fit(X, y)
        best_model = gs.best_estimator_

        # ---- Feature bookkeeping ----------------------------------------
        removed_cols = best_model.named_steps.get("outlier_removal", FeatureOutlierRemover()).get_removed_features()
        selector     = best_model.named_steps.get("selection")
        if selector and hasattr(selector, "get_support"):
            selected_features = list(np.array(X.columns)[selector.get_support()])
            logger.debug("Seed %d picked %d features: %s", seed, len(selected_features), selected_features[:5])

        else:
            selected_features = list(X.columns)

        # ---- Predictions & confusion matrix -----------------------------
        y_cv_pred = cross_val_predict(gs, X, y, cv=outer_cv, n_jobs=-1)
        cv_cm     = confusion_matrix(y, y_cv_pred)
        cv_f1     = f1_score(y, y_cv_pred, average="weighted")
        cv_auc    = None
        if hasattr(best_model.named_steps["est"], "predict_proba"):
            y_cv_prob = cross_val_predict(gs, X, y, cv=outer_cv, method="predict_proba", n_jobs=-1)[:, 1]
            cv_auc = roc_auc_score(y, y_cv_prob)

        # -----------------------------------------------------------------
        # Save model
        joblib.dump(best_model, seed_run_folder / "model.pkl")

        # -----------------------------------------------------------------
        # Generate plots
        images: list[str] = []
        extra_files: list[str] = []

        if cfg.pipeline.explain.lower() == "shap":
            shap_png, shap_html = shap_explain(
                best_model,
                X,
                selected_features,          
                seed_run_folder / "shap",
                max_samples=100
            )
            images.append(str(shap_png))
            extra_files.append(str(shap_html))


        if cfg.pipeline.run_extra_analysis:
            grad_png, ice_pngs = run_gradient_analysis(best_model, X, selected_features, seed_run_folder)
            perm_png = plot_permutation_importance(best_model, X, y, selected_features, seed_run_folder)
            perf_pngs = integrate_into_pipeline(X, y, selected_features,
                                                best_model.named_steps["est"], seed_run_folder)

            images.extend([str(grad_png), str(perm_png), *map(str, ice_pngs), *map(str, perf_pngs)])

        cm_png = seed_run_folder / "cv_confusion.png"
        plt.figure(figsize=(8, 6))
        sns.heatmap(cv_cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
        plt.title("CV Confusion Matrix")
        plt.tight_layout()
        plt.savefig(cm_png); plt.close()
        images.append(str(cm_png))


        results_text = (
            f"Top Model Results\n"
            f"Model: {algorithm}\n"
            f"Feature Selection: {feat_sel_method}\n"
            f"Use Outlier Detection: {cfg.pipeline.use_outlier_detection}\n"
            f"Outlier Method: {cfg.pipeline.outlier_method or 'N/A'}\n"
            f"Removed Outlier Features: {removed_cols}\n"
            f"Seed: {seed}\n"
            f"Resampling Method: {sampling_method}\n"
            f"Selected Features: {selected_features}\n"
            f"Number of Selected Features: {len(selected_features)}\n"
            f"Hyperparameters: {gs.best_params_}\n"
            f"CV MCC: {cv_mcc_scores.mean():.4f}\n"
            f"CV F1 Score: {cv_f1:.4f}\n"
            f"CV AUC: {cv_auc if cv_auc is not None else 'N/A'}\n"
        )



        # -----------------------------------------------------------------
        # PDF report
        create_pdf_report(
            algorithm,
            feat_sel_method,
            results_text,
            images,
            seed_run_folder / "report.pdf",
            cms={1: cv_cm},        # ← now only keyword
            extra_files=extra_files,
        )

        # -----------------------------------------------------------------
        # summary row
        all_results.append({
            "seed": seed,
            "cv_mcc": cv_mcc_scores.mean(),
            "cv_f1":  cv_f1,
            "cv_auc": cv_auc,
            "n_features": len(selected_features),
        })
        seed_progress.update(1)

    seed_progress.close()

    # ───────────────────────── 12) combine + save summary ────────────────
    pd.DataFrame(all_results).to_csv(output_root / "all_seeds_results.csv", index=False)
    logger.info("Combined summary saved to %s", output_root / "all_seeds_results.csv")
    logger.info("Pipeline completed successfully.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run ML pipeline with optional outlier detection and extra analysis.")
    parser.add_argument('--algorithm', required=True, help='Algorithm name, e.g., svm, cart, etc.')
    parser.add_argument('--feature_selection', required=True, help='Feature selection method, e.g., anova, lasso.')
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
