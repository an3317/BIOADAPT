# pipeline/transformers.py

import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

# -----------------------------
# Log2 Normalization & Batch Correction
# -----------------------------

def combat_batch_correction(data: pd.DataFrame, batch_info: pd.Series) -> pd.DataFrame:
    """
    A placeholder batch correction: for each batch, subtract the batch-specific mean.
    Replace with pyCombat or similar in production.
    """
    corrected = data.copy()
    for batch in batch_info.unique():
        mask = batch_info == batch
        batch_mean = data.loc[mask].mean()
        corrected.loc[mask] = data.loc[mask] - batch_mean
    return corrected

class Log2NormalizationAndBatchCorrectionTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that applies log2(x+1) normalization safely, then batch-correction if a batch column exists.

    - If X is a single DataFrame:
        * Without a 'batch' column: only log2 normalization.
        * With 'batch': log2 then combat_batch_correction.
    - If X is a list of DataFrames: each is normalized, concatenated, then batch-corrected if 'batch' exists.
    """
    def __init__(self, batch_col: str = 'batch'):
        self.batch_col = batch_col

    def safe_log2_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Apply log2(x+1) to numeric cols, replace inf/NaN with column median
        df_num = df.select_dtypes(include=[np.number])
        transformed = np.log2(df_num + 1).replace([np.inf, -np.inf], np.nan)
        for col in transformed.columns:
            median = transformed[col].median()
            transformed[col] = transformed[col].fillna(median)
        df = df.copy()
        df[transformed.columns] = transformed
        return df

    def fit(self, X, y=None):
        return self  # stateless

    def transform(self, X, y=None):
        # Handle list of DataFrames
        if isinstance(X, list):
            dfs = [self.safe_log2_transform(pd.DataFrame(df)) for df in X]
            merged = pd.concat(dfs, ignore_index=True)
            if self.batch_col in merged.columns:
                batch_info = merged[self.batch_col]
                num_cols = merged.select_dtypes(include=[np.number]).columns
                corrected = combat_batch_correction(merged[num_cols], batch_info)
                merged[num_cols] = corrected
            return merged

        # Single DataFrame case
        if isinstance(X, pd.DataFrame):
            df = self.safe_log2_transform(X.copy())
            if self.batch_col in df.columns:
                batch_info = df[self.batch_col]
                num_cols = df.select_dtypes(include=[np.number]).columns
                corrected = combat_batch_correction(df[num_cols], batch_info)
                df[num_cols] = corrected
            return df

        raise ValueError("Input must be a DataFrame or list of DataFrames")

# -----------------------------
# Feature Outlier Remover
# -----------------------------

class FeatureOutlierRemover(BaseEstimator, TransformerMixin):
    """
    Remove noisy/outlier-prone features by one of several methods:
      - 'iqr': drop features with too many IQR outliers
      - 'zscore': drop features with too many extreme z-scores
      - 'isolation_forest': drop features flagged by IsolationForest
      - 'pca': drop features with high PCA reconstruction error
    """
    def __init__(
        self,
        method: str = 'iqr',
        iqr_threshold: float = 0.05,
        zscore_threshold: float = 0.05,
        zscore_limit: float = 3.0,
        iso_forest_threshold: float = 0.05,
        pca_reconstruction_error_threshold: float = 0.1,
        n_components_pca: int | None = None
    ):
        self.method = method
        self.iqr_threshold = iqr_threshold
        self.zscore_threshold = zscore_threshold
        self.zscore_limit = zscore_limit
        self.iso_forest_threshold = iso_forest_threshold
        self.pca_reconstruction_error_threshold = pca_reconstruction_error_threshold
        self.n_components_pca = n_components_pca
        self.features_to_remove_ = []

    def fit(self, X, y=None):
        df = pd.DataFrame(X) if not hasattr(X, 'columns') else X.copy()
        self.features_to_remove_ = []

        if self.method == 'iqr':
            Q1, Q3 = df.quantile(0.25), df.quantile(0.75)
            IQR = Q3 - Q1
            for col in df.columns:
                lower, upper = Q1[col] - 1.5 * IQR[col], Q3[col] + 1.5 * IQR[col]
                prop = ((df[col] < lower) | (df[col] > upper)).mean()
                if prop > self.iqr_threshold:
                    self.features_to_remove_.append(col)

        elif self.method == 'zscore':
            for col in df.columns:
                prop = (df[col].abs() > self.zscore_limit).mean()
                if prop > self.zscore_threshold:
                    self.features_to_remove_.append(col)

        elif self.method == 'isolation_forest':
            for col in df.columns:
                iso = IsolationForest(random_state=42)
                preds = iso.fit_predict(df[[col]])
                prop = (preds == -1).mean()
                if prop > self.iso_forest_threshold:
                    self.features_to_remove_.append(col)

        elif self.method == 'pca':
            n_comp = self.n_components_pca or min(10, df.shape[1])
            pca = PCA(n_components=n_comp)
            recon = pca.inverse_transform(pca.fit_transform(df))
            errors = np.mean(np.abs(df.values - recon), axis=0)
            for i, col in enumerate(df.columns):
                if errors[i] > self.pca_reconstruction_error_threshold:
                    self.features_to_remove_.append(col)

        else:
            raise ValueError(f"Unknown method '{self.method}'")

        return self

    def transform(self, X):
        df = pd.DataFrame(X) if not hasattr(X, 'columns') else X.copy()
        return df.drop(columns=self.features_to_remove_, errors='ignore')

    def get_removed_features(self) -> list[str]:
        return self.features_to_remove_
