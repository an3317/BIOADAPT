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

class Log2OrZscoreTransformer(BaseEstimator, TransformerMixin):
    """
    * If the combined numeric matrix contains ANY negative value,
      use z-score normalisation ( (x-μ)/σ ).
    * Otherwise use log₂(x+1) normalisation.
    * In both cases, NaN/inf/-inf are replaced with the column median,
      then (if still NaN) with 0.0 so downstream sklearn never sees NaNs.
    * Optional batch-mean subtraction if 'batch' column exists.
    """

    def __init__(self, batch_col: str = "batch"):
        self.batch_col = batch_col
        self._use_zscore = None  # set in fit

    # ---------- helpers -------------------------------------------------
    @staticmethod
    def _replace_bad(values: pd.DataFrame) -> pd.DataFrame:
        values = values.replace([np.inf, -np.inf], np.nan)
        for c in values.columns:
            median = values[c].median()
            values[c] = values[c].fillna(median if not np.isnan(median) else 0.0)
        return values

    def _combat(self, df: pd.DataFrame, batch: pd.Series) -> pd.DataFrame:
        """very simple mean-centering per batch label"""
        corrected = df.copy()
        for b in batch.unique():
            mask = batch == b
            corrected.loc[mask] = df.loc[mask] - df.loc[mask].mean()
        return corrected
    # --------------------------------------------------------------------

    # sklearn API
    def fit(self, X, y=None):
        df = pd.concat(X, ignore_index=True) if isinstance(X, list) else X.copy()
        num = df.select_dtypes(include=[np.number])
        self._use_zscore = (num < 0).any().any()
        return self  # stateless w.r.t parameters

    def transform(self, X, y=None):
        if isinstance(X, list):
            frames = [self.transform(df) for df in X]
            return pd.concat(frames, ignore_index=True)

        df = X.copy()
        num_cols = df.select_dtypes(include=[np.number]).columns

        if self._use_zscore:
            df[num_cols] = (df[num_cols] - df[num_cols].mean()) / df[num_cols].std(ddof=0)
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                df[num_cols] = np.log2(df[num_cols] + 1)

        df[num_cols] = self._replace_bad(df[num_cols])

        if self.batch_col in df.columns:
            df[num_cols] = self._combat(df[num_cols], df[self.batch_col])

        return df


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
