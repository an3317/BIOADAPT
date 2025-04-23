# pipeline/utils.py
"""
Utility helpers shared across the BIOADAPT pipeline.
"""

from __future__ import annotations
import os
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel
)
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


# ------------------------------------------------------------------ #
# 1)  File & directory helpers
# ------------------------------------------------------------------ #

def ensure_directory_exists(path: str | Path) -> None:
    """Create `path` and any parent directory if they don't already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ #
# 2)  Basic DataFrame helpers
# ------------------------------------------------------------------ #

def show_feature_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Return min/max for every column in a DataFrame."""
    return df.agg(['min', 'max'])


# ------------------------------------------------------------------ #
# 3)  Feature‑selection step factory
# ------------------------------------------------------------------ #

def get_feature_selection_step(method: str):
    if method == 'anova':
        return ('selection', SelectKBest(score_func=f_classif))
    if method == 'mutual_info':
        return ('selection', SelectKBest(score_func=mutual_info_classif))
    if method == 'lasso':
        return ('selection', SelectFromModel(Lasso(max_iter=10_000)))
    if method == 'rfe':
        return ('selection', RFE(estimator=LogisticRegression(max_iter=1_000)))
    raise ValueError(f"Invalid feature selection method '{method}'")


# ------------------------------------------------------------------ #
# 4)  Model & hyper‑parameter helpers
# ------------------------------------------------------------------ #

def get_model(algorithm: str):
    match algorithm:
        case 'logistic_regression':
            return LogisticRegression(max_iter=1_000)
        case 'svm':
            return SVC(probability=True)
        case 'xgboost':
            return XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        case 'random_forest':
            return RandomForestClassifier()
        case 'cart':
            return DecisionTreeClassifier()
    raise ValueError(f"Invalid algorithm '{algorithm}'")

def get_param_grid(fs_method: str, algorithm: str, k_range: List[int]):
    """Return a GridSearch param grid keyed to feature‑selection + algorithm."""
    if algorithm == 'logistic_regression':
        model_grid = {'est__C': [0.01, 0.1, 1, 10], 'est__penalty': ['l2']}
    elif algorithm == 'svm':
        model_grid = {'est__C': [0.1, 1, 10], 'est__kernel': ['linear', 'rbf']}
    elif algorithm == 'xgboost':
        model_grid = {
            'est__n_estimators': [100, 200],
            'est__max_depth': [3, 6],
            'est__learning_rate': [0.01, 0.1],
        }
    elif algorithm == 'random_forest':
        model_grid = {'est__n_estimators': [100, 200], 'est__max_depth': [None, 10, 20]}
    elif algorithm == 'cart':
        model_grid = {'est__max_depth': [None, 10, 20], 'est__min_samples_split': [2, 10]}
    else:
        raise ValueError(f"Invalid algorithm '{algorithm}' chosen.")

    if fs_method in {'anova', 'mutual_info'}:
        fs_grid = {'selection__k': k_range}
    elif fs_method == 'lasso':
        fs_grid = {'selection__estimator__alpha': [0.001, 0.01, 0.1, 1]}
    elif fs_method == 'rfe':
        fs_grid = {'selection__n_features_to_select': k_range}
    else:
        fs_grid = {}

    return {**model_grid, **fs_grid}
