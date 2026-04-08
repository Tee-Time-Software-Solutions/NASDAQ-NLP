"""
evaluation/metrics.py — Evaluation metrics for all models.

WHAT WE MEASURE
----------------
We have two types of models:

1. REGRESSION models (OLS sentiment → CAR)
   → Metric: R² (coefficient of determination)
   → Also: out-of-sample R² to detect overfitting

2. CLASSIFICATION models (NB, LR → predict direction CAR > 0)
   → Metrics: accuracy, F1-score, precision, recall
   → Also: out-of-sample accuracy on the 2019-2020 holdout

MATH — R²
-----------
R² measures how much of the variance in y the model explains.
It ranges from -∞ to 1.0 (1.0 = perfect fit, 0 = same as predicting the mean).

    R² = 1 - SS_residual / SS_total

where:
    SS_residual = Σ (y_i - ŷ_i)²     (sum of squared errors of the model)
    SS_total    = Σ (y_i - ȳ)²       (sum of squared deviations from the mean)

    If the model is no better than guessing the mean: R² ≈ 0
    If the model perfectly predicts: R² = 1
    If the model is WORSE than the mean (e.g. out-of-sample): R² < 0

Out-of-sample R² (OOS R²):
    Same formula, but ŷ_i are predictions made on held-out test data
    using a model fit only on training data.
    OOS R² < in-sample R² is normal (generalisation gap).
    OOS R² < 0 means the model generalises worse than the mean.

WALD TEST FOR ASYMMETRY
-------------------------
We want to test: |β_neg| > |β_pos|?
A Wald test checks if a linear restriction on regression coefficients is
significantly different from zero.

    H0: β_neg = -β_pos   (symmetric effect: negative and positive equally strong)
    H1: β_neg ≠ -β_pos   (asymmetric effect: negative dominates)

The test statistic is:
    W = (Rβ - r)' [R V R']^{-1} (Rβ - r)
where R is the restriction matrix, β is the coefficient vector, V is the covariance matrix.

Under H0, W ~ χ²(q) where q = number of restrictions.
We compute the p-value from the chi-squared distribution.
A p-value < 0.05 rejects H0 (supports asymmetry).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination (R²).

    R² = 1 - SS_res / SS_tot
    where SS_res = Σ(y - ŷ)² and SS_tot = Σ(y - ȳ)².

    Notes
    -----
    - Can be negative (model worse than the mean predictor).
    - Perfect prediction → R² = 1.
    - Predicting the mean → R² = 0.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1 - ss_res / ss_tot)


def oos_r_squared(
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_pred_test: np.ndarray,
) -> float:
    """Out-of-sample R² (Campbell & Thompson 2008 convention).

    The benchmark for OOS R² is always the *training mean* (not the test mean).
    This is the standard in financial predictability studies.

    OOS R² = 1 - MSE_model / MSE_mean
    where MSE_mean uses the training mean as the forecast.

    Parameters
    ----------
    y_train : np.ndarray — training targets (to compute the mean benchmark)
    y_test  : np.ndarray — test targets (actual values)
    y_pred_test : np.ndarray — model predictions on test set

    Returns
    -------
    float — OOS R². Can be negative.
    """
    train_mean = np.mean(y_train)                             # the naive forecast
    mse_model = np.mean((y_test - y_pred_test) ** 2)         # model MSE
    mse_mean = np.mean((y_test - train_mean) ** 2)           # mean-model MSE

    if mse_mean == 0:
        return 1.0 if mse_model == 0 else float("-inf")
    return float(1 - mse_model / mse_mean)


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def classification_report_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict:
    """Compute a dictionary of classification metrics.

    Parameters
    ----------
    y_true : np.ndarray — true binary labels (0 or 1)
    y_pred : np.ndarray — predicted binary labels
    y_prob : np.ndarray | None — predicted probabilities for class 1 (for AUC)

    Returns
    -------
    dict with: accuracy, f1, precision, recall, auc (if y_prob given)
    """
    report = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        try:
            report["auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            report["auc"] = float("nan")
    return report


# ---------------------------------------------------------------------------
# Benchmark table builder
# ---------------------------------------------------------------------------

def build_benchmark_table(results: list[dict]) -> pd.DataFrame:
    """Assemble a comparison table from a list of model result dicts.

    Each dict should have at minimum: 'model', plus relevant metrics.
    Missing metrics are filled with NaN.

    Example input:
    [
        {'model': 'Null',          'target': 'CAR[0,3]', 'train_r2': None, 'test_r2': None},
        {'model': 'LM OLS',        'target': 'CAR[0,3]', 'train_r2': 0.12, 'test_r2': 0.04},
        {'model': 'TF-IDF + NB',   'target': 'Direction', 'train_acc': 0.68, 'test_acc': 0.57},
    ]

    Returns
    -------
    pd.DataFrame sorted by model type.
    """
    df = pd.DataFrame(results)
    # Round numeric columns to 4 decimal places for display
    numeric_cols = df.select_dtypes(include=[float]).columns
    df[numeric_cols] = df[numeric_cols].round(4)
    return df
