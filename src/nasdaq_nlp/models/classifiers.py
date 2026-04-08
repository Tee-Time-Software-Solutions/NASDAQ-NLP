"""
models/classifiers.py — Naive Bayes and Logistic Regression for earnings call direction.

TASK FORMULATION
-----------------
We convert the regression problem (predict CAR magnitude) into a classification problem:

    Binary label: did the stock OUTPERFORM after the call?
        y = 1  if CAR[0,3] > 0  (positive abnormal return → call was bullish)
        y = 0  if CAR[0,3] ≤ 0  (negative abnormal return → call was bearish)

Features: TF-IDF bag-of-words representation of the transcript (500 dimensions).

We train two classifiers (Sessions 9-10):
    1. Naive Bayes (Multinomial)
    2. Logistic Regression

Both are evaluated on the 2019-2020 holdout set.

SESSION 9 — NAIVE BAYES
--------------------------
Naive Bayes is a probabilistic classifier based on Bayes' theorem:

    P(y | x) ∝ P(y) × Π P(x_i | y)

where:
    P(y)      = prior probability of class y (fraction of train docs with label y)
    P(x_i | y) = likelihood of feature x_i given class y

"Naive" = we assume features x_i are conditionally independent given y.
This is false (words co-occur), but the model works surprisingly well in practice
because it only needs to estimate per-feature likelihoods, not all joint distributions.

For text with TF-IDF features, we use Complement Naive Bayes (CNB):
    CNB is more robust when classes are imbalanced (skewed positive/negative split).
    It estimates P(x_i | NOT y) instead of P(x_i | y).

SESSION 10 — LOGISTIC REGRESSION
-----------------------------------
Logistic Regression models the probability that y=1 directly:

    P(y=1 | x) = 1 / (1 + exp(−(β0 + β1x1 + ... + βnxn)))
                                                  ↑
                                           "logit" (log-odds)

The sigmoid function (1/(1+exp(-z))) squashes any real number to (0,1),
which we interpret as a probability.

Coefficients β are estimated by maximum likelihood (not OLS):
we find β that maximises Σ log P(y_i | x_i; β) over training examples.

LR works well for text because:
    - Handles high-dimensional sparse features (TF-IDF matrix)
    - Produces calibrated probabilities (not just 0/1 labels)
    - Regularisation (C parameter) prevents overfitting on small datasets
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import MinMaxScaler

from nasdaq_nlp.config import (
    EVENT_STUDY_PATH,
    TFIDF_FEATURES_PATH,
    TRAIN_YEARS,
    TEST_YEARS,
)
from nasdaq_nlp.evaluation.metrics import classification_report_dict


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_classification_data(
    study_path: Path = EVENT_STUDY_PATH,
    tfidf_path: Path = TFIDF_FEATURES_PATH,
    target_col: str = "car_03",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Merge event study + TF-IDF features and split into train/test.

    The binary label:
        y = 1  if CAR[0,3] > 0   (stock outperformed market-model expectation)
        y = 0  if CAR[0,3] ≤ 0   (stock underperformed)

    Time-based split:
        Train: 2016–2018
        Test:  2019–2020

    Returns
    -------
    X_train, X_test : np.ndarray — TF-IDF feature matrices
    y_train, y_test : np.ndarray — binary labels
    events_test : pd.DataFrame — metadata for the test set
    """
    # Load event study (has CAR values and metadata)
    study = pd.read_csv(study_path, parse_dates=["event_trading_day"])
    study["year"] = study["event_trading_day"].dt.year

    # Load TF-IDF features
    tfidf = pd.read_csv(tfidf_path)
    tfidf["event_trading_day"] = pd.to_datetime(tfidf["event_trading_day"])

    # Merge on file_name (unique per event)
    merged = study.merge(tfidf, on=["ticker", "file_name", "event_trading_day"], how="inner")

    # Drop rows with missing target
    merged = merged.dropna(subset=[target_col])

    # Binary label: did the stock outperform?
    merged["direction"] = (merged[target_col] > 0).astype(int)

    # Feature matrix: all TF-IDF columns
    feature_cols = [c for c in merged.columns if c.startswith("tfidf_")]
    X = merged[feature_cols].values
    y = merged["direction"].values

    # Train/test split by year
    train_mask = merged["year"].isin(TRAIN_YEARS)
    test_mask = merged["year"].isin(TEST_YEARS)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    events_test = merged[test_mask][["ticker", "file_name", "event_trading_day", target_col, "direction"]]

    print(f"Train: {len(y_train)} events  |  Test: {len(y_test)} events")
    print(f"Train class balance: {y_train.mean():.2%} positive")
    print(f"Test  class balance: {y_test.mean():.2%} positive")

    return X_train, X_test, y_train, y_test, events_test


# ---------------------------------------------------------------------------
# Naive Bayes classifier
# ---------------------------------------------------------------------------

def train_naive_bayes(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> ComplementNB:
    """Train a Complement Naive Bayes classifier.

    ComplementNB requires non-negative features.
    TF-IDF values are already non-negative (≥0), so no transformation needed.

    The alpha parameter is Laplace smoothing:
        P(word_i | class) = (count_i + alpha) / (total_count + alpha × V)
    This prevents zero probabilities for words not seen in training.
    alpha=1.0 is standard (add-one smoothing).

    Parameters
    ----------
    X_train : np.ndarray — TF-IDF feature matrix (n_train, n_features)
    y_train : np.ndarray — binary labels

    Returns
    -------
    Trained ComplementNB model.
    """
    # TF-IDF values can be very small — scale to [0, 1] range for CNB stability
    # CNB's add-alpha smoothing is calibrated for count-like inputs
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = ComplementNB(alpha=1.0)
    model.fit(X_scaled, y_train)
    # Store the scaler as an attribute so we can use it on test data
    model._scaler = scaler  # type: ignore[attr-defined]

    return model


def evaluate_naive_bayes(
    model: ComplementNB,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Evaluate NB model on train and test sets."""
    X_train_s = model._scaler.transform(X_train)  # type: ignore[attr-defined]
    X_test_s = model._scaler.transform(X_test)    # type: ignore[attr-defined]

    train_pred = model.predict(X_train_s)
    test_pred = model.predict(X_test_s)
    test_prob = model.predict_proba(X_test_s)[:, 1]

    return {
        "model": "Naive Bayes (TF-IDF)",
        "target": "direction (CAR[0,3] > 0)",
        "train_accuracy": classification_report_dict(y_train, train_pred)["accuracy"],
        **{f"test_{k}": v for k, v in
           classification_report_dict(y_test, test_pred, test_prob).items()},
    }


# ---------------------------------------------------------------------------
# Logistic Regression classifier
# ---------------------------------------------------------------------------

def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
) -> LogisticRegression:
    """Train a Logistic Regression classifier with L2 regularisation.

    The C parameter controls regularisation strength:
        Small C → strong regularisation → simpler model (high bias, low variance)
        Large C → weak regularisation  → more complex model (low bias, high variance)
        C = 1.0 is a sensible default for normalised TF-IDF features.

    We use L2 regularisation (ridge) which shrinks all coefficients toward 0.
    This is appropriate for high-dimensional TF-IDF inputs.

    Parameters
    ----------
    X_train : np.ndarray
    y_train : np.ndarray
    C : float — inverse regularisation strength

    Returns
    -------
    Trained LogisticRegression model.
    """
    model = LogisticRegression(
        C=C,
        penalty="l2",           # L2 (ridge) regularisation
        solver="lbfgs",         # quasi-Newton optimiser — efficient for small datasets
        max_iter=1000,          # increase from default 100 to ensure convergence
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_logistic_regression(
    model: LogisticRegression,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Evaluate LR model on train and test sets."""
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    test_prob = model.predict_proba(X_test)[:, 1]

    return {
        "model": "Logistic Regression (TF-IDF)",
        "target": "direction (CAR[0,3] > 0)",
        "train_accuracy": classification_report_dict(y_train, train_pred)["accuracy"],
        **{f"test_{k}": v for k, v in
           classification_report_dict(y_test, test_pred, test_prob).items()},
    }


# ---------------------------------------------------------------------------
# Convenience: run both classifiers and return results
# ---------------------------------------------------------------------------

def run_classifiers(
    study_path: Path = EVENT_STUDY_PATH,
    tfidf_path: Path = TFIDF_FEATURES_PATH,
) -> tuple[dict, dict]:
    """Train and evaluate both classifiers. Returns (nb_results, lr_results)."""
    X_train, X_test, y_train, y_test, _ = prepare_classification_data(study_path, tfidf_path)

    print("\n── Naive Bayes ──")
    nb = train_naive_bayes(X_train, y_train)
    nb_results = evaluate_naive_bayes(nb, X_train, y_train, X_test, y_test)
    print(f"  Train accuracy: {nb_results['train_accuracy']:.3f}")
    print(f"  Test  accuracy: {nb_results['test_accuracy']:.3f}")
    print(f"  Test  F1:       {nb_results['test_f1']:.3f}")

    print("\n── Logistic Regression ──")
    lr = train_logistic_regression(X_train, y_train)
    lr_results = evaluate_logistic_regression(lr, X_train, y_train, X_test, y_test)
    print(f"  Train accuracy: {lr_results['train_accuracy']:.3f}")
    print(f"  Test  accuracy: {lr_results['test_accuracy']:.3f}")
    print(f"  Test  F1:       {lr_results['test_f1']:.3f}")

    return nb_results, lr_results
