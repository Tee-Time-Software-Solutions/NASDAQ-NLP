"""
models/classifiers.py — Modular classification framework for earnings call direction.

TASK FORMULATION
-----------------
Convert the regression problem into binary classification:

    y = 1  if CAR[0,3] > 0  → stock OUTperformed market-model expectation after the call
    y = 0  if CAR[0,3] ≤ 0  → stock UNDERperformed

IMPORTANT: y=1 does NOT mean the transcript was positive-sounding.
It means the MARKET REACTION was above what the market model predicted.
The model tries to predict that reaction from the transcript text.

A good classifier finds transcripts whose language predicts market outperformance
even after accounting for market-wide moves.

HOW TO ADD A NEW CLASSIFIER
-----------------------------
    from sklearn.svm import SVC
    CLASSIFIERS["SVM"] = lambda: SVC(kernel="rbf", probability=True)

Then call run_classifier_experiment(..., classifier_name="SVM").

METRICS AVAILABLE
-----------------
Pass a subset of these to results_to_df(results, metrics=[...]):
    "accuracy", "f1", "precision", "recall", "auc", "train_accuracy"
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

from nasdaq_nlp.config import TEST_YEARS, TRAIN_YEARS
from nasdaq_nlp.evaluation.metrics import classification_report_dict
from nasdaq_nlp.models.regression import resolve_features

# ---------------------------------------------------------------------------
# Classifier registry
# ---------------------------------------------------------------------------
# Each entry is a zero-argument factory so every call gets a fresh unfitted model.
# Add any sklearn-compatible classifier here.

CLASSIFIERS: dict[str, object] = {
    "NaiveBayes": lambda: ComplementNB(alpha=1.0),
    "LogReg": lambda: LogisticRegression(
        C=1.0, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42
    ),
    "Tree": lambda: DecisionTreeClassifier(max_depth=5, random_state=42),
    "RF": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    "MLP": lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42),
}

# Metrics returned for every classifier experiment.
# All are always computed internally — display is filtered in results_to_df().
_ALL_METRICS = ["accuracy", "f1", "precision", "recall", "auc"]


# ---------------------------------------------------------------------------
# Core classifier experiment runner
# ---------------------------------------------------------------------------


def run_classifier_experiment(
    df: pd.DataFrame,
    feature_set_names: list[str],
    classifier_name: str,
    target_col: str = "car_03",
    experiment_name: str | None = None,
) -> dict:
    """Run one classification experiment.

    The binary label:
        y = 1 if CAR > 0  (market reacted positively to this call, above model prediction)
        y = 0 if CAR ≤ 0

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_modelling_dataset().
    feature_set_names : list[str]
        Feature sets to combine.
        Valid names: "lexicon", "finbert", "tfidf", "embeddings", "controls"
    classifier_name : str
        Which classifier. Valid names: "NaiveBayes", "LogReg", "Tree", "RF", "MLP"
    target_col : str
        CAR column to binarize. "car_03" or "car_01".
    experiment_name : str | None
        Label in results table. Auto-generated if None.

    Returns
    -------
    dict with all metrics:
        model, target, n_train, n_test, train_accuracy,
        test_accuracy, test_f1, test_precision, test_recall, test_auc
    """
    if classifier_name not in CLASSIFIERS:
        raise ValueError(f"Unknown classifier '{classifier_name}'. Available: {list(CLASSIFIERS)}")

    name = experiment_name or f"{' + '.join(feature_set_names)} [{classifier_name}]"

    # Resolve feature columns (also handles year dummies if "controls" requested)
    df, feature_cols = resolve_features(df, feature_set_names)
    if not feature_cols:
        raise ValueError(f"No valid columns found for feature sets: {feature_set_names}")

    # Build dataset
    sub = df[["year"] + feature_cols + [target_col]].dropna().copy()
    sub["label"] = (sub[target_col] > 0).astype(int)  # y=1: stock beat market-model prediction

    train_mask = sub["year"].isin(TRAIN_YEARS)
    test_mask = sub["year"].isin(TEST_YEARS)

    X_train = sub.loc[train_mask, feature_cols].values
    X_test = sub.loc[test_mask, feature_cols].values
    y_train = sub.loc[train_mask, "label"].values
    y_test = sub.loc[test_mask, "label"].values

    print(
        f"\n[{name}]  "
        f"train={len(y_train)} ({y_train.mean():.1%} pos) | "
        f"test={len(y_test)} ({y_test.mean():.1%} pos) | "
        f"features={len(feature_cols)}"
    )

    # ComplementNB requires non-negative inputs → scale everything to [0, 1].
    # Word2Vec embeddings can be negative, so we apply this for NaiveBayes regardless
    # of which feature set is used.
    if classifier_name == "NaiveBayes":
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model = CLASSIFIERS[classifier_name]()
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    train_metrics = classification_report_dict(y_train, y_pred_train)
    test_metrics = classification_report_dict(y_test, y_pred_test, y_prob_test)

    return {
        "model": name,
        "target": f"{target_col} > 0",
        "classifier": classifier_name,
        "features": " + ".join(feature_set_names),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "train_accuracy": round(train_metrics["accuracy"], 4),
        **{f"test_{k}": round(v, 4) for k, v in test_metrics.items()},
    }


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

# Default columns to show — pass a custom list to override
DEFAULT_CLF_METRICS = [
    "train_accuracy",
    "test_accuracy",
    "test_f1",
    "test_precision",
    "test_recall",
    "test_auc",
]


def results_to_df(
    results: list[dict],
    metrics: list[str] = DEFAULT_CLF_METRICS,
) -> pd.DataFrame:
    """Convert a list of classifier result dicts into a display DataFrame.

    Parameters
    ----------
    results : list[dict]
        Outputs of run_classifier_experiment().
    metrics : list[str]
        Which metric columns to include. Defaults to all main metrics.
        Full set: "train_accuracy", "test_accuracy", "test_f1",
                  "test_precision", "test_recall", "test_auc"
    """
    df = pd.DataFrame(results)
    id_cols = ["model", "target", "n_train", "n_test"]
    metric_cols = [m for m in metrics if m in df.columns]
    return df[id_cols + metric_cols]
