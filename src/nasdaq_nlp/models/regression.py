"""
models/regression.py — Modular OLS and sklearn regression framework.

RESEARCH QUESTION
------------------
Does sentiment extracted from earnings call transcripts predict
post-call market reactions (CAR[0,3])?

  CAR_{i,[0,3]} = β0 + β1·NegRate + β2·PosRate + controls + ε

ASYMMETRY HYPOTHESIS
---------------------
|β_neg| > |β_pos|: negative language predicts returns more strongly than positive.
Tested via a Wald test: H0: β_neg + β_pos = 0 (symmetric effect).

HOW TO ADD A NEW REGRESSOR
----------------------------
    from sklearn.svm import SVR
    REGRESSORS["SVR"] = lambda: SVR(kernel="rbf")

Then call run_experiment(..., regressor_name="SVR").

HOW TO ADD A NEW FEATURE SET
------------------------------
    FEATURE_SETS["my_features"] = ["col_a", "col_b"]

Or for dynamically-named columns, add a branch in resolve_features().
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

from nasdaq_nlp.config import (
    ASYMMETRY_RESULTS_PATH,
    BENCHMARK_TABLE_PATH,
    EMBEDDINGS_PATH,
    EVENT_STUDY_PATH,
    FINBERT_FEATURES_PATH,
    LEXICON_FEATURES_PATH,
    TFIDF_FEATURES_PATH,
    TRAIN_YEARS,
    TEST_YEARS,
    ensure_output_dirs,
)
from nasdaq_nlp.evaluation.metrics import r_squared, oos_r_squared


# ---------------------------------------------------------------------------
# Feature set registry
# ---------------------------------------------------------------------------
# Maps a name to a list of STATIC column names that must exist in the df.
# Two special names are resolved dynamically in resolve_features():
#   "tfidf"      → all columns starting with "tfidf_"
#   "embeddings" → all columns starting with "emb_"
# "controls" also triggers construction of year-dummy columns at runtime.

FEATURE_SETS: dict[str, list[str]] = {
    "lexicon":      ["neg_rate", "pos_rate"],
    "lexicon_pres": ["neg_rate_pres", "pos_rate_pres"],  # management prepared remarks only
    "lexicon_qa":   ["neg_rate_qa",   "pos_rate_qa"],    # analyst Q&A only (more candid)
    "finbert":      ["finbert_neg_mean", "finbert_pos_mean"],
    "controls":     ["pre_vol"],   # year dummies added dynamically
    "tfidf":        [],            # resolved at runtime from df column names
    "embeddings":   [],            # resolved at runtime from df column names
}

# Pairs eligible for the Wald asymmetry test.
# If both columns are present in the active feature set, the test runs automatically.
_ASYMMETRY_PAIRS = [
    ("neg_rate",         "pos_rate"),
    ("neg_rate_pres",    "pos_rate_pres"),
    ("neg_rate_qa",      "pos_rate_qa"),
    ("finbert_neg_mean", "finbert_pos_mean"),
]


# ---------------------------------------------------------------------------
# Regressor registry
# ---------------------------------------------------------------------------
# "OLS"  → statsmodels (gives p-values, Wald test, HC3 heteroskedasticity-robust errors)
# Others → sklearn-compatible (fit/predict interface, no p-values)

REGRESSORS: dict[str, object] = {
    "OLS":   "statsmodels",
    "Ridge": lambda: Ridge(alpha=1.0),
    "RF":    lambda: RandomForestRegressor(n_estimators=100, random_state=42),
    "MLP":   lambda: MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42),
}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class RegressionResult(NamedTuple):
    model_name:    str
    target:        str
    features_used: list[str]
    n_train:       int
    n_test:        int
    coef:          pd.Series      # named coefficients (empty for non-linear models)
    pvalues:       pd.Series      # p-values (empty for sklearn models)
    train_r2:      float
    test_r2:       float
    oos_r2:        float
    mae:           float          # mean absolute error on test set
    rmse:          float          # root mean squared error on test set
    wald_pvalue:   float | None   # asymmetry test p-value (OLS only)
    sm_result:     object         # raw statsmodels result or None


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def load_modelling_dataset(
    study_path:      Path = EVENT_STUDY_PATH,
    lexicon_path:    Path = LEXICON_FEATURES_PATH,
    tfidf_path:      Path = TFIDF_FEATURES_PATH,
    finbert_path:    Path = FINBERT_FEATURES_PATH,
    embeddings_path: Path = EMBEDDINGS_PATH,
) -> pd.DataFrame:
    """Merge event study + all available feature CSVs into one modelling DataFrame.

    Feature files that don't exist are skipped with a warning so you can
    run experiments on whatever features are currently available.

    Returns one row per event with all feature columns attached.
    """
    study = pd.read_csv(study_path, parse_dates=["event_trading_day"])
    study["year"] = study["event_trading_day"].dt.year
    print(f"Event study: {len(study)} rows")

    def _load(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        df["event_trading_day"] = pd.to_datetime(df["event_trading_day"])
        return df

    _merge_keys = ["ticker", "file_name", "event_trading_day"]

    # Lexicon (required — core of the asymmetry hypothesis)
    if not lexicon_path.exists():
        raise FileNotFoundError(f"Run build_lexicon_features() first: {lexicon_path}")
    study = study.merge(_load(lexicon_path), on=_merge_keys, how="left")
    print(f"Lexicon features merged ✓")

    # TF-IDF (optional)
    if tfidf_path.exists():
        study = study.merge(_load(tfidf_path), on=_merge_keys, how="left")
        n_tfidf = sum(1 for c in study.columns if c.startswith("tfidf_"))
        print(f"TF-IDF features merged ✓  ({n_tfidf} columns)")
    else:
        print(f"WARN: TF-IDF not found at {tfidf_path} — skipping")

    # FinBERT (optional)
    if finbert_path.exists():
        study = study.merge(_load(finbert_path), on=_merge_keys, how="left")
        print(f"FinBERT features merged ✓")
    else:
        print(f"WARN: FinBERT not found at {finbert_path} — skipping")
        for col in ["finbert_pos_mean", "finbert_neg_mean", "finbert_neu_mean"]:
            study[col] = np.nan

    # Word2Vec embeddings (optional)
    if embeddings_path.exists():
        emb = _load(embeddings_path)
        emb_cols = [c for c in emb.columns if c.startswith("emb_")]
        study = study.merge(emb[_merge_keys + emb_cols], on=_merge_keys, how="left")
        print(f"Embeddings merged ✓  ({len(emb_cols)} dimensions)")
    else:
        print(f"WARN: Embeddings not found at {embeddings_path} — skipping")

    return study


# ---------------------------------------------------------------------------
# Feature resolution
# ---------------------------------------------------------------------------

def resolve_features(
    df: pd.DataFrame,
    feature_set_names: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Resolve feature set names to actual column names present in df.

    - Static sets ("lexicon", "finbert", "controls") use FEATURE_SETS dict
    - "tfidf" → detects all "tfidf_*" columns in df
    - "embeddings" → detects all "emb_*" columns in df
    - "controls" also builds year dummy columns and appends them to df

    Returns
    -------
    (df_with_any_new_dummy_cols, list_of_resolved_column_names)
    """
    df = df.copy()
    cols: list[str] = []

    for name in feature_set_names:
        if name not in FEATURE_SETS:
            raise ValueError(f"Unknown feature set '{name}'. Available: {list(FEATURE_SETS)}")

        if name == "tfidf":
            cols += [c for c in df.columns if c.startswith("tfidf_")]
        elif name == "embeddings":
            cols += [c for c in df.columns if c.startswith("emb_")]
        elif name == "controls":
            if "pre_vol" in df.columns:
                cols.append("pre_vol")
            # Year fixed effects: dummy-encode the year column, drop first to avoid multicollinearity
            yr_dummies = pd.get_dummies(df["year"], prefix="yr", drop_first=True).astype(int)
            for col in yr_dummies.columns:
                df[col] = yr_dummies[col]
            cols += yr_dummies.columns.tolist()
        else:
            # Static feature set — filter to columns that actually exist in df
            cols += [c for c in FEATURE_SETS[name] if c in df.columns]

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_cols = [c for c in cols if not (c in seen or seen.add(c))]  # type: ignore[func-returns-value]

    missing = [c for c in unique_cols if c not in df.columns]
    if missing:
        print(f"  WARN: columns not in df (skipped): {missing}")
        unique_cols = [c for c in unique_cols if c in df.columns]

    return df, unique_cols


# ---------------------------------------------------------------------------
# Core experiment runners
# ---------------------------------------------------------------------------

def run_experiment(
    df: pd.DataFrame,
    feature_set_names: list[str],
    regressor_name: str,
    target_col: str = "car_03",
    experiment_name: str | None = None,
) -> RegressionResult:
    """Run one regression experiment: (feature sets) × (regressor) → RegressionResult.

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_modelling_dataset().
    feature_set_names : list[str]
        Feature sets to combine. E.g. ["lexicon", "controls"].
        Valid names: "lexicon", "finbert", "tfidf", "embeddings", "controls"
    regressor_name : str
        Model to use. Valid names: "OLS", "Ridge", "RF", "MLP"
    target_col : str
        Regression target. "car_03" (4-day CAR) or "car_01" (2-day CAR).
    experiment_name : str | None
        Label in the results table. Auto-generated from inputs if None.

    Returns
    -------
    RegressionResult
    """
    if regressor_name not in REGRESSORS:
        raise ValueError(f"Unknown regressor '{regressor_name}'. Available: {list(REGRESSORS)}")

    name = experiment_name or f"{' + '.join(feature_set_names)} [{regressor_name}]"

    df, feature_cols = resolve_features(df, feature_set_names)
    if not feature_cols:
        raise ValueError(f"No valid columns found for feature sets: {feature_set_names}")

    sub = df[["year"] + feature_cols + [target_col]].dropna()
    train_mask = sub["year"].isin(TRAIN_YEARS)
    test_mask  = sub["year"].isin(TEST_YEARS)

    y_train = sub.loc[train_mask, target_col].values
    y_test  = sub.loc[test_mask,  target_col].values
    X_train = sub.loc[train_mask, feature_cols].values
    X_test  = sub.loc[test_mask,  feature_cols].values

    print(f"\n[{name}]  train={len(y_train)}, test={len(y_test)}, features={len(feature_cols)}")

    if regressor_name == "OLS":
        return _run_ols(name, feature_cols, X_train, X_test, y_train, y_test, target_col)
    else:
        return _run_sklearn(name, feature_cols, X_train, X_test, y_train, y_test, target_col, regressor_name)


def run_null_model(df: pd.DataFrame, target_col: str = "car_03") -> RegressionResult:
    """Null baseline: predict the training mean for every observation.

    R² = 0 by construction on training data.
    OOS R² ≤ 0 — every other model must beat this to be useful.
    """
    sub = df[[target_col, "year"]].dropna()
    train_mask = sub["year"].isin(TRAIN_YEARS)
    test_mask  = sub["year"].isin(TEST_YEARS)

    y_train    = sub.loc[train_mask, target_col].values
    y_test     = sub.loc[test_mask,  target_col].values
    train_mean = float(np.mean(y_train))
    y_pred     = np.full_like(y_test, fill_value=train_mean)

    return RegressionResult(
        model_name="Null (mean)", target=target_col, features_used=[],
        n_train=int(train_mask.sum()), n_test=int(test_mask.sum()),
        coef=pd.Series({"const": train_mean}), pvalues=pd.Series(dtype=float),
        train_r2=0.0,
        test_r2=r_squared(y_test, y_pred),
        oos_r2=oos_r_squared(y_train, y_test, y_pred),
        mae=float(np.mean(np.abs(y_test - y_pred))),
        rmse=float(np.sqrt(np.mean((y_test - y_pred) ** 2))),
        wald_pvalue=None, sm_result=None,
    )


def _run_ols(
    name: str,
    feature_cols: list[str],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    target_col: str,
) -> RegressionResult:
    """Statsmodels OLS with HC3 heteroskedasticity-robust standard errors."""
    X_train_c = sm.add_constant(X_train, has_constant="add")
    X_test_c  = sm.add_constant(X_test,  has_constant="add")

    # HC3 robust errors: correct for non-constant variance in financial returns
    ols = sm.OLS(y_train, X_train_c).fit(cov_type="HC3")

    coef_names = ["const"] + feature_cols
    coef    = pd.Series(ols.params,  index=coef_names)
    pvalues = pd.Series(ols.pvalues, index=coef_names)

    # Wald test: H0: β_neg + β_pos = 0  (symmetric sentiment effect)
    wald_p = None
    for neg_col, pos_col in _ASYMMETRY_PAIRS:
        if neg_col in feature_cols and pos_col in feature_cols:
            try:
                neg_idx = feature_cols.index(neg_col) + 1  # +1 for const column
                pos_idx = feature_cols.index(pos_col) + 1
                R = np.zeros(len(coef_names))
                R[neg_idx] = 1
                R[pos_idx] = 1
                wald_p = float(ols.t_test(R).pvalue)
                print(f"  Wald: β_neg={coef.iloc[neg_idx]:.4f}, "
                      f"β_pos={coef.iloc[pos_idx]:.4f}, "
                      f"p={wald_p:.4f}  "
                      f"{'→ ASYMMETRIC ✓' if wald_p < 0.10 else '→ symmetric'}")
            except Exception as e:
                print(f"  WARN: Wald test failed: {e}")
            break  # only test first matching pair

    y_pred_train = ols.predict(X_train_c)
    y_pred_test  = ols.predict(X_test_c)

    return RegressionResult(
        model_name=name, target=target_col, features_used=feature_cols,
        n_train=len(y_train), n_test=len(y_test),
        coef=coef, pvalues=pvalues,
        train_r2=r_squared(y_train, y_pred_train),
        test_r2=r_squared(y_test,   y_pred_test),
        oos_r2=oos_r_squared(y_train, y_test, y_pred_test),
        mae=float(np.mean(np.abs(y_test - y_pred_test))),
        rmse=float(np.sqrt(np.mean((y_test - y_pred_test) ** 2))),
        wald_pvalue=wald_p, sm_result=ols,
    )


def _run_sklearn(
    name: str,
    feature_cols: list[str],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    target_col: str,
    regressor_name: str,
) -> RegressionResult:
    """Generic sklearn regressor (Ridge / RF / MLP)."""
    model = REGRESSORS[regressor_name]()
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    coef = (pd.Series(model.coef_, index=feature_cols)
            if hasattr(model, "coef_") else pd.Series(dtype=float))

    return RegressionResult(
        model_name=name, target=target_col, features_used=feature_cols,
        n_train=len(y_train), n_test=len(y_test),
        coef=coef, pvalues=pd.Series(dtype=float),
        train_r2=r_squared(y_train, y_pred_train),
        test_r2=r_squared(y_test,   y_pred_test),
        oos_r2=oos_r_squared(y_train, y_test, y_pred_test),
        mae=float(np.mean(np.abs(y_test - y_pred_test))),
        rmse=float(np.sqrt(np.mean((y_test - y_pred_test) ** 2))),
        wald_pvalue=None, sm_result=None,
    )


# ---------------------------------------------------------------------------
# Time-series cross-validation
# ---------------------------------------------------------------------------

def run_timeseries_cv(
    df: pd.DataFrame,
    feature_set_names: list[str],
    regressor_name: str,
    target_col: str = "car_03",
    experiment_name: str | None = None,
) -> pd.DataFrame:
    """Expanding-window time-series cross-validation.

    With a single 2016–18 / 2019–20 split, OOS R² is estimated from only 69
    test observations — one unlucky test period can dominate the result.
    Expanding-window CV gives one OOS estimate per test year, which averages
    out period-specific noise and gives a more stable performance estimate.

    Folds (expanding train window):
        Fold 1: train 2016      → test 2017
        Fold 2: train 2016–17   → test 2018
        Fold 3: train 2016–18   → test 2019
        Fold 4: train 2016–19   → test 2020

    Parameters
    ----------
    df, feature_set_names, regressor_name, target_col
        Same as run_experiment().

    Returns
    -------
    pd.DataFrame with one row per fold + a summary row, columns:
        fold, train_years, test_year, n_train, n_test, oos_r2, mae
    """
    name = experiment_name or f"{' + '.join(feature_set_names)} [{regressor_name}] CV"

    df, feature_cols = resolve_features(df, feature_set_names)
    if not feature_cols:
        raise ValueError(f"No valid columns found for feature sets: {feature_set_names}")

    sub = df[["year"] + feature_cols + [target_col]].dropna()
    all_years = sorted(sub["year"].unique())

    # Need at least 2 years to form one fold
    if len(all_years) < 2:
        raise ValueError(f"Need ≥2 years of data, got: {all_years}")

    fold_rows = []
    all_y_test_pooled:  list[np.ndarray] = []
    all_y_pred_pooled:  list[np.ndarray] = []
    all_y_train_pooled: list[np.ndarray] = []

    for i in range(1, len(all_years)):
        train_years = all_years[:i]
        test_year   = all_years[i]

        train_mask = sub["year"].isin(train_years)
        test_mask  = sub["year"] == test_year

        y_train = sub.loc[train_mask, target_col].values
        y_test  = sub.loc[test_mask,  target_col].values
        X_train = sub.loc[train_mask, feature_cols].values
        X_test  = sub.loc[test_mask,  feature_cols].values

        if len(y_train) < 5 or len(y_test) < 3:
            continue  # skip folds with too few observations

        # Fit and predict using the same backend as run_experiment
        if regressor_name == "OLS":
            import statsmodels.api as sm
            X_train_c = sm.add_constant(X_train, has_constant="add")
            X_test_c  = sm.add_constant(X_test,  has_constant="add")
            ols = sm.OLS(y_train, X_train_c).fit(cov_type="HC3")
            y_pred = ols.predict(X_test_c)
        else:
            model = REGRESSORS[regressor_name]()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        fold_oos = oos_r_squared(y_train, y_test, y_pred)
        fold_mae = float(np.mean(np.abs(y_test - y_pred)))

        fold_rows.append({
            "fold":        i,
            "train_years": f"{train_years[0]}–{train_years[-1]}",
            "test_year":   test_year,
            "n_train":     len(y_train),
            "n_test":      len(y_test),
            "oos_r2":      round(fold_oos, 4),
            "mae":         round(fold_mae, 4),
        })

        all_y_train_pooled.append(y_train)
        all_y_test_pooled.append(y_test)
        all_y_pred_pooled.append(y_pred)

    # Pooled OOS R² across all folds
    y_train_all = np.concatenate(all_y_train_pooled)
    y_test_all  = np.concatenate(all_y_test_pooled)
    y_pred_all  = np.concatenate(all_y_pred_pooled)
    pooled_oos  = oos_r_squared(y_train_all, y_test_all, y_pred_all)
    pooled_mae  = float(np.mean(np.abs(y_test_all - y_pred_all)))

    fold_rows.append({
        "fold":        "pooled",
        "train_years": "all folds",
        "test_year":   "all folds",
        "n_train":     len(y_train_all),
        "n_test":      len(y_test_all),
        "oos_r2":      round(pooled_oos, 4),
        "mae":         round(pooled_mae, 4),
    })

    result_df = pd.DataFrame(fold_rows)
    print(f"\n[{name}] Time-series CV results:")
    print(result_df.to_string(index=False))
    return result_df


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

# Default metrics shown — pass a custom list to results_to_df() to override.
# Full set: "train_r2", "test_r2", "oos_r2", "mae", "rmse", "wald_p"
DEFAULT_REG_METRICS = ["train_r2", "test_r2", "oos_r2", "wald_p"]


def results_to_df(
    results: list[RegressionResult],
    metrics: list[str] = DEFAULT_REG_METRICS,
) -> pd.DataFrame:
    """Summarise a list of RegressionResults into a benchmark table DataFrame.

    Parameters
    ----------
    results : list[RegressionResult]
    metrics : list[str]
        Which metric columns to include.
        Full set: "train_r2", "test_r2", "oos_r2", "mae", "rmse", "wald_p"
    """
    all_rows = []
    for r in results:
        all_rows.append({
            "model":    r.model_name,
            "target":   r.target,
            "n_train":  r.n_train,
            "n_test":   r.n_test,
            "train_r2": round(r.train_r2, 4),
            "test_r2":  round(r.test_r2,  4),
            "oos_r2":   round(r.oos_r2,   4),
            "mae":      round(r.mae,       4),
            "rmse":     round(r.rmse,      4),
            "wald_p":   round(r.wald_pvalue, 4) if r.wald_pvalue is not None else None,
        })
    df = pd.DataFrame(all_rows)
    id_cols = ["model", "target", "n_train", "n_test"]
    metric_cols = [m for m in metrics if m in df.columns]
    return df[id_cols + metric_cols]
