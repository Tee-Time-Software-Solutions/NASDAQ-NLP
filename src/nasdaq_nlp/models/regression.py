"""
models/regression.py — OLS regression of sentiment features on market reactions,
                        plus Wald test for the asymmetry hypothesis.

RESEARCH QUESTION
------------------
Does negative sentiment predict market reactions more strongly than positive sentiment?

In equation form, we test:
    |β_neg| > |β_pos|   (null: |β_neg| = |β_pos|, symmetric effect)

REGRESSION SPECIFICATION
--------------------------
Main model:
    CAR_{i,[0,3]} = β0 + β1 × NegRate_i + β2 × PosRate_i + controls + ε_i

where:
    CAR_i,[0,3]  = cumulative abnormal return over days 0-3 for event i
    NegRate_i    = fraction of words that are LM-negative in transcript i
    PosRate_i    = fraction of words that are LM-positive in transcript i
    controls     = pre-event volatility (pre_vol) + year/quarter fixed effects

The asymmetry hypothesis predicts:
    β1 < 0  (more negative words → lower returns)
    β2 > 0  (more positive words → higher returns)
    |β1| > |β2|  (negative effect stronger)

WALD TEST MATH
---------------
We want to test the linear restriction:
    H0: β1 + β2 = 0   (i.e. |β_neg| = |β_pos|, symmetric)
    H1: β1 + β2 ≠ 0   (asymmetric)

This is equivalent to testing whether the NegRate and PosRate coefficients
are equal in magnitude but opposite in sign.

The Wald statistic is:
    W = (Rβ)' (R Var(β) R')^{-1} (Rβ)

where R = [0, 1, 1, 0, ...] picks out β1 and β2 (the restriction β1 + β2 = 0).
Under H0, W ~ χ²(1).

statsmodels provides this via the t_test() method on OLS results.

BASELINE MODELS
-----------------
We compare sentiment models against:
    1. Null model: predict the mean (ŷ = ȳ for all events) → R² = 0 by definition
    2. Market-only: pre-event volatility as the only predictor
       (controls for "uncertain events cause bigger moves regardless of sentiment")
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from nasdaq_nlp.config import (
    ASYMMETRY_RESULTS_PATH,
    BENCHMARK_TABLE_PATH,
    EMBEDDINGS_PATH,
    EVENT_STUDY_PATH,
    FINBERT_FEATURES_PATH,
    LEXICON_FEATURES_PATH,
    TRAIN_YEARS,
    TEST_YEARS,
    ensure_output_dirs,
)
from nasdaq_nlp.evaluation.metrics import r_squared, oos_r_squared


# ---------------------------------------------------------------------------
# Data assembler
# ---------------------------------------------------------------------------

def load_modelling_dataset(
    study_path: Path = EVENT_STUDY_PATH,
    lexicon_path: Path = LEXICON_FEATURES_PATH,
    finbert_path: Path = FINBERT_FEATURES_PATH,
    embeddings_path: Path = EMBEDDINGS_PATH,
) -> pd.DataFrame:
    """Merge event study + all feature sources into one modelling DataFrame.

    Not all feature files may exist (e.g. FinBERT may not have run yet).
    Missing files produce a warning and are skipped.

    Returns
    -------
    pd.DataFrame with one row per event.
    """
    # Base: event study (CAR, volatility, year, quarter)
    study = pd.read_csv(study_path, parse_dates=["event_trading_day"])
    study["year"] = study["event_trading_day"].dt.year
    print(f"Event study: {len(study)} rows")

    # Helper: ensure event_trading_day is datetime before merging
    def _load_features(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        df["event_trading_day"] = pd.to_datetime(df["event_trading_day"])
        return df

    # Lexicon features (required)
    if not lexicon_path.exists():
        raise FileNotFoundError(f"Run build_lexicon_features() first: {lexicon_path}")
    lexicon = _load_features(lexicon_path)
    study = study.merge(lexicon, on=["ticker", "file_name", "event_trading_day"], how="left")
    print(f"After lexicon merge: {len(study)} rows")

    # FinBERT features (optional)
    if finbert_path.exists():
        finbert = _load_features(finbert_path)
        study = study.merge(finbert, on=["ticker", "file_name", "event_trading_day"], how="left")
        print(f"FinBERT features merged ✓")
    else:
        print(f"WARN: FinBERT features not found at {finbert_path} — skipping")
        study["finbert_pos_mean"] = np.nan
        study["finbert_neg_mean"] = np.nan
        study["finbert_neu_mean"] = np.nan

    # Word2Vec embeddings (optional — used for LR + embeddings model)
    if embeddings_path.exists():
        emb = _load_features(embeddings_path)
        emb_cols = [c for c in emb.columns if c.startswith("emb_")]
        study = study.merge(
            emb[["ticker", "file_name", "event_trading_day"] + emb_cols],
            on=["ticker", "file_name", "event_trading_day"],
            how="left",
        )
        print(f"Embeddings merged ✓ ({len(emb_cols)} dimensions)")
    else:
        print(f"WARN: Embeddings not found — skipping")

    return study


# ---------------------------------------------------------------------------
# OLS regression wrapper
# ---------------------------------------------------------------------------

class RegressionResult(NamedTuple):
    """Container for OLS regression output."""
    model_name: str
    target: str
    n_train: int
    n_test: int
    coef: pd.Series           # named coefficient estimates
    pvalues: pd.Series        # p-values for each coefficient
    train_r2: float
    test_r2: float
    oos_r2: float
    wald_pvalue: float | None  # p-value of asymmetry Wald test (None if not applicable)
    sm_result: object          # the raw statsmodels result (for detailed inspection)


def run_ols(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_name: str,
    neg_col: str | None = None,
    pos_col: str | None = None,
) -> RegressionResult:
    """Run OLS regression and compute train/test metrics.

    Steps:
    1. Drop rows with NaN in features or target.
    2. Split by year (train: 2016-2018, test: 2019-2020).
    3. Fit OLS on training data using statsmodels (includes p-values, std errors).
    4. Predict on test data, compute OOS R².
    5. If neg_col and pos_col are specified, run Wald test for asymmetry.

    Parameters
    ----------
    df : pd.DataFrame — merged modelling dataset
    feature_cols : list[str] — names of predictor columns (must exist in df)
    target_col : str — name of the outcome variable (e.g. 'car_03')
    model_name : str — label for the benchmark table
    neg_col : str | None — name of the negative sentiment feature (for Wald test)
    pos_col : str | None — name of the positive sentiment feature (for Wald test)

    Returns
    -------
    RegressionResult
    """
    # Select relevant columns and drop rows with any NaN
    cols_needed = ["year"] + feature_cols + [target_col]
    sub = df[cols_needed].dropna()

    # Year-based train/test split
    train_mask = sub["year"].isin(TRAIN_YEARS)
    test_mask = sub["year"].isin(TEST_YEARS)

    if train_mask.sum() < 10 or test_mask.sum() < 5:
        print(f"  WARN: {model_name} — insufficient data "
              f"(train={train_mask.sum()}, test={test_mask.sum()})")

    y_train = sub.loc[train_mask, target_col].values
    y_test = sub.loc[test_mask, target_col].values
    X_train = sub.loc[train_mask, feature_cols].values
    X_test = sub.loc[test_mask, feature_cols].values

    # Add intercept column (statsmodels doesn't add one automatically)
    # OLS: y = β0 + β1*x1 + β2*x2 + ...
    X_train_c = sm.add_constant(X_train, has_constant="add")
    X_test_c = sm.add_constant(X_test, has_constant="add")

    # Fit OLS on training data
    ols = sm.OLS(y_train, X_train_c).fit(
        cov_type="HC3",  # HC3 heteroskedasticity-robust standard errors
                         # (financial returns often have non-constant variance)
    )

    # Predictions
    y_pred_train = ols.predict(X_train_c)
    y_pred_test = ols.predict(X_test_c)

    # Metrics
    train_r2 = r_squared(y_train, y_pred_train)
    test_r2 = r_squared(y_test, y_pred_test)
    oos_r2_val = oos_r_squared(y_train, y_test, y_pred_test)

    # Coefficient names (const + feature names)
    coef_names = ["const"] + feature_cols
    coef = pd.Series(ols.params, index=coef_names)
    pvalues = pd.Series(ols.pvalues, index=coef_names)

    # ── Wald test for asymmetry: H0: β_neg + β_pos = 0 ──
    # This tests whether negative and positive sentiment have equal and opposite effects.
    # Rejection (p < 0.05) means they're NOT equal → asymmetry.
    wald_p = None
    if neg_col is not None and pos_col is not None:
        try:
            # Build restriction string: "neg_col = -pos_col" → "neg_col + pos_col = 0"
            neg_idx = feature_cols.index(neg_col) + 1  # +1 for const
            pos_idx = feature_cols.index(pos_col) + 1

            # R vector: 0s everywhere except 1 at neg and 1 at pos position
            R = np.zeros(len(coef_names))
            R[neg_idx] = 1
            R[pos_idx] = 1

            # t_test with a restriction matrix performs the Wald test
            # H0: Rβ = 0  (i.e. β_neg + β_pos = 0 → symmetric)
            t_test = ols.t_test(R)
            wald_p = float(t_test.pvalue)

            print(f"  [{model_name}] β_neg={coef.iloc[neg_idx]:.4f} "
                  f"β_pos={coef.iloc[pos_idx]:.4f} "
                  f"Wald p={wald_p:.4f} "
                  f"{'→ ASYMMETRIC ✓' if wald_p < 0.10 else '→ symmetric'}")
        except Exception as e:
            print(f"  WARN: Wald test failed for {model_name}: {e}")

    return RegressionResult(
        model_name=model_name,
        target=target_col,
        n_train=int(train_mask.sum()),
        n_test=int(test_mask.sum()),
        coef=coef,
        pvalues=pvalues,
        train_r2=train_r2,
        test_r2=test_r2,
        oos_r2=oos_r2_val,
        wald_pvalue=wald_p,
        sm_result=ols,
    )


# ---------------------------------------------------------------------------
# All regression models
# ---------------------------------------------------------------------------

def run_all_regressions(
    df: pd.DataFrame,
    target_col: str = "car_03",
) -> list[RegressionResult]:
    """Run all OLS regression models and return a list of results.

    Models:
        1. Null: no features (ŷ = ȳ → R² = 0 by construction)
        2. Market-only: pre-event volatility
        3. LM Lexicon: NegRate + PosRate
        4. LM + Controls: NegRate + PosRate + pre_vol + year/quarter dummies
        5. FinBERT: finbert_neg_mean + finbert_pos_mean
        6. FinBERT + Controls
    """
    results: list[RegressionResult] = []

    # -- 1. Null model baseline (predicts training mean for all observations)
    #    R² = 0 by definition; we just record it for the table
    sub = df[[target_col, "year"]].dropna()
    train_mean = sub.loc[sub["year"].isin(TRAIN_YEARS), target_col].mean()
    y_test = sub.loc[sub["year"].isin(TEST_YEARS), target_col].values
    y_pred_null = np.full_like(y_test, fill_value=train_mean)
    results.append(RegressionResult(
        model_name="Null (mean)",
        target=target_col,
        n_train=int(sub["year"].isin(TRAIN_YEARS).sum()),
        n_test=int(sub["year"].isin(TEST_YEARS).sum()),
        coef=pd.Series({"const": train_mean}),
        pvalues=pd.Series({"const": np.nan}),
        train_r2=0.0,
        test_r2=r_squared(y_test, y_pred_null),
        oos_r2=oos_r_squared(
            df.loc[df["year"].isin(TRAIN_YEARS), target_col].dropna().values,
            y_test, y_pred_null,
        ),
        wald_pvalue=None,
        sm_result=None,
    ))

    # -- 2. Market-only (pre-event volatility as sole predictor)
    if "pre_vol" in df.columns:
        r = run_ols(df, ["pre_vol"], target_col, "Market-only (pre_vol)")
        results.append(r)

    # -- 3. LM Lexicon (main model)
    if {"neg_rate", "pos_rate"}.issubset(df.columns):
        r = run_ols(
            df, ["neg_rate", "pos_rate"], target_col,
            "LM Lexicon OLS",
            neg_col="neg_rate", pos_col="pos_rate",
        )
        results.append(r)

    # -- 4. LM Lexicon + Controls (pre_vol + year fixed effect)
    if {"neg_rate", "pos_rate", "pre_vol"}.issubset(df.columns):
        # Create year dummies but keep 'year' for the train/test split inside run_ols
        yr_dummies = pd.get_dummies(df["year"], prefix="yr", drop_first=True).astype(int)
        df_with_dummies = pd.concat([df, yr_dummies], axis=1)
        yr_cols = yr_dummies.columns.tolist()
        feat_cols = ["neg_rate", "pos_rate", "pre_vol"] + yr_cols
        r = run_ols(
            df_with_dummies, feat_cols, target_col,
            "LM Lexicon + Controls",
            neg_col="neg_rate", pos_col="pos_rate",
        )
        results.append(r)

    # -- 5. FinBERT (if available)
    if df["finbert_neg_mean"].notna().any():
        r = run_ols(
            df, ["finbert_neg_mean", "finbert_pos_mean"], target_col,
            "FinBERT OLS",
            neg_col="finbert_neg_mean", pos_col="finbert_pos_mean",
        )
        results.append(r)

    return results


# ---------------------------------------------------------------------------
# Pipeline function
# ---------------------------------------------------------------------------

def run_regression_pipeline(
    study_path: Path = EVENT_STUDY_PATH,
    lexicon_path: Path = LEXICON_FEATURES_PATH,
    finbert_path: Path = FINBERT_FEATURES_PATH,
    embeddings_path: Path = EMBEDDINGS_PATH,
    benchmark_output: Path = BENCHMARK_TABLE_PATH,
    asymmetry_output: Path = ASYMMETRY_RESULTS_PATH,
) -> tuple[pd.DataFrame, list[RegressionResult]]:
    """Full regression pipeline: load data → run models → save results.

    Returns
    -------
    (benchmark_df, regression_results)
    """
    ensure_output_dirs()

    # Load merged dataset
    df = load_modelling_dataset(study_path, lexicon_path, finbert_path, embeddings_path)

    print("\n=== Running OLS Regressions: target = CAR[0,3] ===")
    results_03 = run_all_regressions(df, target_col="car_03")

    print("\n=== Running OLS Regressions: target = CAR[0,1] ===")
    results_01 = run_all_regressions(df, target_col="car_01")

    all_results = results_03 + results_01

    # Build benchmark summary table
    rows = []
    for r in all_results:
        rows.append({
            "model": r.model_name,
            "target": r.target,
            "n_train": r.n_train,
            "n_test": r.n_test,
            "train_r2": round(r.train_r2, 4),
            "test_r2": round(r.test_r2, 4),
            "oos_r2": round(r.oos_r2, 4),
            "wald_p": round(r.wald_pvalue, 4) if r.wald_pvalue is not None else None,
        })

    benchmark_df = pd.DataFrame(rows)
    benchmark_df.to_csv(benchmark_output, index=False)
    print(f"\nSaved benchmark table → {benchmark_output}")
    print(benchmark_df.to_string(index=False))

    # Save detailed asymmetry results (coefficients for LM models)
    asym_rows = []
    for r in all_results:
        if r.wald_pvalue is not None:
            row = {
                "model": r.model_name,
                "target": r.target,
                "wald_p": r.wald_pvalue,
                "asymmetric": r.wald_pvalue < 0.10,
            }
            # Add sentiment coefficients
            for name, coef in r.coef.items():
                row[f"coef_{name}"] = coef
            for name, pval in r.pvalues.items():
                row[f"pval_{name}"] = pval
            asym_rows.append(row)

    if asym_rows:
        asym_df = pd.DataFrame(asym_rows)
        asym_df.to_csv(asymmetry_output, index=False)
        print(f"Saved asymmetry results → {asymmetry_output}")

    return benchmark_df, all_results
