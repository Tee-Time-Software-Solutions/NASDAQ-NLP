import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_modelling_dataset() -> pd.DataFrame:
    base = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "event_study_dataset.csv")
    lex = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "lexicon_sentiment_features.csv")
    fin = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "finbert_sentiment_features.csv")

    key = ["ticker", "file_name", "event_trading_day_final"]
    df = (
        base.merge(lex, on=key, how="inner")
        .merge(fin, on=key, how="inner")
    )

    # Basic derived controls
    df["abs_CAR_01"] = df["CAR_01"].abs()
    # Use numpy directly instead of deprecated pd.np
    df["log_pre_vol"] = np.log(df["pre_volatility"].clip(lower=1e-8))
    df["year"] = df["year"].astype(int)
    return df


def time_split(df: pd.DataFrame, train_end_year: int = 2018):
    train = df[df["year"] <= train_end_year].copy()
    test = df[df["year"] > train_end_year].copy()
    return train, test


def fit_ols(train: pd.DataFrame, test: pd.DataFrame, y_col: str, x_cols: list[str], add_const: bool = True):
    def prep(df: pd.DataFrame):
        X = df[x_cols].copy()
        if add_const:
            X = sm.add_constant(X, has_constant="add")
        y = df[y_col]
        return X, y

    X_train, y_train = prep(train)
    X_test, y_test = prep(test)

    model = sm.OLS(y_train, X_train).fit()

    train_r2 = model.rsquared
    y_pred_test = model.predict(X_test)
    ss_res = ((y_test - y_pred_test) ** 2).sum()
    ss_tot = ((y_test - y_test.mean()) ** 2).sum()
    test_r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float("nan")

    return model, train_r2, test_r2


def main() -> None:
    df = load_modelling_dataset()

    train, test = time_split(df, train_end_year=2018)
    print(f"Train events: {len(train)}, Test events: {len(test)}")

    specs = {
        "car01_market_only": {
            "y": "CAR_01",
            "X": ["pre_volatility"],
        },
        "car01_lexicon_only": {
            "y": "CAR_01",
            "X": ["neg_rate_lm", "pos_rate_lm"],
        },
        "car01_finbert_only": {
            "y": "CAR_01",
            "X": ["finbert_neg_mean", "finbert_pos_mean"],
        },
        "car01_full_lexicon": {
            "y": "CAR_01",
            "X": ["neg_rate_lm", "pos_rate_lm", "pre_volatility"],
        },
        "car01_full_finbert": {
            "y": "CAR_01",
            "X": ["finbert_neg_mean", "finbert_pos_mean", "pre_volatility"],
        },
    }
    print(f"Total model specs to run: {len(specs)}")

    results_rows = []
    for idx, (name, spec) in enumerate(specs.items(), start=1):
        model, r2_train, r2_test = fit_ols(train, test, spec["y"], spec["X"])
        print(f"\n=== {name} ===")
        print(model.summary())
        results_rows.append(
            {
                "spec": name,
                "y": spec["y"],
                "X": ",".join(spec["X"]),
                "train_r2": r2_train,
                "test_r2": r2_test,
                "beta_neg": model.params.get("neg_rate_lm", pd.NA),
                "beta_pos": model.params.get("pos_rate_lm", pd.NA),
                "beta_finbert_neg": model.params.get("finbert_neg_mean", pd.NA),
                "beta_finbert_pos": model.params.get("finbert_pos_mean", pd.NA),
            }
        )
        print(f"Finished spec {idx} / {len(specs)}")

    results = pd.DataFrame(results_rows)
    out_path = PROJECT_ROOT / "data" / "processed" / "model_results_car01.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f"\nSaved summary model results to: {out_path}")


if __name__ == "__main__":
    main()

