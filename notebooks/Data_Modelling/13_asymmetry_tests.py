import pandas as pd
from pathlib import Path
import statsmodels.api as sm


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_modelling_dataset() -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / "data" / "processed" / "event_study_dataset.csv")


def load_features() -> pd.DataFrame:
    lex = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "lexicon_sentiment_features.csv")
    fin = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "finbert_sentiment_features.csv")
    key = ["ticker", "file_name", "event_trading_day_final"]
    return lex.merge(fin, on=key, how="inner")


def main() -> None:
    base = load_modelling_dataset()
    feats = load_features()
    key = ["ticker", "file_name", "event_trading_day_final"]
    df = base.merge(feats, on=key, how="inner")

    df["year"] = df["year"].astype(int)
    train = df[df["year"] <= 2018].copy()
    print(f"Running asymmetry test on {len(train)} training events")

    # Example: CAR_01 with lexicon features
    y = train["CAR_01"]
    X = train[["neg_rate_lm", "pos_rate_lm", "pre_volatility"]].copy()
    X = sm.add_constant(X, has_constant="add")

    print("Fitting CAR_01 ~ neg_rate_lm + pos_rate_lm + pre_volatility ...")
    model = sm.OLS(y, X).fit()
    print(model.summary())

    # Wald test: H0: beta_neg = beta_pos (no asymmetry)
    print("Running Wald test for beta_neg = beta_pos ...")
    wald_res = model.wald_test("neg_rate_lm = pos_rate_lm")
    print("\nWald test for symmetry (beta_neg = beta_pos):")
    print(wald_res)

    out_path = PROJECT_ROOT / "data" / "processed" / "asymmetry_tests_car01_lexicon.txt"
    out_path.write_text(
        model.summary().as_text()
        + "\n\nWald test (beta_neg = beta_pos):\n"
        + str(wald_res)
    )
    print(f"\nSaved asymmetry results to: {out_path}")


if __name__ == "__main__":
    main()

