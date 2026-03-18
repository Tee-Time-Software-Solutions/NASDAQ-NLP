import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    car_results_path = PROJECT_ROOT / "data" / "processed" / "model_results_car01.csv"
    if car_results_path.is_file():
        car_results = pd.read_csv(car_results_path)
    else:
        car_results = pd.DataFrame()

    summary = []
    if not car_results.empty:
        for _, row in car_results.iterrows():
            spec = row["spec"]
            desc = {
                "spec": spec,
                "train_r2": row.get("train_r2"),
                "test_r2": row.get("test_r2"),
                "beta_neg_or_finbert_neg": row.get("beta_neg", row.get("beta_finbert_neg")),
                "beta_pos_or_finbert_pos": row.get("beta_pos", row.get("beta_finbert_pos")),
            }
            summary.append(desc)

    summary_df = pd.DataFrame(summary)
    print(f"Building results summary from {len(summary_df)} rows")
    out_path = PROJECT_ROOT / "data" / "processed" / "results_summary_for_paper.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"Saved concise results summary to: {out_path}")


if __name__ == "__main__":
    main()

