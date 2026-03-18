import re
from pathlib import Path

import pandas as pd


def load_events(event_path: Path) -> pd.DataFrame:
    """Load the final event-level dataset used for modelling."""
    df = pd.read_csv(event_path)
    required_cols = {"ticker", "file_name", "event_trading_day_final", "file_path"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in event dataset: {missing}")
    return df


def load_lm_dictionary(csv_path: Path | None = None) -> tuple[set[str], set[str]]:
    """
    Load the Loughran–McDonald dictionary and return positive/negative word sets.

    If `csv_path` is provided, it should point to a local copy of the official
    LM Master Dictionary (e.g. `LoughranMcDonald_MasterDictionary_*.csv`).

    Otherwise this function falls back to a **small built-in mini-dictionary**
    so that the script runs out of the box. For proper research use you should
    download the full dictionary from:
        https://sraf.nd.edu/loughranmcdonald-master-dictionary/
    and pass its path via the `LM_CSV_PATH` environment variable or by editing
    the call in `main()`.
    """
    if csv_path is not None and csv_path.is_file():
        lm = pd.read_csv(csv_path)
        # Common column names in LM master dictionary
        word_col = "Word"
        pos_col = "Positive"
        neg_col = "Negative"
        for col in (word_col, pos_col, neg_col):
            if col not in lm.columns:
                raise ValueError(f"Expected column '{col}' in LM dictionary CSV.")

        lm[word_col] = lm[word_col].astype(str).str.lower()
        pos_words = set(lm.loc[lm[pos_col] > 0, word_col])
        neg_words = set(lm.loc[lm[neg_col] > 0, word_col])
        return pos_words, neg_words

    # Minimal fallback lists (for quick experimentation only)
    neg_words = {
        "loss",
        "losses",
        "decline",
        "declines",
        "risk",
        "uncertain",
        "negative",
        "downturn",
        "weak",
        "concern",
        "headwind",
    }
    pos_words = {
        "profit",
        "profits",
        "growth",
        "strong",
        "opportunity",
        "opportunities",
        "improve",
        "improving",
        "record",
        "robust",
        "positive",
        "upside",
    }
    return pos_words, neg_words


def compute_lexicon_features(events: pd.DataFrame) -> pd.DataFrame:
    """
    Compute token counts and LM positive/negative rates per event.

    The returned frame has one row per event and can be merged back to the
    event dataset using (`ticker`, `file_name`, `event_trading_day_final`).
    """
    word_re = re.compile(r"[A-Za-z']+")

    pos_words, neg_words = load_lm_dictionary()

    rows: list[dict] = []
    for _, row in events.iterrows():
        path = Path(row["file_path"])
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
        except FileNotFoundError:
            # Skip events whose underlying transcript is missing
            continue

        tokens = word_re.findall(text)
        total_tokens = len(tokens)
        if total_tokens == 0:
            total_tokens = 1  # avoid division by zero; counts stay at zero

        neg_count = sum(1 for t in tokens if t in neg_words)
        pos_count = sum(1 for t in tokens if t in pos_words)

        rows.append(
            {
                "ticker": row["ticker"],
                "file_name": row["file_name"],
                "event_trading_day_final": row["event_trading_day_final"],
                "total_tokens": total_tokens,
                "neg_count_lm": neg_count,
                "pos_count_lm": pos_count,
                "neg_rate_lm": neg_count / total_tokens,
                "pos_rate_lm": pos_count / total_tokens,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    events_path = project_root / "data" / "processed" / "event_study_dataset.csv"
    output_path = project_root / "data" / "processed" / "lexicon_sentiment_features.csv"

    events = load_events(events_path)
    features = compute_lexicon_features(events)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)

    print(f"Saved lexicon sentiment features to: {output_path}")
    print(f"Rows: {len(features)}")


if __name__ == "__main__":
    main()

