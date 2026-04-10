"""
features/lexicon.py — Loughran–McDonald financial sentiment lexicon features.

THE LOUGHRAN–MCDONALD DICTIONARY
----------------------------------
Most general-purpose sentiment dictionaries (e.g. Harvard General Inquirer)
were built for news/social media and perform poorly on financial text.
For example, "liability" is neutral in everyday speech but strongly negative
in financial filings.

Loughran and McDonald (2011, Journal of Finance) manually reviewed 10-K filings
and built a dictionary of words classified into multiple categories (Negative,
Positive, Uncertainty, Litigious, Strong_Modal, Weak_Modal, Constraining).

We use only Positive and Negative because the research question is specifically
about directional sentiment asymmetry (|β_neg| > |β_pos|). The other categories
have no direct theoretical link to CAR and are discarded.

MATH (what we compute per transcript)
---------------------------------------
Let W = list of all word tokens in the transcript.
Let N_neg = number of tokens in the Negative word set.
Let N_pos = number of tokens in the Positive word set.

    NegRate = N_neg / |W|   (fraction of all words that are negative)
    PosRate = N_pos / |W|   (fraction of all words that are positive)

These are the two main features in our regression:
    CAR_{i,t} = β0 + β1 × NegRate_i + β2 × PosRate_i + controls + ε

THE ASYMMETRY HYPOTHESIS: |β1| > |β2|
  If negative words have a stronger effect on returns than positive words,
  β1 (the NegRate coefficient) should have larger magnitude than β2 (PosRate).

FULL DICTIONARY
----------------
Expected at: dataset/Loughran-McDonald_MasterDictionary_1993-2025.csv
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from nasdaq_nlp.config import (
    EVENT_STUDY_PATH,
    LEXICON_FEATURES_PATH,
    ensure_output_dirs,
)
from nasdaq_nlp.preprocessing.text import preprocess_transcript

# ---------------------------------------------------------------------------
# Dictionary loader
# ---------------------------------------------------------------------------

# Path to the LM master dictionary CSV in the dataset folder
_LM_CSV_PATH = (
    Path(__file__).parents[3] / "dataset" / "Loughran-McDonald_MasterDictionary_1993-2025.csv"
)


def load_lm_dictionary(csv_path: Path | None = None) -> tuple[set[str], set[str]]:
    """Load the Loughran–McDonald dictionary and return (positive_words, negative_words).

    Parameters
    ----------
    csv_path : Path | None
        Explicit path to the LM CSV. If None, uses the default dataset location.

    Returns
    -------
    (pos_words, neg_words) — both lowercase sets of strings.
    """
    path = csv_path or _LM_CSV_PATH

    if not path.is_file():
        raise FileNotFoundError(
            f"LM dictionary not found at {path}.\n"
            "Download from: https://sraf.nd.edu/loughranmcdonald-master-dictionary/\n"
            "and place it at: dataset/Loughran-McDonald_MasterDictionary_1993-2025.csv"
        )

    lm = pd.read_csv(path)

    # The CSV has a 'Word' column and numeric category columns.
    # Non-zero value = the word belongs to that category.
    for col in ("Word", "Positive", "Negative"):
        if col not in lm.columns:
            raise ValueError(f"Expected column '{col}' in LM dictionary at {path}")

    lm["Word"] = lm["Word"].astype(str).str.lower()
    pos_words = set(lm.loc[lm["Positive"] > 0, "Word"])
    neg_words = set(lm.loc[lm["Negative"] > 0, "Word"])
    print(f"Loaded LM dictionary: {len(pos_words)} positive, {len(neg_words)} negative words")
    return pos_words, neg_words


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

# Tokeniser: letters and apostrophes only (same as preprocessing/text.py)
_WORD_RE = re.compile(r"[A-Za-z']+")


def compute_lexicon_features(
    file_path: str | Path,
    section: str = "full",
    pos_words: set[str] | None = None,
    neg_words: set[str] | None = None,
) -> dict:
    """Compute Loughran–McDonald sentiment features for one transcript.

    Parameters
    ----------
    file_path : str | Path
        Path to the raw transcript .txt file.
    section : str
        'full', 'presentation', or 'qa' — which part of the transcript to score.
    pos_words, neg_words : set[str] | None
        Pre-loaded word sets (pass these in when processing many transcripts
        to avoid reloading the dictionary every time).

    Returns
    -------
    dict with keys:
        total_tokens, neg_count, pos_count, neg_rate, pos_rate
    """
    if pos_words is None or neg_words is None:
        pos_words, neg_words = load_lm_dictionary()

    # Load and preprocess the transcript
    raw_text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    result = preprocess_transcript(raw_text, section=section, remove_stopwords=False)

    # Tokenise the processed text (already lowercased in preprocess_transcript)
    tokens = _WORD_RE.findall(
        result["full_raw"]
        if section == "full"
        else result.get(f"{section}_raw", result["full_raw"])
    )
    total_tokens = max(len(tokens), 1)  # avoid division by zero

    # Count matches against dictionary sets
    neg_count = sum(1 for t in tokens if t in neg_words)
    pos_count = sum(1 for t in tokens if t in pos_words)

    return {
        "total_tokens": total_tokens,
        "neg_count": neg_count,
        "pos_count": pos_count,
        # Rates = counts normalised by total tokens
        # This controls for transcript length (longer calls have more words,
        # so raw counts would be misleading)
        "neg_rate": neg_count / total_tokens,
        "pos_rate": pos_count / total_tokens,
    }


# ---------------------------------------------------------------------------
# Pipeline function
# ---------------------------------------------------------------------------


def build_lexicon_features(
    study_path: Path = EVENT_STUDY_PATH,
    output_path: Path = LEXICON_FEATURES_PATH,
) -> pd.DataFrame:
    """Compute LM sentiment features for all events and save to CSV.

    Computes three sets of rates in one pass per transcript:
        full         → neg_rate,      pos_rate       (whole transcript)
        presentation → neg_rate_pres, pos_rate_pres  (management prepared remarks)
        qa           → neg_rate_qa,   pos_rate_qa    (analyst Q&A section)

    The presentation section is scripted and optimistic; Q&A is unscripted
    and more revealing. Comparing their coefficients tests whether market
    reactions track the candid Q&A tone more than the prepared remarks.

    Returns
    -------
    pd.DataFrame with one row per event.
    """
    ensure_output_dirs()

    # Load dictionary once (avoids reloading for each of 188 transcripts)
    pos_words, neg_words = load_lm_dictionary()

    events = pd.read_csv(study_path)
    print(f"Computing lexicon features for {len(events)} events (full + presentation + Q&A)...")

    rows = []
    for _, event in events.iterrows():
        file_path = Path(event["file_path"])
        if not file_path.is_file():
            print(f"  WARN: file not found — {file_path}")
            continue

        # Compute all three sections in one transcript read
        full = compute_lexicon_features(
            file_path, section="full", pos_words=pos_words, neg_words=neg_words
        )
        pres = compute_lexicon_features(
            file_path, section="presentation", pos_words=pos_words, neg_words=neg_words
        )
        qa = compute_lexicon_features(
            file_path, section="qa", pos_words=pos_words, neg_words=neg_words
        )

        rows.append(
            {
                "ticker": event["ticker"],
                "file_name": event["file_name"],
                "event_trading_day": event["event_trading_day"],
                # Full transcript
                "total_tokens": full["total_tokens"],
                "neg_count": full["neg_count"],
                "pos_count": full["pos_count"],
                "neg_rate": full["neg_rate"],
                "pos_rate": full["pos_rate"],
                # Presentation section
                "neg_rate_pres": pres["neg_rate"],
                "pos_rate_pres": pres["pos_rate"],
                # Q&A section
                "neg_rate_qa": qa["neg_rate"],
                "pos_rate_qa": qa["pos_rate"],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved lexicon features → {output_path}  ({len(df)} rows)")
    print(
        f"  Full    — Avg NegRate: {df['neg_rate'].mean():.4f}  PosRate: {df['pos_rate'].mean():.4f}"
    )
    print(
        f"  Pres    — Avg NegRate: {df['neg_rate_pres'].mean():.4f}  PosRate: {df['pos_rate_pres'].mean():.4f}"
    )
    print(
        f"  Q&A     — Avg NegRate: {df['neg_rate_qa'].mean():.4f}  PosRate: {df['pos_rate_qa'].mean():.4f}"
    )

    return df
