"""
features/lexicon.py — Loughran–McDonald financial sentiment lexicon features.

THE LOUGHRAN–MCDONALD DICTIONARY
----------------------------------
Most general-purpose sentiment dictionaries (e.g. Harvard General Inquirer)
were built for news/social media and perform poorly on financial text.
For example, "liability" is neutral in everyday speech but strongly negative
in financial filings.

Loughran and McDonald (2011, Journal of Finance) manually reviewed 10-K filings
and built a dictionary of ~2,700 words classified as:
    Negative, Positive, Uncertainty, Litigious, Constraining, Superfluous

We use only Positive and Negative for our asymmetry hypothesis.

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
The full LM dictionary CSV can be downloaded free from:
    https://sraf.nd.edu/loughranmcdonald-master-dictionary/

If not found, this module falls back to a built-in mini-dictionary that
captures the most common financial sentiment words. For academic research
quality, always use the full dictionary.
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

# Path where the full LM CSV might live (optional — falls back to mini-dict)
_LM_CSV_CANDIDATES = [
    Path(__file__).parents[3] / "dataset" / "LoughranMcDonald_MasterDictionary.csv",
    Path(__file__).parents[3] / "dataset" / "LM_Master_Dictionary.csv",
]


def load_lm_dictionary(csv_path: Path | None = None) -> tuple[set[str], set[str]]:
    """Load the Loughran–McDonald dictionary and return (positive_words, negative_words).

    Parameters
    ----------
    csv_path : Path | None
        Explicit path to the LM CSV. If None, tries standard candidate paths.
        If still not found, uses the built-in mini-dictionary.

    Returns
    -------
    (pos_words, neg_words) — both lowercase sets of strings.
    """
    # Try caller-provided path first, then standard locations
    search_paths = [csv_path] + _LM_CSV_CANDIDATES if csv_path else _LM_CSV_CANDIDATES

    for path in search_paths:
        if path is not None and path.is_file():
            lm = pd.read_csv(path)

            # LM CSV has a 'Word' column and numeric columns 'Negative', 'Positive'
            # (non-zero value = belongs to that category)
            for col in ("Word", "Positive", "Negative"):
                if col not in lm.columns:
                    raise ValueError(f"Expected column '{col}' in LM dictionary at {path}")

            lm["Word"] = lm["Word"].astype(str).str.lower()
            pos_words = set(lm.loc[lm["Positive"] > 0, "Word"])
            neg_words = set(lm.loc[lm["Negative"] > 0, "Word"])
            print(f"Loaded full LM dictionary from {path}: "
                  f"{len(pos_words)} positive, {len(neg_words)} negative words")
            return pos_words, neg_words

    # -----------------------------------------------------------------------
    # Fallback: built-in mini-dictionary
    # The most common high-frequency financial sentiment words, sufficient for
    # a 188-transcript corpus but less precise than the full 2,700-word list.
    # -----------------------------------------------------------------------
    print("LM dictionary CSV not found — using built-in mini-dictionary.")
    print("Download the full dictionary from: https://sraf.nd.edu/loughranmcdonald-master-dictionary/")

    neg_words = {
        # Loss / decline
        "loss", "losses", "decline", "declines", "declining", "decreased",
        "decrease", "decreases", "deteriorate", "deterioration",
        # Risk / uncertainty
        "risk", "risks", "uncertain", "uncertainty", "uncertainties",
        "unpredictable", "volatile", "volatility",
        # Failure / problems
        "fail", "failed", "failure", "failures", "impair", "impairment",
        "weak", "weakness", "weakening", "concern", "concerns", "worried",
        # Negative outcomes
        "adverse", "adversely", "headwind", "headwinds", "downturn",
        "downside", "slowdown", "shortfall", "disappointing", "disappoints",
        "below", "missed", "charges", "write-off", "writeoff",
        # Legal / regulatory
        "litigation", "litigious", "lawsuit", "penalty", "penalties",
    }

    pos_words = {
        # Profit / growth
        "profit", "profits", "profitable", "growth", "grew", "growing",
        "increase", "increases", "increased", "gains", "gain",
        # Strength
        "strong", "strength", "robust", "solid", "healthy", "positive",
        # Opportunity / momentum
        "opportunity", "opportunities", "momentum", "accelerate", "accelerating",
        # Achievement
        "record", "exceeded", "outperformed", "upside", "beat", "beats",
        "improve", "improved", "improvement", "improvements",
        # Future confidence
        "confident", "confidence", "optimistic", "upbeat", "favorable",
    }

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
    tokens = _WORD_RE.findall(result["full_raw"] if section == "full" else result.get(f"{section}_raw", result["full_raw"]))
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
    section: str = "full",
) -> pd.DataFrame:
    """Compute LM sentiment features for all events and save to CSV.

    Returns
    -------
    pd.DataFrame with one row per event.
    """
    ensure_output_dirs()

    # Load dictionary once (avoids reloading for each of 188 transcripts)
    pos_words, neg_words = load_lm_dictionary()

    events = pd.read_csv(study_path)
    print(f"Computing lexicon features for {len(events)} events...")

    rows = []
    for _, event in events.iterrows():
        file_path = Path(event["file_path"])
        if not file_path.is_file():
            print(f"  WARN: file not found — {file_path}")
            continue

        feats = compute_lexicon_features(file_path, section=section,
                                          pos_words=pos_words, neg_words=neg_words)
        rows.append({
            "ticker": event["ticker"],
            "file_name": event["file_name"],
            "event_trading_day": event["event_trading_day"],
            **feats,
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved lexicon features → {output_path}  ({len(df)} rows)")
    print(f"  Avg NegRate: {df['neg_rate'].mean():.4f}")
    print(f"  Avg PosRate: {df['pos_rate'].mean():.4f}")

    return df
