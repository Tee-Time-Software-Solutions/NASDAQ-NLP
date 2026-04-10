"""
features/tfidf.py — TF-IDF representation of earnings call transcripts.

WHAT IS TF-IDF?  (Session 7)
-------------------------------
TF-IDF = Term Frequency × Inverse Document Frequency.
It's a smarter way to represent how important a word is to a specific document
(compared to plain word counts which just measure how often it appears).

The intuition:
  - A word that appears often in document d is important to d. (high TF)
  - But if that word appears in *every* document, it's not discriminative. (low IDF)
  - TF-IDF weights up words that are frequent in d but rare across the corpus.

MATH
-----
Term Frequency (TF):
    TF(t, d) = count(t in d) / |d|
    = fraction of words in document d that are term t

Inverse Document Frequency (IDF):
    IDF(t) = log( N / df(t) )
    where N = total number of documents
    and df(t) = number of documents containing t

    The log dampens the effect of very rare or very common terms.
    High IDF: term appears in few documents → more discriminative.
    Low IDF:  term appears in many documents → less discriminative.

TF-IDF score:
    TFIDF(t, d) = TF(t, d) × IDF(t)

Example:
    The word "revenue" appears in every earnings call → low IDF → TF-IDF ≈ 0
    The word "blockchain" appears in only 5 transcripts → high IDF → TF-IDF high for those 5

WHY TF-IDF FOR OUR TASK?
    TF-IDF features are the input to our Naive Bayes and Logistic Regression classifiers.
    They let these models identify which financial terms most reliably distinguish
    transcripts that preceded positive vs negative market reactions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from nasdaq_nlp.config import (
    EVENT_STUDY_PATH,
    TFIDF_FEATURES_PATH,
    TFIDF_MAX_FEATURES,
    ensure_output_dirs,
)
from nasdaq_nlp.preprocessing.text import preprocess_transcript


# ---------------------------------------------------------------------------
# TF-IDF feature builder
# ---------------------------------------------------------------------------

def build_tfidf_matrix(
    file_paths: list[Path],
    max_features: int = TFIDF_MAX_FEATURES,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    sublinear_tf: bool = True,
) -> tuple[np.ndarray, list[str], TfidfVectorizer]:
    """Build a TF-IDF feature matrix from a list of transcript files.

    Parameters
    ----------
    file_paths : list[Path]
    max_features : int
        Keep top max_features terms by IDF (default 500 from config).
    ngram_range : tuple
        Include unigrams and bigrams (1,2) by default.
    min_df : int
        Minimum number of documents a term must appear in (filters noise).
    sublinear_tf : bool
        If True, use log(1 + TF) instead of raw TF — reduces the dominance
        of very frequent terms (common in long documents like earnings calls).

    Returns
    -------
    X : np.ndarray, shape (n_docs, max_features)
    feature_names : list[str]
    vectorizer : TfidfVectorizer (fitted)
    """
    print(f"Building TF-IDF matrix for {len(file_paths)} transcripts...")

    # Preprocess: remove stopwords for TF-IDF (we want discriminative terms)
    corpus: list[str] = []
    for path in file_paths:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        processed = preprocess_transcript(raw, section="full", remove_stopwords=True)
        corpus.append(" ".join(processed["tokens"]))

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,  # use log(1+TF) for smoother term weighting
        norm="l2",                  # L2-normalise each document vector (unit length)
                                    # so document length doesn't dominate similarity
        token_pattern=r"[a-z']+",
    )

    X_sparse = vectorizer.fit_transform(corpus)
    X = X_sparse.toarray()
    feature_names = vectorizer.get_feature_names_out().tolist()

    print(f"TF-IDF matrix: {X.shape[0]} docs × {X.shape[1]} features")

    return X, feature_names, vectorizer


# ---------------------------------------------------------------------------
# Top terms per ticker (for visualisation in notebook 02)
# ---------------------------------------------------------------------------

def  top_tfidf_terms_by_ticker(
    events: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    X: np.ndarray,
    top_n: int = 10,
) -> pd.DataFrame:
    """Find the highest TF-IDF terms for each ticker, averaged across its documents.

    This tells us: what words are most characteristic of each company's
    earnings calls, compared to all other companies?

    Parameters
    ----------
    events : pd.DataFrame — must have a 'ticker' column
    vectorizer : TfidfVectorizer (fitted)
    X : np.ndarray — TF-IDF matrix aligned with events
    top_n : int

    Returns
    -------
    pd.DataFrame with columns ['ticker', 'term', 'mean_tfidf', 'rank']
    """
    feature_names = vectorizer.get_feature_names_out()
    rows = []

    for ticker in sorted(events["ticker"].unique()):
        # Get indices of events for this ticker
        mask = (events["ticker"] == ticker).values
        if mask.sum() == 0:
            continue

        # Average TF-IDF vector across all documents for this ticker
        mean_tfidf = X[mask].mean(axis=0)  # shape: (n_features,)

        # Top-n terms by mean TF-IDF
        top_indices = np.argsort(mean_tfidf)[::-1][:top_n]
        for rank, idx in enumerate(top_indices, start=1):
            rows.append({
                "ticker": ticker,
                "term": feature_names[idx],
                "mean_tfidf": float(mean_tfidf[idx]),
                "rank": rank,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pipeline function
# ---------------------------------------------------------------------------

def build_tfidf_features(
    study_path: Path = EVENT_STUDY_PATH,
    output_path: Path = TFIDF_FEATURES_PATH,
) -> tuple[pd.DataFrame, TfidfVectorizer]:
    """Build and save TF-IDF features for all events.

    Saves the feature matrix as a wide CSV (one row per event, one column per term).
    Also returns the fitted vectorizer for use in classifiers.

    Returns
    -------
    (features_df, vectorizer)
    """
    ensure_output_dirs()

    events = pd.read_csv(study_path)
    file_paths = [Path(p) for p in events["file_path"]]

    # Build TF-IDF matrix
    X, feature_names, vectorizer = build_tfidf_matrix(file_paths)

    # Package as DataFrame: identifier columns + one column per TF-IDF feature
    feature_df = pd.DataFrame(X, columns=[f"tfidf_{f}" for f in feature_names])
    result = pd.concat(
        [events[["ticker", "file_name", "event_trading_day"]].reset_index(drop=True),
         feature_df],
        axis=1,
    )

    result.to_csv(output_path, index=False)
    print(f"Saved TF-IDF features → {output_path}  ({result.shape})")

    return result, vectorizer
