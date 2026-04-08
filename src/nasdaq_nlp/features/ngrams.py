"""
features/ngrams.py — N-gram frequency features for earnings call transcripts.

WHAT IS AN N-GRAM?  (Session 3)
---------------------------------
An n-gram is a contiguous sequence of n words. For a sentence like
"revenue growth exceeded expectations":
    Unigrams (n=1): ['revenue', 'growth', 'exceeded', 'expectations']
    Bigrams  (n=2): ['revenue growth', 'growth exceeded', 'exceeded expectations']
    Trigrams (n=3): ['revenue growth exceeded', 'growth exceeded expectations']

WHY N-GRAMS FOR SENTIMENT?
    Unigrams capture individual sentiment words: "strong", "decline"
    Bigrams can capture negation: "not strong", "no growth"
    Bigrams also capture compound financial terms: "revenue growth", "market share"

    In practice, bigrams add marginal value over unigrams for this task —
    earnings call language is direct enough that unigrams dominate.
    We include bigrams as a robustness check (Session 3 concept coverage).

HOW WE REPRESENT N-GRAMS
    For each document, we count how often each n-gram appears.
    This produces a vector of counts — a "bag of n-grams" representation.
    The vocabulary (which n-grams to count) is fitted on the training corpus.

    scikit-learn's CountVectorizer handles this efficiently.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from nasdaq_nlp.config import EVENT_STUDY_PATH, ensure_output_dirs
from nasdaq_nlp.preprocessing.text import preprocess_transcript


# ---------------------------------------------------------------------------
# N-gram feature builder
# ---------------------------------------------------------------------------

def build_ngram_matrix(
    file_paths: list[Path],
    ngram_range: tuple[int, int] = (1, 2),
    max_features: int = 500,
    min_df: int = 2,
) -> tuple[np.ndarray, list[str], CountVectorizer]:
    """Build a count-based n-gram feature matrix from a list of transcripts.

    Parameters
    ----------
    file_paths : list[Path]
        List of transcript file paths. Documents are processed in this order.
    ngram_range : tuple[int, int]
        (min_n, max_n) for n-gram sizes.
        (1, 1) = unigrams only
        (1, 2) = unigrams + bigrams
    max_features : int
        Keep only the top max_features n-grams by corpus frequency.
        This prevents the vocabulary from exploding (our corpus is small).
    min_df : int
        Ignore n-grams that appear in fewer than min_df documents.
        Rare n-grams are usually noise (proper nouns, typos).

    Returns
    -------
    X : np.ndarray, shape (n_documents, max_features)
        Raw count matrix — each row is one document, each column is one n-gram.
    feature_names : list[str]
        N-gram strings corresponding to each column.
    vectorizer : CountVectorizer
        Fitted vectorizer (use vectorizer.transform() on new documents).
    """
    # Preprocess each transcript into a single string
    # We use remove_stopwords=True here because raw counts of 'the', 'and' etc.
    # dominate the vocabulary and mask meaningful financial terms
    print(f"Preprocessing {len(file_paths)} transcripts for n-gram features...")
    corpus: list[str] = []
    for path in file_paths:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        processed = preprocess_transcript(raw, section="full", remove_stopwords=True)
        # Join tokens back into a string (CountVectorizer expects strings)
        corpus.append(" ".join(processed["tokens"]))

    # Fit CountVectorizer on the full corpus
    vectorizer = CountVectorizer(
        ngram_range=ngram_range,    # e.g. (1,2) for unigrams + bigrams
        max_features=max_features,  # keep top 500 n-grams
        min_df=min_df,              # must appear in ≥2 documents
        analyzer="word",
        token_pattern=r"[a-z']+",  # already lowercased, no punctuation
    )

    # fit_transform: learn vocabulary AND transform corpus in one pass
    # X is a sparse matrix (most entries are 0 — most n-grams don't appear in most docs)
    X_sparse = vectorizer.fit_transform(corpus)

    # Convert to dense for easier manipulation (small corpus, so this is fine)
    X = X_sparse.toarray()
    feature_names = vectorizer.get_feature_names_out().tolist()

    print(f"N-gram matrix: {X.shape[0]} documents × {X.shape[1]} features")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    return X, feature_names, vectorizer


def get_top_ngrams(
    vectorizer: CountVectorizer,
    X: np.ndarray,
    top_n: int = 20,
) -> pd.DataFrame:
    """Return the top n-grams by total corpus frequency.

    Useful for a quick sanity check: are the most common n-grams
    financial terms (good) or stopwords (bad preprocessing)?

    Parameters
    ----------
    vectorizer : CountVectorizer (fitted)
    X : np.ndarray — count matrix
    top_n : int — how many top n-grams to return

    Returns
    -------
    pd.DataFrame with columns ['ngram', 'total_count', 'doc_frequency']
    """
    feature_names = vectorizer.get_feature_names_out()
    total_counts = X.sum(axis=0)           # sum across all documents per n-gram
    doc_freq = (X > 0).sum(axis=0)        # number of documents containing each n-gram

    df = pd.DataFrame({
        "ngram": feature_names,
        "total_count": total_counts,
        "doc_frequency": doc_freq,
    })
    return df.sort_values("total_count", ascending=False).head(top_n).reset_index(drop=True)
