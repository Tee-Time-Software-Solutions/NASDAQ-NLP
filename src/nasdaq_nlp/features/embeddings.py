"""
features/embeddings.py — Word2Vec document embeddings.  (Session 13)

WHAT IS WORD2VEC?
------------------
Word2Vec (Mikolov et al., 2013) is a neural network that learns dense vector
representations of words by predicting context. Unlike bag-of-words (TF-IDF),
Word2Vec embeddings capture *semantic relationships*:

    vector("growth") ≈ vector("expansion")         (similar meaning)
    vector("risk") - vector("certain") ≈ vector("uncertain") - vector("certainty")

Two training approaches:
    Skip-gram: predict surrounding words given a center word
               Works well with small datasets (our 188-transcript corpus)
    CBOW: predict center word from surrounding words
          Faster, better for large corpora

We use skip-gram (gensim default for small corpora).

HOW DOES THIS PRODUCE DOCUMENT EMBEDDINGS?
--------------------------------------------
Word2Vec gives us one vector per word. To represent an entire transcript
(document), we take the average of all its word vectors:

    doc_embedding = (1/|tokens|) × Σ w2v(token_i)  for all tokens in document

This is called "average pooling" and, despite its simplicity, produces
document vectors that work well for classification tasks.

The resulting vector has dimension 100 (W2V_VECTOR_SIZE from config.py).

WHY USE THIS INSTEAD OF TF-IDF?
    TF-IDF: sparse, high-dimensional, no semantic similarity
    Word2Vec: dense, low-dimensional (100 vs 500), semantically aware

    "strong revenue growth" and "robust revenue expansion" would have very
    different TF-IDF vectors but similar Word2Vec document embeddings
    (because "strong" ≈ "robust" and "growth" ≈ "expansion" in embedding space).

NEAREST NEIGHBOR CHECK
-----------------------
A sanity check: find the words most similar to "revenue", "risk", "growth", etc.
If our model is working, similar financial concepts should be close in vector space.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from nasdaq_nlp.config import (
    EMBEDDINGS_PATH,
    EVENT_STUDY_PATH,
    W2V_EPOCHS,
    W2V_MIN_COUNT,
    W2V_VECTOR_SIZE,
    W2V_WINDOW,
    WORD2VEC_MODEL_PATH,
    ensure_output_dirs,
)
from nasdaq_nlp.preprocessing.text import preprocess_transcript


# ---------------------------------------------------------------------------
# Corpus builder
# ---------------------------------------------------------------------------

def build_tokenised_corpus(file_paths: list[Path]) -> list[list[str]]:
    """Load and tokenise all transcripts for Word2Vec training.

    Returns
    -------
    list of token lists — one list per document.
    Word2Vec expects this format: [['word1', 'word2', ...], ['word1', ...], ...]
    """
    print(f"Tokenising {len(file_paths)} transcripts for Word2Vec...")
    corpus = []
    for path in file_paths:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        processed = preprocess_transcript(raw, section="full", remove_stopwords=True)
        tokens = processed["tokens"]
        if tokens:  # skip empty documents
            corpus.append(tokens)
    print(f"Corpus: {len(corpus)} documents, "
          f"{sum(len(t) for t in corpus):,} total tokens")
    return corpus


# ---------------------------------------------------------------------------
# Word2Vec training
# ---------------------------------------------------------------------------

def train_word2vec(
    corpus: list[list[str]],
    vector_size: int = W2V_VECTOR_SIZE,
    window: int = W2V_WINDOW,
    min_count: int = W2V_MIN_COUNT,
    epochs: int = W2V_EPOCHS,
    seed: int = 42,
) -> Word2Vec:
    """Train a Word2Vec model on the tokenised corpus.

    Parameters
    ----------
    corpus : list[list[str]]
        Tokenised documents (output of build_tokenised_corpus).
    vector_size : int
        Dimensionality of word vectors. 100 is a good default for small corpora.
        Larger values (e.g. 300) capture more nuance but need more data.
    window : int
        Context window size. Words within ±5 positions of the target word
        are considered "context". Financial text has domain-specific collocations
        like "revenue growth", "earnings per share" — window=5 captures these.
    min_count : int
        Ignore words that appear fewer than min_count times in the corpus.
        With 188 documents, rare words (appearing once) are likely noise.
    epochs : int
        Number of full passes over the corpus during training.
        More epochs → better convergence but slower training.
    seed : int
        Random seed for reproducibility (Word2Vec has stochastic initialisation).

    Returns
    -------
    Trained gensim Word2Vec model.
    """
    print(f"Training Word2Vec: vector_size={vector_size}, window={window}, "
          f"min_count={min_count}, epochs={epochs}")

    model = Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=1,           # sg=1: skip-gram (better for small datasets)
                        # sg=0: CBOW (faster, better for large datasets)
        workers=4,      # parallel training threads
        seed=seed,
        epochs=epochs,
    )

    vocab_size = len(model.wv.key_to_index)
    print(f"Word2Vec vocabulary: {vocab_size} words")
    return model


# ---------------------------------------------------------------------------
# Document embedding (average pooling)
# ---------------------------------------------------------------------------

def document_embedding(tokens: list[str], model: Word2Vec) -> np.ndarray:
    """Compute a document vector as the average of its word vectors.

    Words not in the Word2Vec vocabulary are skipped (they were filtered
    out during training by min_count, so they'd be noise anyway).

    Parameters
    ----------
    tokens : list[str]
        Token list for one document.
    model : Word2Vec
        Trained model with vocabulary and vectors.

    Returns
    -------
    np.ndarray of shape (vector_size,)
    If no tokens are in the vocabulary, returns a zero vector.
    """
    # Get embeddings for each token that is in the model vocabulary
    word_vectors = [
        model.wv[token]
        for token in tokens
        if token in model.wv.key_to_index
    ]

    if not word_vectors:
        # All tokens were OOV (out-of-vocabulary) — return zero vector
        return np.zeros(model.vector_size)

    # Average pool: take the mean across all word vectors
    # Shape: (n_in_vocab, vector_size) → (vector_size,)
    return np.mean(word_vectors, axis=0)


def nearest_neighbors(
    model: Word2Vec,
    word: str,
    top_n: int = 10,
) -> list[tuple[str, float]]:
    """Find the most similar words to 'word' in the embedding space.

    Similarity is measured by cosine similarity:
        cos_sim(a, b) = (a · b) / (||a|| × ||b||)
    Values range from -1 (opposite) to +1 (identical direction).

    This is a sanity check: similar financial terms should cluster together.
    E.g. nearest neighbors of 'growth' might include 'expansion', 'revenue'.

    Parameters
    ----------
    model : Word2Vec
    word : str
    top_n : int

    Returns
    -------
    list of (similar_word, cosine_similarity) tuples.
    """
    if word not in model.wv.key_to_index:
        return []
    return model.wv.most_similar(word, topn=top_n)


# ---------------------------------------------------------------------------
# Pipeline function
# ---------------------------------------------------------------------------

def build_embedding_features(
    study_path: Path = EVENT_STUDY_PATH,
    output_path: Path = EMBEDDINGS_PATH,
    model_path: Path = WORD2VEC_MODEL_PATH,
) -> tuple[pd.DataFrame, Word2Vec]:
    """Train Word2Vec on the full corpus and save per-document embeddings.

    Each event gets a vector of shape (vector_size,) obtained by averaging
    the Word2Vec vectors of all tokens in its transcript.

    Returns
    -------
    (embeddings_df, model)
        embeddings_df: one row per event, columns: ticker, file_name, event_trading_day,
                       emb_0, emb_1, ..., emb_{vector_size-1}
    """
    ensure_output_dirs()

    events = pd.read_csv(study_path)
    file_paths = [Path(p) for p in events["file_path"]]

    # Build tokenised corpus for training
    corpus = build_tokenised_corpus(file_paths)

    # Train Word2Vec
    model = train_word2vec(corpus)

    # Save the trained model (gensim binary format)
    model.save(str(model_path))
    print(f"Saved Word2Vec model → {model_path}")

    # Compute document embeddings for each event
    print("Computing document embeddings (average pooling)...")
    embeddings = []
    for path in file_paths:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        processed = preprocess_transcript(raw, section="full", remove_stopwords=True)
        vec = document_embedding(processed["tokens"], model)
        embeddings.append(vec)

    emb_array = np.vstack(embeddings)  # shape: (n_events, vector_size)

    # Package as DataFrame
    emb_cols = [f"emb_{i}" for i in range(emb_array.shape[1])]
    emb_df = pd.DataFrame(emb_array, columns=emb_cols)
    result = pd.concat(
        [events[["ticker", "file_name", "event_trading_day"]].reset_index(drop=True),
         emb_df],
        axis=1,
    )

    result.to_csv(output_path, index=False)
    print(f"Saved embeddings → {output_path}  ({result.shape})")

    # Print nearest neighbors for a few financial terms as sanity check
    for probe_word in ["growth", "risk", "revenue", "strong"]:
        neighbors = nearest_neighbors(model, probe_word, top_n=5)
        if neighbors:
            nn_str = ", ".join(f"{w}({s:.2f})" for w, s in neighbors)
            print(f"  Nearest to '{probe_word}': {nn_str}")

    return result, model
