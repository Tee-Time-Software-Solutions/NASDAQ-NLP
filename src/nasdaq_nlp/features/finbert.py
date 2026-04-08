"""
features/finbert.py — FinBERT transformer-based sentiment classification.  (Sessions 19-20)

WHAT IS FINBERT?
-----------------
FinBERT (Araci, 2019) is a BERT model fine-tuned on financial text
from financial news, earnings releases, and analyst reports.
It was released as 'ProsusAI/finbert' on HuggingFace.

BERT background (Session 19-20):
    BERT = Bidirectional Encoder Representations from Transformers.
    Unlike Word2Vec which gives a static vector per word regardless of context,
    BERT uses self-attention to produce context-sensitive representations:

        "The company reported record EARNINGS despite supply chain RISKS"
        → 'risk' here is in a mixed positive/negative context — BERT captures this

    FinBERT adds financial domain knowledge on top of BERT's general English
    representations through fine-tuning on labelled financial text.

OUTPUT
-------
For each sentence, FinBERT outputs a probability distribution over three classes:
    P(positive), P(negative), P(neutral)   (sum to 1.0)

We aggregate across all sentences in a transcript by averaging:
    finbert_pos_mean = mean of P(positive) across all sentences
    finbert_neg_mean = mean of P(negative) across all sentences
    finbert_neu_mean = mean of P(neutral)  across all sentences

LEXICON vs FINBERT COMPARISON
--------------------------------
    Loughran-McDonald: rule-based, counts exact words in a fixed dictionary.
        Fast, interpretable, works on any text length.
        Can miss: context ("not strong" counted as positive if "strong" is in the dict).

    FinBERT: neural, context-aware, handles negation and idioms.
        Slow (runs a transformer), needs GPU for large batches.
        Better at: nuanced language, negation, financial jargon.

    We run both and compare their coefficients in the regression.
    This is a key methodological contribution of the paper.

PERFORMANCE NOTE
-----------------
FinBERT on CPU with 188 transcripts (each ~8000 tokens → ~500 sentences) takes
~15-30 minutes total. We cache results to FINBERT_FEATURES_PATH so this runs once.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd

from nasdaq_nlp.config import (
    EVENT_STUDY_PATH,
    FINBERT_BATCH_SIZE,
    FINBERT_FEATURES_PATH,
    FINBERT_MAX_LENGTH,
    FINBERT_MODEL_NAME,
    ensure_output_dirs,
)
from nasdaq_nlp.preprocessing.text import strip_header, strip_boilerplate_lines


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class FinBertConfig:
    """Hyperparameters for FinBERT inference."""
    model_name: str = FINBERT_MODEL_NAME         # "ProsusAI/finbert"
    max_length: int = FINBERT_MAX_LENGTH         # 256 tokens per sentence (BERT max = 512)
    batch_size: int = FINBERT_BATCH_SIZE         # sentences per forward pass


# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------

# Simple regex-based sentence splitter.
# We avoid SpaCy here to keep the dependency lighter — SpaCy adds 400MB+ models.
# The regex works well enough for formal financial text (few abbreviations mid-sentence).
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences for FinBERT processing.

    FinBERT is designed for sentence-level inputs (not full documents).
    Each sentence is independently classified, then we average.

    Filters:
    - Empty sentences
    - Very short sentences (<10 chars) — likely headers or fragments
    - Very long sentences (>500 chars) — likely garbled lines; truncate to max_length anyway

    Parameters
    ----------
    text : str
    Returns
    -------
    list[str] — cleaned sentences ready for the FinBERT pipeline.
    """
    parts = _SENTENCE_END_RE.split(text.strip())
    sentences = []
    for part in parts:
        s = part.strip()
        if len(s) < 10:
            continue  # too short to carry sentiment
        sentences.append(s)
    return sentences


# ---------------------------------------------------------------------------
# Batch iterator
# ---------------------------------------------------------------------------

def batch_iter(items: list[str], batch_size: int) -> Iterator[list[str]]:
    """Yield successive chunks of size batch_size from items."""
    for i in range(0, len(items), batch_size):
        yield items[i: i + batch_size]


# ---------------------------------------------------------------------------
# FinBERT pipeline (lazy import — only load torch/transformers when actually needed)
# ---------------------------------------------------------------------------

def build_finbert_pipeline(cfg: FinBertConfig):
    """Build and return the HuggingFace text-classification pipeline for FinBERT.

    The pipeline handles tokenisation, model inference, and softmax in one call.
    First call downloads the model (~500MB) and caches it in ~/.cache/huggingface/.

    Parameters
    ----------
    cfg : FinBertConfig

    Returns
    -------
    transformers.Pipeline
    """
    # Lazy import: don't load torch at module import time (slow, even if unused)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    print(f"Loading FinBERT model: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name)

    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        max_length=cfg.max_length,
        top_k=None,          # return all class scores (not just top-1)
    )
    print("FinBERT loaded ✓")
    return clf


# ---------------------------------------------------------------------------
# Score aggregation
# ---------------------------------------------------------------------------

def aggregate_sentence_scores(batch_results: list) -> dict[str, float]:
    """Average FinBERT scores across all sentences in a transcript.

    batch_results is a list of per-sentence score lists, e.g.:
        [ [{'label': 'positive', 'score': 0.8}, {'label': 'negative', 'score': 0.1}, ...],
          [{'label': 'neutral',  'score': 0.9}, ...],
          ... ]

    We sum each label's probability across sentences, then divide by n_sentences.
    This gives us the mean probability of each sentiment class across the transcript.

    Parameters
    ----------
    batch_results : list
        Raw output from the HuggingFace pipeline (top_k=None mode).

    Returns
    -------
    dict with keys: finbert_pos_mean, finbert_neg_mean, finbert_neu_mean
    """
    from collections import defaultdict
    sums: dict[str, float] = defaultdict(float)
    n_sentences = 0

    for sentence_scores in batch_results:
        # Each sentence_scores is a list of {label, score} dicts
        if isinstance(sentence_scores, dict):
            sentence_scores = [sentence_scores]  # handle single-label output format

        n_sentences += 1
        for item in sentence_scores:
            label = str(item.get("label", "")).lower()
            score = float(item.get("score", 0.0))
            sums[label] += score

    if n_sentences == 0:
        return {"finbert_pos_mean": 0.0, "finbert_neg_mean": 0.0, "finbert_neu_mean": 0.0}

    return {
        "finbert_pos_mean": sums.get("positive", 0.0) / n_sentences,
        "finbert_neg_mean": sums.get("negative", 0.0) / n_sentences,
        "finbert_neu_mean": sums.get("neutral", 0.0) / n_sentences,
    }


# ---------------------------------------------------------------------------
# Score one transcript
# ---------------------------------------------------------------------------

def score_transcript(
    raw_text: str,
    clf,
    cfg: FinBertConfig,
) -> dict[str, float]:
    """Score a single transcript with FinBERT.

    Parameters
    ----------
    raw_text : str
        Full raw transcript text (will be cleaned and split into sentences).
    clf : transformers.Pipeline
    cfg : FinBertConfig

    Returns
    -------
    dict with finbert_pos_mean, finbert_neg_mean, finbert_neu_mean
    """
    # Clean the transcript before sentence splitting
    cleaned = strip_header(raw_text)
    cleaned = strip_boilerplate_lines(cleaned)

    # Split into sentences
    sentences = split_sentences(cleaned)
    if not sentences:
        return {"finbert_pos_mean": 0.0, "finbert_neg_mean": 0.0, "finbert_neu_mean": 0.0}

    # Process in batches (avoids OOM on long transcripts)
    all_scores = []
    for batch in batch_iter(sentences, cfg.batch_size):
        batch_scores = clf(batch)
        all_scores.extend(batch_scores)

    return aggregate_sentence_scores(all_scores)


# ---------------------------------------------------------------------------
# Pipeline function
# ---------------------------------------------------------------------------

def build_finbert_features(
    study_path: Path = EVENT_STUDY_PATH,
    output_path: Path = FINBERT_FEATURES_PATH,
) -> pd.DataFrame:
    """Run FinBERT over all transcripts and save sentiment features.

    This function checks if the output file already exists and skips
    already-processed events — useful if the process was interrupted.

    WARNING: This is slow on CPU (~10-30 minutes for 188 transcripts).
    Run once and the output is cached.

    Returns
    -------
    pd.DataFrame with columns: ticker, file_name, event_trading_day,
                                finbert_pos_mean, finbert_neg_mean, finbert_neu_mean
    """
    ensure_output_dirs()

    events = pd.read_csv(study_path)
    cfg = FinBertConfig()

    # Check for existing partial results (resume if interrupted)
    if output_path.exists():
        existing = pd.read_csv(output_path)
        done_files = set(existing["file_name"].tolist())
        events_todo = events[~events["file_name"].isin(done_files)]
        print(f"Resuming: {len(done_files)} already done, {len(events_todo)} remaining")
    else:
        existing = pd.DataFrame()
        events_todo = events
        print(f"Processing {len(events_todo)} transcripts with FinBERT")

    if events_todo.empty:
        print("All transcripts already processed ✓")
        return pd.read_csv(output_path)

    # Load FinBERT pipeline (downloads model on first run)
    clf = build_finbert_pipeline(cfg)

    rows = []
    total = len(events_todo)
    for i, (_, event) in enumerate(events_todo.iterrows(), start=1):
        file_path = Path(event["file_path"])
        if not file_path.is_file():
            print(f"  WARN: {file_path.name} not found — skipping")
            continue

        raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
        scores = score_transcript(raw_text, clf, cfg)

        rows.append({
            "ticker": event["ticker"],
            "file_name": event["file_name"],
            "event_trading_day": event["event_trading_day"],
            **scores,
        })

        # Progress + incremental save every 10 events (in case of interruption)
        if i % 10 == 0 or i == total:
            print(f"  [{i}/{total}] {event['ticker']} {event['file_name']}")
            # Save incrementally (append new rows to existing)
            checkpoint = pd.concat(
                [existing, pd.DataFrame(rows)], ignore_index=True
            )
            checkpoint.to_csv(output_path, index=False)

    # Final save
    result = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    result.to_csv(output_path, index=False)
    print(f"Saved FinBERT features → {output_path}  ({len(result)} rows)")

    return result
