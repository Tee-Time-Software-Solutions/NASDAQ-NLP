from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


@dataclass
class FinBertConfig:
    model_name: str = "ProsusAI/finbert"
    max_length: int = 256
    batch_size: int = 16


def load_events(event_path: Path) -> pd.DataFrame:
    df = pd.read_csv(event_path)
    required_cols = {"ticker", "file_name", "event_trading_day_final", "file_path"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in event dataset: {missing}")
    return df


def read_transcript(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def simple_sentence_split(text: str) -> list[str]:
    # Lightweight splitter to avoid spaCy dependency at runtime if desired.
    # Users can replace this with a spaCy-based splitter for better accuracy.
    import re

    # Split on sentence-ending punctuation followed by space/capital
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def batched(iterable: Iterable[str], batch_size: int) -> Iterable[list[str]]:
    batch: list[str] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_finbert_pipeline(cfg: FinBertConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name)
    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        max_length=cfg.max_length,
        return_all_scores=True,
    )
    return clf


def aggregate_probs(all_scores) -> dict[str, float]:
    """
    Aggregate FinBERT probabilities across sentences.

    Handles both of the following shapes that the transformers pipeline can
    return:
    - list[list[{'label': ..., 'score': ...}]]  (when return_all_scores=True)
    - list[{'label': ..., 'score': ...}]       (older / default behaviour)
    """
    import collections

    sums = collections.Counter()
    n = 0
    for sent_scores in all_scores:
        # If we received a dict instead of a list of dicts, wrap it
        if isinstance(sent_scores, dict):
            sent_iter = [sent_scores]
        else:
            sent_iter = sent_scores

        n += 1
        for s in sent_iter:
            label = str(s.get("label", "")).lower()
            score = float(s.get("score", 0.0))
            if not label:
                continue
            sums[label] += score
    if n == 0:
        return {"finbert_pos_mean": 0.0, "finbert_neg_mean": 0.0, "finbert_neu_mean": 0.0}

    return {
        "finbert_pos_mean": sums.get("positive", 0.0) / n,
        "finbert_neg_mean": sums.get("negative", 0.0) / n,
        "finbert_neu_mean": sums.get("neutral", 0.0) / n,
    }


def main() -> None:
    cfg = FinBertConfig()

    project_root = Path(__file__).resolve().parents[1]
    events_path = project_root / "data" / "processed" / "event_study_dataset.csv"
    output_path = project_root / "data" / "processed" / "finbert_sentiment_features.csv"

    events = load_events(events_path)
    total_events = len(events)
    if total_events == 0:
        raise RuntimeError("No events found in event_study_dataset.csv")

    print(f"Total events to score with FinBERT: {total_events}")

    clf = build_finbert_pipeline(cfg)

    rows: list[dict] = []
    for idx, row in events.iterrows():
        path = Path(row["file_path"])
        if not path.is_file():
            continue

        text = read_transcript(path)
        sentences = simple_sentence_split(text)
        if not sentences:
            continue

        all_scores: list[list[dict]] = []
        for batch in batched(sentences, cfg.batch_size):
            batch_scores = clf(batch)
            all_scores.extend(batch_scores)

        agg = aggregate_probs(all_scores)
        rows.append(
            {
                "ticker": row["ticker"],
                "file_name": row["file_name"],
                "event_trading_day_final": row["event_trading_day_final"],
                **agg,
            }
        )

        # Progress hint every 10 events (or on the last event)
        if (idx + 1) % 10 == 0 or (idx + 1) == total_events:
            print(f"Finished {idx + 1} / {total_events} events")

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved FinBERT sentiment features to: {output_path}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    # Optionally allow overriding model name via env var
    model_name = os.getenv("FINBERT_MODEL_NAME")
    if model_name:
        FinBertConfig.model_name = model_name  # type: ignore[attr-defined]
    main()

