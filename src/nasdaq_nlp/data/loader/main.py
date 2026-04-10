"""
loader/main.py — Unified transcript scanner for both dataset sources.

This is the single entry point for loading transcripts.
It scans the original Thomson Reuters dataset and the ECT Kaggle dataset,
returning a combined list of TranscriptRecord objects with a `source` field
identifying which dataset each record came from.

Usage
-----
    from nasdaq_nlp.data.loader import scan_transcripts

    records = scan_transcripts()           # both sources
    records = scan_transcripts(ect=False)  # original only
    records = scan_transcripts(original=False)  # ECT only

    # Filter by source
    original = [r for r in records if r.source == "original"]
    ect_only = [r for r in records if r.source == "ect"]
"""

from __future__ import annotations

from pathlib import Path

from nasdaq_nlp.config import DATASET_DIR
from nasdaq_nlp.data.loader.ect import ECT_DATASET_DIR, scan_ect_transcripts
from nasdaq_nlp.data.loader.original import (
    count_by_ticker,
    scan_original_transcripts,
    transcripts_by_ticker,
)
from nasdaq_nlp.data.loader.schemas import TranscriptRecord

__all__ = [
    "TranscriptRecord",
    "scan_transcripts",
    "transcripts_by_ticker",
    "count_by_ticker",
]


def scan_transcripts(
    dataset_dir: Path = DATASET_DIR,
    ect_dir: Path = ECT_DATASET_DIR,
    original: bool = True,
    ect: bool = True,
) -> list[TranscriptRecord]:
    """Scan both transcript datasets and return a combined list.

    Parameters
    ----------
    dataset_dir : Path
        Root of the original Thomson Reuters dataset.
    ect_dir : Path
        Root of the cleaned_ECTs Kaggle dataset.
    original : bool
        Include original-source transcripts (default True).
    ect : bool
        Include ECT-source transcripts (default True).
        Silently skipped if ect_dir does not exist.

    Returns
    -------
    list[TranscriptRecord]
        Sorted by (ticker, year, quarter). Each record has a `source` field
        of 'original' or 'ect' so callers can handle them differently if needed.
    """
    records: list[TranscriptRecord] = []

    if original:
        orig_records = scan_original_transcripts(dataset_dir)
        records.extend(orig_records)

    if ect and ect_dir.exists():
        ect_records = scan_ect_transcripts(ect_dir)
        records.extend(ect_records)

    records.sort(key=lambda r: (r.ticker, r.year, r.quarter or r.month_str))
    return records
