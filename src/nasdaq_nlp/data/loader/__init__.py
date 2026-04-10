"""
loader/ — Transcript scanning for both dataset sources.

Layout
------
    main.py     — unified scan_transcripts() entry point (start here)
    original.py — original Thomson Reuters dataset scanner
    ect.py      — ECT Kaggle dataset scanner + metadata builders
    schemas.py  — TranscriptRecord Pydantic model

Public API (all importable from nasdaq_nlp.data.loader):
    scan_transcripts(dataset_dir, ect_dir, original, ect)
    TranscriptRecord
    transcripts_by_ticker(records)
    count_by_ticker(records)
"""

from nasdaq_nlp.data.loader.main import (
    TranscriptRecord,
    count_by_ticker,
    scan_transcripts,
    transcripts_by_ticker,
)

__all__ = [
    "TranscriptRecord",
    "scan_transcripts",
    "transcripts_by_ticker",
    "count_by_ticker",
]
