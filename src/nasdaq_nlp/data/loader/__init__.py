"""
loader/ — Scan the transcript dataset and return validated TranscriptRecord objects.

Public API (unchanged from the old loader.py — all callers keep working):
    scan_transcripts(dataset_dir) → list[TranscriptRecord]
    parse_filename(file_name)     → dict | None
    transcripts_by_ticker(records) → dict[str, list[TranscriptRecord]]
    count_by_ticker(records)       → dict[str, int]
    TranscriptRecord               (re-exported from schemas.py)

Internal layout:
    schemas.py  — Pydantic model + FILENAME_RE (what a record looks like)
    __init__.py — scanning / parsing logic (how we build records from disk)
"""

from __future__ import annotations

import warnings
from pathlib import Path

from nasdaq_nlp.config import DATASET_DIR

# Re-export the schema so callers can still do:
#   from nasdaq_nlp.data.loader import TranscriptRecord
from nasdaq_nlp.data.loader.schemas import FILENAME_RE, TranscriptRecord

__all__ = [
    "TranscriptRecord",
    "parse_filename",
    "scan_transcripts",
    "transcripts_by_ticker",
    "count_by_ticker",
]


# ---------------------------------------------------------------------------
# Filename parser
# ---------------------------------------------------------------------------

def parse_filename(file_name: str) -> dict | None:
    """Extract structured fields from an earnings-call filename.

    Parameters
    ----------
    file_name : str
        Bare filename, e.g. '2020-Jan-28-AAPL.txt'.

    Returns
    -------
    dict with keys {year, month_str, day, ticker} if the filename matches the
    expected pattern, or None if it does not.
    """
    m = FILENAME_RE.match(file_name)
    if m is None:
        warnings.warn(
            f"Filename '{file_name}' does not match expected pattern "
            f"(e.g. '2020-Jan-28-AAPL.txt') — skipping."
        )
        return None

    return {
        "year": int(m.group("year")),
        "month_str": m.group("month").capitalize(),  # 'jan' → 'Jan'
        "day": int(m.group("day")),
        "ticker": m.group("ticker").upper(),
    }


# ---------------------------------------------------------------------------
# Dataset scanner
# ---------------------------------------------------------------------------

def scan_transcripts(
    dataset_dir: Path = DATASET_DIR,
) -> list[TranscriptRecord]:
    """Walk dataset_dir and return one validated TranscriptRecord per .txt file.

    Directory structure expected:
        dataset_dir/
            AAPL/
                2020-Jan-28-AAPL.txt
                ...
            AMZN/
                ...

    Files whose names do not match the expected pattern are skipped with a
    warning (so stray READMEs or hidden files do not crash the pipeline).
    Pydantic validation runs at object construction — any remaining bad data
    (e.g. day=99) raises a ValidationError immediately, not silently later.

    Parameters
    ----------
    dataset_dir : Path
        Root directory to scan. Defaults to DATASET_DIR from config.py.

    Returns
    -------
    list[TranscriptRecord]
        Sorted by (ticker, year, month_str, day) for reproducibility.
    """
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}\n"
            "Make sure dataset/Transcripts/ exists."
        )

    records: list[TranscriptRecord] = []

    for ticker_dir in sorted(dataset_dir.iterdir()):
        if not ticker_dir.is_dir():
            continue  # skip stray files at the top level

        for txt_file in sorted(ticker_dir.glob("*.txt")):
            fields = parse_filename(txt_file.name)
            if fields is None:
                continue  # warning already issued by parse_filename

            # Pydantic construction — validators run here
            record = TranscriptRecord(
                ticker=fields["ticker"],
                year=fields["year"],
                month_str=fields["month_str"],
                day=fields["day"],
                file_name=txt_file.name,
                file_path=txt_file.resolve(),
            )
            records.append(record)

    records.sort(key=lambda r: (r.ticker, r.year, r.month_str, r.day))
    return records


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def transcripts_by_ticker(records: list[TranscriptRecord]) -> dict[str, list[TranscriptRecord]]:
    """Group a flat list of TranscriptRecords by ticker symbol."""
    groups: dict[str, list[TranscriptRecord]] = {}
    for rec in records:
        groups.setdefault(rec.ticker, []).append(rec)
    return groups


def count_by_ticker(records: list[TranscriptRecord]) -> dict[str, int]:
    """Return {ticker: count} — useful for quick data-quality checks."""
    return {ticker: len(recs) for ticker, recs in transcripts_by_ticker(records).items()}
