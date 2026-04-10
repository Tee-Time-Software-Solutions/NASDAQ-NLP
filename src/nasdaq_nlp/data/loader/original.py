"""
loader/original.py — Scan the Thomson Reuters StreetEvents transcript dataset.

Expected directory layout:
    dataset/Transcripts/
        AAPL/
            2020-Jan-28-AAPL.txt
        AMZN/
            ...

Public API:
    scan_original_transcripts(dataset_dir) → list[TranscriptRecord]
    parse_filename(file_name)              → dict | None
    transcripts_by_ticker(records)         → dict[str, list[TranscriptRecord]]
    count_by_ticker(records)               → dict[str, int]
"""

from __future__ import annotations

import warnings
from pathlib import Path

from nasdaq_nlp.config import DATASET_DIR
from nasdaq_nlp.data.loader.schemas import FILENAME_RE, TranscriptRecord
from nasdaq_nlp.data.metadata.schemas import MONTH_MAP, QUARTER_MAP

__all__ = [
    "parse_filename",
    "scan_original_transcripts",
    "transcripts_by_ticker",
    "count_by_ticker",
]

# Exchange / index for the 10 original NASDAQ tickers
_ORIGINAL_EXCHANGE = "NASDAQ"
_ORIGINAL_MARKET_IDX = "^IXIC"


def parse_filename(file_name: str) -> dict | None:
    """Extract fields from original-format filename (2020-Jan-28-AAPL.txt).

    Returns dict with year, month_str, day, ticker or None on mismatch.
    """
    m = FILENAME_RE.match(file_name)
    if m is None:
        warnings.warn(
            f"Filename '{file_name}' does not match expected pattern "
            "(e.g. '2020-Jan-28-AAPL.txt') — skipping."
        )
        return None
    return {
        "year": int(m.group("year")),
        "month_str": m.group("month").capitalize(),
        "day": int(m.group("day")),
        "ticker": m.group("ticker").upper(),
    }


def scan_original_transcripts(
    dataset_dir: Path = DATASET_DIR,
) -> list[TranscriptRecord]:
    """Walk dataset_dir and return one TranscriptRecord per .txt file.

    Files not matching the expected filename pattern are skipped with a warning.
    """
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}\nMake sure dataset/Transcripts/ exists."
        )

    records: list[TranscriptRecord] = []
    for ticker_dir in sorted(dataset_dir.iterdir()):
        if not ticker_dir.is_dir():
            continue
        for txt_file in sorted(ticker_dir.glob("*.txt")):
            fields = parse_filename(txt_file.name)
            if fields is None:
                continue
            month_num = MONTH_MAP.get(fields["month_str"].upper(), 1)
            quarter = QUARTER_MAP[month_num]
            records.append(
                TranscriptRecord(
                    ticker=fields["ticker"],
                    year=fields["year"],
                    month_str=fields["month_str"],
                    day=fields["day"],
                    quarter=quarter,
                    exchange=_ORIGINAL_EXCHANGE,
                    market_index=_ORIGINAL_MARKET_IDX,
                    file_name=txt_file.name,
                    file_path=txt_file.resolve(),
                    source="original",
                )
            )

    records.sort(key=lambda r: (r.ticker, r.year, r.month_str, r.day))
    return records


def transcripts_by_ticker(
    records: list[TranscriptRecord],
) -> dict[str, list[TranscriptRecord]]:
    """Group a flat list of TranscriptRecords by ticker symbol."""
    groups: dict[str, list[TranscriptRecord]] = {}
    for rec in records:
        groups.setdefault(rec.ticker, []).append(rec)
    return groups


def count_by_ticker(records: list[TranscriptRecord]) -> dict[str, int]:
    """Return {ticker: count}."""
    return {t: len(rs) for t, rs in transcripts_by_ticker(records).items()}
