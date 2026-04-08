"""
Objective: read transcripts per ticker from the dataset 
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from nasdaq_nlp.config import DATASET_DIR


# *** DATA STRUCTURE ***

@dataclass
class TranscriptRecord:
    """One row of data per earnings call transcript.

    Fields
    ------
    ticker : str
        Stock ticker symbol (e.g. 'AAPL'), derived from the parent folder name.
    year : int
        Calendar year extracted from the filename (e.g. 2020).
    month_str : str
        Three-letter month abbreviation as-is from filename (e.g. 'Jan').
    day : int
        Day of month from filename.
    file_name : str
        The bare filename without directory (e.g. '2020-Jan-28-AAPL.txt').
    file_path : Path
        Absolute path to the transcript file.
    raw_text : str
        Full raw file contents (lazily loaded — empty string until load() is called
        or eager=True is passed to scan_transcripts).
    """
    ticker: str
    year: int
    month_str: str
    day: int
    file_name: str
    file_path: Path
    raw_text: str = ""  # filled only when eager=True or you call record.load()

    def load(self) -> "TranscriptRecord":
        """Read the file from disk and populate raw_text in-place.

        Returns self so calls can be chained:  record = record.load()
        """
        self.raw_text = self.file_path.read_text(encoding="utf-8", 
                                                errors="ignore" # How to handle chars that are not correct for the given encoding
                                                )
        return self

    @property
    def date_label(self) -> str:
        """Human-readable date string, e.g. '2020-Jan-28'."""
        return f"{self.year}-{self.month_str}-{self.day:02d}"


# *** PARSING ***

# Expected filename pattern:  YYYY-Mon-DD-TICKER.txt
# Examples: 2020-Jan-28-AAPL.txt, 2019-Oct-24-GOOGL.txt
_FILENAME_RE = re.compile(
    r"^(?P<year>\d{4})-(?P<month>[A-Za-z]{3})-(?P<day>\d{1,2})-(?P<ticker>[A-Z]+)\.txt$"
)


def parse_filename(file_name: str) -> dict | None:
    """Extract structured fields from an earnings-call filename.

    Parameters
    ----------
    file_name : str
        Bare filename, e.g. '2020-Jan-28-AAPL.txt'.

    Returns
    -------
    dict with keys {year, month_str, day, ticker} if the filename matches,
    or None if the filename does not match the expected pattern.
    """
    m = _FILENAME_RE.match(file_name)
    if m is None:
        return None  # skip non-matching files (e.g. .DS_Store, README.md)
    return {
        "year": int(m.group("year")),
        "month_str": m.group("month").capitalize(),   # normalise e.g. 'jan' → 'Jan'
        "day": int(m.group("day")),
        "ticker": m.group("ticker").upper(),
    }


# ---------------------------------------------------------------------------
# Main scanner
# ---------------------------------------------------------------------------

def scan_transcripts(
    dataset_dir: Path = DATASET_DIR,
    *,
    eager: bool = False,
) -> list[TranscriptRecord]:
    """Walk dataset_dir and return one TranscriptRecord per .txt file found.

    The function searches one level deep: dataset_dir/<TICKER>/<date-file>.txt.
    Files that do not match the expected naming pattern are silently skipped
    (so stray README files or hidden files don't cause errors).

    Parameters
    ----------
    dataset_dir : Path
        Root directory to scan.  Defaults to DATASET_DIR from config.py.
    eager : bool
        If True, read every transcript into memory immediately.
        If False (default), raw_text is left empty; call record.load() when needed.
        Eager mode is convenient for small corpora but uses more RAM.

    Returns
    -------
    list[TranscriptRecord]
        Sorted by (ticker, year, month, day) for reproducibility.
    """
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}\n"
            "Make sure dataset/Transcripts/ exists relative to the project root."
        )

    records: list[TranscriptRecord] = []

    # Iterate over ticker subdirectories (e.g. AAPL/, AMZN/, …)
    for ticker_dir in sorted(dataset_dir.iterdir()):
        if not ticker_dir.is_dir():
            continue  # skip stray files at the top level

        # Iterate over transcript files inside each ticker folder
        for txt_file in sorted(ticker_dir.glob("*.txt")):
            fields = parse_filename(txt_file.name)
            if fields is None:
                # Skip files that don't match YYYY-Mon-DD-TICKER.txt
                continue

            record = TranscriptRecord(
                ticker=fields["ticker"],
                year=fields["year"],
                month_str=fields["month_str"],
                day=fields["day"],
                file_name=txt_file.name,
                file_path=txt_file.resolve(),
            )

            if eager:
                record.load()

            records.append(record)

    # Sort for reproducibility: ticker first, then chronologically
    records.sort(key=lambda r: (r.ticker, r.year, r.month_str, r.day))
    return records


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def transcripts_by_ticker(records: list[TranscriptRecord]) -> dict[str, list[TranscriptRecord]]:
    """Group a list of TranscriptRecords by ticker symbol.

    Example
    -------
    >>> recs = scan_transcripts()
    >>> by_ticker = transcripts_by_ticker(recs)
    >>> len(by_ticker['AAPL'])   # → number of AAPL earnings calls
    """
    groups: dict[str, list[TranscriptRecord]] = {}
    for rec in records:
        groups.setdefault(rec.ticker, []).append(rec)
    return groups


def count_by_ticker(records: list[TranscriptRecord]) -> dict[str, int]:
    """Return {ticker: count} mapping — useful for quick data-quality checks."""
    return {ticker: len(recs) for ticker, recs in transcripts_by_ticker(records).items()}
