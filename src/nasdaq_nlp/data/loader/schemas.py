"""
loader/schemas.py — Pydantic model for a single transcript record.

Covers both dataset sources:
  - "original": Thomson Reuters StreetEvents, filename 2020-Jan-28-AAPL.txt
  - "ect":      Kaggle cleaned_ECTs, filename 2018_Q1_aapl_processed.txt

Fields present in both sources: ticker, year, quarter, exchange, market_index,
    file_name, file_path, source, raw_text.
Fields only populated for "original" source: month_str, day.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# Filename regexes
# ---------------------------------------------------------------------------

FILENAME_RE = re.compile(
    r"^(?P<year>\d{4})-(?P<month>[A-Za-z]{3})-(?P<day>\d{1,2})-(?P<ticker>[A-Z]+)\.txt$"
)
"""Original dataset: ``2020-Jan-28-AAPL.txt``"""

ECT_FILENAME_RE = re.compile(
    r"^(?P<year>\d{4})_Q(?P<quarter>\d)_(?P<ticker_lower>[a-z]+)_processed\.txt$"
)
"""ECT dataset: ``2018_Q1_aapl_processed.txt``"""


# ---------------------------------------------------------------------------
# Unified Pydantic model
# ---------------------------------------------------------------------------


class TranscriptRecord(BaseModel):
    """One transcript file from either dataset source.

    Common fields (both sources)
    ----------------------------
    ticker        : stock symbol, always upper-case
    year          : calendar year of the earnings call
    quarter       : 'Q1' … 'Q4'
    exchange      : 'NASDAQ' or 'NYSE'
    market_index  : benchmark ticker, e.g. '^IXIC' or '^GSPC'
    file_name     : bare filename
    file_path     : absolute path on disk
    source        : 'original' or 'ect'
    raw_text      : full file contents, lazily loaded via load()

    Original-only fields (empty string / 0 for ECT records)
    ---------------------------------------------------------
    month_str     : three-letter month, e.g. 'Jan'
    day           : day-of-month 1–31
    """

    ticker: str
    year: int
    quarter: str = ""
    exchange: str = "NASDAQ"
    market_index: str = "^IXIC"
    file_name: str
    file_path: Path
    source: Literal["original", "ect"] = "original"
    raw_text: str = ""

    # Original-only
    month_str: str = ""
    day: int = 0

    model_config = {"arbitrary_types_allowed": True}

    # -----------------------------------------------------------------------
    # Validators
    # -----------------------------------------------------------------------

    @field_validator("ticker")
    @classmethod
    def ticker_must_be_uppercase(cls, v: str) -> str:
        return v.upper()

    @field_validator("month_str")
    @classmethod
    def month_title_case(cls, v: str) -> str:
        return v.capitalize()

    @field_validator("year")
    @classmethod
    def year_in_valid_range(cls, v: int) -> int:
        if not (2000 <= v <= 2100):
            raise ValueError(f"year {v} outside 2000–2100")
        return v

    @field_validator("day")
    @classmethod
    def day_in_valid_range(cls, v: int) -> int:
        if not (0 <= v <= 31):  # 0 is valid for ECT records (no exact day)
            raise ValueError(f"day {v} out of range 0–31")
        return v

    # -----------------------------------------------------------------------
    # Methods
    # -----------------------------------------------------------------------

    def load(self) -> "TranscriptRecord":
        """Read file from disk into raw_text. Returns self for chaining."""
        self.raw_text = self.file_path.read_text(encoding="utf-8", errors="ignore")
        return self

    @property
    def date_label(self) -> str:
        """Human-readable date: '2020-Jan-28' for original, '2018-Q1' for ECT."""
        if self.month_str:
            return f"{self.year}-{self.month_str}-{self.day:02d}"
        return f"{self.year}-{self.quarter}"
