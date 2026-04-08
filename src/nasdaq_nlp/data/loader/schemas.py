"""
loader/schemas.py — Pydantic model for a single transcript record, plus the
                    filename regex that belongs with the model it validates.

WHY PYDANTIC HERE
-----------------
Pydantic v2 gives us:
  - Automatic type coercion (string "2020" → int 2020)
  - Field validators that run at construction time — bad data is caught early,
    not silently propagated through the pipeline
  - A .model_dump() / .model_json() for free serialisation
  - IDE-friendly type hints without runtime overhead

The dataclass version had no validation: a bad filename could create a record
with month_str="" or day=0 and the error would only surface 3 steps later.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, field_validator


# ---------------------------------------------------------------------------
# Filename regex — lives here because it is part of the schema contract.
# If the filename format ever changes, update the regex AND the model together.
# ---------------------------------------------------------------------------

FILENAME_RE = re.compile(
    r"^(?P<year>\d{4})-(?P<month>[A-Za-z]{3})-(?P<day>\d{1,2})-(?P<ticker>[A-Z]+)\.txt$"
)
"""Compiled regex for the expected earnings-call filename format.

Expected: ``2020-Jan-28-AAPL.txt``
Groups:   year (4 digits) · month (3 alpha) · day (1-2 digits) · ticker (UPPER)
"""


# ---------------------------------------------------------------------------
# Pydantic model
# ---------------------------------------------------------------------------

class TranscriptRecord(BaseModel):
    """One transcript file, with all metadata extracted from its filename.

    Fields
    ------
    ticker : str
        Stock ticker symbol, always upper-case (e.g. 'AAPL').
    year : int
        Calendar year of the earnings call (e.g. 2020).
    month_str : str
        Three-letter month abbreviation, Title-cased (e.g. 'Jan').
    day : int
        Day-of-month (1–31).
    file_name : str
        Bare filename without directory (e.g. '2020-Jan-28-AAPL.txt').
    file_path : Path
        Absolute path to the transcript on disk.
    raw_text : str
        Full file contents, lazily loaded. Empty until ``load()`` is called.
    """

    ticker: str
    year: int
    month_str: str
    day: int
    file_name: str
    file_path: Path
    raw_text: str = ""

    # Pydantic v2 config — allow Path objects (arbitrary types) and mutation
    # (we mutate raw_text in load()).
    model_config = {"arbitrary_types_allowed": True}

    # -----------------------------------------------------------------------
    # Field validators — run at construction time, before the object exists
    # -----------------------------------------------------------------------

    @field_validator("ticker")
    @classmethod
    def ticker_must_be_uppercase(cls, v: str) -> str:
        """Normalise ticker to uppercase regardless of source capitalisation."""
        return v.upper()

    @field_validator("month_str")
    @classmethod
    def month_must_be_title_case(cls, v: str) -> str:
        """Normalise month to Title-case (e.g. 'JAN' → 'Jan')."""
        return v.capitalize()

    @field_validator("year")
    @classmethod
    def year_in_valid_range(cls, v: int) -> int:
        """Guard against obviously-wrong years (typos in filenames)."""
        if not (2000 <= v <= 2100):
            raise ValueError(f"year {v} is outside the expected 2000–2100 range")
        return v

    @field_validator("day")
    @classmethod
    def day_in_valid_range(cls, v: int) -> int:
        if not (1 <= v <= 31):
            raise ValueError(f"day {v} is out of range 1–31")
        return v

    # -----------------------------------------------------------------------
    # Methods
    # -----------------------------------------------------------------------

    def load(self) -> "TranscriptRecord":
        """Read the file from disk into raw_text (mutates in place).

        Returns self so calls can be chained: ``record = record.load()``
        """
        self.raw_text = self.file_path.read_text(encoding="utf-8", errors="ignore")
        return self

    @property
    def date_label(self) -> str:
        """Human-readable date string, e.g. '2020-Jan-28'."""
        return f"{self.year}-{self.month_str}-{self.day:02d}"
