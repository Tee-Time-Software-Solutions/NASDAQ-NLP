"""
metadata/schemas.py — Constants, regex, and the Pydantic model for one row of
                      event metadata (one earnings call → one trading-day event).

WHY SEPARATE FROM __init__.py
-------------------------------
The build logic in __init__.py is ~100 lines of date-math; it changes when the
pipeline logic changes.  The constants and model here change only when the data
contract changes (new columns, new markets, new date range).  Keeping them
separate makes each file's responsibility clear and limits merge conflicts.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# Timezone
# ---------------------------------------------------------------------------

EASTERN = ZoneInfo("America/New_York")
"""US Eastern timezone (handles EST↔EDT transitions via IANA DB)."""

# ---------------------------------------------------------------------------
# Market constants
# ---------------------------------------------------------------------------

MARKET_CLOSE_HOUR_ET = 16
"""NYSE/NASDAQ regular-session close: 4:00 PM Eastern Time."""

# ---------------------------------------------------------------------------
# Trading holidays (NYSE, 2016-2020 — covers the full dataset)
# ---------------------------------------------------------------------------

TRADING_HOLIDAYS: set[datetime] = {
    # 2016
    datetime(2016, 1, 1),
    datetime(2016, 1, 18),
    datetime(2016, 2, 15),
    datetime(2016, 5, 30),
    datetime(2016, 7, 4),
    datetime(2016, 9, 5),
    datetime(2016, 11, 24),
    datetime(2016, 12, 26),
    # 2017
    datetime(2017, 1, 2),
    datetime(2017, 1, 16),
    datetime(2017, 2, 20),
    datetime(2017, 5, 29),
    datetime(2017, 7, 4),
    datetime(2017, 9, 4),
    datetime(2017, 11, 23),
    datetime(2017, 12, 25),
    # 2018
    datetime(2018, 1, 1),
    datetime(2018, 1, 15),
    datetime(2018, 2, 19),
    datetime(2018, 5, 28),
    datetime(2018, 7, 4),
    datetime(2018, 9, 3),
    datetime(2018, 11, 22),
    datetime(2018, 12, 25),
    # 2019
    datetime(2019, 1, 1),
    datetime(2019, 1, 21),
    datetime(2019, 2, 18),
    datetime(2019, 5, 27),
    datetime(2019, 7, 4),
    datetime(2019, 9, 2),
    datetime(2019, 11, 28),
    datetime(2019, 12, 25),
    # 2020
    datetime(2020, 1, 1),
    datetime(2020, 1, 20),
    datetime(2020, 2, 17),
    datetime(2020, 5, 25),
    datetime(2020, 7, 3),
    datetime(2020, 9, 7),
    datetime(2020, 11, 26),
    datetime(2020, 12, 25),
}

# ---------------------------------------------------------------------------
# Month name → numeric (for parsing "JANUARY 28, 2020 / 10:00PM GMT")
# ---------------------------------------------------------------------------

MONTH_MAP: dict[str, int] = {
    "JANUARY": 1,
    "FEBRUARY": 2,
    "MARCH": 3,
    "APRIL": 4,
    "MAY": 5,
    "JUNE": 6,
    "JULY": 7,
    "AUGUST": 8,
    "SEPTEMBER": 9,
    "OCTOBER": 10,
    "NOVEMBER": 11,
    "DECEMBER": 12,
    # Three-letter abbreviations
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}

# ---------------------------------------------------------------------------
# Calendar month → fiscal quarter label
# ---------------------------------------------------------------------------

QUARTER_MAP: dict[int, str] = {
    1: "Q1",
    2: "Q1",
    3: "Q1",
    4: "Q2",
    5: "Q2",
    6: "Q2",
    7: "Q3",
    8: "Q3",
    9: "Q3",
    10: "Q4",
    11: "Q4",
    12: "Q4",
}

# ---------------------------------------------------------------------------
# Header datetime regex
# ---------------------------------------------------------------------------

HEADER_DT_RE = re.compile(
    r"(?P<month>[A-Z]+)\s+(?P<day>\d{1,2}),\s*(?P<year>\d{4})"
    r"\s*/\s*"
    r"(?P<hour>\d{1,2}):(?P<minute>\d{2})(?P<ampm>AM|PM)\s+GMT",
    re.IGNORECASE,
)
"""Regex to extract the call datetime from transcript headers.

Target format: ``JANUARY 28, 2020 / 10:00PM GMT``
Handles: AM/PM, single-digit day, varying minutes.
"""


# ---------------------------------------------------------------------------
# Pydantic schema for one event-metadata row
# ---------------------------------------------------------------------------


class EventMetadataRecord(BaseModel):
    """One row of event_metadata.csv — one earnings call, fully resolved.

    This is the output schema of the metadata pipeline.  Every field has
    a clear type so downstream modules can trust what they receive.

    Fields
    ------
    ticker : str
        Stock ticker (e.g. 'AAPL').
    file_name : str
        Source transcript filename.
    file_path : str
        Absolute path to the transcript file.
    year : int
        Calendar year of the call.
    quarter : str
        Fiscal quarter label derived from the call month (e.g. 'Q3').
    call_datetime_gmt : str | None
        ISO-like string 'YYYY-MM-DD HH:MM:SS' in UTC. None if header not found.
    call_datetime_et : str | None
        Same but converted to US Eastern time.
    call_time_et : str | None
        'HH:MM' for display purposes.
    after_market_close : bool
        True if the call happened at or after 4 PM ET.
    event_trading_day : str
        'YYYY-MM-DD' of the NYSE trading day that 'counts' for this call.
    """

    ticker: str
    file_name: str
    file_path: str
    year: int
    quarter: str
    call_datetime_gmt: Optional[str] = None
    call_datetime_et: Optional[str] = None
    call_time_et: Optional[str] = None
    after_market_close: bool = False
    event_trading_day: str

    @field_validator("quarter")
    @classmethod
    def quarter_must_be_valid(cls, v: str) -> str:
        if v not in {"Q1", "Q2", "Q3", "Q4"}:
            raise ValueError(f"quarter '{v}' is not one of Q1/Q2/Q3/Q4")
        return v

    @field_validator("event_trading_day")
    @classmethod
    def event_day_must_be_iso(cls, v: str) -> str:
        """Ensure the date string is parseable as YYYY-MM-DD."""
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"event_trading_day '{v}' is not a valid YYYY-MM-DD date")
        return v
