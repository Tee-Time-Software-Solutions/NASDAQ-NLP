"""
metadata.py — Parse transcript headers to extract precise call timestamps,
              then determine the correct event trading day for each call.

WHY THIS MATTERS FOR THE EVENT STUDY
--------------------------------------
An earnings call that happens at 10 PM on Tuesday affects the market on
Wednesday (next open). One at 2 PM on Tuesday affects the same day.
The threshold in US markets is 4:00 PM Eastern Time (ET).

  call before 4 PM ET  →  event_trading_day = same calendar day
  call at/after 4 PM ET  →  event_trading_day = next business day
  call on weekend  →  event_trading_day = next Monday (or Tuesday if Monday is holiday)

Getting this right is critical: if we assign the wrong day, the CAR measure
picks up the wrong return window and the entire analysis is corrupted.

PIPELINE
---------
scan_transcripts() → list[TranscriptRecord]
    ↓ (metadata.py)
build_event_metadata() → outputs/processed/event_metadata.csv
    columns: ticker, file_name, year, quarter, call_datetime_gmt,
             call_datetime_et, after_market_close, event_trading_day
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from zoneinfo import ZoneInfo  # Python 3.9+ standard library — no extra install needed

from nasdaq_nlp.config import DATASET_DIR, EVENT_METADATA_PATH, ensure_output_dirs
from nasdaq_nlp.data.loader import TranscriptRecord, scan_transcripts


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# US Eastern timezone object (handles EST↔EDT transitions automatically via IANA database)
EASTERN = ZoneInfo("America/New_York")

# NYSE/NASDAQ market close time
MARKET_CLOSE_HOUR_ET = 16   # 4:00 PM ET

# US federal holidays that are NYSE trading holidays (approx — covers 2016-2020).
# A proper production system would use a trading-calendar library (e.g. exchange_calendars),
# but for this 5-ticker, 2016-2020 study the simple list is sufficient.
TRADING_HOLIDAYS: set[datetime] = {
    # 2016
    datetime(2016, 1, 1), datetime(2016, 1, 18), datetime(2016, 2, 15),
    datetime(2016, 5, 30), datetime(2016, 7, 4), datetime(2016, 9, 5),
    datetime(2016, 11, 24), datetime(2016, 12, 26),
    # 2017
    datetime(2017, 1, 2), datetime(2017, 1, 16), datetime(2017, 2, 20),
    datetime(2017, 5, 29), datetime(2017, 7, 4), datetime(2017, 9, 4),
    datetime(2017, 11, 23), datetime(2017, 12, 25),
    # 2018
    datetime(2018, 1, 1), datetime(2018, 1, 15), datetime(2018, 2, 19),
    datetime(2018, 5, 28), datetime(2018, 7, 4), datetime(2018, 9, 3),
    datetime(2018, 11, 22), datetime(2018, 12, 25),
    # 2019
    datetime(2019, 1, 1), datetime(2019, 1, 21), datetime(2019, 2, 18),
    datetime(2019, 5, 27), datetime(2019, 7, 4), datetime(2019, 9, 2),
    datetime(2019, 11, 28), datetime(2019, 12, 25),
    # 2020
    datetime(2020, 1, 1), datetime(2020, 1, 20), datetime(2020, 2, 17),
    datetime(2020, 5, 25), datetime(2020, 7, 3), datetime(2020, 9, 7),
    datetime(2020, 11, 26), datetime(2020, 12, 25),
}

# Months 3-letter → numeric, for parsing "JANUARY 28, 2020" style headers
_MONTH_MAP = {
    "JANUARY": 1, "FEBRUARY": 2, "MARCH": 3, "APRIL": 4,
    "MAY": 5, "JUNE": 6, "JULY": 7, "AUGUST": 8,
    "SEPTEMBER": 9, "OCTOBER": 10, "NOVEMBER": 11, "DECEMBER": 12,
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
    "JUN": 6, "JUL": 7, "AUG": 8,
    "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

# Quarter map: calendar month → fiscal quarter label
_QUARTER_MAP = {1: "Q1", 2: "Q1", 3: "Q1",
                4: "Q2", 5: "Q2", 6: "Q2",
                7: "Q3", 8: "Q3", 9: "Q3",
                10: "Q4", 11: "Q4", 12: "Q4"}


# ---------------------------------------------------------------------------
# Header datetime parser
# ---------------------------------------------------------------------------

# Target format in transcript files:  "JANUARY 28, 2020 / 10:00PM GMT"
# Variations observed: 12-hour clock, both AM/PM, single-digit day, minutes vary
_HEADER_DT_RE = re.compile(
    r"(?P<month>[A-Z]+)\s+(?P<day>\d{1,2}),\s*(?P<year>\d{4})"
    r"\s*/\s*"
    r"(?P<hour>\d{1,2}):(?P<minute>\d{2})(?P<ampm>AM|PM)\s+GMT",
    re.IGNORECASE,
)


def parse_header_datetime(raw_text: str) -> datetime | None:
    """Scan the first lines of the trasncript. Provided dataset
    always provides datetime at the top

    Parameters
    ----------
    raw_text : str
        Full transcript text.

    Returns
    -------
    datetime (UTC/GMT timezone-aware) if found, None otherwise.
    """
    # Only search the first 20 lines — the date is always in the header
    for line in raw_text.splitlines()[:20]:
        m = _HEADER_DT_RE.search(line.upper())
        if m is None:
            continue

        month_num = _MONTH_MAP.get(m.group("month").upper())
        if month_num is None:
            continue  # unknown month string — skip

        hour = int(m.group("hour"))
        minute = int(m.group("minute"))
        ampm = m.group("ampm").upper()

        # Convert 12-hour clock to 24-hour
        if ampm == "PM" and hour != 12:
            hour += 12
        elif ampm == "AM" and hour == 12:
            hour = 0


        dt_utc = datetime(
            year=int(m.group("year")),
            month=month_num,
            day=int(m.group("day")),
            hour=hour,
            minute=minute,
            tzinfo=timezone.utc,   # header says GMT = UTC
        )

        return dt_utc

    raise Exception("Could not extract datetime for given file")

# ---------------------------------------------------------------------------
# Event trading day assignment
# ---------------------------------------------------------------------------

def next_business_day(dt: datetime) -> datetime:
    """Return the next calendar day that is a NYSE trading day.

    Skips weekends (Saturday=5, Sunday=6) and known trading holidays.
    Returns a date-only datetime (midnight, no timezone).
    """
    # Strip time and timezone — we just want the calendar date
    candidate = datetime(dt.year, dt.month, dt.day) + timedelta(days=1)
    while candidate.weekday() >= 5 or candidate in TRADING_HOLIDAYS:
        candidate += timedelta(days=1)
    return candidate


def assign_event_trading_day(call_dt_et: datetime) -> datetime:
    """Determine which trading day 'counts' for an earnings call.

    Logic:
        - Before 4 PM ET → same calendar day (if it is itself a trading day)
        - At/after 4 PM ET, or on a non-trading day → next business day

    Parameters
    ----------
    call_dt_et : datetime
        Call time in US Eastern timezone (timezone-aware).

    Returns
    -------
    datetime (midnight, no timezone) representing the event trading day.
    """
    call_date = datetime(call_dt_et.year, call_dt_et.month, call_dt_et.day)
    is_weekend = call_date.weekday() >= 5
    is_holiday = call_date in TRADING_HOLIDAYS
    after_close = call_dt_et.hour >= MARKET_CLOSE_HOUR_ET

    if is_weekend or is_holiday or after_close:
        # Market is closed 
        return next_business_day(call_dt_et)
    else:
        # Market is open 
        return call_date


# ---------------------------------------------------------------------------
# Build metadata DataFrame
# ---------------------------------------------------------------------------

def _record_to_metadata_row(record: TranscriptRecord) -> dict:
    """Convert one TranscriptRecord into a metadata dict (one row in the CSV).

    This function reads the file if raw_text is empty, extracts the header
    datetime, converts to Eastern, and assigns the event trading day.
    """
    # Read content of the record (lazy loading)
    if not record.raw_text:
        record.load()

    row: dict = {
        "ticker": record.ticker,
        "file_name": record.file_name,
        "file_path": str(record.file_path),
        "year": record.year,
        "quarter": _QUARTER_MAP[
            _MONTH_MAP.get(record.month_str.upper(), 1)
        ],
    }

    # --- Parse the call datetime from the file header ---
    call_dt_utc = parse_header_datetime(record.raw_text)

    # Convert from UTC to US Eastern (ZoneInfo handles DST automatically)
    call_dt_et = call_dt_utc.astimezone(EASTERN)
    row["call_datetime_gmt"] = call_dt_utc.strftime("%Y-%m-%d %H:%M:%S")
    row["call_datetime_et"] = call_dt_et.strftime("%Y-%m-%d %H:%M:%S")
    row["call_time_et"] = call_dt_et.strftime("%H:%M")
    row["after_market_close"] = call_dt_et.hour >= MARKET_CLOSE_HOUR_ET
    event_day = assign_event_trading_day(call_dt_et)

    row["event_trading_day"] = event_day.strftime("%Y-%m-%d")
    return row


def build_event_metadata(
    dataset_dir: Path = DATASET_DIR,
    output_path: Path = EVENT_METADATA_PATH,
) -> pd.DataFrame:
    """Scan all transcripts, extract metadata, save to CSV, and return DataFrame.

    This is the first step in the pipeline:
        scan → parse header datetimes → assign event trading days → CSV

    Returns
    -------
    pd.DataFrame with one row per earnings call.
    """
    ensure_output_dirs()

    records = scan_transcripts(dataset_dir)
    print(f"Found {len(records)} transcripts across {len({r.ticker for r in records})} tickers")

    rows = []
    for record in records:
        try:
            row = _record_to_metadata_row(record)
            rows.append(row)
        except Exception as exc:
            # Don't crash the whole pipeline for a single bad file
            print(f"  WARN: skipping {record.file_name}: {exc}")

    df = pd.DataFrame(rows)

    # Convert event_trading_day to a proper date column
    df["event_trading_day"] = pd.to_datetime(df["event_trading_day"])

    # Sort for reproducibility
    df.sort_values(["ticker", "event_trading_day"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Save
    df.to_csv(output_path, index=False)
    print(f"Saved event metadata → {output_path}  ({len(df)} rows)")

    return df
