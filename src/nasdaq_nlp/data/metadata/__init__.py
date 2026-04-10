"""
metadata/ — Parse transcript headers to extract call timestamps, assign each
            call to the correct NYSE trading day, and write event_metadata.csv.

WHY THIS MATTERS FOR THE EVENT STUDY
--------------------------------------
An earnings call at 10 PM Tuesday affects the market on Wednesday (next open).
One at 2 PM Tuesday affects the same day.  Getting this right is critical —
the wrong event day corrupts the entire CAR measure.

Rule:
    call before 4 PM ET  →  event_trading_day = same calendar day
    call at/after 4 PM ET →  event_trading_day = next business day
    call on weekend/holiday → event_trading_day = next business day

Public API:
    build_event_metadata(dataset_dir, output_path) → pd.DataFrame
    parse_header_datetime(raw_text)                → datetime | None
    EventMetadataRecord                            (re-exported from schemas.py)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from nasdaq_nlp.config import DATASET_DIR, EVENT_METADATA_PATH, ensure_output_dirs

if TYPE_CHECKING:
    from nasdaq_nlp.data.loader.schemas import TranscriptRecord

# NOTE: loader imports are done lazily inside functions to avoid circular imports
# (loader → ect → metadata.schemas → metadata.__init__ → loader)
# Re-export schema so callers can import from this package
from nasdaq_nlp.data.metadata.schemas import (
    EASTERN,
    HEADER_DT_RE,
    MARKET_CLOSE_HOUR_ET,
    MONTH_MAP,
    QUARTER_MAP,
    TRADING_HOLIDAYS,
    EventMetadataRecord,
)

__all__ = [
    "EventMetadataRecord",
    "parse_header_datetime",
    "assign_event_trading_day",
    "build_event_metadata",
]


# ---------------------------------------------------------------------------
# Header datetime parser
# ---------------------------------------------------------------------------


def parse_header_datetime(raw_text: str) -> datetime | None:
    """Scan the first 20 lines of a transcript and extract the call datetime.

    Parameters
    ----------
    raw_text : str
        Full transcript text (only the header is searched).

    Returns
    -------
    timezone-aware datetime in UTC if the header is found, None otherwise.
    """
    for line in raw_text.splitlines()[:20]:
        m = HEADER_DT_RE.search(line.upper())
        if m is None:
            continue

        month_num = MONTH_MAP.get(m.group("month").upper())
        if month_num is None:
            continue

        hour = int(m.group("hour"))
        minute = int(m.group("minute"))
        ampm = m.group("ampm").upper()

        # Convert 12-hour → 24-hour
        if ampm == "PM" and hour != 12:
            hour += 12
        elif ampm == "AM" and hour == 12:
            hour = 0

        try:
            dt_utc = datetime(
                year=int(m.group("year")),
                month=month_num,
                day=int(m.group("day")),
                hour=hour,
                minute=minute,
                tzinfo=timezone.utc,
            )
        except ValueError:
            continue

        return dt_utc

    return None


# ---------------------------------------------------------------------------
# Event trading day assignment
# ---------------------------------------------------------------------------


def next_business_day(dt: datetime) -> datetime:
    """Return the next calendar day that is a NYSE trading day (date only)."""
    candidate = datetime(dt.year, dt.month, dt.day) + timedelta(days=1)
    while candidate.weekday() >= 5 or candidate in TRADING_HOLIDAYS:
        candidate += timedelta(days=1)
    return candidate


def assign_event_trading_day(call_dt_et: datetime) -> datetime:
    """Determine which trading day 'counts' for an earnings call.

    Logic:
        - Before 4 PM ET on a trading day → same calendar day
        - At/after 4 PM ET, weekend, or holiday → next business day

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
        return next_business_day(call_dt_et)
    else:
        return call_date


# ---------------------------------------------------------------------------
# Build one metadata row (validated via Pydantic)
# ---------------------------------------------------------------------------


def _record_to_metadata_row(record: "TranscriptRecord") -> EventMetadataRecord:
    """Convert one TranscriptRecord into a validated EventMetadataRecord."""
    if not record.raw_text:
        record.load()

    month_num = MONTH_MAP.get(record.month_str.upper(), 1)
    quarter = QUARTER_MAP[month_num]

    call_dt_utc = parse_header_datetime(record.raw_text)

    if call_dt_utc is not None:
        call_dt_et = call_dt_utc.astimezone(EASTERN)
        event_day = assign_event_trading_day(call_dt_et)
        row_kwargs = dict(
            call_datetime_gmt=call_dt_utc.strftime("%Y-%m-%d %H:%M:%S"),
            call_datetime_et=call_dt_et.strftime("%Y-%m-%d %H:%M:%S"),
            call_time_et=call_dt_et.strftime("%H:%M"),
            after_market_close=call_dt_et.hour >= MARKET_CLOSE_HOUR_ET,
        )
    else:
        # Fallback: no parseable header — assume midday, same-day event
        fallback_dt = datetime(record.year, month_num, record.day, 12, 0, tzinfo=EASTERN)
        event_day = assign_event_trading_day(fallback_dt)
        row_kwargs = dict(
            call_datetime_gmt=None,
            call_datetime_et=None,
            call_time_et=None,
            after_market_close=False,
        )

    # Pydantic validates all fields on construction
    return EventMetadataRecord(
        ticker=record.ticker,
        file_name=record.file_name,
        file_path=str(record.file_path),
        year=record.year,
        quarter=quarter,
        event_trading_day=event_day.strftime("%Y-%m-%d"),
        **row_kwargs,
    )


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def build_event_metadata(
    dataset_dir: Path = DATASET_DIR,
    output_path: Path = EVENT_METADATA_PATH,
) -> pd.DataFrame:
    """Scan all transcripts → parse headers → assign trading days → save CSV.

    Returns
    -------
    pd.DataFrame with one row per earnings call (188 rows for the full dataset).
    """
    ensure_output_dirs()

    from nasdaq_nlp.data.loader.original import scan_original_transcripts as scan_transcripts

    records = scan_transcripts(dataset_dir)
    print(f"Found {len(records)} transcripts across {len({r.ticker for r in records})} tickers")

    rows: list[dict] = []
    for record in records:
        try:
            meta = _record_to_metadata_row(record)
            # Use .model_dump() (Pydantic v2) instead of dataclass asdict()
            rows.append(meta.model_dump())
        except Exception as exc:
            print(f"  WARN: skipping {record.file_name}: {exc}")

    df = pd.DataFrame(rows)
    df["event_trading_day"] = pd.to_datetime(df["event_trading_day"])
    df.sort_values(["ticker", "event_trading_day"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_csv(output_path, index=False)
    print(f"Saved event metadata → {output_path}  ({len(df)} rows)")

    return df
