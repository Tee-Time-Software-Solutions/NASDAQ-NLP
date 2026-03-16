from pathlib import Path
import calendar
import re

import pandas as pd
from pandas.tseries.offsets import BDay


RAW_TRANSCRIPTS_DIR = Path("data/raw/earnings-call-transcripts/Transcripts")
OUTPUT_PATH = Path("data/processed/event_metadata_final.csv")

FILENAME_PATTERN = re.compile(
    r"(?P<year>\d{4})-(?P<month>[A-Za-z]{3})-(?P<day>\d{2})-(?P<ticker>[A-Z]+)\.txt$"
)

HEADER_DATETIME_PATTERN = re.compile(
    r"([A-Z]+ \d{1,2}, \d{4} / \d{1,2}:\d{2}[AP]M GMT)"
)


def quarter_from_month(month: int) -> str:
    if month in [1, 2, 3]:
        return "Q1"
    if month in [4, 5, 6]:
        return "Q2"
    if month in [7, 8, 9]:
        return "Q3"
    return "Q4"


def parse_filename(file_path: Path) -> dict:
    match = FILENAME_PATTERN.search(file_path.name)
    if not match:
        raise ValueError(f"Filename does not match expected pattern: {file_path.name}")

    year = int(match.group("year"))
    month_abbr = match.group("month")
    day = int(match.group("day"))
    ticker = match.group("ticker")

    month_num = list(calendar.month_abbr).index(month_abbr)
    if month_num == 0:
        raise ValueError(f"Invalid month abbreviation in filename: {file_path.name}")

    call_date = pd.Timestamp(year=year, month=month_num, day=day)

    return {
        "ticker": ticker,
        "file_name": file_path.name,
        "file_path": str(file_path),
        "call_date_from_filename": call_date.date().isoformat(),
        "year": year,
        "quarter_label": quarter_from_month(month_num),
    }


def parse_header_datetime(text: str) -> tuple[str | None, pd.Timestamp | None]:
    match = HEADER_DATETIME_PATTERN.search(text)
    if not match:
        return None, None

    raw_value = match.group(1)

    parsed_value = pd.to_datetime(
        raw_value,
        format="%B %d, %Y / %I:%M%p GMT",
        errors="coerce",
    )

    if pd.isna(parsed_value):
        return raw_value, None

    parsed_value = parsed_value.tz_localize("UTC")
    return raw_value, parsed_value


def compute_event_fields(
    call_dt_gmt: pd.Timestamp | None,
    fallback_date: str,
) -> dict:
    """
    Compute timezone-aware event fields using America/New_York.

    Pipeline rule:
    - Convert GMT timestamp to New York time
    - If the call occurs strictly after 4:00 PM ET,
      assign the event to the next business day
    - Otherwise assign the event to the same calendar day in ET

    Notes:
    - BDay handles weekends, but not exchange holidays
    - final holiday-safe reconciliation should happen when merging
      against actual stock-price dates
    """
    if call_dt_gmt is None or pd.isna(call_dt_gmt):
        fallback_ts = pd.Timestamp(fallback_date)
        return {
            "call_datetime_gmt": None,
            "call_datetime_et": None,
            "call_time_gmt": None,
            "call_time_et": None,
            "after_market_close_et": None,
            "event_trading_day_final": fallback_ts.date().isoformat(),
        }

    call_dt_et = call_dt_gmt.tz_convert("America/New_York")

    market_close_et = call_dt_et.normalize() + pd.Timedelta(hours=16)
    after_market_close_et = call_dt_et > market_close_et

    if after_market_close_et:
        event_day = (call_dt_et.normalize() + BDay(1)).date().isoformat()
    else:
        event_day = call_dt_et.date().isoformat()

    return {
        "call_datetime_gmt": call_dt_gmt.isoformat(),
        "call_datetime_et": call_dt_et.isoformat(),
        "call_time_gmt": call_dt_gmt.strftime("%H:%M:%S"),
        "call_time_et": call_dt_et.strftime("%H:%M:%S"),
        "after_market_close_et": bool(after_market_close_et),
        "event_trading_day_final": event_day,
    }


def build_event_metadata() -> pd.DataFrame:
    rows = []
    transcript_files = sorted(RAW_TRANSCRIPTS_DIR.rglob("*.txt"))

    for file_path in transcript_files:
        try:
            meta = parse_filename(file_path)

            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

            raw_header_dt, parsed_header_dt = parse_header_datetime(text)
            event_fields = compute_event_fields(
                parsed_header_dt,
                meta["call_date_from_filename"],
            )

            row = {
                **meta,
                "call_datetime_header_raw": raw_header_dt,
                "call_datetime_parsed": (
                    parsed_header_dt.isoformat() if parsed_header_dt is not None else None
                ),
                **event_fields,
                "error": None,
            }

            rows.append(row)

        except Exception as e:
            rows.append(
                {
                    "ticker": None,
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "call_date_from_filename": None,
                    "year": None,
                    "quarter_label": None,
                    "call_datetime_header_raw": None,
                    "call_datetime_parsed": None,
                    "call_datetime_gmt": None,
                    "call_datetime_et": None,
                    "call_time_gmt": None,
                    "call_time_et": None,
                    "after_market_close_et": None,
                    "event_trading_day_final": None,
                    "error": str(e),
                }
            )

    df = pd.DataFrame(rows)

    column_order = [
        "ticker",
        "file_name",
        "file_path",
        "call_date_from_filename",
        "year",
        "quarter_label",
        "call_datetime_header_raw",
        "call_datetime_parsed",
        "call_datetime_gmt",
        "call_datetime_et",
        "call_time_gmt",
        "call_time_et",
        "after_market_close_et",
        "event_trading_day_final",
        "error",
    ]

    return df[column_order].sort_values(
        ["ticker", "call_date_from_filename", "file_name"]
    ).reset_index(drop=True)


if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = build_event_metadata()
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved final event metadata to: {OUTPUT_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Rows with errors: {df['error'].notna().sum()}")
    print(df.head())