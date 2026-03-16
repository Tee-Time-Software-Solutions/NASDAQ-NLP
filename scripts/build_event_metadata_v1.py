from pathlib import Path
import pandas as pd
import re
from datetime import datetime, timedelta
import calendar


RAW_TRANSCRIPTS_DIR = Path("data/raw/earnings-call-transcripts/Transcripts")
OUTPUT_PATH = Path("data/processed/event_metadata_v1_rawrule.csv")


FILENAME_PATTERN = re.compile(
    r"(?P<year>\d{4})-(?P<month>[A-Za-z]{3})-(?P<day>\d{2})-(?P<ticker>[A-Z]+)\.txt$"
)

HEADER_DATETIME_PATTERN = re.compile(
    r"([A-Z]+ \d{1,2}, \d{4} / \d{1,2}:\d{2}[AP]M GMT)"
)


def next_business_day(date_obj: datetime) -> datetime:
    next_day = date_obj + timedelta(days=1)
    while next_day.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        next_day += timedelta(days=1)
    return next_day


def quarter_from_month(month: int) -> str:
    if month in [1, 2, 3]:
        return "Q1"
    elif month in [4, 5, 6]:
        return "Q2"
    elif month in [7, 8, 9]:
        return "Q3"
    else:
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
    call_date = datetime(year, month_num, day)

    return {
        "ticker": ticker,
        "file_name": file_path.name,
        "file_path": str(file_path),
        "call_date_from_filename": call_date.date().isoformat(),
        "year": year,
        "quarter_label": quarter_from_month(month_num),
    }


def parse_header_datetime(text: str) -> tuple[str | None, datetime | None]:
    match = HEADER_DATETIME_PATTERN.search(text)
    if not match:
        return None, None

    raw_value = match.group(1)
    parsed_value = datetime.strptime(raw_value, "%B %d, %Y / %I:%M%p GMT")
    return raw_value, parsed_value


def estimate_event_trading_day(call_dt_gmt: datetime | None, fallback_date: str) -> str:
    """
    Approximation:
    - 4:00 PM ET is roughly 8:00 PM GMT during daylight saving time
    - 4:00 PM ET is roughly 9:00 PM GMT during standard time

    For a first-pass metadata file, use a simple threshold:
    if GMT hour >= 21, move to next business day
    else same day
    """
    if call_dt_gmt is None:
        return fallback_date

    event_day = call_dt_gmt

    if call_dt_gmt.hour >= 21:
        event_day = next_business_day(call_dt_gmt)

    return event_day.date().isoformat()


def build_event_metadata() -> pd.DataFrame:
    rows = []

    transcript_files = sorted(RAW_TRANSCRIPTS_DIR.rglob("*.txt"))

    for file_path in transcript_files:
        try:
            meta = parse_filename(file_path)

            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

            raw_header_dt, parsed_header_dt = parse_header_datetime(text)

            row = {
                **meta,
                "call_datetime_header_raw": raw_header_dt,
                "call_datetime_parsed": parsed_header_dt.isoformat() if parsed_header_dt else None,
                "call_time_gmt": parsed_header_dt.strftime("%H:%M:%S") if parsed_header_dt else None,
                "event_trading_day": estimate_event_trading_day(
                    parsed_header_dt,
                    meta["call_date_from_filename"]
                ),
            }

            rows.append(row)

        except Exception as e:
            rows.append({
                "ticker": None,
                "file_name": file_path.name,
                "file_path": str(file_path),
                "call_date_from_filename": None,
                "year": None,
                "quarter_label": None,
                "call_datetime_header_raw": None,
                "call_datetime_parsed": None,
                "call_time_gmt": None,
                "event_trading_day": None,
                "error": str(e),
            })

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = build_event_metadata()
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved event metadata to: {OUTPUT_PATH}")
    print(f"Rows: {len(df)}")
    print(df.head())