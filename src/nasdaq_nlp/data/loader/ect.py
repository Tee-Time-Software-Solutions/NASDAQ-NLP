"""
loader/ect.py — Scan the cleaned_ECTs Kaggle dataset and build event metadata.

File naming: {year}_Q{quarter}_{ticker_lower}_processed.txt

MARKET INDEX MAPPING
---------------------
    NASDAQ stocks  → ^IXIC (NASDAQ Composite)
    NYSE stocks    → ^GSPC (S&P 500)

EVENT DATE ESTIMATION
----------------------
ECT filenames have year+quarter but no exact date.
We use yfinance get_earnings_dates() when available; otherwise we fall back to
the last business day of the reporting month for that quarter:
    Q1 (Jan–Mar) reports in late April → April 30
    Q2 (Apr–Jun) reports in late July  → July 31
    Q3 (Jul–Sep) reports in late Oct   → October 31
    Q4 (Oct–Dec) reports in late Jan   → January 31 (year+1)

Public API:
    scan_ect_transcripts(ect_dir)            → list[TranscriptRecord]
    build_ect_metadata(ect_dir, output_path) → pd.DataFrame
    build_combined_metadata(...)             → pd.DataFrame
    COMPANY_TO_TICKER, INDEX_FOR_EXCHANGE    (constants)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from nasdaq_nlp.config import PROCESSED_DIR, ensure_output_dirs
from nasdaq_nlp.data.loader.schemas import ECT_FILENAME_RE, TranscriptRecord
from nasdaq_nlp.data.metadata.schemas import TRADING_HOLIDAYS

__all__ = [
    "COMPANY_TO_TICKER",
    "INDEX_FOR_EXCHANGE",
    "scan_ect_transcripts",
    "build_ect_metadata",
    "build_combined_metadata",
]

# ---------------------------------------------------------------------------
# Dataset location
# ---------------------------------------------------------------------------

ECT_DATASET_DIR: Path = Path.home() / "Downloads" / "archive (1)" / "cleaned_ECTs_dataset"
ECT_METADATA_PATH: Path = PROCESSED_DIR / "ect_event_metadata.csv"

# ---------------------------------------------------------------------------
# Ticker / exchange / index mapping
# ---------------------------------------------------------------------------

COMPANY_TO_TICKER: dict[str, tuple[str, str]] = {
    # ── NASDAQ ───────────────────────────────────────────────────────────────
    "AMD": ("AMD", "NASDAQ"),
    "Adobe": ("ADBE", "NASDAQ"),
    "Alphabet": ("GOOGL", "NASDAQ"),
    "Amazon": ("AMZN", "NASDAQ"),
    "Apple": ("AAPL", "NASDAQ"),
    "Cisco": ("CSCO", "NASDAQ"),
    "Costco": ("COST", "NASDAQ"),
    "META": ("META", "NASDAQ"),
    "Microsoft": ("MSFT", "NASDAQ"),
    "Nvidia": ("NVDA", "NASDAQ"),
    "Lululemon": ("LULU", "NASDAQ"),
    "PayPal": ("PYPL", "NASDAQ"),
    "Booking": ("BKNG", "NASDAQ"),
    "Qualcomm": ("QCOM", "NASDAQ"),
    "Starbucks": ("SBUX", "NASDAQ"),
    "Netflix": ("NFLX", "NASDAQ"),
    "Intel": ("INTC", "NASDAQ"),
    "Micron": ("MU", "NASDAQ"),
    # ── NYSE ─────────────────────────────────────────────────────────────────
    "IBM": ("IBM", "NYSE"),
    "Oracle": ("ORCL", "NYSE"),
    "Salesforce": ("CRM", "NYSE"),
    "Accenture": ("ACN", "NYSE"),
    # ── European / other (excluded — no reliable yfinance index coverage) ───
    "ASML": ("ASML", "BOVESPA"),
    "SAP": ("SAP", "DAX"),
    "Siemens": ("SIE", "DAX"),
    "Allianz": ("ALV", "DAX"),
    "LVMH": ("MC", "CAC40"),
    "TotalEnergies": ("TTE", "CAC40"),
    "AXA": ("CS", "CAC40"),
    "HSBC": ("HSBA", "FTSE100"),
    "BP": ("BP", "FTSE100"),
    "Shell": ("SHEL", "FTSE100"),
    "Vodafone": ("VOD", "FTSE100"),
    "Inditex": ("ITX", "IBEX35"),
    "Banco_Santander": ("SAN", "IBEX35"),
}

INDEX_FOR_EXCHANGE: dict[str, str | None] = {
    "NASDAQ": "^IXIC",
    "NYSE": "^GSPC",
    "DAX": "^GDAXI",
    "CAC40": "^FCHI",
    "FTSE100": "^FTSE",
    "IBEX35": "^IBEX",
    "BOVESPA": None,  # unreliable yfinance coverage → excluded
    "OMX": None,
}

_EXCLUDED_EXCHANGES = {k for k, v in INDEX_FOR_EXCHANGE.items() if v is None}

# Quarter → (report_month, report_day) — last day of reporting month
_QUARTER_REPORT_DATE = {"Q1": (4, 30), "Q2": (7, 31), "Q3": (10, 31), "Q4": (1, 31)}


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------


def _next_trading_day(dt: datetime) -> datetime:
    while dt.weekday() >= 5 or datetime(dt.year, dt.month, dt.day) in TRADING_HOLIDAYS:
        dt += timedelta(days=1)
    return dt


def _approx_event_date(year: int, quarter: str) -> datetime:
    report_month, report_day = _QUARTER_REPORT_DATE[quarter]
    report_year = year + 1 if quarter == "Q4" else year
    try:
        dt = datetime(report_year, report_month, report_day)
    except ValueError:
        dt = datetime(report_year, report_month, 28)
    return _next_trading_day(dt)


def _yfinance_event_date(ticker: str, year: int, quarter: str) -> datetime | None:
    try:
        import yfinance as yf

        dates = yf.Ticker(ticker).get_earnings_dates(limit=40)
        if dates is None or dates.empty:
            return None
        dates.index = pd.to_datetime(
            dates.index.tz_localize(None) if dates.index.tz is not None else dates.index
        )
        q_month, _ = _QUARTER_REPORT_DATE[quarter]
        q_year = year + 1 if quarter == "Q4" else year
        target = datetime(q_year, q_month, 15)
        for dt in sorted(dates.index):
            if abs((dt - target).days) < 60:
                return _next_trading_day(dt)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Scanner — returns TranscriptRecord objects (same type as original source)
# ---------------------------------------------------------------------------


def scan_ect_transcripts(
    ect_dir: Path = ECT_DATASET_DIR,
) -> list[TranscriptRecord]:
    """Walk the ECT directory and return one TranscriptRecord per file.

    Files whose company folder is not in COMPANY_TO_TICKER, or whose exchange
    is in _EXCLUDED_EXCHANGES, are silently skipped.
    """
    if not ect_dir.exists():
        raise FileNotFoundError(
            f"ECT dataset not found at {ect_dir}.\n"
            "Download from Kaggle and place at the expected location."
        )

    records: list[TranscriptRecord] = []
    for company_dir in sorted(ect_dir.iterdir()):
        if not company_dir.is_dir():
            continue
        company = company_dir.name
        if company not in COMPANY_TO_TICKER:
            continue
        ticker, exchange = COMPANY_TO_TICKER[company]
        if exchange in _EXCLUDED_EXCHANGES:
            continue

        market_index = INDEX_FOR_EXCHANGE[exchange]
        for txt_file in sorted(company_dir.glob("*_processed.txt")):
            m = ECT_FILENAME_RE.match(txt_file.name)
            if not m:
                continue
            year = int(m.group("year"))
            quarter = f"Q{m.group('quarter')}"
            records.append(
                TranscriptRecord(
                    ticker=ticker,
                    year=year,
                    quarter=quarter,
                    exchange=exchange,
                    market_index=market_index,
                    file_name=txt_file.name,
                    file_path=txt_file.resolve(),
                    source="ect",
                )
            )

    records.sort(key=lambda r: (r.ticker, r.year, r.quarter))
    return records


# ---------------------------------------------------------------------------
# Pipeline step: build ect_event_metadata.csv
# ---------------------------------------------------------------------------


def build_ect_metadata(
    ect_dir: Path = ECT_DATASET_DIR,
    output_path: Path = ECT_METADATA_PATH,
    use_yfinance: bool = True,
) -> pd.DataFrame:
    """Scan ECT transcripts → resolve event trading days → save CSV."""
    ensure_output_dirs()
    records = scan_ect_transcripts(ect_dir)
    print(f"Found {len(records)} ECT transcripts across {len({r.ticker for r in records})} tickers")

    rows = []
    for r in records:
        if use_yfinance:
            event_day = _yfinance_event_date(r.ticker, r.year, r.quarter)
            source = "yfinance" if event_day else "approx"
            event_day = event_day or _approx_event_date(r.year, r.quarter)
        else:
            event_day = _approx_event_date(r.year, r.quarter)
            source = "approx"

        rows.append(
            {
                "ticker": r.ticker,
                "file_name": r.file_name,
                "file_path": str(r.file_path),
                "year": r.year,
                "quarter": r.quarter,
                "exchange": r.exchange,
                "market_index": r.market_index,
                "event_trading_day": event_day.strftime("%Y-%m-%d"),
                "date_source": source,
                "call_datetime_gmt": None,
                "call_datetime_et": None,
                "call_time_et": None,
                "after_market_close": False,
            }
        )

    df = pd.DataFrame(rows)
    df["event_trading_day"] = pd.to_datetime(df["event_trading_day"])
    df.sort_values(["ticker", "event_trading_day"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(output_path, index=False)

    print(f"Saved ECT metadata → {output_path}  ({len(df)} rows)")
    yf_count = (df["date_source"] == "yfinance").sum()
    print(f"  {yf_count} dates from yfinance, {len(df) - yf_count} approximated")
    return df


# ---------------------------------------------------------------------------
# Pipeline step: combine original + ECT metadata
# ---------------------------------------------------------------------------


def build_combined_metadata(
    original_path: Path = PROCESSED_DIR / "event_metadata.csv",
    ect_path: Path = ECT_METADATA_PATH,
    output_path: Path = PROCESSED_DIR / "combined_event_metadata.csv",
) -> pd.DataFrame:
    """Merge original + ECT metadata, dedup by (ticker, event_trading_day)."""
    orig = pd.read_csv(original_path)

    _ticker_to_exchange = {t: ex for (t, ex) in COMPANY_TO_TICKER.values()}
    orig["exchange"] = orig["ticker"].map(_ticker_to_exchange).fillna("NASDAQ")
    orig["market_index"] = orig["exchange"].map(INDEX_FOR_EXCHANGE).fillna("^IXIC")
    orig["date_source"] = "original"

    ect = pd.read_csv(ect_path)
    combined = pd.concat([orig, ect], ignore_index=True)
    combined["event_trading_day"] = pd.to_datetime(combined["event_trading_day"])

    # Keep original over ECT when same ticker+day
    combined.sort_values("date_source", ascending=True, inplace=True)
    combined.drop_duplicates(subset=["ticker", "event_trading_day"], keep="first", inplace=True)
    combined.sort_values(["ticker", "event_trading_day"], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    combined.to_csv(output_path, index=False)
    print(
        f"Combined metadata → {output_path}  "
        f"({len(combined)} rows, {len(combined['ticker'].unique())} tickers)"
    )
    return combined
