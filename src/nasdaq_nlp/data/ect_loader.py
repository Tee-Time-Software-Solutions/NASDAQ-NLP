"""
data/ect_loader.py — Import the cleaned_ECTs Kaggle dataset into the pipeline.

Dataset structure (already downloaded):
    cleaned_ECTs_dataset/
        Apple/
            2018_Q1_aapl_processed.txt
            2018_Q2_aapl_processed.txt
            ...
        Cisco/
            ...

File naming: {year}_Q{quarter}_{ticker_lower}_processed.txt

MARKET INDEX MAPPING
---------------------
Stocks belong to different exchanges and should be benchmarked against different indices.
We record the correct index per ticker in the metadata so market_model.py can use it.

    NASDAQ stocks  → ^IXIC (NASDAQ Composite)
    NYSE stocks    → ^GSPC (S&P 500)
    Non-US stocks  → excluded by default (their index is not in our pipeline)

EVENT DATE ESTIMATION
----------------------
The ECT filenames have year+quarter but no exact date.
We use yfinance get_earnings_dates() to get the real date; if that fails we fall
back to the last business day of the reporting month for that quarter:
    Q1 (Jan–Mar) reports in late April → April 30
    Q2 (Apr–Jun) reports in late July  → July 31
    Q3 (Jul–Sep) reports in late Oct   → October 31
    Q4 (Oct–Dec) reports in late Jan   → January 31 (year+1)

HOW TO USE
-----------
    from nasdaq_nlp.data.ect_loader import build_ect_metadata
    ect_df = build_ect_metadata()   # saves to outputs/processed/ect_event_metadata.csv

Then call build_combined_metadata() to merge it with the original dataset.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from nasdaq_nlp.config import PROCESSED_DIR, ensure_output_dirs
from nasdaq_nlp.data.metadata.schemas import TRADING_HOLIDAYS

# ---------------------------------------------------------------------------
# Dataset location
# ---------------------------------------------------------------------------

ECT_DATASET_DIR: Path = (
    Path.home() / "Downloads" / "archive (1)" / "cleaned_ECTs_dataset"
)
ECT_METADATA_PATH: Path = PROCESSED_DIR / "ect_event_metadata.csv"

# ---------------------------------------------------------------------------
# Ticker / exchange mapping
# ---------------------------------------------------------------------------
# company folder name → (yfinance ticker, exchange)
# exchange is "NASDAQ" or "NYSE"; non-US companies are excluded (no entry here).

COMPANY_TO_TICKER: dict[str, tuple[str, str]] = {
    # ── NASDAQ ────────────────────────────────────────────────────────────────
    "AMD":            ("AMD",   "NASDAQ"),
    "Adobe":          ("ADBE",  "NASDAQ"),
    "Alphabet":       ("GOOGL", "NASDAQ"),
    "Amazon":         ("AMZN",  "NASDAQ"),
    "Apple":          ("AAPL",  "NASDAQ"),
    "Cisco":          ("CSCO",  "NASDAQ"),
    "Costco":         ("COST",  "NASDAQ"),
    "META":           ("META",  "NASDAQ"),
    "Microsoft":      ("MSFT",  "NASDAQ"),
    "Nvidia":         ("NVDA",  "NASDAQ"),
    "Lululemon":      ("LULU",  "NASDAQ"),
    "PAYPAL":         ("PYPL",  "NASDAQ"),
    # ── NYSE ──────────────────────────────────────────────────────────────────
    "AXP":            ("AXP",   "NYSE"),
    "Bank of America":("BAC",   "NYSE"),
    "Citi":           ("C",     "NYSE"),
    "Ford":           ("F",     "NYSE"),
    "GM":             ("GM",    "NYSE"),
    "IBM":            ("IBM",   "NYSE"),
    "JPM":            ("JPM",   "NYSE"),
    "Mastercard":     ("MA",    "NYSE"),
    "Marriott":       ("MAR",   "NYSE"),
    "Nike":           ("NKE",   "NYSE"),
    "Oracle":         ("ORCL",  "NYSE"),
    "SalesForce":     ("CRM",   "NYSE"),
    "UnitedHealth":   ("UNH",   "NYSE"),
    "Walt Disney":    ("DIS",   "NYSE"),
    "Walmart":        ("WMT",   "NYSE"),
    # ── European (Frankfurt/Paris/London) ────────────────────────────────────
    # Using local exchange tickers — benchmarked against their national index.
    # These prices are in local currency (EUR/GBP) which is correct because the
    # market model only uses *returns* (percentage changes), not price levels.
    "Airbus":          ("AIR.PA",  "CAC40"),        # Euronext Paris
    "BMW":             ("BMW.DE",  "DAX"),           # Frankfurt
    "BP":              ("BP.L",    "FTSE100"),       # London
    "Deutsche Bank":   ("DBK.DE",  "DAX"),           # Frankfurt
    "EDF":             ("EDF.PA",  "CAC40"),         # Paris
    "Engie":           ("ENGI.PA", "CAC40"),         # Paris
    "Loreal":          ("OR.PA",   "CAC40"),         # Paris
    "Louis Vuitton":   ("MC.PA",   "CAC40"),         # LVMH, Paris
    "Shell":           ("SHEL.L",  "FTSE100"),       # London
    "SIE":             ("SIE.DE",  "DAX"),           # Siemens, Frankfurt
    "Schneider Electric": ("SU.PA","CAC40"),         # Paris
    # ── Other / mixed ─────────────────────────────────────────────────────────
    "Accenture":       ("ACN",    "NYSE"),
    "Allstate":        ("ALL",    "NYSE"),
    "BKNG":            ("BKNG",   "NASDAQ"),
    "BAM":             ("BAM",    "NYSE"),            # Brookfield
    "Branco":          ("BBAS3.SA","BOVESPA"),        # Brazilian — excluded below
    "BBVA":            ("BBVA",   "NYSE"),            # NYSE ADR
    "Cardinal Health": ("CAH",    "NYSE"),
    "Compass Group":   ("CPG.L",  "FTSE100"),
    "Elevance Health": ("ELV",    "NYSE"),
    "Amadeus":         ("AMS.MC", "IBEX35"),         # Madrid
    "Capgemini":       ("CAP.PA", "CAC40"),
    "Volvo":           ("VOLV-B.ST","OMX"),          # Stockholm — excluded below
}

# Which yfinance index ticker to use per exchange
INDEX_FOR_EXCHANGE: dict[str, str] = {
    "NASDAQ":  "^IXIC",
    "NYSE":    "^GSPC",
    "DAX":     "^GDAXI",
    "CAC40":   "^FCHI",
    "FTSE100": "^FTSE",
    "IBEX35":  "^IBEX",
    # Exclude exchanges where yfinance coverage is unreliable
    "BOVESPA": None,   # Brazilian stocks — skip
    "OMX":     None,   # Nordic — skip
}

# Exchanges to exclude entirely (index = None above means skip)
_EXCLUDED_EXCHANGES = {k for k, v in INDEX_FOR_EXCHANGE.items() if v is None}

# Quarter → (reporting month, day) — approximate earnings release date
# Q1 earnings (Jan–Mar period) → released ~late April
_QUARTER_REPORT_DATE: dict[str, tuple[int, int]] = {
    "Q1": (4, 30),   # April 30
    "Q2": (7, 31),   # July 31
    "Q3": (10, 31),  # October 31
    "Q4": (1, 31),   # January 31 of year+1
}

_FILENAME_RE = re.compile(r"^(\d{4})_Q(\d)_(\w+)_processed\.txt$")

# ---------------------------------------------------------------------------
# Date helpers (reuse the same logic as metadata/__init__.py)
# ---------------------------------------------------------------------------

def _next_trading_day(dt: datetime) -> datetime:
    """Advance to the next NYSE trading day (skip weekends and holidays)."""
    candidate = datetime(dt.year, dt.month, dt.day)
    while candidate.weekday() >= 5 or candidate in TRADING_HOLIDAYS:
        candidate += timedelta(days=1)
    return candidate


def _approx_event_date(year: int, quarter: str) -> datetime:
    """Best-guess trading day for a given year + quarter."""
    report_month, report_day = _QUARTER_REPORT_DATE[quarter]
    report_year = year + 1 if quarter == "Q4" else year
    try:
        dt = datetime(report_year, report_month, report_day)
    except ValueError:
        # e.g., Feb 30 → use last day of Feb
        dt = datetime(report_year, report_month, 28)
    return _next_trading_day(dt)


def _yfinance_event_date(ticker: str, year: int, quarter: str) -> datetime | None:
    """Try to get the real earnings date from yfinance."""
    try:
        import yfinance as yf
        dates = yf.Ticker(ticker).get_earnings_dates(limit=40)
        if dates is None or dates.empty:
            return None
        dates.index = pd.to_datetime(dates.index.tz_localize(None)
                                     if dates.index.tz is not None
                                     else dates.index)
        # Filter to the target year+quarter reporting window
        q_month, _ = _QUARTER_REPORT_DATE[quarter]
        q_year = year + 1 if quarter == "Q4" else year
        # Look for a date within ±3 months of the expected reporting month
        target = datetime(q_year, q_month, 15)
        for dt in sorted(dates.index):
            if abs((dt - target).days) < 60:
                return _next_trading_day(dt)
    except Exception:
        pass
    return None

# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

def scan_ect_transcripts(ect_dir: Path = ECT_DATASET_DIR) -> list[dict]:
    """Walk the ECTs directory and return one dict per transcript file.

    Returns
    -------
    list of dicts with keys: company, ticker, exchange, file_path, file_name,
                              year, quarter
    """
    if not ect_dir.exists():
        raise FileNotFoundError(
            f"ECT dataset not found at {ect_dir}.\n"
            "Set ECT_DATASET_DIR or move the folder to the expected location."
        )

    rows = []
    for company_dir in sorted(ect_dir.iterdir()):
        if not company_dir.is_dir():
            continue

        company = company_dir.name
        if company not in COMPANY_TO_TICKER:
            continue  # unmapped — skip

        ticker, exchange = COMPANY_TO_TICKER[company]
        if exchange in _EXCLUDED_EXCHANGES:
            continue  # exchange not reliably supported by yfinance

        for txt_file in sorted(company_dir.glob("*_processed.txt")):
            m = _FILENAME_RE.match(txt_file.name)
            if not m:
                continue
            year, q_num, _ticker_lower = int(m.group(1)), int(m.group(2)), m.group(3)
            quarter = f"Q{q_num}"

            rows.append({
                "company":     company,
                "ticker":      ticker,
                "exchange":    exchange,
                "market_index": INDEX_FOR_EXCHANGE[exchange],
                "file_path":   str(txt_file),
                "file_name":   txt_file.name,
                "year":        year,
                "quarter":     quarter,
            })

    return rows


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def build_ect_metadata(
    ect_dir: Path = ECT_DATASET_DIR,
    output_path: Path = ECT_METADATA_PATH,
    use_yfinance: bool = True,
) -> pd.DataFrame:
    """Scan ECTs → resolve event trading days → save ect_event_metadata.csv.

    Parameters
    ----------
    use_yfinance : bool
        If True, try to fetch the real earnings date from yfinance first.
        Falls back to quarter-end approximation if yfinance fails or has no data.

    Returns
    -------
    pd.DataFrame in the same column format as event_metadata.csv, plus
    an `exchange` column (NASDAQ / NYSE) and `market_index` column (^IXIC / ^GSPC).
    """
    ensure_output_dirs()

    raw = scan_ect_transcripts(ect_dir)
    print(f"Found {len(raw)} ECT transcripts across "
          f"{len({r['ticker'] for r in raw})} tickers")

    rows = []
    for r in raw:
        if use_yfinance:
            event_day = _yfinance_event_date(r["ticker"], r["year"], r["quarter"])
            if event_day:
                source = "yfinance"
            else:
                event_day = _approx_event_date(r["year"], r["quarter"])
                source = "approx"
        else:
            event_day = _approx_event_date(r["year"], r["quarter"])
            source = "approx"

        rows.append({
            "ticker":            r["ticker"],
            "file_name":         r["file_name"],
            "file_path":         r["file_path"],
            "year":              r["year"],
            "quarter":           r["quarter"],
            "exchange":          r["exchange"],
            "market_index":      r["market_index"],
            "event_trading_day": event_day.strftime("%Y-%m-%d"),
            "date_source":       source,
            # Fields present in original metadata — not available for ECT
            "call_datetime_gmt": None,
            "call_datetime_et":  None,
            "call_time_et":      None,
            "after_market_close": False,
        })

    df = pd.DataFrame(rows)
    df["event_trading_day"] = pd.to_datetime(df["event_trading_day"])
    df.sort_values(["ticker", "event_trading_day"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_csv(output_path, index=False)
    print(f"Saved ECT metadata → {output_path}  ({len(df)} rows)")
    yf_count = (df["date_source"] == "yfinance").sum()
    print(f"  Event dates: {yf_count} from yfinance, {len(df) - yf_count} approximated")
    print(f"  Exchanges: {df['exchange'].value_counts().to_dict()}")

    return df


# ---------------------------------------------------------------------------
# Merge with original metadata
# ---------------------------------------------------------------------------

def build_combined_metadata(
    original_path: Path = PROCESSED_DIR / "event_metadata.csv",
    ect_path: Path = ECT_METADATA_PATH,
    output_path: Path = PROCESSED_DIR / "combined_event_metadata.csv",
) -> pd.DataFrame:
    """Concatenate original + ECT metadata, deduplicate by (ticker, event_trading_day).

    The original dataset has no exchange/market_index columns — these are filled in
    using COMPANY_TO_TICKER so the market model uses the right benchmark.

    Returns
    -------
    Combined DataFrame written to combined_event_metadata.csv.
    """
    orig = pd.read_csv(original_path)

    # Tag original events with their exchange + market index
    _ticker_to_exchange = {t: ex for (t, ex) in COMPANY_TO_TICKER.values()}
    orig["exchange"] = orig["ticker"].map(_ticker_to_exchange).fillna("NASDAQ")
    orig["market_index"] = orig["exchange"].map(INDEX_FOR_EXCHANGE).fillna("^IXIC")
    orig["date_source"] = "original"

    ect = pd.read_csv(ect_path)

    combined = pd.concat([orig, ect], ignore_index=True)
    combined["event_trading_day"] = pd.to_datetime(combined["event_trading_day"])

    # Deduplicate: same ticker + same trading day → keep original if available
    combined.sort_values("date_source", ascending=True, inplace=True)  # "original" < "yfinance" < "approx"
    combined.drop_duplicates(subset=["ticker", "event_trading_day"], keep="first", inplace=True)
    combined.sort_values(["ticker", "event_trading_day"], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    combined.to_csv(output_path, index=False)
    print(f"Combined metadata → {output_path}  ({len(combined)} rows, "
          f"{len(combined['ticker'].unique())} tickers)")
    return combined
