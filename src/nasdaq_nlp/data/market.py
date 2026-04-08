"""
market.py — Download historical stock prices from Yahoo Finance and compute daily returns.

WHAT THIS MODULE DOES
----------------------
1. Reads the event metadata CSV to find which tickers and date range we need.
2. Downloads Adjusted Close prices from yfinance for each stock and the NASDAQ index.
   - "Adjusted Close" accounts for stock splits and dividends, so returns are
     not distorted by corporate actions.
3. Computes the simple daily return for each stock and the index:

       R_t = (P_t - P_{t-1}) / P_{t-1}   =   P_t / P_{t-1}  - 1

   This is the percentage price change from one day to the next.
   We use "simple" returns (not log returns) because they add up nicely over
   short windows (which is what we need for CAR = AR_0 + AR_1 + ...).

OUTPUT FILES
-------------
outputs/processed/stock_returns.csv   — daily returns for each stock
outputs/processed/index_returns.csv   — daily returns for NASDAQ (^IXIC)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from nasdaq_nlp.config import (
    EVENT_METADATA_PATH,
    INDEX_RAW_PATH,
    INDEX_RETURNS_PATH,
    PRICES_RAW_PATH,
    STOCK_RETURNS_PATH,
    ensure_output_dirs,
)

# NASDAQ Composite index ticker on Yahoo Finance
NASDAQ_INDEX_TICKER = "^IXIC"

# Extra calendar days to download beyond the event window.
# We need 120 trading days BEFORE the earliest call for the estimation window,
# plus some buffer for weekends/holidays. 200 calendar days ≈ 140 trading days.
DOWNLOAD_BUFFER_DAYS = 200


def _compute_returns(prices_df: pd.DataFrame, ticker_col: str | None = None) -> pd.DataFrame:
    """Compute simple daily returns from an Adjusted Close price series.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Must have a DatetimeIndex and either:
        - a 'ticker' column + 'adj_close' column (for stocks), or
        - just 'adj_close' (for the index).
    ticker_col : str | None
        If provided, group by this column before computing returns
        (each ticker is its own price series, so we must not diff across tickers).

    Returns
    -------
    pd.DataFrame with a new 'return' column (NaN for first row of each group).
    """
    df = prices_df.copy()

    if ticker_col is not None:
        # Compute return within each ticker group separately
        # pct_change() = (P_t - P_{t-1}) / P_{t-1}
        df["return"] = df.groupby(ticker_col)["adj_close"].pct_change()
    else:
        df["return"] = df["adj_close"].pct_change()

    return df


def download_stock_prices(
    tickers: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Download Adjusted Close prices for a list of tickers using yfinance.

    Downloads one ticker at a time and concatenates to avoid bulk-download API
    issues with yfinance 0.2.x (rate limits, timezone errors when mixing tickers).

    Returns
    -------
    pd.DataFrame with columns [date, ticker, adj_close] sorted by (ticker, date).
    """
    print(f"Downloading prices for: {tickers}")
    print(f"  Date range: {start_date.date()} → {end_date.date()}")

    dfs = []
    for ticker in tickers:
        print(f"  → {ticker}", end="", flush=True)
        try:
            hist = yf.Ticker(ticker).history(
                start=start_date,
                end=end_date,
                auto_adjust=True,
            )
            if hist.empty:
                print(f" EMPTY — skipping")
                continue

            close = hist[["Close"]].reset_index()
            close.columns = ["date", "adj_close"]
            close["ticker"] = ticker
            # Strip timezone info from index (tz-aware → tz-naive date for CSV compatibility)
            close["date"] = pd.to_datetime(close["date"]).dt.tz_localize(None)
            dfs.append(close.dropna(subset=["adj_close"])) # drop days where stock didnt trade (if any)
            print(f" ✓ ({len(close)} rows)")
        except Exception as exc:
            print(f" ERROR: {exc}")
            continue

    if not dfs:
        raise RuntimeError("No ticker data downloaded successfully.")

    result = pd.concat(dfs, ignore_index=True)
    result = result.sort_values(["ticker", "date"]).reset_index(drop=True)
    return result


def download_index_prices(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Download NASDAQ Composite (^IXIC) Adjusted Close prices."""
    print(f"Downloading NASDAQ index ({NASDAQ_INDEX_TICKER})")

    hist = yf.Ticker(NASDAQ_INDEX_TICKER).history(
        start=start_date,
        end=end_date,
        auto_adjust=True,
    )

    if hist.empty:
        raise RuntimeError("Failed to download NASDAQ index data.")

    df = hist[["Close"]].reset_index()
    df.columns = ["date", "adj_close"]
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values("date").reset_index(drop=True)

    return df


def build_market_returns(
    metadata_path: Path = EVENT_METADATA_PATH,
    stock_output: Path = STOCK_RETURNS_PATH,
    index_output: Path = INDEX_RETURNS_PATH,
    prices_raw_output: Path = PRICES_RAW_PATH,
    index_raw_output: Path = INDEX_RAW_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end: load metadata → download prices → compute returns → save CSVs.

    Returns
    -------
    (stock_returns_df, index_returns_df)
    """
    ensure_output_dirs()

    # --- Load event metadata to determine tickers and date range ---
    meta = pd.read_csv(metadata_path, parse_dates=["event_trading_day"])
    tickers = sorted(meta["ticker"].unique().tolist())

    # We need prices starting 200+ calendar days before the earliest event
    # (for the 120-trading-day estimation window) through 30 days after the last
    earliest_event = meta["event_trading_day"].min()
    latest_event = meta["event_trading_day"].max()
    start_date = earliest_event - pd.Timedelta(days=DOWNLOAD_BUFFER_DAYS)
    end_date = latest_event + pd.Timedelta(days=30)

    # --- Download ---
    stocks_raw = download_stock_prices(tickers, start_date, end_date)
    index_raw = download_index_prices(start_date, end_date)

    # Save raw price files (useful for debugging)
    stocks_raw.to_csv(prices_raw_output, index=False)
    index_raw.to_csv(index_raw_output, index=False)
    print(f"Saved raw prices → {prices_raw_output}  ({len(stocks_raw)} rows)")

    # --- Compute returns ---
    # Simple daily return: R_t = (P_t / P_{t-1}) - 1
    # pct_change() does this in one step; first row per ticker is NaN (no prior day)
    stock_returns = _compute_returns(stocks_raw, ticker_col="ticker")
    index_returns = _compute_returns(index_raw)

    # Drop the initial NaN row (first day per ticker has no prior price)
    stock_returns = stock_returns.dropna(subset=["return"]).reset_index(drop=True)
    index_returns = index_returns.dropna(subset=["return"]).reset_index(drop=True)

    # Save
    stock_returns.to_csv(stock_output, index=False)
    index_returns.to_csv(index_output, index=False)
    print(f"Saved stock returns  → {stock_output}  ({len(stock_returns)} rows)")
    print(f"Saved index returns  → {index_output}  ({len(index_returns)} rows)")

    # Quick sanity check
    null_count = stock_returns["return"].isna().sum()
    assert null_count == 0, f"Unexpected NaN returns: {null_count}"

    return stock_returns, index_returns
