import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
import json


def main():
    # -------------------------
    # Paths
    # -------------------------
    metadata_path = Path("data/processed/event_metadata_final.csv")
    raw_output_dir = Path("data/raw/market_data")
    log_output_dir = Path("outputs/logs")

    raw_output_dir.mkdir(parents=True, exist_ok=True)
    log_output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Load metadata
    # -------------------------
    metadata = pd.read_csv(metadata_path)

    required_cols = [
        "ticker",
        "event_trading_day_final",
        "after_market_close_et",
        "call_datetime_et",
    ]
    missing_cols = [col for col in required_cols if col not in metadata.columns]
    if missing_cols:
        raise ValueError(
            f"event_metadata_final.csv is missing required columns: {missing_cols}"
        )

    metadata["event_trading_day_final"] = pd.to_datetime(
        metadata["event_trading_day_final"], errors="coerce"
    )

    if metadata["event_trading_day_final"].isna().any():
        bad_cols = [
            c for c in ["ticker", "file_name", "event_trading_day_final"]
            if c in metadata.columns
        ]
        bad_rows = metadata.loc[
            metadata["event_trading_day_final"].isna(),
            bad_cols,
        ]
        raise ValueError(
            "Some event_trading_day_final values could not be parsed.\n"
            f"{bad_rows.to_string(index=False)}"
        )

    # Keep only valid, unique tickers
    tickers = sorted(
        metadata["ticker"]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s.ne("")]
        .unique()
        .tolist()
    )

    if not tickers:
        raise ValueError("No valid tickers found in event_metadata_final.csv")

    # -------------------------
    # Define download range
    # -------------------------
    event_dates = metadata["event_trading_day_final"]

    start_date_dt = event_dates.min() - pd.Timedelta(days=200)
    end_date_dt = event_dates.max() + pd.Timedelta(days=30)

    start_date = start_date_dt.strftime("%Y-%m-%d")
    end_date = end_date_dt.strftime("%Y-%m-%d")

    # -------------------------
    # Download stock prices
    # -------------------------
    all_prices = []
    failed_tickers = []
    empty_tickers = []

    for ticker in tickers:
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=False,
                progress=False,
                actions=False,
                threads=False,
            )

            if df.empty:
                empty_tickers.append(ticker)
                print(f"[WARNING] No data returned for {ticker}")
                continue

            df = df.reset_index()

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [
                    "_".join([str(level) for level in col if str(level) != ""]).strip("_")
                    for col in df.columns
                ]

            rename_map = {}
            for col in df.columns:
                col_lower = col.lower()
                if col_lower.startswith("date"):
                    rename_map[col] = "Date"
                elif col_lower.startswith("open"):
                    rename_map[col] = "Open"
                elif col_lower.startswith("high"):
                    rename_map[col] = "High"
                elif col_lower.startswith("low"):
                    rename_map[col] = "Low"
                elif col_lower.startswith("close") and "adj" not in col_lower:
                    rename_map[col] = "Close"
                elif "adj close" in col_lower or col_lower.startswith("adj_close"):
                    rename_map[col] = "Adj Close"
                elif col_lower.startswith("volume"):
                    rename_map[col] = "Volume"

            df = df.rename(columns=rename_map)

            expected_core_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
            missing_price_cols = [col for col in expected_core_cols if col not in df.columns]
            if missing_price_cols:
                failed_tickers.append(
                    {
                        "ticker": ticker,
                        "reason": f"Missing expected columns: {missing_price_cols}",
                    }
                )
                print(f"[ERROR] {ticker}: missing columns {missing_price_cols}")
                continue

            if "Adj Close" not in df.columns:
                df["Adj Close"] = df["Close"]

            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"])
            df["ticker"] = ticker

            df = df[
                ["Date", "ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
            ].copy()

            all_prices.append(df)

        except Exception as e:
            failed_tickers.append({"ticker": ticker, "reason": str(e)})
            print(f"[ERROR] {ticker}: {e}")

    if not all_prices:
        raise RuntimeError("No stock price data was downloaded successfully.")

    prices = pd.concat(all_prices, ignore_index=True)
    prices = prices.sort_values(["ticker", "Date"]).drop_duplicates(
        subset=["ticker", "Date"], keep="last"
    )

    prices.to_csv(raw_output_dir / "prices_raw.csv", index=False)

    # -------------------------
    # Download NASDAQ index
    # -------------------------
    nasdaq = yf.download(
        "^IXIC",
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
        actions=False,
        threads=False,
    )

    if nasdaq.empty:
        raise RuntimeError("No NASDAQ index data was returned for ^IXIC")

    nasdaq = nasdaq.reset_index()

    if isinstance(nasdaq.columns, pd.MultiIndex):
        nasdaq.columns = [
            "_".join([str(level) for level in col if str(level) != ""]).strip("_")
            for col in nasdaq.columns
        ]

    rename_map = {}
    for col in nasdaq.columns:
        col_lower = col.lower()
        if col_lower.startswith("date"):
            rename_map[col] = "Date"
        elif col_lower.startswith("open"):
            rename_map[col] = "Open"
        elif col_lower.startswith("high"):
            rename_map[col] = "High"
        elif col_lower.startswith("low"):
            rename_map[col] = "Low"
        elif col_lower.startswith("close") and "adj" not in col_lower:
            rename_map[col] = "Close"
        elif "adj close" in col_lower or col_lower.startswith("adj_close"):
            rename_map[col] = "Adj Close"
        elif col_lower.startswith("volume"):
            rename_map[col] = "Volume"

    nasdaq = nasdaq.rename(columns=rename_map)

    if "Adj Close" not in nasdaq.columns and "Close" in nasdaq.columns:
        nasdaq["Adj Close"] = nasdaq["Close"]

    nasdaq["Date"] = pd.to_datetime(nasdaq["Date"], errors="coerce")
    nasdaq = nasdaq.dropna(subset=["Date"])
    nasdaq = nasdaq.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")

    nasdaq.to_csv(raw_output_dir / "nasdaq_index_raw.csv", index=False)

    # -------------------------
    # Audit log
    # -------------------------
    run_timestamp_utc = datetime.now(timezone.utc).isoformat()
    timestamp_slug = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    audit = {
        "run_timestamp_utc": run_timestamp_utc,
        "metadata_file": str(metadata_path),
        "raw_output_dir": str(raw_output_dir),
        "download_source": "yfinance / Yahoo Finance",
        "event_anchor_variable": "event_trading_day_final",
        "date_range_rule": {
            "pull_start": "min(event_trading_day_final) - 200 calendar days",
            "pull_end": "max(event_trading_day_final) + 30 calendar days",
        },
        "download_window": {
            "start_date": start_date,
            "end_date": end_date,
        },
        "event_date_summary": {
            "min_event_trading_day_final": event_dates.min().strftime("%Y-%m-%d"),
            "max_event_trading_day_final": event_dates.max().strftime("%Y-%m-%d"),
            "num_events": int(len(metadata)),
            "num_unique_tickers": int(len(tickers)),
        },
        "stock_download_summary": {
            "tickers_requested": tickers,
            "tickers_requested_count": int(len(tickers)),
            "tickers_with_no_data": empty_tickers,
            "tickers_with_no_data_count": int(len(empty_tickers)),
            "failed_tickers": failed_tickers,
            "failed_tickers_count": int(len(failed_tickers)),
            "rows_downloaded": int(len(prices)),
            "min_stock_date": prices["Date"].min().strftime("%Y-%m-%d"),
            "max_stock_date": prices["Date"].max().strftime("%Y-%m-%d"),
        },
        "index_download_summary": {
            "ticker": "^IXIC",
            "rows_downloaded": int(len(nasdaq)),
            "min_index_date": nasdaq["Date"].min().strftime("%Y-%m-%d"),
            "max_index_date": nasdaq["Date"].max().strftime("%Y-%m-%d"),
        },
    }

    with open(log_output_dir / "market_data_log.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    with open(log_output_dir / f"market_data_log_{timestamp_slug}.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    with open(log_output_dir / "market_data_log.txt", "w", encoding="utf-8") as f:
        f.write("Market data download completed\n")
        f.write(f"Run timestamp (UTC): {run_timestamp_utc}\n")
        f.write("Anchor variable: event_trading_day_final\n")
        f.write(f"Date range: {start_date} to {end_date}\n")
        f.write(f"Unique tickers requested: {len(tickers)}\n")
        f.write(f"Stock rows downloaded: {len(prices)}\n")
        f.write(f"NASDAQ index rows downloaded: {len(nasdaq)}\n")
        f.write(f"Tickers with no data: {empty_tickers}\n")
        f.write(f"Failed tickers: {failed_tickers}\n")

    with open(log_output_dir / f"market_data_log_{timestamp_slug}.txt", "w", encoding="utf-8") as f:
        f.write("Market data download completed\n")
        f.write(f"Run timestamp (UTC): {run_timestamp_utc}\n")
        f.write("Anchor variable: event_trading_day_final\n")
        f.write(f"Date range: {start_date} to {end_date}\n")
        f.write(f"Unique tickers requested: {len(tickers)}\n")
        f.write(f"Stock rows downloaded: {len(prices)}\n")
        f.write(f"NASDAQ index rows downloaded: {len(nasdaq)}\n")
        f.write(f"Tickers with no data: {empty_tickers}\n")
        f.write(f"Failed tickers: {failed_tickers}\n")

    print("Download complete.")


if __name__ == "__main__":
    main()