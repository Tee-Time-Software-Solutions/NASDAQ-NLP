"""
market_model.py — Estimate the market model for each event, compute Abnormal Returns,
                  Cumulative Abnormal Returns (CAR), and Volatility Change (ΔVol).

THE MATH (from scratch)
========================

1. MARKET MODEL (OLS regression)
----------------------------------
For each earnings call event i, we fit a linear regression using stock and
market returns in the *estimation window* (trading days -120 to -20 relative
to the call date):

    R_{i,t} = α_i + β_i × R_{m,t} + ε_{i,t}

where:
    R_{i,t}  = stock i's return on day t
    R_{m,t}  = NASDAQ index return on day t (the "market portfolio proxy")
    α_i      = stock i's excess return above the market (intercept)
    β_i      = sensitivity to market moves (slope)
               β > 1 means the stock amplifies market swings
               β < 1 means it's more stable than the market
    ε_{i,t}  = residual — the part the model can't explain

We estimate α̂_i and β̂_i by ordinary least squares (OLS), which minimises
the sum of squared residuals over the estimation window.

WHY NOT USE THE FULL SAMPLE?
We exclude the event window from estimation so that the earnings call itself
doesn't contaminate the market model coefficients.

2. ABNORMAL RETURN (AR)
------------------------
The abnormal return on day t is what the stock actually returned minus
what the market model *predicted* it would return given the market's movement:

    AR_{i,t} = R_{i,t} - (α̂_i + β̂_i × R_{m,t})

If AR > 0: the stock did better than the market-model expected → positive "surprise"
If AR < 0: the stock underperformed expectations → negative "surprise"

3. CUMULATIVE ABNORMAL RETURN (CAR)
--------------------------------------
We sum ARs over an event window to smooth out daily noise:

    CAR_i[0,1] = AR_{i,0} + AR_{i,1}        (2-day window)
    CAR_i[0,3] = AR_{i,0} + AR_{i,1} + AR_{i,2} + AR_{i,3}   (4-day window)

Day 0 = the event trading day (as assigned in metadata.py).

4. VOLATILITY CHANGE (ΔVol)
-----------------------------
Volatility = standard deviation of daily returns over a window.
We compare the 10 days before vs 10 days after the call:

    ΔVol_i = std(R_{i,+1..+10}) - std(R_{i,-10..-1})

If ΔVol > 0: the stock became more volatile after the earnings call.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.linalg import lstsq

from nasdaq_nlp.config import (
    EST_END,
    EST_START,
    EVENT_END_LONG,
    EVENT_END_SHORT,
    EVENT_METADATA_PATH,
    EVENT_START,
    EVENT_STUDY_PATH,
    INDEX_RETURNS_PATH,
    MARKET_MODEL_PATH,
    STOCK_RETURNS_PATH,
    VOL_POST_END,
    VOL_POST_START,
    VOL_PRE_END,
    VOL_PRE_START,
    ensure_output_dirs,
)

# ---------------------------------------------------------------------------
# Event-time panel builder
# ---------------------------------------------------------------------------


def build_event_panel(
    meta: pd.DataFrame,
    stock_returns: pd.DataFrame,
    index_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Align stock and market returns to event time for each earnings call.

    For each event, we:
    1. Find the event trading day (day 0).
    2. Map all surrounding trading days to relative integers (-120, ..., +10).
    3. Merge stock return + market return for each relative day.

    This creates a long-format DataFrame where each row is
    (event_id, relative_day, stock_return, market_return).

    Parameters
    ----------
    meta : pd.DataFrame
        Output of build_event_metadata() — one row per call.
    stock_returns : pd.DataFrame
        Output of build_market_returns() — long format with [date, ticker, return].
    index_returns : pd.DataFrame
        Output of build_market_returns() — long format with [date, adj_close, return].

    Returns
    -------
    pd.DataFrame with columns:
        event_id, ticker, file_name, event_trading_day,
        date, relative_day, stock_return, market_return
    """
    # Build a (index_ticker, date) → return lookup.
    # If index_returns has an `index_ticker` column (multi-index run), group by it.
    # Otherwise, treat the whole series as a single unnamed index (backwards compat).
    if "index_ticker" in index_returns.columns:
        index_maps: dict[str, dict] = {
            idx: dict(zip(pd.to_datetime(grp["date"]), grp["return"]))
            for idx, grp in index_returns.groupby("index_ticker")
        }
        default_index = next(iter(index_maps))
    else:
        _single = dict(zip(pd.to_datetime(index_returns["date"]), index_returns["return"]))
        index_maps = {"^IXIC": _single}
        default_index = "^IXIC"

    # Build a (ticker, date) → stock_return lookup
    stock_returns_indexed = stock_returns.set_index(["ticker", "date"])["return"]

    rows = []
    for event_id, row in meta.iterrows():
        ticker = row["ticker"]
        event_day = pd.Timestamp(row["event_trading_day"])
        # Use the event's assigned market index; fall back to default
        mkt_idx = (
            row.get("market_index", default_index) if "market_index" in row.index else default_index
        )
        index_map = index_maps.get(mkt_idx, index_maps[default_index])

        # Get all trading dates for this ticker, sorted
        ticker_dates = sorted(stock_returns.loc[stock_returns["ticker"] == ticker, "date"].tolist())
        ticker_dates_arr = pd.to_datetime(ticker_dates)

        # Find the position of event_day in the ticker's trading calendar
        date_to_pos = {d: i for i, d in enumerate(ticker_dates_arr)}
        if event_day not in date_to_pos:
            # Event day not in our price data — skip this event
            continue
        event_pos = date_to_pos[event_day]

        # We need relative days from EST_START to max(EVENT_END_LONG, VOL_POST_END)
        rel_start = EST_START  # -120
        rel_end = max(EVENT_END_LONG, VOL_POST_END)  # +10

        for rel_day in range(rel_start, rel_end + 1):
            abs_pos = event_pos + rel_day
            if abs_pos < 0 or abs_pos >= len(ticker_dates_arr):
                continue  # out of our data range — skip

            cal_date = ticker_dates_arr[abs_pos]

            # Look up returns
            stock_ret = stock_returns_indexed.get((ticker, cal_date), np.nan)
            market_ret = index_map.get(cal_date, np.nan)

            rows.append(
                {
                    "event_id": event_id,
                    "ticker": ticker,
                    "file_name": row["file_name"],
                    "event_trading_day": event_day,
                    "date": cal_date,
                    "relative_day": rel_day,
                    "stock_return": stock_ret,
                    "market_return": market_ret,
                }
            )

    panel = pd.DataFrame(rows)
    return panel


# ---------------------------------------------------------------------------
# OLS market model estimation
# ---------------------------------------------------------------------------


def estimate_market_model(panel: pd.DataFrame) -> pd.DataFrame:
    """Estimate α and β for each event using the estimation window (days -120 to -20).

    OLS formula (in matrix notation):
        [α, β] = (X'X)^{-1} X'y
    where X = [1, R_market] and y = R_stock over the estimation window.

    Returns
    -------
    pd.DataFrame with columns [event_id, ticker, file_name, event_trading_day, alpha, beta, n_obs]
    """
    results = []

    for event_id, event_df in panel.groupby("event_id"):
        # Restrict to the estimation window: days -120 to -20
        est = event_df[
            (event_df["relative_day"] >= EST_START) & (event_df["relative_day"] <= EST_END)
        ].dropna(subset=["stock_return", "market_return"])

        n_obs = len(est)
        if n_obs < 30:
            # Too few observations — unreliable OLS; skip this event
            continue

        # Design matrix X: column of ones (intercept) + market return
        X = np.column_stack([np.ones(n_obs), est["market_return"].values])
        y = est["stock_return"].values

        # OLS solution: minimise ||Xb - y||^2
        coeffs, _, _, _ = lstsq(X, y, rcond=None)
        alpha, beta = coeffs

        results.append(
            {
                "event_id": event_id,
                "ticker": est["ticker"].iloc[0],
                "file_name": est["file_name"].iloc[0],
                "event_trading_day": est["event_trading_day"].iloc[0],
                "alpha": alpha,
                "beta": beta,
                "n_obs_estimation": n_obs,
            }
        )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Abnormal returns and CAR
# ---------------------------------------------------------------------------


def compute_abnormal_returns(
    panel: pd.DataFrame,
    model_params: pd.DataFrame,
) -> pd.DataFrame:
    """Compute AR for each (event, relative_day) in the event window.

    AR_{i,t} = R_{i,t} - (α̂_i + β̂_i × R_{m,t})

    Returns a DataFrame with all event-window rows and an 'ar' column.
    """
    # Merge model parameters (alpha, beta) onto the panel
    panel_with_params = panel.merge(
        model_params[["event_id", "alpha", "beta"]],
        on="event_id",
        how="inner",  # only keep events that have a valid model
    )

    # Filter to event window: days 0 to EVENT_END_LONG (+3)
    event_window = panel_with_params[
        (panel_with_params["relative_day"] >= EVENT_START)
        & (panel_with_params["relative_day"] <= EVENT_END_LONG)
    ].copy()

    # AR = actual return - expected return
    # expected = alpha + beta * market_return
    event_window["expected_return"] = (
        event_window["alpha"] + event_window["beta"] * event_window["market_return"]
    )
    event_window["ar"] = event_window["stock_return"] - event_window["expected_return"]

    return event_window


def compute_car(ar_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate AR into CAR over two event windows: [0,1] and [0,3].

    For each event, sum the ARs over the specified days.
    One row per event in the output.

    CAR[0,1] = AR_0 + AR_1        (captures immediate reaction)
    CAR[0,3] = AR_0 + ... + AR_3  (captures post-announcement drift)
    """
    car_rows = []

    for event_id, ev in ar_df.groupby("event_id"):
        # Extract AR for each relative day (pivot by relative_day)
        ar_by_day = ev.set_index("relative_day")["ar"]

        def sum_ar(start: int, end: int) -> float:
            """Sum AR over a range of relative days, ignoring missing days."""
            vals = [ar_by_day.get(d, np.nan) for d in range(start, end + 1)]
            clean = [v for v in vals if not np.isnan(v)]
            return float(np.sum(clean)) if clean else np.nan

        car_rows.append(
            {
                "event_id": event_id,
                "car_01": sum_ar(0, EVENT_END_SHORT),  # CAR[0,1]
                "car_03": sum_ar(0, EVENT_END_LONG),  # CAR[0,3]
                "ar_0": ar_by_day.get(0, np.nan),  # day-0 AR only (immediate)
            }
        )

    return pd.DataFrame(car_rows)


# ---------------------------------------------------------------------------
# Volatility change
# ---------------------------------------------------------------------------


def compute_volatility_change(
    panel: pd.DataFrame,
    model_params: pd.DataFrame,
) -> pd.DataFrame:
    """Compute pre-event and post-event return volatility for each event.

    Volatility = standard deviation of daily stock returns over the window.

    Pre-event:  trading days -10 to -1
    Post-event: trading days +1 to +10

    ΔVol = post_vol - pre_vol

    A positive ΔVol means the stock became more volatile after the call.
    This often happens regardless of direction — earnings calls resolve
    uncertainty but can introduce new uncertainty too.
    """
    results = []

    for event_id in model_params["event_id"]:
        ev = panel[panel["event_id"] == event_id].set_index("relative_day")

        pre_returns = [
            ev.loc[d, "stock_return"]
            for d in range(VOL_PRE_START, VOL_PRE_END + 1)
            if d in ev.index and not np.isnan(ev.loc[d, "stock_return"])
        ]
        post_returns = [
            ev.loc[d, "stock_return"]
            for d in range(VOL_POST_START, VOL_POST_END + 1)
            if d in ev.index and not np.isnan(ev.loc[d, "stock_return"])
        ]

        pre_vol = float(np.std(pre_returns, ddof=1)) if len(pre_returns) >= 3 else np.nan
        post_vol = float(np.std(post_returns, ddof=1)) if len(post_returns) >= 3 else np.nan
        delta_vol = (
            post_vol - pre_vol if (not np.isnan(pre_vol) and not np.isnan(post_vol)) else np.nan
        )

        results.append(
            {
                "event_id": event_id,
                "pre_vol": pre_vol,
                "post_vol": post_vol,
                "delta_vol": delta_vol,
            }
        )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------


def build_event_study(
    metadata_path=EVENT_METADATA_PATH,
    stock_returns_path=STOCK_RETURNS_PATH,
    index_returns_path=INDEX_RETURNS_PATH,
    model_output_path=MARKET_MODEL_PATH,
    study_output_path=EVENT_STUDY_PATH,
) -> pd.DataFrame:
    """Run the full event-study pipeline and save results to CSV.

    Steps:
        1. Load metadata, stock returns, index returns
        2. Build event-time panel (align to relative trading days)
        3. Estimate market model (OLS α, β) per event
        4. Compute AR, CAR[0,1], CAR[0,3]
        5. Compute volatility change (ΔVol)
        6. Merge everything → event_study_dataset.csv

    Returns the final merged DataFrame.
    """
    ensure_output_dirs()

    # Load inputs
    meta = pd.read_csv(metadata_path, parse_dates=["event_trading_day"])
    stocks = pd.read_csv(stock_returns_path, parse_dates=["date"])
    index = pd.read_csv(index_returns_path, parse_dates=["date"])

    print(f"Events: {len(meta)} | Stock return rows: {len(stocks)} | Index rows: {len(index)}")

    # Step 1: build event-time panel
    print("Building event-time panel...")
    panel = build_event_panel(meta, stocks, index)
    print(f"Panel: {len(panel)} rows across {panel['event_id'].nunique()} events")

    # Step 2: OLS market model
    print("Estimating market models (OLS)...")
    model_params = estimate_market_model(panel)
    print(f"  Fitted {len(model_params)} models (events with ≥30 estimation observations)")

    # Step 3: AR and CAR
    print("Computing abnormal returns and CAR...")
    ar_df = compute_abnormal_returns(panel, model_params)
    car_df = compute_car(ar_df)

    # Step 4: Volatility change
    print("Computing volatility change...")
    vol_df = compute_volatility_change(panel, model_params)

    # Step 5: Merge everything
    event_study = model_params.merge(car_df, on="event_id", how="left").merge(
        vol_df, on="event_id", how="left"
    )

    # Attach file_path from metadata for use in feature extraction
    event_study = event_study.merge(
        meta[["file_name", "file_path", "year", "quarter"]],
        on="file_name",
        how="left",
    )

    # Save model parameters separately (useful for reproducibility checks)
    model_params.to_csv(model_output_path, index=False)

    # Save final event-study dataset
    event_study.to_csv(study_output_path, index=False)
    print(f"Saved market model   → {model_output_path}")
    print(f"Saved event study    → {study_output_path}  ({len(event_study)} rows)")

    # Quick sanity checks
    nan_car = event_study["car_03"].isna().sum()
    if nan_car > 0:
        print(f"  WARN: {nan_car} events have NaN CAR[0,3]")

    return event_study
