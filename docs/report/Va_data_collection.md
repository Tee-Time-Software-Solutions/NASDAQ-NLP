# Data Collection & Cleaning
**Part V(a) — Modelling Report**
*Asymmetric Sentiment Effects in Earnings Calls — NLP Course, Group Project*

---

## Overview

The dataset combines two transcript corpora with daily stock price data from Yahoo Finance. The goal was to get enough events to run a meaningful event study while keeping the sample homogeneous enough that a single sentiment-return model makes sense.

## Transcript Sources

### Thomson Reuters StreetEvents (Primary)

The primary source is a set of 188 earnings call transcripts for 10 NASDAQ-listed technology firms covering 2016–2020. Tickers: AAPL, AMZN, GOOG, META, MSFT, NFLX, NVDA, TSLA, CRM, AMD. Files follow the naming convention `YYYY-Mon-DD-TICKER.txt`.

### Kaggle cleaned_ECTs (Supplementary)

To extend coverage and date range, we added transcripts from the publicly available Kaggle `cleaned_ECTs` corpus. This adds 9 technology-adjacent firms including IBM, Oracle, Salesforce, and Accenture, extending the date range back to 2007. Files follow the convention `YYYY_Qn_ticker_processed.txt`.

### Combined Dataset

After filtering both sources to technology and technology-adjacent firms only (to minimize cross-sector heterogeneity) and removing any duplicates, the combined corpus contains **620 events** across **19 tickers** spanning 2007–2025.

| Attribute | Value |
|-----------|-------|
| Total events | 620 |
| Unique tickers | 19 |
| Date range | 2007–2025 |
| Training window | 2016–2018 |
| Test window | 2019–2020 |

## Market Price Data

Daily adjusted closing prices are downloaded from Yahoo Finance via `yfinance` for each of the 19 tickers plus their benchmark indices: `^IXIC` (NASDAQ Composite) for NASDAQ-listed firms and `^GSPC` (S&P 500) for NYSE-listed firms. We download data from 2006 onward to ensure sufficient pre-event history for the estimation window.

## Event Trading Day Assignment

A critical cleaning step is assigning each transcript to the correct trading day. An earnings call held at 10 PM Tuesday affects Wednesday's open; a call at 2 PM is priced the same day. We parse each transcript header for a timestamp and apply:

- Call before 4:00 PM ET on a trading day → event day = same calendar day.
- Call at/after 4:00 PM ET, on a weekend, or on a market holiday → event day = next business day.

If no header timestamp is found, we fall back to noon on the calendar date of the file (same-day assignment). This rule is implemented in `src/nasdaq_nlp/data/metadata/__init__.py`.

## Data Quality

Events are dropped if the stock or benchmark has fewer than 60 valid trading days in the estimation window [−120, −20] relative to the event day. This removes transcripts near IPO dates or during trading halts. The final 620 events all have complete estimation windows.

## Code Reference

- `src/nasdaq_nlp/data/loader/` — transcript scanning for both sources.
- `src/nasdaq_nlp/data/metadata/` — header parsing and event day assignment.
- `src/nasdaq_nlp/data/market/` — price download and return computation.
- Entry point: `make pipeline` or `notebooks/01_data_pipeline.ipynb`.
