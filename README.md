# NASDAQ-NLP

## Project overview
This project investigates whether **negative sentiment in earnings call transcripts has a stronger impact on market reactions than positive sentiment**.

The current repo state focuses on building a **clean, auditable event study backbone**:
- **Transcript ingestion (Kaggle)** → local raw transcript files
- **Event metadata** (timezone conversion + “after market close” logic) → `data/processed/event_metadata_final.csv`
- **Market data download** (stocks + NASDAQ index) anchored to the event day → `data/raw/market_data/*.csv`
- **Validation notebooks** that assert the pipeline outputs are internally consistent

## Setup

### Prerequisites
- **Python**: 3.10+ recommended
- **Kaggle account + API token**: required to download transcripts
- **Internet access**: required for Kaggle + Yahoo Finance (via `yfinance`)

### Install
Create and activate a virtual environment, then install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create your environment file:

```bash
cp .env.example .env
```

Edit `.env` and set:

```text
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

## Quickstart (end-to-end)

### 1) Download transcripts from Kaggle
Downloads and extracts the Kaggle dataset into `data/raw/earnings-call-transcripts/`.

```bash
python scripts/download_data.py
```

Dataset source: `https://www.kaggle.com/datasets/ashwinm500/earnings-call-transcripts`

### 2) Build canonical event metadata
Parses transcript filenames + transcript headers, converts timestamps to ET, flags after-hours calls, and assigns the event trading day.

```bash
python scripts/build_event_metadata.py
```

Output:
- `data/processed/event_metadata_final.csv`

### 3) Download market data (stocks + NASDAQ index)
Downloads OHLCV for all tickers in `event_metadata_final.csv` and the NASDAQ Composite index (`^IXIC`).

```bash
python scripts/download_market_data.py
```

Outputs:
- `data/raw/market_data/prices_raw.csv`
- `data/raw/market_data/nasdaq_index_raw.csv`
- `outputs/logs/market_data_log.json` (+ timestamped copies)

### 4) Run validations (recommended)
Start Jupyter and run the validation notebooks:

```bash
jupyter notebook
```

Run in order:
- `notebooks/01_validate_event_metadata_final.ipynb`
- `notebooks/02_validate_market_data.ipynb`

## What the pipeline does (the process you followed)

### A) Transcript ingestion
- **Goal**: get a consistent local corpus of earnings call transcripts.
- **How**: `scripts/download_data.py` uses the Kaggle CLI and your `.env` credentials to download + unzip the dataset into `data/raw/earnings-call-transcripts/`.

### B) Event metadata (the “event study spine”)
You built a canonical event table where **each row is an earnings call event** and the key output is a single anchor date used for market alignment.

**Final rule implemented in** `scripts/build_event_metadata.py`:
- Parse a GMT call timestamp from the transcript header when available.
- Convert to **`America/New_York`** (ET).
- Compute `after_market_close_et` using **strictly after 4:00 PM ET**.
- Assign `event_trading_day_final`:
  - if after market close → **next business day** (weekday-only via `BDay`)
  - else → same calendar day (ET)

**Historical note (v1)**:
- `scripts/build_event_metadata_v1.py` produced an earlier “first pass” metadata file using a crude GMT hour threshold (no explicit timezone conversion). The notebook `notebooks/01_event_metadata_exploration.ipynb` documents why this approach was insufficient (DST/edge times) and motivated the final implementation.

### C) Market data download anchored to event day
`scripts/download_market_data.py` downloads stock and index data **relative to the event anchor**, not the raw calendar date in the transcript filename.

**Anchor variable**: `event_trading_day_final`

**Download window rule** (implemented in the script and logged):
- start = `min(event_trading_day_final) - 200 calendar days`
- end = `max(event_trading_day_final) + 30 calendar days`

This supports common event-study windows (estimation + event windows) and gives a buffer for alignment.

### D) Validation notebooks (audit, not transformation)
The notebooks in this repo are primarily **audit/validation artifacts**: they assert that the pipeline outputs are consistent and complete before downstream return calculations and modeling.

## Repository map (what each file/folder corresponds to)

### Root
- **`README.md`**: this document (runbook + file map).
- **`.env.example`**: template for Kaggle credentials.
- **`.gitignore`**: ignores `.env`, virtualenvs, and all of `data/raw/` + `data/processed/` (data is meant to be generated locally).
- **`requirements.txt`**: Python dependencies used by scripts and notebooks.

### `scripts/` (reproducible pipeline steps)
- **`scripts/download_data.py`**: downloads the Kaggle “earnings call transcripts” dataset and extracts it to `data/raw/earnings-call-transcripts/`.
- **`scripts/build_event_metadata_v1.py`**: first-pass event metadata builder (heuristic GMT threshold). Output: `data/processed/event_metadata_v1_rawrule.csv`.
- **`scripts/build_event_metadata.py`**: canonical event metadata builder (timezone-aware, ET market-close logic). Output: `data/processed/event_metadata_final.csv`.
- **`scripts/download_market_data.py`**: downloads stock OHLCV for event tickers + NASDAQ index (`^IXIC`) using `yfinance`. Outputs raw CSVs and an audit log in `outputs/logs/`.

### `notebooks/` (exploration + validation)
- **`notebooks/01_data_collection_cleaning.ipynb`**: currently empty (placeholder).
- **`notebooks/01_event_metadata_exploration.ipynb`**: exploratory validation of the *v1* metadata logic; shows DST/after-hours edge cases and motivates the final rule.
- **`notebooks/01_validate_event_metadata_final.ipynb`**: validates `data/processed/event_metadata_final.csv` (parsing success, timezone conversion, after-hours flag, business-day assignment consistency).
- **`notebooks/02_validate_market_data.ipynb`**: validates `prices_raw.csv` and `nasdaq_index_raw.csv` (schema checks, duplicates, missing values, ticker coverage, and event-day coverage).

### `data/` (generated artifacts)
Note: `data/raw/` and `data/processed/` are gitignored in this repo; they’re meant to be generated locally by running the scripts above.

- **`data/raw/earnings-call-transcripts/`**: extracted Kaggle dataset (per-ticker transcript `.txt` files).
- **`data/processed/event_metadata_v1_rawrule.csv`**: v1 metadata output (heuristic rule; kept for comparison/diagnostics).
- **`data/processed/event_metadata_final.csv`**: canonical event table used to anchor market alignment.
- **`data/raw/market_data/prices_raw.csv`**: downloaded OHLCV for each ticker across the computed date range.
- **`data/raw/market_data/nasdaq_index_raw.csv`**: downloaded OHLCV for `^IXIC` across the same date range.

### `outputs/`
- **`outputs/logs/market_data_log.json`**: audit log emitted by `scripts/download_market_data.py` (includes date window, ticker coverage, failures, and min/max dates). Timestamped copies are also written alongside it.

## Current state / what’s next
At the moment, the repo contains the **data acquisition + event/market alignment foundation** plus validation notebooks. The next natural steps (not yet implemented here) are:
- compute returns, abnormal returns, and event-window CAR/volatility using the downloaded market data
- extract sentiment features from transcripts (e.g., dictionary-based or model-based sentiment)
- run regression/event study analyses to test asymmetric impact of negative vs positive sentiment

