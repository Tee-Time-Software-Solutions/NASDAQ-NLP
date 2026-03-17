# NASDAQ-NLP

## Project overview

This project investigates whether **negative sentiment in earnings call transcripts has a stronger impact on market reactions than positive sentiment**.

The pipeline builds a **clean, auditable event-study backbone**:

1. **Transcript ingestion** (Kaggle) → raw transcript files  
2. **Event metadata** (timezone + “after market close” logic) → event table with `event_trading_day_final`  
3. **Market data** (stocks + NASDAQ index) → raw OHLCV, then daily returns  
4. **Event windows** → event-time panel aligned to each earnings call  
5. **Market model** → OLS α, β per event (estimation window [-120,-20])  
6. **Abnormal returns** → expected vs actual returns in event time  
7. **CAR & volatility change** → CAR[0,1], CAR[0,3], Δ volatility (post − pre)  
8. **Final dataset** → one row per event: metadata + CAR + volatility, ready for sentiment merge and regression  

---

## Setup (for teammates)

Follow these steps exactly so your environment matches the project and all notebooks run in order.

### 1. Clone the repository

```bash
git clone <repository-url>
cd NASDAQ-NLP
```

Use the same branch the team has agreed on (e.g. `main`).

### 2. Python version

- **Required:** Python **3.10 or higher** (3.11 or 3.12 is fine).
- Check: `python3 --version` or `python --version`.

If you need a different version, use `pyenv`, `conda`, or your system package manager; the rest of the setup assumes `python3` is 3.10+.

### 3. Virtual environment (recommended)

Create and activate a venv so dependencies don’t conflict with other projects:

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (Command Prompt):**

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

You should see `(.venv)` (or similar) in your prompt. All following commands assume this environment is active.

### 4. Install dependencies

From the project root (`NASDAQ-NLP/`):

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs Jupyter so you can run notebooks in the browser with `jupyter notebook` (or use VS Code / Cursor to run them in the editor).

### 5. Kaggle API credentials (required for data download)

The transcript dataset is downloaded from Kaggle. You need a Kaggle account and an API key.

1. Sign up at [kaggle.com](https://www.kaggle.com) if needed.  
2. Open **Account** → **API** → **Create New Token**. This downloads `kaggle.json`.  
3. Copy the project env template and add your credentials:

   ```bash
   cp .env.example .env
   ```

4. Open `.env` and set (replace with your values):

   ```text
   KAGGLE_USERNAME=your_kaggle_username
   KAGGLE_KEY=your_kaggle_api_key
   ```

   **Important:**  
   - The download script expects **`KAGGLE_USERNAME`** and **`KAGGLE_KEY`** (not `KAGGLE_API_KEY`).  
   - Do **not** commit `.env`; it is listed in `.gitignore`.  
   - On some systems the Kaggle CLI also looks for `~/.kaggle/kaggle.json`; the script uses `.env` so the repo stays self-contained.

### 6. Verify setup

- **Scripts:** From project root, run (this will fail if `.env` or Kaggle is missing, but confirms Python and paths):

  ```bash
  python scripts/download_data.py
  ```

  After a successful run you should see `data/raw/earnings-call-transcripts/` with extracted transcript files.

- **Notebooks:** Start Jupyter (if installed) and open e.g. `notebooks/01_validate_event_metadata_final.ipynb`. Run the first few cells; they load `../data/processed/event_metadata_final.csv`. That file only exists after you have run the **scripts** and **notebooks** in the order below, so for a first check you can just confirm the kernel runs and that `pandas`/`pathlib` work.

### 7. Directory layout expectations

- Scripts and notebooks assume they are run **from the project root** (scripts) or that notebooks use paths like `../data/...` (i.e. repo root is the parent of `notebooks/`).  
- **Data:** `data/raw/` and `data/processed/` are gitignored. You (or a teammate) must generate them by running the pipeline.  
- **Outputs:** Logs and other outputs go under `outputs/` (e.g. `outputs/logs/`).

---

## Pipeline overview

The pipeline has two parts: **scripts** (data download + event metadata) and **notebooks** (returns, event windows, market model, abnormal returns, CAR, volatility, final dataset). Scripts must be run first; then notebooks must be run in the order given below because each notebook depends on outputs of the previous ones.

### Scripts (run from project root)

| Step | Command | What it does |
|------|---------|----------------|
| 1 | `python scripts/download_data.py` | Downloads Kaggle earnings-call transcripts → `data/raw/earnings-call-transcripts/` |
| 2 | `python scripts/build_event_metadata.py` | Builds event table → `data/processed/event_metadata_final.csv` |
| 3 | `python scripts/download_market_data.py` | Downloads OHLCV (stocks + NASDAQ index) → `data/raw/market_data/prices_raw.csv`, `nasdaq_index_raw.csv` |

### Notebooks (run in this order)

Notebooks are run **in numerical order**; later notebooks expect the output files produced by earlier ones. Run all cells in each notebook before moving to the next.

| Order | Notebook | Main inputs | Main outputs |
|-------|----------|-------------|--------------|
| 1 | `01_validate_event_metadata_final.ipynb` | `event_metadata_final.csv` | — (validation only) |
| 2 | `02_validate_market_data.ipynb` | `prices_raw.csv`, `nasdaq_index_raw.csv`, `event_metadata_final.csv` | — (validation only) |
| 3 | `03_compute_returns.ipynb` | `prices_raw.csv`, `nasdaq_index_raw.csv` | `market_data_with_returns.csv`, `index_returns.csv` |
| 4 | `04_build_event_windows.ipynb` | `event_metadata_final.csv`, `market_data_with_returns.csv`, `index_returns.csv` | `event_windows.csv`, `event_window_eligibility.csv` |
| 5 | `05_estimate_market_model.ipynb` | `event_windows.csv`, `event_window_eligibility.csv` | `event_market_model.csv` |
| 6 | `06_compute_abnormal_returns.ipynb` | `event_windows.csv`, `event_market_model.csv` | `event_abnormal_returns.csv` |
| 7 | `07_compute_CAR.ipynb` | `event_abnormal_returns.csv` | `event_CAR.csv` |
| 8 | `08_compute_volatility_change.ipynb` | `event_abnormal_returns.csv` | `event_volatility_change.csv` |
| 9 | `09_assemble_event_study_dataset.ipynb` | `event_metadata_final.csv`, `event_CAR.csv`, `event_volatility_change.csv` | `event_study_dataset.csv` |

Optional/exploratory (do not affect the pipeline):

- `01_data_collection_cleaning.ipynb` — placeholder.  
- `01_event_metadata_exploration.ipynb` — explores v1 metadata and motivates the final event-day rule; no pipeline outputs.

---

## Notebook run order (detailed)

Run notebooks **in this exact order** and run **all cells** in each notebook. If you skip or reorder, later notebooks will fail (missing or wrong columns/files).

1. **01_validate_event_metadata_final.ipynb**  
   - **Inputs:** `data/processed/event_metadata_final.csv` (from `scripts/build_event_metadata.py`).  
   - **Purpose:** Checks parsing, timezone conversion, after-hours flag, and event-day assignment. Read-only.

2. **02_validate_market_data.ipynb**  
   - **Inputs:** `event_metadata_final.csv`, `data/raw/market_data/prices_raw.csv`, `nasdaq_index_raw.csv`.  
   - **Purpose:** Validates raw market data (schema, dates, ticker coverage). Read-only.

3. **03_compute_returns.ipynb**  
   - **Inputs:** `prices_raw.csv`, `nasdaq_index_raw.csv`.  
   - **Outputs:** `data/processed/market_data_with_returns.csv`, `data/processed/index_returns.csv`.  
   - **Purpose:** Daily simple returns from Adjusted Close (by ticker and for NASDAQ index).

4. **04_build_event_windows.ipynb**  
   - **Inputs:** `event_metadata_final.csv`, `market_data_with_returns.csv`, `index_returns.csv`.  
   - **Outputs:** `data/processed/event_windows.csv`, `data/processed/event_window_eligibility.csv`.  
   - **Purpose:** Event-time panel (relative day `t`) and event-level eligibility flags for estimation/CAR/volatility windows.

5. **05_estimate_market_model.ipynb**  
   - **Inputs:** `event_windows.csv`, `event_window_eligibility.csv`.  
   - **Outputs:** `data/processed/event_market_model.csv`.  
   - **Purpose:** OLS market model \(R_{stock,t} = \alpha + \beta R_{market,t} + \epsilon_t\) over estimation window \(t \in [-120, -20]\); one \((\alpha, \beta)\) per event.

6. **06_compute_abnormal_returns.ipynb**  
   - **Inputs:** `event_windows.csv`, `event_market_model.csv`.  
   - **Outputs:** `data/processed/event_abnormal_returns.csv`.  
   - **Purpose:** Expected return from market model; abnormal return = actual − expected, in event time.

7. **07_compute_CAR.ipynb**  
   - **Inputs:** `event_abnormal_returns.csv`.  
   - **Outputs:** `data/processed/event_CAR.csv`.  
   - **Purpose:** CAR[0,1] and CAR[0,3] per event (sum of abnormal returns over windows [0,1] and [0,3]).

8. **08_compute_volatility_change.ipynb**  
   - **Inputs:** `event_abnormal_returns.csv`.  
   - **Outputs:** `data/processed/event_volatility_change.csv`.  
   - **Purpose:** Pre-event volatility (std of returns, \(t \in [-10,-1]\)), post-event volatility (\(t \in [1,10]\)), and Δ volatility (post − pre) per event.

9. **09_assemble_event_study_dataset.ipynb**  
   - **Inputs:** `event_metadata_final.csv`, `event_CAR.csv`, `event_volatility_change.csv`.  
   - **Outputs:** `data/processed/event_study_dataset.csv`.  
   - **Purpose:** One row per event: metadata + CAR[0,1], CAR[0,3], volatility change; ready to merge with sentiment and run regressions.

Dependency chain (summary):

```text
event_metadata_final.csv, prices_raw, nasdaq_index_raw
    → 03 → market_data_with_returns, index_returns
    → 04 → event_windows, event_window_eligibility
    → 05 → event_market_model
    → 06 → event_abnormal_returns
    → 07 → event_CAR
    → 08 → event_volatility_change
    → 09 → event_study_dataset
```

---

## What each part of the pipeline does

### Scripts

- **download_data.py** — Uses Kaggle CLI + `.env` credentials; downloads and extracts the earnings-call transcript dataset to `data/raw/earnings-call-transcripts/`.
- **build_event_metadata.py** — Parses transcript filenames and headers (GMT call time), converts to ET, sets `after_market_close_et`, and assigns `event_trading_day_final` (same day or next business day after 4 PM ET). Writes `event_metadata_final.csv`.
- **build_event_metadata_v1.py** — Old heuristic (GMT hour threshold); kept for reference. Output: `event_metadata_v1_rawrule.csv`.
- **download_market_data.py** — Uses `event_metadata_final.csv` to get ticker list and date range (min/max event date ± buffer), downloads OHLCV via yfinance for stocks and ^IXIC. Writes `prices_raw.csv`, `nasdaq_index_raw.csv`, and logs under `outputs/logs/`.

### Event metadata and market alignment

- **Event anchor:** All event windows and market data alignment use `event_trading_day_final` (trading day on which the market reacts to the call).
- **Download window:** `min(event_trading_day_final) - 200` to `max(event_trading_day_final) + 30` calendar days so estimation and event windows are covered.

### Notebooks (short description)

- **01_validate_event_metadata_final** — Validates `event_metadata_final.csv` (no writes).  
- **02_validate_market_data** — Validates raw market files and coverage (no writes).  
- **03_compute_returns** — Daily returns from Adjusted Close; writes `market_data_with_returns.csv`, `index_returns.csv`.  
- **04_build_event_windows** — Builds event-time panel and eligibility table; writes `event_windows.csv`, `event_window_eligibility.csv`.  
- **05_estimate_market_model** — OLS α, β per event on estimation window; writes `event_market_model.csv`.  
- **06_compute_abnormal_returns** — Expected and abnormal returns in event time; writes `event_abnormal_returns.csv`.  
- **07_compute_CAR** — CAR[0,1] and CAR[0,3]; writes `event_CAR.csv`.  
- **08_compute_volatility_change** — Pre/post volatility and Δ volatility; writes `event_volatility_change.csv`.  
- **09_assemble_event_study_dataset** — Merges metadata + CAR + volatility into one event-level file; writes `event_study_dataset.csv`.

---

## Repository map

### Root

- **README.md** — This file (setup + run order + file map).  
- **.env.example** — Template for `KAGGLE_USERNAME` and `KAGGLE_KEY`.  
- **.gitignore** — Excludes `.env`, `data/raw/`, `data/processed/`, `.venv`, etc.  
- **requirements.txt** — Python dependencies for scripts and notebooks.

### `scripts/`

- **download_data.py** — Kaggle transcript download → `data/raw/earnings-call-transcripts/`.  
- **build_event_metadata.py** — Event table → `event_metadata_final.csv`.  
- **build_event_metadata_v1.py** — Legacy v1 metadata → `event_metadata_v1_rawrule.csv`.  
- **download_market_data.py** — Stock + index OHLCV → `prices_raw.csv`, `nasdaq_index_raw.csv` + logs.

### `notebooks/`

- **01_data_collection_cleaning.ipynb** — Placeholder (optional).  
- **01_event_metadata_exploration.ipynb** — Exploratory validation of v1 rule (optional).  
- **01_validate_event_metadata_final.ipynb** — Validates `event_metadata_final.csv`.  
- **02_validate_market_data.ipynb** — Validates raw market data.  
- **03_compute_returns.ipynb** — Daily returns → `market_data_with_returns.csv`, `index_returns.csv`.  
- **04_build_event_windows.ipynb** — Event-time panel → `event_windows.csv`, `event_window_eligibility.csv`.  
- **05_estimate_market_model.ipynb** — Market model (α, β) → `event_market_model.csv`.  
- **06_compute_abnormal_returns.ipynb** — Abnormal returns → `event_abnormal_returns.csv`.  
- **07_compute_CAR.ipynb** — CAR[0,1], CAR[0,3] → `event_CAR.csv`.  
- **08_compute_volatility_change.ipynb** — Volatility change → `event_volatility_change.csv`.  
- **09_assemble_event_study_dataset.ipynb** — Final event-level dataset → `event_study_dataset.csv`.

### `data/` (all generated; gitignored)

**Raw**

- **data/raw/earnings-call-transcripts/** — Extracted Kaggle transcripts (per-ticker `.txt`).  
- **data/raw/market_data/prices_raw.csv** — OHLCV by ticker.  
- **data/raw/market_data/nasdaq_index_raw.csv** — NASDAQ index OHLCV.

**Processed**

- **event_metadata_final.csv** — Canonical event table (`event_trading_day_final`, etc.).  
- **event_metadata_v1_rawrule.csv** — V1 metadata (optional).  
- **market_data_with_returns.csv** — Per-ticker daily returns.  
- **index_returns.csv** — Daily NASDAQ returns.  
- **event_windows.csv** — Event-time panel (stock + market returns, relative day `t`).  
- **event_window_eligibility.csv** — Event-level window coverage flags.  
- **event_market_model.csv** — α, β (and related) per event.  
- **event_abnormal_returns.csv** — Abnormal returns in event time.  
- **event_CAR.csv** — CAR[0,1], CAR[0,3] per event.  
- **event_volatility_change.csv** — Pre/post volatility and Δ volatility per event.  
- **event_study_dataset.csv** — Final event-level dataset (metadata + CAR + volatility).

### `outputs/`

- **outputs/logs/** — Market-data download logs (e.g. `market_data_log.json`, timestamped copies).

---

## Current state and next steps

- **Done:** Full event-study data pipeline from transcripts and market data to `event_study_dataset.csv` (one row per event, with CAR and volatility change).  
- **Next:**  
  - Merge transcript-based sentiment (e.g. dictionary or model) into `event_study_dataset.csv`.  
  - Run regressions to test whether negative sentiment has a stronger association with market reactions (CAR, volatility) than positive sentiment.

---

## Reproducibility

- Recreate environment: `pip install -r requirements.txt` (with same Python version).  
- Data: Run the three scripts, then notebooks 01–09 in order; do not skip or reorder.  
- `.env` must contain valid Kaggle credentials before `download_data.py`.
