"""
config.py — Central configuration for the NASDAQ-NLP project.

All file paths are defined here as pathlib.Path constants so that every
other module imports from this file rather than hard-coding strings.
If you ever move the project root, only this file needs updating.

Directory layout recap:
    NASDAQ-NLP/
    ├── dataset/               raw earnings-call transcripts (static, never modified)
    ├── outputs/
    │   ├── processed/         intermediate CSVs written by the pipeline
    │   └── results/           final model outputs, charts, tables
    └── src/nasdaq_nlp/        this package
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------

# PROJECT_ROOT: the NASDAQ-NLP/ folder, two levels above this file
#   this_file  →  src/nasdaq_nlp/config.py
#   .parent    →  src/nasdaq_nlp/
#   .parent    →  src/
#   .parent    →  NASDAQ-NLP/   ← PROJECT_ROOT
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# Raw transcript data — never write here, only read
DATASET_DIR: Path = PROJECT_ROOT / "dataset" / "Transcripts"

# Generated outputs — safe to delete and regenerate
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
PROCESSED_DIR: Path = OUTPUTS_DIR / "processed" 
RESULTS_DIR: Path = OUTPUTS_DIR / "results"
LOGS_DIR: Path = OUTPUTS_DIR / "logs"

# ---------------------------------------------------------------------------
# Intermediate CSV paths  (written by each pipeline step)
# ---------------------------------------------------------------------------

# Step 1 — event metadata: one row per earnings call, with the trading day
EVENT_METADATA_PATH: Path = PROCESSED_DIR / "event_metadata.csv" # CSV with metadata on trading day after shareholder meeting occured
 
# Step 2 — market data: daily stock + index OHLCV
PRICES_RAW_PATH: Path = PROCESSED_DIR / "prices_raw.csv" # cleaned yfinance data on price of stock
INDEX_RAW_PATH: Path = PROCESSED_DIR / "nasdaq_index_raw.csv" # cleaned yfinance data on price of index
 
# Step 2b — returns: daily return columns added
STOCK_RETURNS_PATH: Path = PROCESSED_DIR / "stock_returns.csv" # yfinance downloaded data with percentage change on returns
INDEX_RETURNS_PATH: Path = PROCESSED_DIR / "index_returns.csv"

# Step 3 — market model + event-study measures
MARKET_MODEL_PATH: Path = PROCESSED_DIR / "market_model.csv"        # α, β per event
EVENT_STUDY_PATH: Path = PROCESSED_DIR / "event_study_dataset.csv"  # AR, CAR, ΔVol

# Step 4 — lexicon sentiment
LEXICON_FEATURES_PATH: Path = PROCESSED_DIR / "lexicon_features.csv"

# Step 5 — TF-IDF feature matrix (saved as CSV for small corpora)
TFIDF_FEATURES_PATH: Path = PROCESSED_DIR / "tfidf_features.csv"

# Step 6 — Word2Vec document embeddings
EMBEDDINGS_PATH: Path = PROCESSED_DIR / "doc_embeddings.csv"
WORD2VEC_MODEL_PATH: Path = PROCESSED_DIR / "word2vec.model"  # gensim binary

# Step 7 — FinBERT sentiment
FINBERT_FEATURES_PATH: Path = PROCESSED_DIR / "finbert_features.csv"

# Final merged dataset used for all modeling
MODELLING_DATASET_PATH: Path = PROCESSED_DIR / "modelling_dataset.csv"

# ---------------------------------------------------------------------------
# Result paths  (written by modeling notebooks)
# ---------------------------------------------------------------------------

BENCHMARK_TABLE_PATH: Path = RESULTS_DIR / "benchmark_table.csv"
ASYMMETRY_RESULTS_PATH: Path = RESULTS_DIR / "asymmetry_results.csv"
COEFFICIENT_PLOT_PATH: Path = RESULTS_DIR / "coefficient_plot.png"

# ---------------------------------------------------------------------------
# Event-study window parameters
# These constants define the relative trading-day windows used throughout.
# Change them here and the entire pipeline adjusts automatically.
# ---------------------------------------------------------------------------

# Estimation window: days used to fit the market model (OLS) for each event
EST_START: int = -120   # 120 trading days before the call
EST_END: int = -20      # stop 20 days before (buffer so the call itself doesn't contaminate)

# Event window: days over which we measure the market reaction
EVENT_START: int = 0    # day 0 = earnings call day (or next trading day if after close)
EVENT_END_SHORT: int = 1    # CAR[0,1]: 2-day cumulative return
EVENT_END_LONG: int = 3     # CAR[0,3]: 4-day cumulative return

# Volatility windows: pre- vs post-call standard deviation
VOL_PRE_START: int = -10
VOL_PRE_END: int = -1
VOL_POST_START: int = 1
VOL_POST_END: int = 10

# ---------------------------------------------------------------------------
# Modelling parameters
# ---------------------------------------------------------------------------

# Time-based train/test split: train on 2016–2018, test on 2019–2020
TRAIN_YEARS = range(2016, 2019)   # 2016, 2017, 2018
TEST_YEARS = range(2019, 2021)    # 2019, 2020

# TF-IDF vocabulary size
TFIDF_MAX_FEATURES: int = 500

# Word2Vec hyperparameters
W2V_VECTOR_SIZE: int = 100   # embedding dimension
W2V_WINDOW: int = 5          # context window size
W2V_MIN_COUNT: int = 2       # ignore words with fewer occurrences
W2V_EPOCHS: int = 10

# FinBERT
FINBERT_MODEL_NAME: str = "ProsusAI/finbert"
FINBERT_MAX_LENGTH: int = 256   # tokens per sentence (FinBERT max is 512)
FINBERT_BATCH_SIZE: int = 16    # sentences per GPU/CPU batch


def ensure_output_dirs() -> None:
    """Create all output directories if they don't exist."""
    for directory in (PROCESSED_DIR, RESULTS_DIR, LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
