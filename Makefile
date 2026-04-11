.DEFAULT_GOAL := help
.PHONY: help install pipeline finbert notebooks serve lint clean format test all

# ── Python / UV ──────────────────────────────────────────────────────────────
# UV must be installed:  curl -LsSf https://astral.sh/uv/install.sh | sh

help:          ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:       ## Create venv and install all dependencies via UV
	uv sync

# ── Run everything: pipeline then serve ─────────────────────────────────────
all: pipeline serve  ## Run full pipeline then launch dashboard

# ── Data pipeline ────────────────────────────────────────────────────────────
pipeline:      ## Run the full data pipeline (metadata → market → features)
	@echo "── Step 1: build event metadata ──"
	uv run python -c "from nasdaq_nlp.data.metadata import build_event_metadata; build_event_metadata()"
	@echo "── Step 2: download market data + compute returns ──"
	uv run python -c "from nasdaq_nlp.data.market import build_market_returns; build_market_returns()"
	@echo "── Step 3: compute market model (OLS α,β), AR, CAR, ΔVol ──"
	uv run python -c "from nasdaq_nlp.models.market_model import build_event_study; build_event_study()"
	@echo "── Step 4: lexicon sentiment features ──"
	uv run python -c "from nasdaq_nlp.features.lexicon import build_lexicon_features; build_lexicon_features()"
	@echo "── Step 5: n-gram + TF-IDF features ──"
	uv run python -c "from nasdaq_nlp.features.tfidf import build_tfidf_features; build_tfidf_features()"
	@echo "── Step 6: Word2Vec document embeddings ──"
	uv run python -c "from nasdaq_nlp.features.embeddings import build_embedding_features; build_embedding_features()"
	@echo "── Step 7: FinBERT sentiment (slow — runs transformer) ──"
	uv run python -c "from nasdaq_nlp.features.finbert import build_finbert_features; build_finbert_features()"
	@echo ""
	@echo "✓ Pipeline complete.  Outputs in outputs/processed/"

ect-pipeline:  ## Import cleaned_ECTs dataset and run full pipeline on combined data
	@echo "── Step 1: build ECT event metadata (resolves event dates via yfinance) ──"
	uv run python -c "from nasdaq_nlp.data.loader.ect import build_ect_metadata; build_ect_metadata()"
	@echo "── Step 2: merge with original metadata → combined_event_metadata.csv ──"
	uv run python -c "from nasdaq_nlp.data.loader.ect import build_combined_metadata; build_combined_metadata()"
	@echo "── Step 3: download market data for all tickers + indices ──"
	uv run python -c "from pathlib import Path; from nasdaq_nlp.data.market import build_market_returns; build_market_returns(Path('outputs/processed/combined_event_metadata.csv'))"
	@echo "── Step 4: compute market model + CAR for all events ──"
	uv run python -c "from pathlib import Path; from nasdaq_nlp.models.market_model import build_event_study; from nasdaq_nlp.config import EVENT_STUDY_PATH, STOCK_RETURNS_PATH, INDEX_RETURNS_PATH, MARKET_MODEL_PATH; build_event_study(Path('outputs/processed/combined_event_metadata.csv'), STOCK_RETURNS_PATH, INDEX_RETURNS_PATH, MARKET_MODEL_PATH, EVENT_STUDY_PATH)"
	@echo "── Step 5: lexicon features (re-run on all events) ──"
	uv run python -c "from nasdaq_nlp.features.lexicon import build_lexicon_features; build_lexicon_features()"
	@echo "── Step 6: TF-IDF features ──"
	uv run python -c "from nasdaq_nlp.features.tfidf import build_tfidf_features; build_tfidf_features()"
	@echo "── Step 7: Word2Vec embeddings ──"
	uv run python -c "from nasdaq_nlp.features.embeddings import build_embedding_features; build_embedding_features()"
	@echo ""
	@echo "✓ ECT pipeline complete. Run notebooks/03 and notebooks/04 to see updated results."

finbert:       ## Run FinBERT feature extraction (slow — ~15-30 min on CPU)
	@echo "── FinBERT: scoring 188 transcripts ──"
	@echo "   Progress logs stream to stdout. Kill safely — resumes from checkpoint."
	uv run python -c "from nasdaq_nlp.features.finbert import build_finbert_features; build_finbert_features()"
	@echo "✓ FinBERT complete. Output: outputs/processed/finbert_features.csv"

finbert-bg:    ## Run FinBERT in background, streaming logs to outputs/logs/finbert.log
	@echo "Starting FinBERT in background → outputs/logs/finbert.log"
	@mkdir -p outputs/logs
	nohup uv run python -c \
		"from nasdaq_nlp.features.finbert import build_finbert_features; build_finbert_features()" \
		> outputs/logs/finbert.log 2>&1 &
	@echo "PID: $$!  —  tail -f outputs/logs/finbert.log"

# ── Notebooks ────────────────────────────────────────────────────────────────
# Executes all 4 notebooks headlessly; fails loudly if any cell raises.

notebooks:     ## Execute all 4 notebooks headlessly (requires pipeline to run first)
	uv run jupyter nbconvert --to notebook --execute \
		--ExecutePreprocessor.timeout=600 \
		--output-dir notebooks/executed \
		notebooks/01_data_pipeline.ipynb \
		notebooks/02_feature_extraction.ipynb \
		notebooks/03_modeling.ipynb \
		notebooks/04_results.ipynb
	@echo "✓ All notebooks executed cleanly."

# ── Frontend ─────────────────────────────────────────────────────────────────
serve:         ## Launch the Streamlit dashboard (frontend/)
	uv run streamlit run frontend/app.py

# ── Testing ──────────────────────────────────────────────────────────────────
test:          ## Run unit tests with coverage
	pytest tests/ -v --tb=short --cov --cov-report=term-missing

# ── Code quality ─────────────────────────────────────────────────────────────
lint:          ## Run ruff linter + formatter check
	uv run ruff check src/
	uv run ruff format --check src/

format:        ## Auto-fix lint issues and reformat
	uv run ruff check --fix src/
	uv run ruff format src/

# ── Clean ────────────────────────────────────────────────────────────────────
clean:         ## Remove generated data and outputs
	rm -rf outputs/processed/ outputs/results/ outputs/logs/
