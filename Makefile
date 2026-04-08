.PHONY: pipeline serve clean

# ── Full data pipeline ───────────────────────────────────────────────────────
pipeline:
	@echo "=== Step 1/6: Downloading Kaggle transcripts ==="
	python3 scripts/download_data.py

	@echo "=== Step 2/6: Building event metadata ==="
	python3 scripts/build_event_metadata.py

	@echo "=== Step 3/6: Downloading market data ==="
	python3 scripts/download_market_data.py

	@echo "=== Step 4/6: Running processing notebooks ==="
	@# Notebooks use ../data/ relative paths — symlink so they resolve correctly
	@if [ ! -L notebooks/data ]; then ln -s "$$(pwd)/data" notebooks/data; fi
	@for nb in \
		01_validate_event_metadata_final.ipynb \
		02_validate_market_data.ipynb \
		03_compute_returns.ipynb \
		04_build_event_windows.ipynb \
		05_estimate_market_model.ipynb \
		06_compute_abnormal_returns.ipynb \
		07_compute_CAR.ipynb \
		08_compute_volatility_change.ipynb \
		09_assemble_event_study_dataset.ipynb; do \
		echo "  Running $$nb ..."; \
		jupyter nbconvert --to notebook --execute \
			notebooks/Data_Processing/$$nb \
			--ExecutePreprocessor.timeout=600 \
			--output /tmp/nb_out.ipynb > /dev/null 2>&1 || \
			{ echo "  FAILED: $$nb"; exit 1; }; \
	done

	@echo "=== Step 5/6: Computing sentiment features ==="
	python3 scripts/compute_lexicon_sentiment.py
	python3 scripts/compute_finbert_sentiment.py

	@echo "=== Step 6/6: Running models ==="
	python3 notebooks/Data_Modelling/12_model_sentiment_vs_market.py
	python3 notebooks/Data_Modelling/13_asymmetry_tests.py
	python3 notebooks/Data_Modelling/14_results_summary.py

	@echo "=== Copying results to outputs/results/ ==="
	@mkdir -p outputs/results
	cp data/processed/event_study_dataset.csv outputs/results/
	cp data/processed/lexicon_sentiment_features.csv outputs/results/
	cp data/processed/finbert_sentiment_features.csv outputs/results/
	cp data/processed/model_results_car01.csv outputs/results/
	cp data/processed/results_summary_for_paper.csv outputs/results/
	cp data/processed/asymmetry_tests_car01_lexicon.txt outputs/results/

	@echo "=== Pipeline complete ==="

# ── Local Streamlit app ──────────────────────────────────────────────────────
serve:
	streamlit run frontend/app.py
