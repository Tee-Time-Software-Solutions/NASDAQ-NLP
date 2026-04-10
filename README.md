# NASDAQ-NLP — Earnings Sentiment and Asymmetric Market Reactions

**Research question**: Are negative sentiment signals in earnings call transcripts
stronger predictors of short-window market reactions than positive signals?

We test the **asymmetry hypothesis** — |β_neg| > |β_pos| — using NLP features
applied to 620 earnings calls across 19 technology-sector firms (2007–2025).

**Live dashboard**: https://tee-time-software-solutions-nasdaq-nlp-frontendapp-featu-pvfgpc.streamlit.app/

---

## Setup

### Requirements
- Python ≥ 3.10
- [UV](https://docs.astral.sh/uv/) package manager

### Install

```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies and the nasdaq_nlp package
uv sync
```

---

## Run the Pipeline

### Full pipeline (all steps)

```bash
make pipeline
```

This runs in order:
1. Parse transcript metadata → `outputs/processed/event_metadata.csv`
2. Download market data (yfinance) + compute daily returns
3. Estimate OLS market model → compute AR, CAR[0,1/3], ΔVol
4. Compute Loughran–McDonald lexicon sentiment features (full, presentation, Q&A)
5. Compute TF-IDF features (500 features, min_df=3)
6. Train Word2Vec + compute document embeddings (100-dim skip-gram)

### Run notebooks

```bash
uv run jupyter notebook
```

Notebooks in order:
- `01_data_pipeline.ipynb` — data loading and event study
- `02_feature_extraction.ipynb` — sentiment features
- `03_modeling.ipynb` — regression and classification models
- `04_results.ipynb` — asymmetry test, OOS R², plots

### Other targets

```bash
make lint        # ruff linter
make format      # auto-fix formatting
make clean       # remove generated outputs (keeps dataset/ intact)
make serve       # launch Streamlit dashboard (frontend/)
```

---

## Project Structure

```
NASDAQ-NLP/
├── dataset/               Raw transcripts (original Thomson Reuters dataset)
│   └── Transcripts/
│       ├── AAPL/ ... NVDA/
├── outputs/               Generated files (gitignored — run pipeline to recreate)
│   ├── processed/         Intermediate CSVs
│   └── results/           Final tables, plots
├── src/nasdaq_nlp/        Main library
│   ├── config.py          All file paths as Path constants
│   ├── data/              loader, metadata, market data
│   ├── preprocessing/     text cleaning and tokenisation
│   ├── features/          lexicon, tfidf, embeddings
│   ├── models/            market_model, classifiers, regression
│   └── evaluation/        metrics (R², OOS R², Wald test)
├── notebooks/
│   ├── 01_data_pipeline.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_results.ipynb
├── frontend/              Streamlit dashboard
├── docs/
│   ├── paper.tex          LaTeX paper (IEEEtran format)
│   └── report/            Individual submission reports (I–VII)
├── pyproject.toml         UV-compatible package config
└── Makefile               Task runner
```

---

## NLP Concepts Covered

| Course Session | Concept | Implementation |
|---|---|---|
| 3 | N-grams | `features/ngrams.py` |
| 7 | TF-IDF | `features/tfidf.py` |
| 9 | Naive Bayes | `models/classifiers.py` |
| 10 | Logistic Regression | `models/classifiers.py` |
| 11 | Sentiment analysis (LM lexicon) | `features/lexicon.py` |
| 13 | Word2Vec embeddings | `features/embeddings.py` |

---

## Key Results

| Model | Target | OOS R² | Wald p |
|-------|--------|--------|--------|
| Null (mean) | CAR[0,3] | 0.000 | — |
| **LM Lexicon [OLS]** | CAR[0,3] | **+0.057** | 0.485 |
| Word2Vec [Ridge] | CAR[0,3] | +0.050 | — |
| TF-IDF [RF] | CAR[0,3] | +0.045 | — |
| **LM + Controls [OLS]** | CAR[0,3] | −0.086 | **0.042** |

**Asymmetry finding**: β_neg = −8.92 vs β_pos = +3.31 (2.7× larger magnitude).
Wald test p = 0.042 with controls → H₀ rejected at 5% level.
Effect is concentrated in the Q&A section (β_neg_qa = −10.3).

**Classification**: Best accuracy 59.6%, best AUC 0.600 (TF-IDF + LM + Controls [RF]).

---

## Dataset

- Earnings call transcripts: Thomson Reuters StreetEvents + Kaggle cleaned_ECTs
- Market data: Yahoo Finance via `yfinance`
- LM Dictionary: Loughran & McDonald (2011), *Journal of Finance*
