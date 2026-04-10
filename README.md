# NASDAQ-NLP — Earnings Sentiment and Asymmetric Market Reactions

**Research question**: Are negative sentiment signals in earnings call transcripts
stronger predictors of short-window market reactions than positive signals?

We test the **asymmetry hypothesis** — |β_neg| > |β_pos| — using five NLP approaches
applied to earnings calls across NASDAQ-listed firms (2016–2020).

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
make install
# or: uv sync
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
4. Compute Loughran–McDonald lexicon sentiment features
5. Compute TF-IDF features (500 features)
6. Train Word2Vec + compute document embeddings
7. Run FinBERT sentiment (**slow — ~20 min on CPU, runs once**)

### Run notebooks

```bash
make notebooks
```

Or open interactively:
```bash
uv run jupyter notebook
```

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
├── dataset/               Raw transcripts (2016-2020)
│   └── Transcripts/
│       ├── AAPL/ ... NVDA/
├── outputs/               Generated files (gitignored — run pipeline to recreate)
│   ├── processed/         Intermediate CSVs
│   └── results/           Final tables, plots
├── src/nasdaq_nlp/        Main library
│   ├── config.py          All file paths as Path constants
│   ├── data/              loader, metadata, market data
│   ├── preprocessing/     text cleaning and tokenisation
│   ├── features/          lexicon, ngrams, tfidf, embeddings, finbert
│   ├── models/            market_model, classifiers, regression
│   └── evaluation/        metrics (R², OOS R², Wald test)
├── notebooks/
│   ├── 01_data_pipeline.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_results.ipynb
├── docs/
│   ├── paper.tex          LaTeX paper
│   ├── MATH.md            Full math derivations from scratch
│   └── TODO.md            Project checklist
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
| 11 | Sentiment analysis | `features/lexicon.py` |
| 13 | Word2Vec embeddings | `features/embeddings.py` |
| 19-20 | Transformers (FinBERT) | `features/finbert.py` |

---

## Key Results

| Model | Target | OOS R² | Wald p |
|-------|--------|--------|--------|
| Null (mean) | CAR[0,3] | 0.000 | — |
| Market-only | CAR[0,3] | −0.038 | — |
| LM Lexicon OLS | CAR[0,3] | +0.095 | 0.195 |
| LM + Controls | CAR[0,3] | −0.133 | 0.101 |
| Naive Bayes (TF-IDF) | direction | 56.5% acc | — |
| Logistic Regression | direction | 53.6% acc | — |

**Asymmetry finding**: β_neg ≈ −13 vs β_pos ≈ +0.8 (16× larger magnitude).
Wald test p = 0.054 for CAR[0,1] with controls → marginal support for asymmetry (α = 10%).

---

## Math Reference

All mathematical derivations (returns, OLS, AR, CAR, TF-IDF, Naive Bayes,
Logistic Regression, Word2Vec, Wald test, FinBERT) are in [docs/MATH.md](docs/MATH.md).

---

## Dataset

- Earnings call transcripts: Thomson Reuters StreetEvents (via Kaggle)
- Market data: Yahoo Finance via `yfinance`
- LM Dictionary: Loughran & McDonald (2011), *Journal of Finance*
- FinBERT: Araci (2019), `ProsusAI/finbert` (HuggingFace)
