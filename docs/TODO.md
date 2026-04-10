# NASDAQ-NLP — Master TODO

> **Today (codebase)**: Milestones 1–11 ✓  
> **Tomorrow (reports)**: Grading checklist items

---

## Milestone 1 — Scaffolding ✓
- [x] `pyproject.toml` (UV deps, setuptools build)
- [x] `Makefile` (install / pipeline / notebooks / serve / lint / clean)
- [x] `src/nasdaq_nlp/config.py` (all paths as `Path` constants)
- [x] Delete old `notebooks/` and `scripts/`
- [x] Rename `archive/` → `dataset/`, set up `outputs/processed/` + `outputs/results/`
- [x] `docs/TODO.md` (this file)

## Milestone 2 — Data Loading ✓
- [x] `src/nasdaq_nlp/data/loader.py`
- [x] `src/nasdaq_nlp/data/metadata.py`
- [x] Verify: 188 transcripts, 10 tickers, 0 header parse failures

## Milestone 3 — Market Data + Returns ✓
- [x] `src/nasdaq_nlp/data/market.py`
- [x] `src/nasdaq_nlp/models/market_model.py` (OLS α,β — AR — CAR[0,1]/[0,3] — ΔVol)
- [x] Verify: 0 NaN in returns, all 188 events have complete CAR data

## Milestone 4 — Preprocessing + Lexicon ✓
- [x] `src/nasdaq_nlp/preprocessing/text.py`
- [x] `src/nasdaq_nlp/features/lexicon.py`
- [x] Verify: NegRate avg=0.0019, PosRate avg=0.0126

## Milestone 5 — N-grams + TF-IDF ✓
- [x] `src/nasdaq_nlp/features/ngrams.py`
- [x] `src/nasdaq_nlp/features/tfidf.py`
- [x] Verify: (188, 500) TF-IDF matrix

## Milestone 6 — Word2Vec Embeddings ✓
- [x] `src/nasdaq_nlp/features/embeddings.py`
- [x] Verify: (188, 100) doc vectors, 'strong' → solid(0.80), healthy(0.73)

## Milestone 7 — FinBERT ✓
- [x] `src/nasdaq_nlp/features/finbert.py`
- [x] Verify: AAPL 2020 pos=0.33 neg=0.06 neu=0.61, sum=1.00

## Milestone 8 — Classifiers + Evaluation ✓
- [x] `src/nasdaq_nlp/models/classifiers.py` (Complement NB + Logistic Regression)
- [x] `src/nasdaq_nlp/evaluation/metrics.py`
- [x] Verify: NB test acc=56.5%, LR test acc=53.6% (both > 50% baseline)

## Milestone 9 — Regression + Asymmetry ✓
- [x] `src/nasdaq_nlp/models/regression.py`
- [x] Verify: β_neg=-13.3, β_pos=+0.8, Wald p=0.054 (CAR[0,1]+controls) → asymmetric ✓

## Milestone 10 — Notebooks (4 clean) ✓
- [x] `notebooks/01_data_pipeline.ipynb` — executes cleanly
- [x] `notebooks/02_feature_extraction.ipynb` — executes cleanly
- [x] `notebooks/03_modeling.ipynb` — executes cleanly
- [x] `notebooks/04_results.ipynb` — executes cleanly

## Milestone 11 — Docs + Math ✓
- [x] `docs/MATH.md` (13 topics: returns, OLS, AR, CAR, ΔVol, TF-IDF, NB, LR, W2V, Wald, FinBERT)
- [x] `README.md` rewrite (UV setup, run order, results summary)

---

## Grading Checklist — Section V: Modelling (16 pts)

| Sub-section | Weight | Status |
|---|---|---|
| Data Collection & Cleaning | 2% | ✓ loader.py, metadata.py, market.py |
| Preprocessing | 2% | ✓ preprocessing/text.py |
| Feature Extraction | 2% | ✓ lexicon, ngrams, tfidf, embeddings, finbert |
| Modelling (correctness + complexity) | 4% | ✓ NB, LR, OLS, Wald test |
| Evaluation | 2% | ✓ R², OOS R², accuracy, F1, Wald p |
| Deployment | 2% | [ ] Streamlit frontend (Phase 2) |
| Code Documentation | 2% | ✓ heavy inline comments throughout src/ |

---

## Report Sections — Submit Tomorrow

| Section | Weight | Status |
|---|---|---|
| II: Literature Review | 2% | done in paper.tex |
| III: Methodology | 2% | done in paper.tex |
| IV: Introduction | 2% | done in paper.tex |
| VI: Results | 2% | [ ] write using benchmark_table.csv + asymmetry_results.csv |
| VII: Abstract + Conclusion | 2% | [ ] write after results |
| VIII: Final submission ZIP | 2% | [ ] |

---

## Phase 2 — Frontend / Deployment (after codebase)

- [ ] `frontend/app.py` — Streamlit dashboard
  - Ticker selector + date range
  - CAR distribution by ticker
  - Sentiment scatter (NegRate vs CAR[0,3])
  - Coefficient table from regression
- [ ] `make serve` target (already in Makefile)
- [ ] Deployment section in README


## QUESTIONBS
1. Clean directory, notebooks 
3. Content review (contrast with guidelines)
4. Checking for redudnant, only code, errorneous conclusions
2. Review paper
5. Write reports 
6. Merge to main + retry install + deployment 
7. Provide zip folder 