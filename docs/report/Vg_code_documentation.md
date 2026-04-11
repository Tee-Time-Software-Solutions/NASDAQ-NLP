# Code Documentation
**Part V(g) — Modelling Report**
*Asymmetric Sentiment Effects in Earnings Calls — NLP Course, Group Project*

---

## Package Structure

The codebase is organized as an installable Python package (`nasdaq-nlp`) under `src/nasdaq_nlp/`, following the src-layout convention. This prevents accidental imports of the uninstalled source directory.

```
src/nasdaq_nlp/
├── config.py           All output file paths as Path constants
├── data/
│   ├── loader/         Transcript scanning (both dataset sources)
│   ├── metadata/       Header parsing, event trading day assignment
│   └── market/         Price download and return computation
├── preprocessing/      Text cleaning, section split, tokenization
├── features/
│   ├── lexicon.py      LM sentiment rates (full, pres, Q&A)
│   ├── tfidf.py        TF-IDF fitting and transform
│   └── embeddings.py   Word2Vec training and document pooling
├── models/
│   ├── regression.py   OLS, Ridge, RF, MLP, Wald test, time-series CV
│   └── classifiers.py  NB, LogReg, Tree, RF, MLP classifiers
├── evaluation/         OOS R², MAE, AUC utilities
└── visualization.py    All benchmark plots
```

## Module Documentation

Every module begins with a docstring explaining its purpose and public API. Every public function includes `Parameters` and `Returns` sections. Example from `data/loader/main.py`:

```python
def scan_transcripts(
    dataset_dir: Path = DATASET_DIR,
    ect_dir: Path = ECT_DATASET_DIR,
    original: bool = True,
    ect: bool = True,
) -> list[TranscriptRecord]:
    """Scan both transcript datasets and return a combined list.

    Parameters
    ----------
    dataset_dir : Path
        Root of the Thomson Reuters dataset.
    ect_dir : Path
        Root of the Kaggle cleaned_ECTs dataset.
    original : bool
        Include original-source transcripts (default True).
    ect : bool
        Include ECT-source transcripts (default True).

    Returns
    -------
    list[TranscriptRecord]
        Sorted by (ticker, year, quarter).
    """
```

## Setup and Installation

```bash
# Install UV package manager (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies and the nasdaq_nlp package
uv sync
```

## Running the Pipeline

```bash
make pipeline     # full pipeline: metadata -> returns -> features -> models
make serve        # launch Streamlit dashboard
make lint         # ruff linter check
make format       # auto-fix formatting with ruff
make clean        # delete outputs/ (keeps dataset/ intact)
```

Individual pipeline steps can be re-run in isolation. Each step reads its input CSV, applies a single transformation, and writes its output CSV to `outputs/processed/`.

## Reproducibility

- Dependency versions are pinned in `uv.lock`.
- All sklearn models use `random_state=42`.
- Word2Vec is the only non-deterministic step; its output is cached after the first run so subsequent runs are deterministic.
- The full analysis can be reproduced from scratch via:

```bash
uv sync
make pipeline
jupyter nbconvert --execute notebooks/03_modeling.ipynb
jupyter nbconvert --execute notebooks/04_results.ipynb
```

## Code Quality

Static analysis and formatting are enforced by `ruff`:

```bash
make lint     # ruff check + format --check on src/
make format   # ruff check --fix + format src/
```

All modules in `src/nasdaq_nlp/` pass ruff without warnings.

## Repository

Full source code, notebooks, and documentation: https://github.com/Tee-Time-Software-Solutions/NASDAQ-NLP
