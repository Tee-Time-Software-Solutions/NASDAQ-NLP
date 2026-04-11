# Deployment
**Part V(f) — Modelling Report**
*Asymmetric Sentiment Effects in Earnings Calls — NLP Course, Group Project*

---

## Overview

We deploy the project as an interactive Streamlit dashboard hosted on Streamlit Cloud. The app allows anyone to explore the model outputs and data without needing to re-run the pipeline locally.

**Live URL:** https://nasdaq-nlp.streamlit.app/

**Source code:** https://github.com/Tee-Time-Software-Solutions/NASDAQ-NLP

## Architecture

The dashboard is organized into four tabs that mirror the four analysis notebooks:

1. **Data Pipeline** — Call timeline by ticker and year, after-hours call breakdown, market model parameters (α̂, β̂) per ticker, and CAR distribution over time.
2. **Feature Extraction** — Loughran–McDonald sentiment rates by ticker, TF-IDF top terms by model, negative rate heat map (ticker × year), and transcript length analysis.
3. **Modelling** — Interactive benchmark table with train/test R², OOS R², MAE, and Wald p-values; actual vs. predicted CAR scatter plots.
4. **Results** — Headline Wald test verdict, OLS coefficient comparison, OOS R² visualization, and economic interpretation of findings.

A fifth "Try It Yourself" tab allows users to input free text and see live LM sentiment scoring and a predicted CAR direction.

## Data Flow

The offline pipeline (`make pipeline`) generates all intermediate CSV files to `outputs/processed/` and final results to `outputs/results/`. A copy of the results CSVs is committed to `frontend/data/` so the dashboard can load pre-computed results without re-running any modeling.

```bash
make pipeline     # build all data + models -> outputs/
make serve        # streamlit run frontend/app.py
```

## Technology Stack

- **Frontend**: Streamlit (Python), Altair (interactive charts).
- **Hosting**: Streamlit Community Cloud (free tier).
- **Data format**: Pre-computed CSV files; sub-second load times, no live inference.

All charts use Altair for interactive tooltips and zooming. A global sidebar provides ticker filtering and CAR window selection (CAR[0,1] vs. CAR[0,3]), propagating the selection to all tabs.

## Code Reference

- `frontend/app.py` — main Streamlit entry point.
- `frontend/data/` — pre-committed result CSVs served by the app.
