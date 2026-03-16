# NASDAQ-NLP

## Project Overview
This project investigates whether **negative sentiment in earnings call transcripts has a stronger impact on market reactions than positive sentiment**.

The analysis uses earnings call transcripts and applies NLP techniques to extract sentiment signals, which are then linked to stock market reactions such as cumulative abnormal returns (CAR) and volatility.

---

## Repository Structure

```
NASDAQ-NLP/
│
├── data/
│   ├── raw/           # Raw datasets (downloaded automatically, not tracked by git)
│   └── processed/     # Cleaned datasets used for modelling
│
├── notebooks/         # Jupyter notebooks for analysis
│
├── scripts/           # Utility scripts (dataset download, preprocessing)
│
├── src/               # Core project code
│
├── outputs/           # Generated figures, tables, and results
│
├── .env.example       # Template for environment variables
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup

Clone the repository:

```bash
git clone <repo-url>
cd NASDAQ-NLP
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create environment variables file:

```bash
cp .env.example .env
```

Then open `.env` and add your Kaggle credentials:

```
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

---

## Download Dataset

Run the dataset download script:

```bash
python scripts/download_data.py
```

This will download the **Earnings Call Transcripts dataset from Kaggle** and extract it into:

```
data/raw/earnings-call-transcripts/
```

Raw datasets are **not stored in this repository** and must be downloaded locally.

Dataset source:  
https://www.kaggle.com/datasets/ashwinm500/earnings-call-transcripts

---

## Running the Project

Typical workflow:

1. Download the dataset
2. Run preprocessing notebooks
3. Extract sentiment features
4. Train and evaluate models

Start Jupyter:

```bash
jupyter notebook
```

Then open notebooks in the `notebooks/` directory.

---

## Reproducibility

To recreate the environment:

```bash
pip install -r requirements.txt
```

Ensure `.env` contains valid Kaggle credentials before downloading the dataset.

---

## Authors

