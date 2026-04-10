"""
frontend/app.py — NASDAQ-NLP Interactive Dashboard

Five tabs — four mirroring the project notebooks, plus live inference:
  1. Data Pipeline       (notebook 01) — call timeline, after-hours breakdown, market-model stats
  2. Feature Extraction  (notebook 02) — lexicon rates, TF-IDF top terms, sentiment heatmap
  3. Modelling           (notebook 03) — benchmark table, model comparison charts
  4. Results             (notebook 04) — asymmetry test, coefficient plot, interpretation
  5. Try It Yourself     — upload transcripts for live sentiment scoring and CAR prediction

Data source: frontend/data/ (pre-computed CSVs committed to git).
Tab 5 runs live inference using Loughran-McDonald lexicon and optionally FinBERT.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

_FINBERT_AVAILABLE = False
try:
    import transformers  # noqa: F401
    _FINBERT_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NASDAQ-NLP · Earnings Sentiment & Markets",
    page_icon="📈",
    layout="wide",
)

DATA_DIR = Path(__file__).parent / "data"

# ---------------------------------------------------------------------------
# Data loaders — @st.cache_data so CSVs are read only once per session
# ---------------------------------------------------------------------------

@st.cache_data
def load_event_study() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "event_study_dataset.csv",
                     parse_dates=["event_trading_day"])
    return df

@st.cache_data
def load_event_metadata() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "event_metadata.csv",
                     parse_dates=["event_trading_day"])
    return df

@st.cache_data
def load_lexicon() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "lexicon_features.csv",
                       parse_dates=["event_trading_day"])

@st.cache_data
def load_benchmark() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "benchmark_table.csv")

@st.cache_data
def load_asymmetry() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "asymmetry_results.csv")

@st.cache_data
def load_tfidf_top_terms() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "tfidf_top_terms.csv")

@st.cache_data
def load_sentiment_over_time() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "sentiment_over_time.csv")

@st.cache_data
def load_ticker_model_summary() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "ticker_model_summary.csv")

# ---------------------------------------------------------------------------
# Load all data
# ---------------------------------------------------------------------------
study    = load_event_study()
meta     = load_event_metadata()
lex      = load_lexicon()
bench    = load_benchmark()
asym     = load_asymmetry()
top_terms= load_tfidf_top_terms()
sot      = load_sentiment_over_time()
ticker_mm= load_ticker_model_summary()

ALL_TICKERS = sorted(study["ticker"].unique().tolist())

# ---------------------------------------------------------------------------
# Loughran-McDonald mini word lists (for live scoring)
# ---------------------------------------------------------------------------
_NEG_WORDS = {
    "loss", "losses", "decline", "declines", "risk", "uncertain",
    "negative", "downturn", "weak", "concern", "headwind",
}
_POS_WORDS = {
    "profit", "profits", "growth", "strong", "opportunity",
    "opportunities", "improve", "improving", "record", "robust",
    "positive", "upside",
}


@st.cache_resource
def load_finbert():
    from transformers import pipeline
    return pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        return_all_scores=True,
        truncation=True,
        max_length=256,
    )


def score_lexicon(text: str) -> dict:
    tokens = re.findall(r"[A-Za-z']+", text.lower())
    total = len(tokens) or 1
    neg = sum(1 for t in tokens if t in _NEG_WORDS)
    pos = sum(1 for t in tokens if t in _POS_WORDS)
    return {
        "total_tokens": len(tokens),
        "neg_count": neg,
        "pos_count": pos,
        "neg_rate": neg / total,
        "pos_rate": pos / total,
    }


def score_finbert(text: str) -> dict:
    clf = load_finbert()
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    if not sentences:
        return {"finbert_neg_mean": 0.0, "finbert_pos_mean": 0.0, "finbert_neu_mean": 0.0}

    all_scores = clf(sentences, batch_size=16)

    sums: Counter = Counter()
    n = 0
    for sent_scores in all_scores:
        if isinstance(sent_scores, dict):
            sent_scores = [sent_scores]
        n += 1
        for s in sent_scores:
            sums[s["label"].lower()] += s["score"]

    if n == 0:
        return {"finbert_neg_mean": 0.0, "finbert_pos_mean": 0.0, "finbert_neu_mean": 0.0}

    return {
        "finbert_neg_mean": sums.get("negative", 0.0) / n,
        "finbert_pos_mean": sums.get("positive", 0.0) / n,
        "finbert_neu_mean": sums.get("neutral", 0.0) / n,
    }


def predict_car(asym_df: pd.DataFrame, model_name: str, target: str,
                features: dict) -> float | None:
    """Use OLS coefficients from asymmetry_results.csv to predict CAR."""
    row = asym_df[(asym_df["model"] == model_name) & (asym_df["target"] == target)]
    if row.empty:
        return None
    row = row.iloc[0]

    const = row.get("coef_const", 0.0)
    if pd.isna(const):
        const = 0.0

    car = const
    coef_map = {
        "neg_rate": "coef_neg_rate",
        "pos_rate": "coef_pos_rate",
    }
    for feat, val in features.items():
        col = coef_map.get(feat)
        if col and pd.notna(row.get(col)):
            car += row[col] * val
    return car

# ---------------------------------------------------------------------------
# Sidebar — global controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("NASDAQ-NLP")
    st.caption("Earnings Sentiment & Market Reactions · 2016–2020")
    st.divider()

    selected_tickers = st.multiselect(
        "Filter tickers",
        options=ALL_TICKERS,
        default=ALL_TICKERS,
    )

    car_target = st.radio(
        "CAR window",
        options=["car_01", "car_03"],
        format_func=lambda x: "CAR[0,1] — 2-day" if x == "car_01" else "CAR[0,3] — 4-day",
        index=1,
    )

    st.divider()
    st.markdown(
        "**Research question**\n\n"
        "Is |β_neg| > |β_pos|?  \n"
        "*(Do markets react more to bad news than good news?)*"
    )

# Apply ticker filter
study_f  = study[study["ticker"].isin(selected_tickers)]
meta_f   = meta[meta["ticker"].isin(selected_tickers)]
lex_f    = lex[lex["ticker"].isin(selected_tickers)]
sot_f    = sot[sot["ticker"].isin(selected_tickers)]
top_f    = top_terms[top_terms["ticker"].isin(selected_tickers)]
mm_f     = ticker_mm[ticker_mm["ticker"].isin(selected_tickers)]

# Merge lexicon onto study once (used in multiple tabs)
study_lex = study_f.merge(
    lex_f[["file_name", "neg_rate", "pos_rate", "total_tokens"]],
    on="file_name", how="left",
)

# ---------------------------------------------------------------------------
# Tabs — same order as the four notebooks
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📦 01 · Data Pipeline",
    "🔬 02 · Feature Extraction",
    "🤖 03 · Modelling",
    "📊 04 · Results",
    "🧪 05 · Try It Yourself",
])


# ===========================================================================
# TAB 1 — Data Pipeline  (mirrors notebook 01_data_pipeline.ipynb)
# ===========================================================================
with tab1:
    st.header("Data Pipeline")
    st.markdown(
        "This tab mirrors **notebook 01**: raw transcript scan → event metadata → "
        "market model → CAR / ΔVol computation."
    )

    # ---- 1A. Dataset summary -----------------------------------------------
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Transcripts", len(study_f))
    col_b.metric("Tickers", len(study_f["ticker"].unique()))
    col_c.metric("Years", f"{int(study_f['year'].min())}–{int(study_f['year'].max())}")
    after_pct = meta_f["after_market_close"].mean() * 100 if len(meta_f) else 0
    col_d.metric("After-hours calls", f"{after_pct:.0f}%")

    st.divider()

    # ---- 1B. Timeline of earnings calls ------------------------------------
    st.subheader("Earnings Call Timeline")
    st.caption("Each dot = one earnings call. Color = ticker.")

    timeline_data = study_f[["ticker", "event_trading_day", car_target]].copy()
    timeline_data.columns = ["Ticker", "Date", "CAR"]

    timeline = (
        alt.Chart(timeline_data)
        .mark_circle(size=70)
        .encode(
            x=alt.X("Date:T", title="Event trading day"),
            y=alt.Y("Ticker:N", title=None),
            color=alt.Color("Ticker:N", legend=None),
            size=alt.Size("CAR:Q", scale=alt.Scale(range=[20, 200]), legend=None),
            tooltip=["Ticker", alt.Tooltip("Date:T", format="%Y-%m-%d"),
                     alt.Tooltip("CAR:Q", format=".3f", title="CAR")],
        )
        .properties(height=280)
        .interactive()
    )
    st.altair_chart(timeline, width="stretch")

    # ---- 1C. After-hours breakdown -----------------------------------------
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Call Timing Distribution")
        if len(meta_f) > 0:
            timing_counts = meta_f["after_market_close"].value_counts().reset_index()
            timing_counts.columns = ["After close", "Count"]
            timing_counts["Label"] = timing_counts["After close"].map(
                {True: "After 4 PM ET", False: "Before 4 PM ET"}
            )
            pie = (
                alt.Chart(timing_counts)
                .mark_arc(innerRadius=50)
                .encode(
                    theta=alt.Theta("Count:Q"),
                    color=alt.Color("Label:N",
                                    scale=alt.Scale(range=["#d62728", "#2ca02c"])),
                    tooltip=["Label", "Count"],
                )
                .properties(height=250)
            )
            st.altair_chart(pie, width="stretch")

    with col_right:
        st.subheader("Calls per Quarter")
        q_counts = study_f.groupby(["year", "quarter"]).size().reset_index(name="n")
        q_counts["period"] = q_counts["year"].astype(str) + "-" + q_counts["quarter"]
        bar_q = (
            alt.Chart(q_counts)
            .mark_bar()
            .encode(
                x=alt.X("period:N", sort=None, title="Year-Quarter"),
                y=alt.Y("n:Q", title="# Calls"),
                color=alt.Color("quarter:N",
                                scale=alt.Scale(scheme="tableau10")),
                tooltip=["period", "n"],
            )
            .properties(height=250)
        )
        st.altair_chart(bar_q, width="stretch")

    st.divider()

    # ---- 1D. Market model parameters (α, β) per ticker --------------------
    st.subheader("Market Model Parameters (α, β) per Ticker")
    st.caption(
        "Estimated on a 100-day pre-event window (days −120 to −20).  "
        "β > 1 = more volatile than the NASDAQ index."
    )

    mm_display = mm_f[["ticker", "alpha", "beta", "n", "mean_car03", "mean_pre_vol"]].copy()
    mm_display.columns = ["Ticker", "α (alpha)", "β (beta)", "Events",
                          "Mean CAR[0,3]", "Mean Pre-vol"]
    for col in ["α (alpha)", "β (beta)", "Mean CAR[0,3]", "Mean Pre-vol"]:
        mm_display[col] = mm_display[col].apply(lambda v: f"{v:.4f}")
    st.dataframe(mm_display, width="stretch", hide_index=True)

    # ---- 1E. CAR distribution over time ------------------------------------
    st.subheader(f"{'CAR[0,1]' if car_target == 'car_01' else 'CAR[0,3]'} Over Time")

    car_time = sot_f[["period", "ticker", car_target]].copy()
    car_time.columns = ["Period", "Ticker", "CAR"]

    car_line = (
        alt.Chart(car_time)
        .mark_line(point=True)
        .encode(
            x=alt.X("Period:N", sort=None, title="Year-Quarter"),
            y=alt.Y("CAR:Q", title="Mean CAR"),
            color=alt.Color("Ticker:N"),
            tooltip=["Ticker", "Period", alt.Tooltip("CAR:Q", format=".4f")],
        )
        .properties(height=320)
        .interactive()
    )
    st.altair_chart(car_line, width="stretch")

    # ---- 1F. Volatility change distribution --------------------------------
    st.subheader("Post-event Volatility Change (ΔVol)")
    st.caption("ΔVol = post-event σ[+1,+10] − pre-event σ[−10,−1]. Positive = more volatile after.")

    vol_data = study_f[["ticker", "delta_vol"]].copy()
    vol_data.columns = ["Ticker", "ΔVol"]

    vol_hist = (
        alt.Chart(vol_data)
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X("ΔVol:Q", bin=alt.Bin(maxbins=30), title="ΔVol"),
            y=alt.Y("count():Q", title="Count"),
            color=alt.Color("Ticker:N"),
            tooltip=["Ticker", "count()"],
        )
        .properties(height=280)
    )
    st.altair_chart(vol_hist, width="stretch")


# ===========================================================================
# TAB 2 — Feature Extraction  (mirrors notebook 02_feature_extraction.ipynb)
# ===========================================================================
with tab2:
    st.header("Feature Extraction")
    st.markdown(
        "This tab mirrors **notebook 02**: lexicon sentiment, TF-IDF top terms, "
        "and sentiment rate comparisons — the five feature methods from the course."
    )

    # ---- 2A. Lexicon sentiment rates by ticker ----------------------------
    st.subheader("Loughran–McDonald Sentiment Rates by Ticker")

    lex_agg = lex_f.groupby("ticker")[["neg_rate", "pos_rate"]].mean().reset_index()
    lex_long = lex_agg.melt(id_vars="ticker", var_name="Type", value_name="Rate")
    lex_long["Type"] = lex_long["Type"].map({"neg_rate": "Negative", "pos_rate": "Positive"})

    lex_bar = (
        alt.Chart(lex_long)
        .mark_bar()
        .encode(
            x=alt.X("ticker:N", title="Ticker"),
            y=alt.Y("Rate:Q", title="Mean word rate"),
            color=alt.Color("Type:N",
                            scale=alt.Scale(domain=["Negative", "Positive"],
                                            range=["#d62728", "#2ca02c"])),
            xOffset=alt.XOffset("Type:N"),
            tooltip=["ticker", "Type", alt.Tooltip("Rate:Q", format=".5f")],
        )
        .properties(height=320, title="Average Negative / Positive Word Rate per Ticker")
    )
    st.altair_chart(lex_bar, width="stretch")

    # ---- 2B. Sentiment rates over time (line chart) -----------------------
    st.subheader("Sentiment Rates Over Time")
    st.caption("Quarterly mean across all selected tickers. Watch for trend breaks around 2020.")

    sot_agg = (sot_f.groupby("period")[["neg_rate", "pos_rate"]]
               .mean().reset_index()
               .melt(id_vars="period", var_name="Type", value_name="Rate"))
    sot_agg["Type"] = sot_agg["Type"].map({"neg_rate": "Negative", "pos_rate": "Positive"})

    sent_line = (
        alt.Chart(sot_agg)
        .mark_line(point=True)
        .encode(
            x=alt.X("period:N", sort=None, title="Year-Quarter"),
            y=alt.Y("Rate:Q", title="Mean word rate"),
            color=alt.Color("Type:N",
                            scale=alt.Scale(domain=["Negative", "Positive"],
                                            range=["#d62728", "#2ca02c"])),
            tooltip=["period", "Type", alt.Tooltip("Rate:Q", format=".5f")],
        )
        .properties(height=300)
        .interactive()
    )
    st.altair_chart(sent_line, width="stretch")

    st.divider()

    # ---- 2C. TF-IDF top terms per ticker -----------------------------------
    st.subheader("TF-IDF Top Terms per Ticker")
    st.caption("Mean TF-IDF score across all earnings calls for a ticker. Higher = more distinctive term.")

    ticker_sel = st.selectbox("Select ticker", options=selected_tickers, key="tfidf_ticker")
    top_ticker = top_f[top_f["ticker"] == ticker_sel].sort_values("mean_tfidf", ascending=False).head(15)

    tfidf_chart = (
        alt.Chart(top_ticker)
        .mark_bar(color="#0066cc")
        .encode(
            x=alt.X("mean_tfidf:Q", title="Mean TF-IDF score"),
            y=alt.Y("term:N", sort="-x", title="Term"),
            tooltip=["term", alt.Tooltip("mean_tfidf:Q", format=".4f", title="Score")],
        )
        .properties(height=380, title=f"Top 15 TF-IDF Terms — {ticker_sel}")
    )
    st.altair_chart(tfidf_chart, width="stretch")

    # ---- 2D. Negative rate scatter vs total tokens (transcript length) ----
    st.divider()
    st.subheader("Negative Rate vs Transcript Length")
    st.caption("Controls for length confound — does a longer call mean more negative words?")

    scatter_data = study_lex[["ticker", "neg_rate", "total_tokens", car_target]].dropna()
    scatter_data.columns = ["Ticker", "NegRate", "Tokens", "CAR"]

    length_scatter = (
        alt.Chart(scatter_data)
        .mark_circle(size=60, opacity=0.7)
        .encode(
            x=alt.X("Tokens:Q", title="Total transcript tokens (length)"),
            y=alt.Y("NegRate:Q", title="Negative word rate"),
            color=alt.Color("Ticker:N"),
            size=alt.Size("CAR:Q", scale=alt.Scale(range=[10, 200]), legend=None),
            tooltip=["Ticker", alt.Tooltip("Tokens:Q"), alt.Tooltip("NegRate:Q", format=".4f"),
                     alt.Tooltip("CAR:Q", format=".4f")],
        )
        .properties(height=350)
        .interactive()
    )
    st.altair_chart(length_scatter, width="stretch")

    # ---- 2E. Sentiment comparison heat map (ticker × year) ----------------
    st.divider()
    st.subheader("Negative Sentiment Rate Heat Map (Ticker × Year)")

    heat_data = (lex_f.groupby(["ticker", lex_f["event_trading_day"].dt.year])["neg_rate"]
                 .mean().reset_index())
    heat_data.columns = ["Ticker", "Year", "NegRate"]

    heatmap = (
        alt.Chart(heat_data)
        .mark_rect()
        .encode(
            x=alt.X("Year:O", title="Year"),
            y=alt.Y("Ticker:N", title=None),
            color=alt.Color("NegRate:Q",
                            scale=alt.Scale(scheme="reds"),
                            title="Mean NegRate"),
            tooltip=["Ticker", "Year", alt.Tooltip("NegRate:Q", format=".5f")],
        )
        .properties(height=300, title="Negative Language Rate by Ticker and Year")
    )
    st.altair_chart(heatmap, width="stretch")


# ===========================================================================
# TAB 3 — Modelling  (mirrors notebook 03_modeling.ipynb)
# ===========================================================================
with tab3:
    st.header("Modelling")
    st.markdown(
        "This tab mirrors **notebook 03**: classification (Naive Bayes, Logistic Regression) "
        "and regression (OLS) across all feature sets. The benchmark table lets you compare "
        "model performance interactively."
    )

    # ---- 3A. Benchmark table -----------------------------------------------
    st.subheader("Model Benchmark Table")

    target_choice = st.radio(
        "Target variable",
        options=["car_03", "car_01"],
        format_func=lambda x: "CAR[0,3] — 4-day window" if x == "car_03" else "CAR[0,1] — 2-day window",
        horizontal=True,
        key="bench_target",
    )

    sub = bench[bench["target"] == target_choice].copy().drop(columns=["target"])
    for col in ["train_r2", "test_r2", "oos_r2", "wald_p"]:
        sub[col] = sub[col].apply(lambda v: f"{v:.4f}" if pd.notna(v) else "—")
    sub = sub.rename(columns={
        "model": "Model", "n_train": "Train n", "n_test": "Test n",
        "train_r2": "Train R²", "test_r2": "Test R²", "oos_r2": "OOS R²", "wald_p": "Wald p",
    })
    st.dataframe(sub, width="stretch", hide_index=True)

    # ---- 3B. R² comparison bar chart (train vs test) ----------------------
    st.subheader("Train R² vs Test R² — All Models")
    st.caption("A large train→test gap = overfitting. OLS models here have very small gaps (low variance).")

    bench_plot = bench[bench["target"] == target_choice].copy()
    bench_long = bench_plot.melt(
        id_vars=["model"], value_vars=["train_r2", "test_r2", "oos_r2"],
        var_name="Split", value_name="R²"
    ).dropna()
    bench_long["Split"] = bench_long["Split"].map({
        "train_r2": "In-sample (train)", "test_r2": "Test", "oos_r2": "OOS R²"
    })

    r2_bar = (
        alt.Chart(bench_long)
        .mark_bar()
        .encode(
            x=alt.X("model:N", title=None,
                    sort=list(bench_plot["model"])),
            y=alt.Y("R²:Q"),
            color=alt.Color("Split:N",
                            scale=alt.Scale(scheme="tableau10")),
            xOffset=alt.XOffset("Split:N"),
            tooltip=["model", "Split", alt.Tooltip("R²:Q", format=".4f")],
        )
        .properties(height=350, title=f"R² by Model — {target_choice.upper()}")
    )
    st.altair_chart(r2_bar, width="stretch")

    st.divider()

    # ---- 3C. CAR scatter: actual vs predicted (NegRate model proxy) --------
    st.subheader("Actual CAR vs NegRate (best linear predictor)")
    st.caption(
        "The OLS line shows the learned relationship. Points far from the line are "
        "earnings calls where sentiment alone fails to explain the market reaction."
    )

    scatter_df = study_lex[["ticker", "neg_rate", car_target]].dropna().copy()
    scatter_df.columns = ["Ticker", "NegRate", "CAR"]

    scatter_base = (
        alt.Chart(scatter_df)
        .mark_circle(size=55, opacity=0.65)
        .encode(
            x=alt.X("NegRate:Q", title="Negative word rate (LM lexicon)"),
            y=alt.Y("CAR:Q", title=f"Actual {'CAR[0,1]' if car_target == 'car_01' else 'CAR[0,3]'}"),
            color=alt.Color("Ticker:N"),
            tooltip=["Ticker", alt.Tooltip("NegRate:Q", format=".5f"),
                     alt.Tooltip("CAR:Q", format=".4f")],
        )
        .properties(height=380)
    )
    reg = scatter_base.transform_regression("NegRate", "CAR").mark_line(
        color="black", strokeDash=[6, 3], size=2
    )
    st.altair_chart((scatter_base + reg).interactive(), width="stretch")

    st.divider()

    # ---- 3D. CAR distribution: positive vs negative events ----------------
    st.subheader("CAR Distribution — Positive vs Negative Calls")
    st.caption("Calls with above-median NegRate vs below-median. Do negative calls cluster below zero?")

    median_neg = study_lex["neg_rate"].median()
    study_lex_copy = study_lex.copy()
    study_lex_copy["Sentiment"] = study_lex_copy["neg_rate"].apply(
        lambda v: "High Negative" if v >= median_neg else "Low Negative"
    )

    dist_data = study_lex_copy[["Sentiment", car_target]].dropna().copy()
    dist_data.columns = ["Sentiment", "CAR"]

    dist_chart = (
        alt.Chart(dist_data)
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X("CAR:Q", bin=alt.Bin(maxbins=25), title="CAR"),
            y=alt.Y("count():Q", title="Count"),
            color=alt.Color("Sentiment:N",
                            scale=alt.Scale(domain=["High Negative", "Low Negative"],
                                            range=["#d62728", "#2ca02c"])),
            tooltip=["Sentiment", "count()"],
        )
        .properties(height=300)
    )
    st.altair_chart(dist_chart, width="stretch")


# ===========================================================================
# TAB 4 — Results  (mirrors notebook 04_results.ipynb)
# ===========================================================================
with tab4:
    st.header("Results: Sentiment Asymmetry")
    st.markdown(
        "This tab mirrors **notebook 04**: the Wald test for asymmetry, coefficient "
        "comparison, OOS R², and the final conclusion on the research question."
    )

    # ---- 4A. Headline result -----------------------------------------------
    best = asym.loc[asym["wald_p"].idxmin()]
    is_sig = best["wald_p"] < 0.10
    asymmetry_ratio = abs(best["coef_neg_rate"] / best["coef_pos_rate"])

    verdict_color = "#d4edda" if is_sig else "#fff3cd"
    border_color  = "#28a745" if is_sig else "#ffc107"
    verdict_text  = "ASYMMETRY CONFIRMED (p < 0.10)" if is_sig else "Marginal evidence (p < 0.15)"

    st.markdown(
        f"""
        <div style="padding:18px;border-radius:8px;
                    background:{verdict_color};border:1px solid {border_color}">
          <b style="font-size:1.15rem">Research finding: {verdict_text}</b><br>
          Best model: <b>{best['model']}</b> on <b>{best['target'].upper()}</b><br>
          Wald p-value: <b>{best['wald_p']:.4f}</b> &nbsp;|&nbsp;
          Asymmetry ratio: <b>|β_neg| / |β_pos| ≈ {asymmetry_ratio:.0f}×</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ---- 4B. Coefficient bar chart ----------------------------------------
    st.subheader("OLS Coefficients — β_neg vs β_pos")
    st.caption(
        "A large negative β_neg (left of zero) combined with a small β_pos confirms asymmetry. "
        "The Wald test formally tests whether they differ."
    )

    coef_rows = []
    for _, row in asym.iterrows():
        label = f"{row['model']} ({row['target'].upper()})"
        coef_rows += [
            {"Model": label, "Coefficient": "β_neg (negative language)", "Value": row["coef_neg_rate"]},
            {"Model": label, "Coefficient": "β_pos (positive language)", "Value": row["coef_pos_rate"]},
        ]
    coef_df = pd.DataFrame(coef_rows)

    coef_chart = (
        alt.Chart(coef_df)
        .mark_bar()
        .encode(
            x=alt.X("Model:N", title=None),
            y=alt.Y("Value:Q", title="Coefficient"),
            color=alt.condition(
                alt.datum.Value >= 0,
                alt.value("#2ca02c"),
                alt.value("#d62728"),
            ),
            xOffset=alt.XOffset("Coefficient:N"),
            tooltip=["Model", "Coefficient", alt.Tooltip("Value:Q", format=".3f")],
        )
        .properties(height=380)
    )
    st.altair_chart(coef_chart, width="stretch")

    # ---- 4C. Full asymmetry table -----------------------------------------
    st.subheader("Full Asymmetry Results Table")

    disp = asym[["model", "target", "coef_neg_rate", "pval_neg_rate",
                  "coef_pos_rate", "pval_pos_rate", "wald_p", "asymmetric"]].copy()
    disp.columns = ["Model", "Target", "β_neg", "p(β_neg)", "β_pos", "p(β_pos)",
                     "Wald p", "Asymmetric?"]
    for c in ["β_neg", "p(β_neg)", "β_pos", "p(β_pos)", "Wald p"]:
        disp[c] = disp[c].apply(lambda v: f"{v:.4f}" if pd.notna(v) else "—")
    disp["Asymmetric?"] = disp["Asymmetric?"].map({True: "Yes ✓", False: "No"})
    st.dataframe(disp, width="stretch", hide_index=True)

    st.divider()

    # ---- 4D. OOS R² chart -------------------------------------------------
    st.subheader("Out-of-Sample R² (Campbell–Thompson Convention)")
    st.caption(
        "OOS R² > 0 means the model beats a simple historical-mean forecast. "
        "Even small positive values are meaningful in financial prediction tasks."
    )

    oos_data = bench[bench["oos_r2"].notna()].copy()
    oos_data = oos_data.rename(columns={"model": "Model", "oos_r2": "OOS R²", "target": "Target"})

    oos_chart = (
        alt.Chart(oos_data)
        .mark_bar()
        .encode(
            x=alt.X("Model:N", title=None),
            y=alt.Y("OOS R²:Q"),
            color=alt.condition(
                alt.datum["OOS R²"] >= 0,
                alt.value("#2ca02c"),
                alt.value("#d62728"),
            ),
            column=alt.Column("Target:N"),
            tooltip=["Model", "Target", alt.Tooltip("OOS R²:Q", format=".4f")],
        )
        .properties(height=320, width=320)
    )
    st.altair_chart(oos_chart)

    st.divider()

    # ---- 4E. Written interpretation ----------------------------------------
    st.subheader("Interpretation & Conclusion")
    st.markdown(
        f"""
        **Key findings:**

        1. **β_neg ≈ {best['coef_neg_rate']:.1f}**: a 1 pp increase in negative word rate is
           associated with a **{abs(best['coef_neg_rate']):.1f} pp decrease** in {best['target'].upper()}.

        2. **β_pos ≈ {best['coef_pos_rate']:.1f}**: positive language has a far weaker
           (statistically insignificant) effect.

        3. **Asymmetry ratio ≈ {asymmetry_ratio:.0f}×**: negative language drives returns
           roughly {asymmetry_ratio:.0f}× harder than positive language per unit rate.

        4. **Wald test p = {best['wald_p']:.3f}**: {'rejects H₀ at 10% — asymmetry is statistically confirmed.' if is_sig else 'borderline (p < 0.15) — consistent with asymmetry but not definitively confirmed.'}

        **Economic interpretation:**
        Markets process bad news efficiently — a spike in negative earnings language
        triggers an immediate, proportionate price decline. Positive language, by
        contrast, is largely discounted by investors who have already priced in optimism.
        This is consistent with the **negativity bias** documented in behavioural finance
        and the finding that analysts give more weight to downside guidance revisions.

        **Limitations:**
        - Small lexicon (mini Loughran–McDonald fallback) dilutes the signal.
        - 188 observations across 5 years limits statistical power.
        - FinBERT sentence-level results (not shown here) provide stronger signal on
          a per-sentence basis and are the recommended feature for future work.
        """
    )


# ===========================================================================
# TAB 5 — Try It Yourself (live model deployment)
# ===========================================================================


def _render_results(container, title: str, text: str, use_fb: bool):
    """Render sentiment scores and CAR prediction into *container*."""
    with container, st.container(border=True):
        st.subheader(title)

        lx = score_lexicon(text)

        st.markdown("**Loughran-McDonald Lexicon**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Negative Rate", f"{lx['neg_rate']:.4f}")
        c2.metric("Positive Rate", f"{lx['pos_rate']:.4f}")
        c3.metric("Neg / Pos Words", f"{lx['neg_count']} / {lx['pos_count']}")
        c4.metric("Total Tokens", f"{lx['total_tokens']:,}")

        fin = None
        if use_fb:
            with st.spinner("Running FinBERT..."):
                fin = score_finbert(text)

            st.markdown("**FinBERT (deep learning)**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Negative", f"{fin['finbert_neg_mean']:.4f}")
            c2.metric("Positive", f"{fin['finbert_pos_mean']:.4f}")
            c3.metric("Neutral", f"{fin['finbert_neu_mean']:.4f}")

        st.markdown("---")
        st.markdown("**Predicted Market Reaction**")

        for target, label in [("car_03", "CAR[0,3]"), ("car_01", "CAR[0,1]")]:
            car = predict_car(
                asym, "LM Lexicon" if target == "car_03" else "LM Lexicon [CAR01]",
                target,
                {"neg_rate": lx["neg_rate"], "pos_rate": lx["pos_rate"]},
            )
            if car is not None:
                st.metric(f"Lexicon Model — {label}", f"{car:+.4%}")


with tab5:
    st.header("Try It Yourself")
    st.markdown(
        "Upload one or two earnings-call transcripts to analyze sentiment "
        "and predict the market reaction. Upload two to compare side by side."
    )

    uploaded_files = st.file_uploader(
        "Upload transcripts (.txt)",
        type=["txt"],
        accept_multiple_files=True,
        help="Upload up to 2 plain-text transcripts to compare.",
    )

    if len(uploaded_files) > 2:
        st.warning("Only the first 2 files will be analyzed.")
        uploaded_files = uploaded_files[:2]

    transcript_paste = ""
    if not uploaded_files:
        transcript_paste = st.text_area(
            "Or paste text directly",
            height=250,
            placeholder="Paste earnings call transcript here...",
        )

    use_finbert = False
    if _FINBERT_AVAILABLE:
        use_finbert = st.toggle(
            "Use FinBERT",
            help="Load the FinBERT transformer model for deep-learning-based sentiment. "
            "First run may take ~60 seconds.",
        )
    else:
        st.info(
            "**FinBERT is unavailable** (requires `transformers` and `torch`). "
            "Lexicon-based analysis is always available.",
        )

    inputs: list[tuple[str, str]] = []
    if uploaded_files:
        for f in uploaded_files:
            inputs.append((f.name, f.getvalue().decode("utf-8", errors="ignore")))
    elif transcript_paste.strip():
        inputs.append(("Pasted text", transcript_paste))

    if st.button("Analyze", type="primary", disabled=not inputs):
        if len(inputs) == 1:
            _render_results(st.container(), inputs[0][0], inputs[0][1], use_finbert)
        else:
            col_left, col_right = st.columns(2)
            _render_results(col_left, inputs[0][0], inputs[0][1], use_finbert)
            _render_results(col_right, inputs[1][0], inputs[1][1], use_finbert)

        st.caption(
            "These predictions are from OLS models trained on a limited "
            "dataset of NASDAQ earnings calls. They should not be used for "
            "investment decisions. CAR = cumulative abnormal return on "
            "the event day and subsequent trading days."
        )
