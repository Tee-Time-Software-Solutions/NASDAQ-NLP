import re

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go  # noqa: F401
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "outputs" / "results"

_FINBERT_AVAILABLE = False
try:
    import transformers  # noqa: F401
    _FINBERT_AVAILABLE = True
except ImportError:
    pass

# Loughran-McDonald mini word lists (same as in compute_lexicon_sentiment.py)
_NEG_WORDS = {
    "loss", "losses", "decline", "declines", "risk", "uncertain",
    "negative", "downturn", "weak", "concern", "headwind",
}
_POS_WORDS = {
    "profit", "profits", "growth", "strong", "opportunity",
    "opportunities", "improve", "improving", "record", "robust",
    "positive", "upside",
}

st.set_page_config(page_title="NASDAQ Earnings NLP", layout="wide")
st.title("NASDAQ Earnings-Call Sentiment & Market Reaction")


@st.cache_data
def load_data():
    events = pd.read_csv(RESULTS_DIR / "event_study_dataset.csv")
    lex = pd.read_csv(RESULTS_DIR / "lexicon_sentiment_features.csv")
    fin = pd.read_csv(RESULTS_DIR / "finbert_sentiment_features.csv")
    models = pd.read_csv(RESULTS_DIR / "model_results_car01.csv")

    key = ["ticker", "file_name", "event_trading_day_final"]
    df = events.merge(lex, on=key, how="left").merge(fin, on=key, how="left")
    return df, models, len(events) - len(df.dropna(subset=["neg_rate_lm", "finbert_neg_mean"]))


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
        "neg_rate_lm": neg / total,
        "pos_rate_lm": pos / total,
    }


def score_finbert(text: str) -> dict:
    clf = load_finbert()
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    if not sentences:
        return {"finbert_neg_mean": 0.0, "finbert_pos_mean": 0.0, "finbert_neu_mean": 0.0}

    all_scores = clf(sentences, batch_size=16)

    from collections import Counter

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


def predict_car(models_df: pd.DataFrame, spec_name: str, features: dict) -> float | None:
    row = models_df[models_df["spec"] == spec_name]
    if row.empty:
        return None
    row = row.iloc[0]

    const = row.get("const", 0.0)
    if pd.isna(const):
        const = 0.0

    beta_map = {
        "neg_rate_lm": "beta_neg",
        "pos_rate_lm": "beta_pos",
        "finbert_neg_mean": "beta_finbert_neg",
        "finbert_pos_mean": "beta_finbert_pos",
    }

    car = const
    for feat, val in features.items():
        beta_col = beta_map.get(feat)
        if beta_col and pd.notna(row.get(beta_col)):
            car += row[beta_col] * val
    return car


df, models, missing_count = load_data()
if missing_count > 0:
    st.warning(f"{missing_count} events are missing sentiment data and are excluded from sentiment charts.")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Market Reaction", "Asymmetry", "Try It Yourself"])

# ── Tab 1: Overview ─────────────────────────────────────────────────────────
with tab1:
    st.header("Model Benchmark")
    st.markdown(
        "OLS regressions predicting **CAR[0,1]** (cumulative abnormal return "
        "on the event day and the next trading day) from sentiment features."
    )

    display = models[["spec", "y", "X", "train_r2", "test_r2"]].copy()
    display.columns = ["Specification", "Target", "Features", "Train R²", "Test R²"]
    display["Train R²"] = display["Train R²"].map("{:.4f}".format)
    display["Test R²"] = display["Test R²"].map("{:.4f}".format)
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Coefficient Highlights")

    coef_rows = []
    for _, row in models.iterrows():
        spec = row["spec"]
        for col, label in [
            ("beta_neg", "neg_rate_lm"),
            ("beta_pos", "pos_rate_lm"),
            ("beta_finbert_neg", "finbert_neg_mean"),
            ("beta_finbert_pos", "finbert_pos_mean"),
        ]:
            val = row.get(col)
            if pd.notna(val):
                coef_rows.append({"Specification": spec, "Coefficient": label, "Value": val})

    if coef_rows:
        coef_df = pd.DataFrame(coef_rows)
        fig_coef = px.bar(
            coef_df,
            x="Coefficient",
            y="Value",
            color="Specification",
            barmode="group",
            title="Sentiment Coefficients Across Specifications",
        )
        st.plotly_chart(fig_coef, use_container_width=True)

    st.subheader("Dataset Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Events", len(df))
    col2.metric("Unique Tickers", df["ticker"].nunique())
    col3.metric("Year Range", f"{df['year'].min()} – {df['year'].max()}")

# ── Tab 2: Market Reaction ──────────────────────────────────────────────────
with tab2:
    st.header("CAR[0,1] Distribution")

    tickers = sorted(df["ticker"].unique())
    selected = st.multiselect("Filter by ticker", tickers, default=tickers)
    filtered = df[df["ticker"].isin(selected)]

    fig_hist = px.histogram(
        filtered,
        x="CAR_01",
        color="ticker",
        nbins=40,
        marginal="rug",
        title="Distribution of CAR[0,1] by Ticker",
        labels={"CAR_01": "Cumulative Abnormal Return [0,1]"},
    )
    fig_hist.add_vline(x=0, line_dash="dash", line_color="grey")
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("CAR[0,1] by Ticker")
    fig_box = px.box(
        filtered,
        x="ticker",
        y="CAR_01",
        color="ticker",
        title="CAR[0,1] Box Plot by Ticker",
        labels={"CAR_01": "Cumulative Abnormal Return [0,1]"},
    )
    st.plotly_chart(fig_box, use_container_width=True)

# ── Tab 3: Asymmetry ────────────────────────────────────────────────────────
with tab3:
    st.header("Sentiment–Return Asymmetry")
    st.markdown(
        "Do negative sentiment signals move stock prices more than positive ones? "
        "The scatter plots below show sentiment scores vs CAR[0,1]."
    )

    sent_choice = st.radio(
        "Sentiment source",
        ["Loughran-McDonald Lexicon", "FinBERT"],
        horizontal=True,
    )

    if sent_choice == "Loughran-McDonald Lexicon":
        neg_col, pos_col = "neg_rate_lm", "pos_rate_lm"
        neg_label, pos_label = "Negative Word Rate (LM)", "Positive Word Rate (LM)"
    else:
        neg_col, pos_col = "finbert_neg_mean", "finbert_pos_mean"
        neg_label, pos_label = "FinBERT Negative Score", "FinBERT Positive Score"

    col_left, col_right = st.columns(2)

    with col_left:
        fig_neg = px.scatter(
            df,
            x=neg_col,
            y="CAR_01",
            color="ticker",
            trendline="ols",
            title=f"{neg_label} vs CAR[0,1]",
            labels={neg_col: neg_label, "CAR_01": "CAR[0,1]"},
            opacity=0.7,
        )
        st.plotly_chart(fig_neg, use_container_width=True)

    with col_right:
        fig_pos = px.scatter(
            df,
            x=pos_col,
            y="CAR_01",
            color="ticker",
            trendline="ols",
            title=f"{pos_label} vs CAR[0,1]",
            labels={pos_col: pos_label, "CAR_01": "CAR[0,1]"},
            opacity=0.7,
        )
        st.plotly_chart(fig_pos, use_container_width=True)

    st.markdown("---")
    st.subheader("Asymmetry Test (Wald)")
    asym_path = RESULTS_DIR / "asymmetry_tests_car01_lexicon.txt"
    if asym_path.exists():
        st.code(asym_path.read_text(), language="text")
    else:
        st.info("Asymmetry test results not found.")

# ── Tab 4: Try It Yourself ─────────────────────────────────────────────────


def _render_results(container, title: str, text: str, use_finbert: bool):
    """Render sentiment scores and CAR prediction into *container*."""
    with container, st.container(border=True):
        st.subheader(title)

        lex = score_lexicon(text)

        st.markdown("**Loughran-McDonald Lexicon**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Negative Rate", f"{lex['neg_rate_lm']:.4f}")
        c2.metric("Positive Rate", f"{lex['pos_rate_lm']:.4f}")
        c3.metric("Neg / Pos Words", f"{lex['neg_count']} / {lex['pos_count']}")
        c4.metric("Total Tokens", f"{lex['total_tokens']:,}")

        fin = None
        if use_finbert:
            with st.spinner("Running FinBERT..."):
                fin = score_finbert(text)

            st.markdown("**FinBERT (deep learning)**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Negative", f"{fin['finbert_neg_mean']:.4f}")
            c2.metric("Positive", f"{fin['finbert_pos_mean']:.4f}")
            c3.metric("Neutral", f"{fin['finbert_neu_mean']:.4f}")

        st.markdown("---")
        st.markdown("**Predicted Market Reaction**")

        lex_car = predict_car(
            models,
            "car01_lexicon_only",
            {"neg_rate_lm": lex["neg_rate_lm"], "pos_rate_lm": lex["pos_rate_lm"]},
        )
        if lex_car is not None:
            st.metric("Lexicon Model — CAR[0,1]", f"{lex_car:+.4%}")

        if fin:
            fin_car = predict_car(
                models,
                "car01_finbert_only",
                {
                    "finbert_neg_mean": fin["finbert_neg_mean"],
                    "finbert_pos_mean": fin["finbert_pos_mean"],
                },
            )
            if fin_car is not None:
                st.metric("FinBERT Model — CAR[0,1]", f"{fin_car:+.4%}")


with tab4:
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

    # Cap at 2 files
    if len(uploaded_files) > 2:
        st.warning("Only the first 2 files will be analyzed.")
        uploaded_files = uploaded_files[:2]

    # Fallback: text area when no files uploaded
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

    # Build list of (name, text) pairs to analyze
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
            "These predictions are from simple OLS models trained on a limited "
            "dataset of NASDAQ earnings calls. They should not be used for "
            "investment decisions. CAR\\[0,1\\] = cumulative abnormal return on "
            "the event day and the next trading day."
        )
