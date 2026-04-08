import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "outputs" / "results"

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


df, models, missing_count = load_data()
if missing_count > 0:
    st.warning(f"{missing_count} events are missing sentiment data and are excluded from sentiment charts.")

tab1, tab2, tab3 = st.tabs(["Overview", "Market Reaction", "Asymmetry"])

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
