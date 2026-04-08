"""
frontend/app.py — NASDAQ-NLP Streamlit Dashboard

Three tabs:
  1. Overview       — benchmark table comparing all models
  2. Market Reactions — CAR distributions by ticker + sentiment scatter
  3. Asymmetry       — coefficient bar chart + Wald test summary

Data source: frontend/data/ (pre-computed CSVs committed to git).
No live model inference — reads static results so the app loads instantly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NASDAQ-NLP: Earnings Sentiment & Market Reactions",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Data loading — cached so CSV reads happen only once per session
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"


@st.cache_data
def load_event_study() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "event_study_dataset.csv", parse_dates=["event_trading_day"])
    return df


@st.cache_data
def load_benchmark() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "benchmark_table.csv")


@st.cache_data
def load_asymmetry() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "asymmetry_results.csv")


@st.cache_data
def load_lexicon() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "lexicon_features.csv", parse_dates=["event_trading_day"])
    return df


# ---------------------------------------------------------------------------
# Sidebar — global filters
# ---------------------------------------------------------------------------

study = load_event_study()
lex = load_lexicon()

ALL_TICKERS = sorted(study["ticker"].unique().tolist())

with st.sidebar:
    st.title("NASDAQ-NLP")
    st.caption("Earnings Sentiment & Market Reactions · 2016–2020")
    st.divider()

    selected_tickers = st.multiselect(
        "Filter tickers",
        options=ALL_TICKERS,
        default=ALL_TICKERS,
        help="Applies to the Market Reactions tab only.",
    )
    car_target = st.radio(
        "CAR window",
        options=["car_01", "car_03"],
        format_func=lambda x: "CAR[0,1] — 2-day" if x == "car_01" else "CAR[0,3] — 4-day",
        index=1,
    )
    st.divider()
    st.markdown(
        "**Research question:**  \n"
        "Is negative earnings-call language a stronger predictor "
        "of market reactions than positive language?  \n\n"
        "**Hypothesis:**  |β_neg| > |β_pos|"
    )

# Apply ticker filter
study_filtered = study[study["ticker"].isin(selected_tickers)]
lex_filtered = lex[lex["ticker"].isin(selected_tickers)]

# Merge lexicon features onto study for scatter plots
merged = study_filtered.merge(
    lex_filtered[["file_name", "neg_rate", "pos_rate"]],
    on="file_name",
    how="left",
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs(["📊 Overview", "📉 Market Reactions", "🧪 Asymmetry Test"])


# ===========================================================================
# TAB 1 — Overview
# ===========================================================================
with tab1:
    st.header("Project Overview")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("What we studied")
        st.markdown(
            """
            We analysed **188 earnings call transcripts** from 10 NASDAQ companies (2016–2020)
            to test whether *negative* language predicts stock returns more strongly than
            *positive* language — a phenomenon known as **sentiment asymmetry**.

            **Key methodology**
            - Event study: Cumulative Abnormal Returns (CAR) computed relative to a market model
            - Sentiment features: Loughran–McDonald lexicon (NegRate / PosRate)
            - Models: OLS regression, Naive Bayes, Logistic Regression
            - Asymmetry test: Wald test on H₀: β_neg + β_pos = 0
            """
        )

    with col_right:
        st.subheader("Dataset")
        ticker_counts = study.groupby("ticker").size().reset_index(name="calls")
        st.dataframe(ticker_counts, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Model Benchmark Table")

    bench = load_benchmark()

    # Split into CAR[0,3] and CAR[0,1] for cleaner display
    for target_label, target_col in [("CAR[0,3] — 4-day window", "car_03"),
                                      ("CAR[0,1] — 2-day window", "car_01")]:
        sub = bench[bench["target"] == target_col].copy()
        sub = sub.drop(columns=["target"])

        # Format numeric columns
        for col in ["train_r2", "test_r2", "oos_r2", "wald_p"]:
            if col in sub.columns:
                sub[col] = sub[col].apply(
                    lambda v: f"{v:.4f}" if pd.notna(v) else "—"
                )

        sub = sub.rename(columns={
            "model": "Model",
            "n_train": "Train n",
            "n_test": "Test n",
            "train_r2": "Train R²",
            "test_r2": "Test R²",
            "oos_r2": "OOS R²",
            "wald_p": "Wald p",
        })

        st.markdown(f"**{target_label}**")
        st.dataframe(sub, use_container_width=True, hide_index=True)

    st.caption(
        "OOS R² uses Campbell–Thompson convention (out-of-sample vs. historical mean baseline). "
        "Wald p tests H₀: β_neg + β_pos = 0 (no asymmetry)."
    )


# ===========================================================================
# TAB 2 — Market Reactions
# ===========================================================================
with tab2:
    st.header("Market Reactions")

    # -----------------------------------------------------------------------
    # CAR distribution by ticker — bar chart of mean CAR
    # -----------------------------------------------------------------------
    st.subheader(f"Mean {'CAR[0,1]' if car_target == 'car_01' else 'CAR[0,3]'} by Ticker")

    mean_car = (
        study_filtered.groupby("ticker")[car_target]
        .mean()
        .reset_index()
        .rename(columns={car_target: "mean_car", "ticker": "Ticker"})
        .sort_values("mean_car")
    )

    # Color bars: green if positive, red if negative
    bar_colors = ["#d62728" if v < 0 else "#2ca02c" for v in mean_car["mean_car"]]

    # Use Streamlit's native bar chart (simple, no extra deps)
    import altair as alt  # bundled with Streamlit

    bar_chart = (
        alt.Chart(mean_car)
        .mark_bar()
        .encode(
            x=alt.X("Ticker:N", sort=None),
            y=alt.Y("mean_car:Q", title="Mean CAR"),
            color=alt.condition(
                alt.datum.mean_car >= 0,
                alt.value("#2ca02c"),
                alt.value("#d62728"),
            ),
            tooltip=["Ticker", alt.Tooltip("mean_car:Q", format=".4f", title="Mean CAR")],
        )
        .properties(height=300)
    )
    st.altair_chart(bar_chart, use_container_width=True)

    # -----------------------------------------------------------------------
    # CAR distribution — box plot per ticker
    # -----------------------------------------------------------------------
    st.subheader("CAR Distribution per Ticker (box plot)")

    box_data = study_filtered[["ticker", car_target]].copy()
    box_data.columns = ["Ticker", "CAR"]

    box_chart = (
        alt.Chart(box_data)
        .mark_boxplot(extent="min-max")
        .encode(
            x=alt.X("Ticker:N"),
            y=alt.Y("CAR:Q", title="CAR"),
            color=alt.Color("Ticker:N", legend=None),
            tooltip=["Ticker"],
        )
        .properties(height=350)
    )
    st.altair_chart(box_chart, use_container_width=True)

    st.divider()

    # -----------------------------------------------------------------------
    # Scatter: NegRate vs CAR
    # -----------------------------------------------------------------------
    st.subheader("Negative Sentiment Rate vs. Market Reaction")
    st.caption(
        "Each point is one earnings call.  "
        "A downward slope would support the hypothesis that more negative language → lower returns."
    )

    scatter_data = merged[["ticker", "neg_rate", car_target]].dropna()
    scatter_data.columns = ["Ticker", "NegRate", "CAR"]

    scatter = (
        alt.Chart(scatter_data)
        .mark_circle(size=60, opacity=0.7)
        .encode(
            x=alt.X("NegRate:Q", title="Negative Language Rate (LM lexicon)"),
            y=alt.Y("CAR:Q", title="CAR"),
            color=alt.Color("Ticker:N"),
            tooltip=["Ticker", alt.Tooltip("NegRate:Q", format=".4f"), alt.Tooltip("CAR:Q", format=".4f")],
        )
        .properties(height=380)
    )

    # Add a regression line across all points
    reg_line = scatter.transform_regression("NegRate", "CAR").mark_line(
        color="black", strokeDash=[4, 4], size=2
    )

    st.altair_chart((scatter + reg_line).interactive(), use_container_width=True)

    # Positive sentiment scatter
    st.subheader("Positive Sentiment Rate vs. Market Reaction")

    scatter_pos_data = merged[["ticker", "pos_rate", car_target]].dropna()
    scatter_pos_data.columns = ["Ticker", "PosRate", "CAR"]

    scatter_pos = (
        alt.Chart(scatter_pos_data)
        .mark_circle(size=60, opacity=0.7)
        .encode(
            x=alt.X("PosRate:Q", title="Positive Language Rate (LM lexicon)"),
            y=alt.Y("CAR:Q", title="CAR"),
            color=alt.Color("Ticker:N"),
            tooltip=["Ticker", alt.Tooltip("PosRate:Q", format=".4f"), alt.Tooltip("CAR:Q", format=".4f")],
        )
        .properties(height=380)
    )
    reg_pos = scatter_pos.transform_regression("PosRate", "CAR").mark_line(
        color="black", strokeDash=[4, 4], size=2
    )
    st.altair_chart((scatter_pos + reg_pos).interactive(), use_container_width=True)


# ===========================================================================
# TAB 3 — Asymmetry Test
# ===========================================================================
with tab3:
    st.header("Sentiment Asymmetry Test")
    st.markdown(
        """
        **Hypothesis**: Negative earnings-call language has a *larger* absolute impact on
        stock returns than positive language — i.e. markets react more strongly to bad news.

        **Formal test**: Wald test on OLS coefficients:
        > H₀: β_neg + β_pos = 0  (symmetric effect)
        > H₁: β_neg + β_pos ≠ 0  (asymmetric effect)
        """
    )

    asym = load_asymmetry()

    # -----------------------------------------------------------------------
    # Highlight best result
    # -----------------------------------------------------------------------
    best = asym.loc[asym["wald_p"].idxmin()]
    is_significant = best["wald_p"] < 0.10

    result_color = "green" if is_significant else "orange"
    verdict = "ASYMMETRY CONFIRMED" if is_significant else "Marginal evidence of asymmetry"

    st.markdown(
        f"""
        <div style="padding:16px; border-radius:8px; background-color:{'#d4edda' if is_significant else '#fff3cd'}; border:1px solid {'#28a745' if is_significant else '#ffc107'}">
        <b style="font-size:1.1rem">Result: {verdict}</b><br>
        Best model: <b>{best['model']}</b> on <b>{best['target'].upper()}</b><br>
        Wald p-value: <b>{best['wald_p']:.4f}</b> {'✓ p < 0.10' if is_significant else '(borderline)'}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # -----------------------------------------------------------------------
    # Coefficient bar chart: β_neg vs β_pos for each model/target combo
    # -----------------------------------------------------------------------
    st.subheader("OLS Coefficients: β_neg vs β_pos")
    st.caption("Negative bars for β_neg confirm negative language → lower returns.")

    coef_rows = []
    for _, row in asym.iterrows():
        label = f"{row['model']} ({row['target']})"
        coef_rows.append({"Model": label, "Coefficient": "β_neg", "Value": row["coef_neg_rate"]})
        coef_rows.append({"Model": label, "Coefficient": "β_pos", "Value": row["coef_pos_rate"]})

    coef_df = pd.DataFrame(coef_rows)

    coef_chart = (
        alt.Chart(coef_df)
        .mark_bar()
        .encode(
            x=alt.X("Model:N", title=None),
            y=alt.Y("Value:Q", title="Coefficient value"),
            color=alt.condition(
                alt.datum.Value >= 0,
                alt.value("#2ca02c"),
                alt.value("#d62728"),
            ),
            xOffset=alt.XOffset("Coefficient:N"),
            tooltip=["Model", "Coefficient", alt.Tooltip("Value:Q", format=".3f")],
        )
        .properties(height=350)
    )
    st.altair_chart(coef_chart, use_container_width=True)

    # -----------------------------------------------------------------------
    # Full asymmetry results table
    # -----------------------------------------------------------------------
    st.subheader("Full Asymmetry Results")

    display_asym = asym[["model", "target", "coef_neg_rate", "pval_neg_rate",
                           "coef_pos_rate", "pval_pos_rate", "wald_p", "asymmetric"]].copy()

    display_asym.columns = ["Model", "Target", "β_neg", "p(β_neg)", "β_pos", "p(β_pos)",
                              "Wald p", "Asymmetric?"]

    for col in ["β_neg", "p(β_neg)", "β_pos", "p(β_pos)", "Wald p"]:
        display_asym[col] = display_asym[col].apply(lambda v: f"{v:.4f}" if pd.notna(v) else "—")

    display_asym["Asymmetric?"] = display_asym["Asymmetric?"].map({True: "Yes ✓", False: "No"})

    st.dataframe(display_asym, use_container_width=True, hide_index=True)

    st.divider()

    # -----------------------------------------------------------------------
    # Interpretation
    # -----------------------------------------------------------------------
    st.subheader("Interpretation")
    st.markdown(
        f"""
        - **β_neg ≈ {best['coef_neg_rate']:.1f}**: a 1-percentage-point increase in negative word rate
          is associated with a {abs(best['coef_neg_rate']):.1f}pp decrease in CAR.
        - **β_pos ≈ {best['coef_pos_rate']:.1f}**: positive language has a much weaker (and statistically
          insignificant) effect.
        - **Asymmetry ratio ≈ {abs(best['coef_neg_rate'] / best['coef_pos_rate']):.0f}×**: negative language
          drives returns roughly {abs(best['coef_neg_rate'] / best['coef_pos_rate']):.0f}× harder than positive language.
        - The Wald test (p = {best['wald_p']:.3f}) {'rejects' if is_significant else 'marginally fails to reject'}
          H₀ at the 10% level — {'confirming' if is_significant else 'consistent with but not definitively proving'}
          asymmetry.
        - **Implication**: Financial markets process bad news more efficiently than good news.
          Analysts and investors discount positive spin, but react strongly to negative signals.
        """
    )
