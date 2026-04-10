"""
visualization.py — Benchmark plot suite for regression and classification results.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nasdaq_nlp.config import RESULTS_DIR

# ---------------------------------------------------------------------------
# Colour maps
# ---------------------------------------------------------------------------

REGRESSOR_COLOURS = {
    "OLS": "#4C72B0",
    "Ridge": "#55A868",
    "RF": "#C44E52",
    "MLP": "#8172B2",
    "Null": "#AAAAAA",
}

CLF_COLOURS = {
    "NaiveBayes": "#4C72B0",
    "LogReg": "#55A868",
    "Tree": "#C44E52",
    "RF": "#8172B2",
    "MLP": "#CCB974",
}


def _bar_colours(names: list[str], cmap: dict) -> list[str]:
    out = []
    for n in names:
        col = "#CCCCCC"
        for k, c in cmap.items():
            if k in n:
                col = c
                break
        out.append(col)
    return out


def _legend(cmap: dict) -> list:
    return [mpatches.Patch(color=c, label=k) for k, c in cmap.items()]


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------


def plot_benchmark(
    reg_df: "pd.DataFrame",
    clf_df: "pd.DataFrame",
    save_path: Path = RESULTS_DIR / "benchmark_plots.png",
) -> Path:
    """Produce the 6-panel benchmark figure and save to disk.

    Panels
    ------
    (0,0) OOS R²                 — main regression comparison vs null
    (0,1) Train vs Test R²       — overfitting check
    (1,0) MAE                    — absolute error in CAR units
    (1,1) Wald p-value           — asymmetry test per OLS model
    (2,0) Classification Acc/F1  — test-set scores
    (2,1) Train vs Test Accuracy — classifier overfitting check

    Parameters
    ----------
    reg_df    : benchmark_table.csv as a DataFrame
    clf_df    : classification_results.csv as a DataFrame
    save_path : where to write the PNG

    Returns
    -------
    Path to the saved figure.
    """
    reg_03 = reg_df[reg_df["target"] == "car_03"].reset_index(drop=True)
    wald_df = reg_03[reg_03["wald_p"].notna()].reset_index(drop=True)

    fig, axes = plt.subplots(3, 2, figsize=(16, 15))
    fig.suptitle("Benchmark: Regression & Classification", fontsize=14, fontweight="bold", y=1.01)

    names = reg_03["model"].tolist()
    cols = _bar_colours(names, REGRESSOR_COLOURS)

    # Panel (0,0) — OOS R²
    ax = axes[0, 0]
    vals = reg_03["oos_r2"].tolist()
    bars = ax.barh(names, vals, color=cols, edgecolor="white", height=0.6)
    ax.axvline(0, color="black", lw=1.2, ls="--", label="Null baseline")
    ax.set_xlabel("OOS R²")
    ax.set_title("OOS R² — Regression (CAR[0,3])")
    ax.legend(handles=_legend(REGRESSOR_COLOURS), fontsize=8, loc="lower right")
    for bar, v in zip(bars, vals):
        ax.text(
            v + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.3f}",
            va="center",
            ha="left",
            fontsize=7,
        )

    # Panel (0,1) — Train vs Test R²
    ax = axes[0, 1]
    x, w = np.arange(len(names)), 0.38
    ax.barh(x + w / 2, reg_03["train_r2"], w, color="#4C72B0", label="Train R²", alpha=0.85)
    ax.barh(x - w / 2, reg_03["test_r2"], w, color="#C44E52", label="Test R²", alpha=0.85)
    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("R²")
    ax.set_title("Train vs Test R² — overfitting check")
    ax.legend(fontsize=9)

    # Panel (1,0) — MAE
    ax = axes[1, 0]
    mae = reg_03["mae"].tolist()
    bars = ax.barh(names, mae, color=cols, edgecolor="white", height=0.6)
    ax.set_xlabel("Mean Absolute Error (CAR units)")
    ax.set_title("MAE — Regression (lower is better)")
    ax.legend(handles=_legend(REGRESSOR_COLOURS), fontsize=8)
    for bar, v in zip(bars, mae):
        ax.text(
            v + 0.0001,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.4f}",
            va="center",
            ha="left",
            fontsize=7,
        )

    # Panel (1,1) — Wald p-value
    ax = axes[1, 1]
    if len(wald_df) > 0:
        wnames = wald_df["model"].tolist()
        wvals = wald_df["wald_p"].tolist()
        wcols = ["#C44E52" if p < 0.10 else "#4C72B0" for p in wvals]
        bars = ax.barh(wnames, wvals, color=wcols, edgecolor="white", height=0.5)
        ax.axvline(0.10, color="red", lw=1.2, ls="--", label="p=0.10")
        ax.axvline(0.05, color="orange", lw=1.0, ls=":", label="p=0.05")
        ax.set_xlabel("Wald p-value (H₀: β_neg + β_pos = 0)")
        ax.set_title("Asymmetry Test — red = |β_neg| ≠ |β_pos|")
        ax.legend(fontsize=9)
        for bar, v in zip(bars, wvals):
            ax.text(
                v + 0.002,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}",
                va="center",
                ha="left",
                fontsize=8,
            )
    else:
        ax.text(
            0.5,
            0.5,
            "No Wald results\n(run OLS + lexicon first)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
        )
        ax.set_title("Asymmetry Test: Wald p-value")

    # Panel (2,0) — Classification accuracy & F1
    ax = axes[2, 0]
    cnames = clf_df["model"].tolist()
    cx, w = np.arange(len(cnames)), 0.38
    ax.barh(
        cx + w / 2, clf_df["test_accuracy"], w, color="#4C72B0", label="Test accuracy", alpha=0.85
    )
    ax.barh(cx - w / 2, clf_df["test_f1"], w, color="#55A868", label="Test F1", alpha=0.85)
    ax.axvline(0.5, color="black", lw=1.0, ls="--", label="Random baseline")
    ax.set_yticks(cx)
    ax.set_yticklabels(cnames, fontsize=8)
    ax.set_xlabel("Score")
    ax.set_title("Classification: Accuracy & F1 (test set)")
    ax.legend(fontsize=9)

    # Panel (2,1) — Train vs Test accuracy
    ax = axes[2, 1]
    ax.barh(
        cx + w / 2, clf_df["train_accuracy"], w, color="#4C72B0", label="Train accuracy", alpha=0.85
    )
    ax.barh(
        cx - w / 2, clf_df["test_accuracy"], w, color="#C44E52", label="Test accuracy", alpha=0.85
    )
    ax.axvline(0.5, color="black", lw=0.8, ls="--")
    ax.set_yticks(cx)
    ax.set_yticklabels(cnames, fontsize=8)
    ax.set_xlabel("Accuracy")
    ax.set_title("Classification: Train vs Test — overfitting check")
    ax.legend(fontsize=9)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {save_path}")
    plt.show()
    return save_path


# ---------------------------------------------------------------------------
# Notebook 04 plots
# ---------------------------------------------------------------------------

# Maps coef column name → (display label, corresponding pval column).
# Ordered so that negative-sentiment bars appear below positive-sentiment bars.
_SENT_COEF_COLS = {
    "coef_neg_rate": ("NegRate", "pval_neg_rate"),
    "coef_pos_rate": ("PosRate", "pval_pos_rate"),
    "coef_neg_rate_pres": ("NegRate (Pres)", "pval_neg_rate_pres"),
    "coef_pos_rate_pres": ("PosRate (Pres)", "pval_pos_rate_pres"),
    "coef_neg_rate_qa": ("NegRate (Q&A)", "pval_neg_rate_qa"),
    "coef_pos_rate_qa": ("PosRate (Q&A)", "pval_pos_rate_qa"),
}


def plot_coefficient(
    asym_df: "pd.DataFrame",
    target: str = "car_03",
    save_path: Path = RESULTS_DIR / "coefficient_plot.png",
) -> Path:
    """Bar chart of β_neg / β_pos for each OLS model that ran the Wald test.

    Bars are red when individual p < 0.10, grey otherwise.
    Title shows Wald p and whether H₀ (symmetric) is rejected.

    Parameters
    ----------
    asym_df   : asymmetry_results.csv loaded as DataFrame
    target    : which CAR window to filter on ("car_03" or "car_01")
    save_path : output PNG path
    """
    import pandas as pd

    models = asym_df[asym_df["target"] == target].reset_index(drop=True)
    n = len(models)
    if n == 0:
        print("No asymmetry models found — run 03_modeling.ipynb first.")
        return save_path

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, models.iterrows()):
        labels, coefs, pvals = [], [], []
        for coef_col, (label, pval_col) in _SENT_COEF_COLS.items():
            if coef_col in row and pd.notna(row[coef_col]):
                labels.append(label)
                coefs.append(float(row[coef_col]))
                pvals.append(float(row.get(pval_col, 1.0)))

        colours = ["#C44E52" if p < 0.10 else "#AAAAAA" for p in pvals]
        ax.barh(labels, coefs, color=colours, edgecolor="white", height=0.5)
        ax.axvline(0, color="black", lw=0.8, ls="--")

        wald_p = float(row["wald_p"])
        ax.set_title(
            f"{row['model']}\nWald p={wald_p:.3f}  "
            f"{'→ ASYMMETRIC ✓' if wald_p < 0.10 else '→ symmetric'}",
            fontsize=10,
        )
        ax.set_xlabel("Coefficient estimate")

        for val, lab, p in zip(coefs, labels, pvals):
            sign = 1 if val >= 0 else -1
            ax.text(
                val + sign * abs(val) * 0.05,
                lab,
                f"p={p:.3f}",
                va="center",
                ha="left" if val >= 0 else "right",
                fontsize=8,
            )

    axes[0].legend(
        handles=[
            mpatches.Patch(color="#C44E52", label="p < 0.10"),
            mpatches.Patch(color="#AAAAAA", label="p ≥ 0.10"),
        ],
        fontsize=8,
    )
    plt.suptitle("Sentiment Coefficients — OLS on CAR[0,3]", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {save_path}")
    plt.show()
    return save_path


def plot_oos_vs_clf(
    benchmark_df: "pd.DataFrame",
    clf_df: "pd.DataFrame",
    save_path: Path = RESULTS_DIR / "oos_vs_clf_plot.png",
) -> Path:
    """Side-by-side: OOS R² (regression) vs Accuracy/F1 (classification).

    Parameters
    ----------
    benchmark_df : benchmark_table.csv filtered to car_03
    clf_df       : classification_results.csv
    save_path    : output PNG path
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left — OOS R²
    ax = axes[0]
    colours = ["#C44E52" if v < 0 else "#4C72B0" for v in benchmark_df["oos_r2"]]
    bars = ax.barh(
        benchmark_df["model"], benchmark_df["oos_r2"], color=colours, edgecolor="white", height=0.6
    )
    ax.axvline(0, color="black", lw=1.0, ls="--", label="Null baseline")
    ax.set_xlabel("OOS R²")
    ax.set_title("Regression OOS R² (CAR[0,3])\nred = worse than null", fontsize=11)
    for bar, v in zip(bars, benchmark_df["oos_r2"]):
        sign = 1 if v >= 0 else -1
        ax.text(
            v + sign * 0.0005,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.4f}",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=7,
        )

    # Right — Classification accuracy & F1
    ax = axes[1]
    x, w = np.arange(len(clf_df)), 0.38
    if "test_accuracy" in clf_df.columns:
        ax.barh(
            x + w / 2,
            clf_df["test_accuracy"],
            w,
            color="#4C72B0",
            label="Test accuracy",
            alpha=0.85,
        )
    if "test_f1" in clf_df.columns:
        ax.barh(x - w / 2, clf_df["test_f1"], w, color="#55A868", label="Test F1", alpha=0.85)
    ax.axvline(0.5, color="black", lw=1.0, ls="--", label="Random baseline (0.5)")
    ax.set_yticks(x)
    ax.set_yticklabels(clf_df["model"], fontsize=8)
    ax.set_xlabel("Score")
    ax.set_title("Classification: Accuracy & F1 (CAR[0,3] > 0)", fontsize=11)
    ax.legend(fontsize=9)

    plt.suptitle(
        "Predictive Performance: Regression vs Classification", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {save_path}")
    plt.show()
    return save_path
