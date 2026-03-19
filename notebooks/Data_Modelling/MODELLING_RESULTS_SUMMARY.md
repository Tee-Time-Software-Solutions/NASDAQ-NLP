# Modelling Results Summary

This note explains the results from `notebooks/Data_Modelling/15_modelling_pipeline.ipynb` for a non-technical audience.

## What the model names mean

- `null`: predicts the average outcome only (no information used).
- `baseline_controls`: uses basic market/context controls (no sentiment).
- `lexicon_sentiment`: uses dictionary sentiment only.
- `finbert_sentiment`: uses FinBERT sentiment only.
- `full_..._plus_controls`: sentiment + baseline controls.
- `extended_..._plus_controls`: sentiment + a larger set of controls.

## How to read performance quickly

- Higher `test_r2` is better (positive means useful prediction on new data).
- Lower `test_rmse` and `test_mae` are better (smaller errors).
- For CAR models, higher `sign_accuracy` means better at predicting up vs down direction.

---

## 1) CAR_01 (very short-term return reaction)

### Best-performing model
- **`finbert_sentiment`** is the best for this target.
- It is the only clearly positive `test_r2` model (`0.0295`) and has the lowest error range among top models.

### Full results (CAR_01)

- `finbert_sentiment`: `test_r2=0.0295`, `rmse=0.0664`, `mae=0.0512`, `sign_accuracy=0.58`
- `lexicon_sentiment`: `test_r2=-0.0149`, `rmse=0.0679`, `mae=0.0525`, `sign_accuracy=0.54`
- `null`: `test_r2=-0.0295`, `rmse=0.0684`, `mae=0.0527`, `sign_accuracy=0.54`
- `full_finbert_plus_controls`: `test_r2=-0.1410`, `rmse=0.0720`, `mae=0.0570`, `sign_accuracy=0.49`
- `baseline_controls`: `test_r2=-0.3197`, `rmse=0.0775`, `mae=0.0596`, `sign_accuracy=0.55`
- `full_lexicon_plus_controls`: `test_r2=-0.3670`, `rmse=0.0789`, `mae=0.0617`, `sign_accuracy=0.54`
- `extended_finbert_plus_controls`: `test_r2=-0.4552`, `rmse=0.0814`, `mae=0.0655`, `sign_accuracy=0.57`
- `extended_lexicon_plus_controls`: `test_r2=-1.3125`, `rmse=0.1026`, `mae=0.0850`, `sign_accuracy=0.55`

### Key takeaway
- For immediate return reaction, **simple FinBERT sentiment alone works best**.
- Adding many controls in this dataset often made prediction worse (over-complex for sample size).

---

## 2) CAR_03 (3-day return reaction)

### Best-performing model
- **`finbert_sentiment`** again performs best.
- It has the highest `test_r2` (`0.0510`) and best overall error/ordering quality.

### Full results (CAR_03)

- `finbert_sentiment`: `test_r2=0.0510`, `rmse=0.0676`, `mae=0.0523`, `sign_accuracy=0.55`
- `lexicon_sentiment`: `test_r2=-0.0297`, `rmse=0.0704`, `mae=0.0555`, `sign_accuracy=0.49`
- `null`: `test_r2=-0.0446`, `rmse=0.0709`, `mae=0.0559`, `sign_accuracy=0.49`
- `full_finbert_plus_controls`: `test_r2=-0.1325`, `rmse=0.0738`, `mae=0.0565`, `sign_accuracy=0.52`
- `extended_finbert_plus_controls`: `test_r2=-0.3316`, `rmse=0.0801`, `mae=0.0625`, `sign_accuracy=0.52`
- `baseline_controls`: `test_r2=-0.3499`, `rmse=0.0806`, `mae=0.0631`, `sign_accuracy=0.51`
- `full_lexicon_plus_controls`: `test_r2=-0.4070`, `rmse=0.0823`, `mae=0.0653`, `sign_accuracy=0.52`
- `extended_lexicon_plus_controls`: `test_r2=-1.2376`, `rmse=0.1038`, `mae=0.0850`, `sign_accuracy=0.52`

### Key takeaway
- Same pattern as CAR_01: **FinBERT-only is strongest**.
- Extra model complexity did not improve out-of-sample performance here.

---

## 3) Volatility change (post-call risk level change)

### Best-performing model
- **`baseline_controls`** is best (`test_r2=0.1729`), closely followed by `full_finbert_plus_controls` (`0.1679`).

### Full results (volatility_change)

- `baseline_controls`: `test_r2=0.1729`, `rmse=0.01137`, `mae=0.00829`
- `full_finbert_plus_controls`: `test_r2=0.1679`, `rmse=0.01140`, `mae=0.00828`
- `full_lexicon_plus_controls`: `test_r2=0.1323`, `rmse=0.01164`, `mae=0.00846`
- `extended_finbert_plus_controls`: `test_r2=0.1108`, `rmse=0.01179`, `mae=0.00928`
- `extended_lexicon_plus_controls`: `test_r2=0.1092`, `rmse=0.01180`, `mae=0.00933`
- `null`: `test_r2=-0.0226`, `rmse=0.01264`, `mae=0.00909`
- `finbert_sentiment`: `test_r2=-0.0403`, `rmse=0.01275`, `mae=0.00917`
- `lexicon_sentiment`: `test_r2=-0.0531`, `rmse=0.01283`, `mae=0.00925`

### Key takeaway
- For volatility, **market/context controls carry most of the predictive signal**.
- Sentiment alone is weak for volatility in this sample.

---

## Asymmetry question: Is negative sentiment stronger than positive sentiment?

Using the stricter clustered test (`cluster_p`):

- **CAR_01 + FinBERT sentiment-only**: significant asymmetry, negative stronger (`cluster_p=0.0179`).
- **CAR_03 + FinBERT sentiment-only**: significant asymmetry, negative stronger (`cluster_p=0.0174`).
- All other model/outcome combinations: no significant asymmetry under clustered inference.

### Key takeaway
- There is **some evidence** that negative sentiment matters more than positive sentiment for short-window returns.
- But this is **not universal across all model setups**, so it should be presented as suggestive, not definitive.

---

## Are these results statistically significant, and do they add value?

### 1) Significance: are we seeing a real effect or random noise?

- For the core asymmetry question (negative vs positive sentiment), significance is tested with p-values.
- In this project, a common threshold is `p < 0.05`.
- Under the stricter clustered test:
  - FinBERT sentiment-only for `CAR_01` and `CAR_03` is significant (`cluster_p` around `0.018`).
  - Most other asymmetry tests are **not** significant.

Meaning: we have **targeted evidence** (not broad evidence) that negative sentiment can matter more than positive sentiment for short-window returns.

### 2) Compared to a simple average baseline

- The `null` model is the baseline that predicts the average outcome only.
- `test_r2` tells us if a model beats that baseline on unseen data:
  - `test_r2 > 0`: better than average-only baseline.
  - `test_r2 = 0`: same as baseline.
  - `test_r2 < 0`: worse than baseline.

What happened here:
- For returns (`CAR_01`, `CAR_03`), only `finbert_sentiment` clearly beats the average baseline.
- For volatility change, controls-based models beat the baseline the most.

### 3) Is practical value being created?

Short answer: **some value, but limited and specific**.

- **Useful where it works:**
  - FinBERT sentiment adds value for short-window return prediction.
  - Controls add value for volatility-change prediction.
- **Not broadly useful yet:**
  - Many larger/extended models underperform the baseline out-of-sample.
  - So this is not yet a strong universal forecasting system.

### Stakeholder conclusion

- The models are **informative in specific areas**, not uniformly informative across all setups.
- Current value is best framed as:
  - a focused signal for short-window return asymmetry (FinBERT sentiment),
  - plus a controls-driven signal for volatility change.
- This is enough to support a careful business/research insight, but not enough to claim robust production-grade forecasting performance yet.

---

## Executive summary (one-minute version)

- Best return models are **simple FinBERT sentiment models**.
- Best volatility model is **controls-based**, not sentiment-only.
- Evidence for “negative > positive” exists mainly in **FinBERT sentiment-only return models**.
- Adding many extra controls improved statistical depth but did **not** improve prediction quality in this sample.

## Source files used for this summary

- `data/processed/model_performance_extended.csv`
- `data/processed/asymmetry_results_hc3_vs_cluster.csv`
- `data/processed/robustness_grid_summary.csv`
- `notebooks/Data_Modelling/15_modelling_pipeline.ipynb`
