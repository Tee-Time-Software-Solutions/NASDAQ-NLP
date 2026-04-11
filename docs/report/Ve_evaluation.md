# Evaluation
**Part V(e) — Modelling Report**
*Asymmetric Sentiment Effects in Earnings Calls — NLP Course, Group Project*

---

## Evaluation Protocol

All models use a strict time-based train/test split to prevent look-ahead bias:

- **Training**: years 2016–2018 (n ≈ 155 events)
- **Test**: years 2019–2020 (n ≈ 187 events)

No test-period data enters feature vocabulary construction, scaling, or hyperparameter selection. This is essential because many events in the test window occurred during or after COVID-19 market disruptions, and any data leakage would make the evaluation unreliable.

## Regression Metrics

### OOS R²

Following Campbell and Thompson (2008), out-of-sample R² is:

```
OOS R² = 1 - Σ(y_t - ŷ_t)² / Σ(y_t - ȳ_train)²
```

A positive value means the model beats the naive historical-mean forecast. This is a stricter benchmark than in-sample R², which can be inflated by overfitting.

### MAE

Mean Absolute Error in CAR units gives an intuitive sense of typical prediction error.

### Wald Test p-value

For OLS models, we test H₀: β_neg + β_pos = 0 using a Wald test with HC3 standard errors. Rejection at p < 0.10 supports the asymmetry hypothesis.

## Regression Results

| Model | OOS R² |
|-------|--------|
| Null (historical mean) | 0.000 |
| **LM Lexicon [OLS]** | **+0.057** |
| Word2Vec [Ridge] | +0.050 |
| TF-IDF [RF] | +0.045 |
| TF-IDF [Ridge] | +0.043 |
| LM + Controls [Ridge] | −0.044 |
| LM + Controls [OLS] | −0.086 |
| TF-IDF [OLS] | −0.469 |

The LM lexicon achieves the best OOS R² of +0.057. Models with year dummy controls overfit with only ~155 training observations, even with Ridge regularization. The Wald test rejects H₀ at p = 0.042 for LM + Controls [OLS], with β_neg = −8.92 vs β_pos = +3.31.

## Classification Metrics

Accuracy, macro F1, and AUC are reported on the 2019–2020 test set. F1 and AUC are more informative than accuracy because the class distribution in the test window skews toward positive CARs (bull market conditions in 2019–2020).

| Model | Accuracy | F1 | AUC |
|-------|----------|----|-----|
| TF-IDF + LM + Controls [RF] | 59.6% | 0.553 | **0.600** |
| TF-IDF [NaiveBayes] | **59.6%** | 0.632 | 0.588 |
| TF-IDF [RF] | 51.9% | 0.468 | 0.563 |
| TF-IDF [LogReg] | 51.9% | 0.590 | 0.578 |
| LM Lexicon [LogReg] | 45.5% | 0.625 | 0.508 |

Best AUC is 0.600; best accuracy is 59.6%. All TF-IDF models beat the 50% random baseline in accuracy.

## Cross-Validation

An expanding-window CV (train on 2016→Y, test on Y+1 for Y=2017–2019) confirms that LM lexicon features maintain positive pooled OOS R² across folds, while TF-IDF and control-heavy models remain more variable across test years.

## Code Reference

- `src/nasdaq_nlp/evaluation/` — OOS R², Wald test, AUC utilities.
- `notebooks/03_modeling.ipynb` — full evaluation runs.
- `notebooks/04_results.ipynb` — result visualization and interpretation.
