# Modelling
**Part V(d) — Modelling Report**
*Asymmetric Sentiment Effects in Earnings Calls — NLP Course, Group Project*

---

## Task Definitions

We frame the prediction problem in two complementary ways:

- **Regression**: predict CAR[0,3] (a continuous return) from transcript features and controls.
- **Classification**: predict whether CAR[0,3] > 0 (binary direction label).

The regression task is the primary one because it supports the formal Wald test of the asymmetry hypothesis.

## Regression Models

### OLS with HC3 Standard Errors

The main specification is:

```
CAR_i[0,3] = β0 + β1·NegRate_i + β2·PosRate_i + x_i·γ + ε_i
```

where controls `x_i` include pre-event volatility and year dummies. OLS is estimated with HC3 heteroskedasticity-consistent standard errors via `statsmodels`. The asymmetry test is a Wald test of H₀: β1 + β2 = 0.

### Ridge Regression

Ridge (ℓ2 penalty, α = 1) is used when high-dimensional features (TF-IDF, year dummies) are included to control overfitting. With only ~155 training events, plain OLS on 500+ features is severely underdetermined.

### Random Forest and MLP

Random Forest (100 trees) and a two-hidden-layer MLP are included as nonlinear baselines. Both use `random_state=42` for reproducibility.

### Null Model

The null model predicts the training-set mean for every test event. Positive OOS R² means a model beats this naive baseline.

## Classification Models

Binary label: y = 1 if CAR[0,3] > 0. We evaluate five classifiers:

- Naive Bayes (Gaussian for LM features; Multinomial for TF-IDF)
- Logistic Regression (`max_iter=1000`)
- Decision Tree
- Random Forest (100 trees)
- Multi-Layer Perceptron (two hidden layers)

All use `random_state=42`. Class imbalance in the test window (positive CARs dominated 2019–2020) means F1 and AUC are more informative than accuracy alone.

## Time-Series Cross-Validation

Beyond the single train/test split, we run an expanding-window cross-validation: train on 2016→year Y, test on year Y+1, for Y ∈ {2017, 2018, 2019}. Pooled OOS R² averages across all folds gives a more robust estimate than a single test window.

## Code Reference

- `src/nasdaq_nlp/models/regression.py` — OLS, Ridge, RF, MLP, Wald test.
- `src/nasdaq_nlp/models/classifiers.py` — classification models.
- `src/nasdaq_nlp/evaluation/` — OOS R², MAE, AUC utilities.
- `notebooks/03_modeling.ipynb` — full model training and result export.
