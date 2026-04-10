# Math From Scratch — NASDAQ-NLP

> **Audience**: 3rd-year CS undergrad. No finance background assumed.
> Each section builds on the previous one.

---

## 1. Simple Daily Returns

We need a way to measure "how much did the stock move today?"
The natural answer is the **percentage change in price**:

$$R_t = \frac{P_t - P_{t-1}}{P_{t-1}} = \frac{P_t}{P_{t-1}} - 1$$

where:
- $P_t$ = Adjusted Close price on day $t$
- $R_t$ = return on day $t$ (a decimal, e.g. 0.02 = 2% gain)

**Why "Adjusted" Close?** Raw prices jump when a company does a stock split
(e.g. 4-for-1 split cuts the price by 75%). Adjusted Close accounts for
splits and dividends so returns are comparable over time.

**Why not log returns?** Log returns ($\ln P_t / P_{t-1}$) are also common.
They're additive across time (over a year = sum of daily log returns).
We use simple returns because they're **additive across assets** at a point in time,
which is what we need for CAR = AR_0 + AR_1 + ...

---

## 2. Event Time

Each earnings call has a date. We call this date **day 0** (the "event day").
We then index all other dates relative to day 0:

```
Day -120  -20   -1   0   +1   +3   +10
  |--------|-----|---|-----|-----|-----|
  estimation window  call  event window  vol window
```

This relative indexing (trading days, not calendar days) lets us compare
events across different dates, quarters, and tickers on the same scale.

---

## 3. Market Model — OLS Regression

**Why do we need a model?**
If AAPL drops 3% on the day of its earnings call, is that because:
(a) the earnings call was bad, or
(b) the whole market dropped 3% that day?

We need to separate the stock-specific reaction from the market-wide move.
The **market model** does this.

**The regression:**

$$R_{i,t} = \alpha_i + \beta_i R_{m,t} + \varepsilon_{i,t}$$

- $R_{i,t}$ = stock $i$'s return on trading day $t$
- $R_{m,t}$ = NASDAQ index return on day $t$ (the "market")
- $\alpha_i$ = intercept (stock $i$'s average return unexplained by market)
- $\beta_i$ = slope (how sensitive is stock $i$ to market moves?)
- $\varepsilon_{i,t}$ = residual (what's left over — noise, idiosyncratic events)

**Estimation window**: We fit this regression using **100 trading days of data**
from day −120 to day −20. We exclude the 20 days nearest the event to avoid
contaminating the model with the earnings call itself.

**Ordinary Least Squares (OLS)** minimises the sum of squared residuals:

$$\min_{\alpha, \beta} \sum_{t \in \text{est. window}} \varepsilon_{i,t}^2 = \sum_{t} (R_{i,t} - \alpha_i - \beta_i R_{m,t})^2$$

The closed-form solution (in matrix notation):

$$\begin{bmatrix} \hat\alpha \\ \hat\beta \end{bmatrix} = (X^\top X)^{-1} X^\top y$$

where $X = [1, R_{m,t}]$ (design matrix with intercept column) and $y = R_{i,t}$ (outcome vector).

**Interpreting $\beta$:**
- $\beta = 1.0$: the stock moves 1% for every 1% market move (average)
- $\beta = 1.5$: 1.5% per 1% market move (amplifies market, "aggressive" stock)
- $\beta = 0.5$: 0.5% per 1% market move (dampens market, "defensive" stock)

---

## 4. Abnormal Return (AR)

Once we have $\hat\alpha_i$ and $\hat\beta_i$, we compute what the stock
**would have returned** on each event-window day if it just moved with the market:

$$\hat{R}_{i,t} = \hat\alpha_i + \hat\beta_i R_{m,t} \quad \text{(expected return)}$$

The **abnormal return** is the difference between what actually happened
and what the market model predicted:

$$AR_{i,t} = R_{i,t} - \hat{R}_{i,t} = R_{i,t} - (\hat\alpha_i + \hat\beta_i R_{m,t})$$

Interpretation:
- $AR > 0$: stock outperformed market-model expectation (positive earnings surprise)
- $AR < 0$: stock underperformed (negative surprise)
- $AR = 0$: stock moved exactly as the market model predicted (no earnings news)

---

## 5. Cumulative Abnormal Return (CAR)

A single-day AR is noisy. We sum over a window to smooth out daily noise
and capture the full market reaction:

$$\text{CAR}_i[0, 3] = AR_{i,0} + AR_{i,1} + AR_{i,2} + AR_{i,3}$$

We compute two windows:
- **CAR[0,1]**: 2-day window (immediate reaction)
- **CAR[0,3]**: 4-day window (captures post-announcement drift)

**Why [0,3]?** Earnings calls often happen after market close (day 0), so the
main price reaction starts on day +1. Including day +2 and +3 catches
delayed reactions as analysts update their models and issue revised ratings.

---

## 6. Volatility Change (ΔVol)

Volatility = **standard deviation of daily returns** over a window.

$$\sigma_\text{pre} = \text{std}(R_{i,-10}, R_{i,-9}, \ldots, R_{i,-1})$$
$$\sigma_\text{post} = \text{std}(R_{i,+1}, R_{i,+2}, \ldots, R_{i,+10})$$

$$\Delta\text{Vol}_i = \sigma_\text{post} - \sigma_\text{pre}$$

$\Delta\text{Vol} > 0$: earnings call increased return variability.
This happens when the call introduces new uncertainty (guidance revision, legal issue).

---

## 7. TF-IDF (Term Frequency–Inverse Document Frequency)

**Problem**: Simple word counts favour long documents and common words ("the", "and").
TF-IDF corrects this with two components:

**Term Frequency (TF):**
$$\text{TF}(t, d) = \frac{\text{count of } t \text{ in document } d}{|\text{words in } d|}$$

Measures how often term $t$ appears *relative to document length*.

**Inverse Document Frequency (IDF):**
$$\text{IDF}(t) = \log\left(\frac{N}{df(t)}\right)$$

- $N$ = total number of documents (188 transcripts)
- $df(t)$ = number of documents containing $t$

High IDF: term is rare across the corpus → more discriminative.
Low IDF: term appears in almost every document ("revenue", "quarter") → less useful for distinguishing documents.

**TF-IDF score:**
$$\text{TFIDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

---

## 8. Naive Bayes

Based on **Bayes' theorem**:

$$P(y \mid x) = \frac{P(x \mid y) \, P(y)}{P(x)}$$

For classification: given the words in the transcript, what's the probability
of a positive market reaction ($y = 1$)?

$$P(y \mid x_1, \ldots, x_n) \propto P(y) \prod_{i=1}^{n} P(x_i \mid y)$$

The **"naive"** assumption: features $x_i$ are **conditionally independent** given $y$.
This is false (words co-occur), but it works surprisingly well in practice.

**Multinomial NB for text**: $P(x_i \mid y)$ is the probability of seeing word $i$
given class $y$, estimated from training counts with Laplace smoothing:

$$P(x_i \mid y) = \frac{\text{count}(x_i \text{ in class } y) + 1}{\sum_j \text{count}(x_j \text{ in class } y) + V}$$

where $V$ is the vocabulary size and the +1 is Laplace (add-one) smoothing
to avoid zero probabilities for unseen words.

---

## 9. Logistic Regression

Models the *probability* of class $y = 1$ directly:

$$P(y = 1 \mid x) = \sigma(\beta_0 + \beta_1 x_1 + \ldots + \beta_n x_n) = \frac{1}{1 + e^{-(\beta^\top x)}}$$

where $\sigma(z) = 1 / (1 + e^{-z})$ is the **sigmoid function** (maps any real number to (0,1)).

The **log-odds** (logit) is linear:
$$\log\frac{P(y=1)}{P(y=0)} = \beta^\top x$$

**Training**: maximize the log-likelihood over training examples:
$$\hat\beta = \arg\max_\beta \sum_i \left[ y_i \log P_i + (1-y_i) \log(1 - P_i) \right]$$

No closed-form solution — solved iteratively (L-BFGS optimizer in our implementation).

**L2 regularisation** (Ridge): adds a penalty $\lambda \|\beta\|^2$ to prevent overfitting
on high-dimensional TF-IDF features. Hyperparameter $C = 1/\lambda$ controls strength.

---

## 10. Word2Vec Embeddings

**Goal**: represent words as dense vectors where similar words are close together.

**Skip-gram** architecture: train a neural network to predict surrounding words
given a center word.

For a window of size 5, given the word "revenue" in a sentence:
$$[\ldots, \text{strong}, \text{revenue}, \underline{\text{growth}}, \text{exceeded}, \ldots]$$
the model predicts "strong", "growth", "exceeded" (context words) from "revenue".

During training, the model learns a weight matrix $W \in \mathbb{R}^{V \times d}$
where $V$ = vocabulary size and $d$ = embedding dimension (100 in our model).
Row $i$ of $W$ is the embedding vector for word $i$.

After training: words in similar contexts end up with similar vectors.
Measured by **cosine similarity**:

$$\text{cos}(a, b) = \frac{a \cdot b}{\|a\| \|b\|} \in [-1, 1]$$

**Document embedding** (average pooling):
$$\vec{d} = \frac{1}{|T|} \sum_{w \in T} \vec{w}$$

where $T$ is the set of tokens in the transcript. This gives one 100-dim vector per transcript.

---

## 11. OLS Regression (Sentiment → Market Reaction)

Our main model for the research question:

$$\text{CAR}_{i,[0,3]} = \beta_0 + \beta_1 \cdot \text{NegRate}_i + \beta_2 \cdot \text{PosRate}_i + \gamma^\top \text{controls}_i + \varepsilon_i$$

**Controls**: pre-event volatility ($\sigma_\text{pre}$), year fixed effects.

OLS solution: $\hat\beta = (X^\top X)^{-1} X^\top y$ (same as the market model).

**R² (coefficient of determination):**
$$R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2} = 1 - \frac{SS_\text{res}}{SS_\text{tot}}$$

- $R^2 = 1$: perfect prediction
- $R^2 = 0$: no better than guessing the mean
- $R^2 < 0$: worse than the mean (possible out-of-sample)

---

## 12. Wald Test for Asymmetry

We want to test: does negative sentiment have a *larger* impact than positive sentiment?

**Null hypothesis** (symmetric):
$$H_0: \beta_1 + \beta_2 = 0 \quad \Leftrightarrow \quad |\beta_1| = |\beta_2|$$

(If $\beta_1 < 0$ and $\beta_2 > 0$, this says they have equal magnitude.)

**Alternative** (asymmetric):
$$H_1: \beta_1 + \beta_2 \neq 0$$

**Wald statistic:**
$$W = \frac{(R\hat\beta)^2}{R \, \widehat{\text{Var}}(\hat\beta) \, R^\top}$$

where $R = [0, 1, 1, 0, \ldots]$ picks out $\beta_1$ and $\beta_2$.

Under $H_0$, $W \sim \chi^2(1)$ (chi-squared with 1 degree of freedom).
We compute $p = P(W > w_{\text{obs}} \mid H_0)$.

- $p < 0.05$: reject $H_0$ at 5% level → strong evidence of asymmetry
- $p < 0.10$: reject at 10% → marginal evidence
- $p > 0.10$: cannot reject → symmetric effects

**Our result**: Wald $p = 0.054$ for CAR[0,1] with controls → marginal support
for the asymmetry hypothesis (significant at 10%, not at 5%).

---

## 13. FinBERT — BERT for Finance

BERT = **Bidirectional Encoder Representations from Transformers**.

Unlike Word2Vec (static vector per word), BERT produces **context-sensitive** representations
via the self-attention mechanism:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

where $Q$ (queries), $K$ (keys), $V$ (values) are linear projections of the input.
Each token attends to all other tokens — hence "bidirectional."

**FinBERT** (Araci 2019) is BERT fine-tuned on financial text (financial news,
analyst reports) to classify sentiment as:
- Positive / Negative / Neutral

Output: a probability distribution summing to 1 for each sentence.
We average across all sentences in the transcript to get document-level scores.

**Advantage over lexicon**: understands negation ("not strong" → not positive),
context-dependence ("charges" is negative in financial context but neutral generally),
and domain jargon beyond a fixed word list.
