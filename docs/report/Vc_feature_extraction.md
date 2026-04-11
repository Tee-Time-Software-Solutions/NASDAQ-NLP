# Feature Extraction
**Part V(c) — Modelling Report**
*Asymmetric Sentiment Effects in Earnings Calls — NLP Course, Group Project*

---

## Overview

We extract three families of features from each transcript: lexicon-based sentiment rates, TF-IDF bag-of-words vectors, and Word2Vec document embeddings. Each captures a different aspect of transcript language, and we evaluate them separately and in combination.

## Loughran–McDonald Lexicon Features

### Why LM and not a general lexicon

Standard sentiment lexicons (VADER, AFINN) are calibrated on social media and news text. Financial transcripts use specialized vocabulary where words like "loss", "liability", and "default" are clearly negative but would be scored neutrally or incorrectly by general-purpose tools. The Loughran–McDonald Master Dictionary was built specifically for financial document analysis and contains 2,345 negative and 347 positive financial-domain words.

### Computed Features

For each transcript we compute negative and positive rates normalized by total token count:

```
NegRate = N_neg / |W|
PosRate = N_pos / |W|
```

We compute these for three text scopes, yielding **6 features per event**:

| Feature | Scope | Key finding |
|---------|-------|-------------|
| `neg_rate` / `pos_rate` | Full transcript | Baseline |
| `neg_rate_pres` / `pos_rate_pres` | Presentation section | Near-zero negative signal |
| `neg_rate_qa` / `pos_rate_qa` | Q&A section | Dominant negative signal |

## TF-IDF Representation

A bag-of-words TF-IDF matrix is fitted on training-set transcripts only (to avoid look-ahead bias) with:

- Vocabulary size: 500 terms (top by document frequency)
- Minimum document frequency: 3
- N-gram range: unigrams only

Each transcript is represented as a 500-dimensional sparse vector. TF-IDF features are used with Ridge, Random Forest, and Naive Bayes models.

## Word2Vec Document Embeddings

A skip-gram Word2Vec model is trained on all training-set transcripts with:

- Embedding dimension: 100
- Context window: 5 tokens
- Training epochs: 10
- Random seed: 42

Each document is represented as the mean of its in-vocabulary token embeddings, yielding a 100-dimensional dense vector. Tokens absent from the vocabulary are skipped during pooling.

## Control Variables

Pre-event return volatility σ[−10,−1] (standard deviation of daily returns over the 10 days before the event) and year dummy variables are included as controls in the regression models. Volatility controls for the confound that high-volatility events tend to produce larger CAR magnitudes regardless of tone.

## What We Did Not Include: FinBERT

We originally planned to include FinBERT as a fourth feature source. Running fine-tuned Transformer inference over 620 transcripts of up to 15,000 tokens each caused memory exhaustion on our local hardware. FinBERT integration is a logical next step for future work.

## Code Reference

- `src/nasdaq_nlp/features/lexicon.py` — LM feature computation.
- `src/nasdaq_nlp/features/tfidf.py` — TF-IDF fitting and transform.
- `src/nasdaq_nlp/features/embeddings.py` — Word2Vec training and pooling.
- `notebooks/02_feature_extraction.ipynb` — full feature extraction walkthrough.
