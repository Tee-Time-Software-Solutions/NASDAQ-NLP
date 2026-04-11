# Preprocessing
**Part V(b) — Modelling Report**
*Asymmetric Sentiment Effects in Earnings Calls — NLP Course, Group Project*

---

## Overview

Each raw transcript goes through four preprocessing stages before any features are extracted. The design principle is to do the minimum necessary to make LM lexicon matching reliable, without stripping information that could matter for TF-IDF or Word2Vec features.

## Stage 1 — Header and Boilerplate Removal

The first lines of every transcript contain operator instructions, legal disclaimers, safe-harbor forward-looking statements, and participant metadata (names, titles, firms). These are removed because they are formulaic and carry no call-specific sentiment. The removal is pattern-based: we strip everything up to and including the operator's opening instruction line, then remove the participant list block.

## Stage 2 — Section Split

This is the most important preprocessing step for our analysis. Each transcript is divided into two sections:

- **Presentation**: Prepared executive remarks, read from a script. Typically the CEO/CFO walking through quarterly results and guidance.
- **Q&A**: Unscripted exchange between analysts and management. Analysts pose questions; executives respond without prepared text.

The boundary is detected using participant turn-marker patterns (e.g., "Questions and Answers", "Operator" handoff lines). Section-level features (`NegRate_pres`, `NegRate_qa`) computed from this split turned out to be among the most informative features in our analysis — the Q&A section carries significantly stronger negative signal than the presentation.

## Stage 3 — Normalization

Text is lowercased and punctuation is standardized. We intentionally **do not** apply stemming or stopword removal. Stemming would corrupt LM lexicon matches ("losses" → "loss" would fail to match the LM entry "LOSSES"). Stopwords are retained so the token count denominator in NegRate/PosRate correctly reflects total transcript length.

## Stage 4 — Tokenization

Words are extracted using the regex `[A-Za-z']+`, which captures alphabetic tokens and contractions while excluding numbers and symbols. This is consistent with the word boundaries used by the LM Master Dictionary.

## Output

The preprocessing pipeline produces, per transcript:

- `clean_text` — full normalized transcript.
- `pres_text` — presentation section only.
- `qa_text` — Q&A section only.
- `tokens` — tokenized full transcript (list of strings).

## Code Reference

- `src/nasdaq_nlp/preprocessing/` — cleaning and tokenization functions.
- `notebooks/02_feature_extraction.ipynb` — preprocessing is run as part of the feature extraction notebook.
