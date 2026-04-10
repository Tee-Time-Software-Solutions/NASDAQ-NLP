"""
text.py — Text preprocessing for earnings call transcripts.

WHY PREPROCESSING MATTERS
--------------------------
Raw transcripts contain a lot of noise that has nothing to do with sentiment:
  - Legal disclaimers ("Forward-looking statements…")
  - Operator instructions ("Please press *1 to ask a question")
  - Metadata headers (event title, participant list)
  - Punctuation, numbers, stop words (for bag-of-words models)

If we feed this noise into our sentiment models, they pick up spurious signals.
For example, the word "statement" appears in both "forward-looking statement"
(boilerplate) and "our results beat expectations" (actual content) — we don't
want the boilerplate to pollute the sentiment score.

WHAT WE DO
-----------
1. Strip boilerplate: remove the header block, participant list, legal notices.
2. Section split: separate Presentation (exec remarks) from Q&A (analyst dialog).
   These sections carry different signal — exec remarks are more scripted and
   controlled; Q&A is more spontaneous and may reveal more information.
3. Normalise: lowercase, remove punctuation (for bag-of-words features).
4. Tokenise: split into word tokens (for n-gram and TF-IDF features).

NOTE: We do NOT remove stopwords or apply stemming for the lexicon/FinBERT models
because those approaches need complete words (e.g. "uncertain" → Loughran-McDonald
negative, "UNCERTAIN" would be missed if we strip the 'UN' prefix via stemming).
Stopwords are removed only for the TF-IDF / Word2Vec models where they add noise.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Boilerplate patterns to strip
# ---------------------------------------------------------------------------

# Lines to delete: forward-looking statements, operator instructions, Thomson header
_BOILERPLATE_PATTERNS = [
    # Thomson Reuters header lines
    re.compile(r"Thomson Reuters StreetEvents", re.IGNORECASE),
    re.compile(r"E\s+D\s+I\s+T\s+E\s+D\s+V\s+E\s+R\s+S\s+I\s+O\s+N", re.IGNORECASE),
    # Section dividers (lines of = signs)
    re.compile(r"^={3,}\s*$"),
    # Forward-looking disclaimer keywords — these paragraphs add no sentiment signal
    re.compile(r"forward.looking\s+statement", re.IGNORECASE),
    re.compile(r"safe\s+harbor\s+statement", re.IGNORECASE),
    re.compile(r"non-GAAP|non\s+GAAP", re.IGNORECASE),
    # Operator / conference call instructions
    re.compile(r"\boperator\b.*instructions?\b", re.IGNORECASE),
    re.compile(r"press\s+\*?\d+\s+to\s+(ask|queue)", re.IGNORECASE),
    # Legalese
    re.compile(r"securities\s+and\s+exchange\s+commission", re.IGNORECASE),
]

# Regex for the header block: everything from start until first "=" divider section
# The header contains title, date, participant list — not transcript content
_HEADER_END_RE = re.compile(
    r"={10,}.*?(?:PRESENTATION|QUESTIONS?\s+AND\s+ANSWERS?|Q\s*&\s*A|OVERVIEW)",
    re.IGNORECASE | re.DOTALL,
)


def strip_header(text: str) -> str:
    """Remove everything before the first content section marker.

    The earnings call file starts with:
        Thomson Reuters StreetEvents...
        <date line>
        ============================
        Corporate Participants
        ============================
        ...participant list...
        ============================
        PRESENTATION
        ============================   ← this is where content starts

    We drop everything up to and including the first major section marker.
    If no marker is found, we return the text unchanged (conservative fallback).
    """
    # Find where the actual content starts
    # Look for '======' lines followed by known section headers
    lines = text.split("\n")
    content_start = 0
    found_marker = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        # Look for section headers that mark the start of real content
        if re.match(r"^\s*={5,}\s*$", stripped):
            # Check if the next non-empty line is a content section header
            for j in range(i + 1, min(i + 5, len(lines))):
                next_line = lines[j].strip().upper()
                if any(
                    keyword in next_line
                    for keyword in [
                        "PRESENTATION",
                        "QUESTIONS AND ANSWERS",
                        "Q&A",
                        "OVERVIEW",
                        "FINANCIAL DATA",
                        "CORPORATE PARTICIPANTS",
                    ]
                ):
                    # Skip participant sections — they're just names
                    if "CORPORATE PARTICIPANTS" in next_line or "CONFERENCE CALL" in next_line:
                        continue
                    content_start = i
                    found_marker = True
                    break
            if found_marker:
                break

    return "\n".join(lines[content_start:]) if found_marker else text


def strip_boilerplate_lines(text: str) -> str:
    """Remove individual lines matching boilerplate patterns.

    Applied after strip_header() to catch remaining noise.
    """
    cleaned_lines = []
    for line in text.split("\n"):
        is_boilerplate = any(pat.search(line) for pat in _BOILERPLATE_PATTERNS)
        if not is_boilerplate:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


# ---------------------------------------------------------------------------
# Section splitter: Presentation vs Q&A
# ---------------------------------------------------------------------------

# These regexes detect where the Q&A section begins in the transcript
_QA_SECTION_RE = re.compile(
    r"(QUESTIONS?\s+AND\s+ANSWERS?|Q\s*&\s*A\s+SESSION|QUESTION\s*-\s*AND\s*-\s*ANSWER)",
    re.IGNORECASE,
)


def split_sections(text: str) -> tuple[str, str]:
    """Split a transcript into (presentation_text, qa_text).

    Returns
    -------
    (presentation, qa) — both may be empty strings if sections not found.
    The presentation section is everything before Q&A begins.
    """
    match = _QA_SECTION_RE.search(text)
    if match is None:
        # No Q&A marker found — treat the whole thing as presentation
        return text, ""

    presentation = text[: match.start()].strip()
    qa = text[match.start() :].strip()
    return presentation, qa


# ---------------------------------------------------------------------------
# Normalisation and tokenisation
# ---------------------------------------------------------------------------

# Tokeniser: match sequences of letters and apostrophes (keeps "don't", "we're")
# This preserves contractions which the LM dictionary handles correctly
_WORD_RE = re.compile(r"[A-Za-z']+")


def normalise(text: str) -> str:
    """Lowercase and remove punctuation, keeping apostrophes for contractions.

    Does NOT remove stopwords here — that is done selectively in each feature module:
    - Lexicon: keep all words (need 'uncertain', 'risk', etc.)
    - TF-IDF / Word2Vec: remove stopwords for bag-of-words features
    """
    # Lowercase
    text = text.lower()
    # Remove non-alphabetic characters except apostrophes and whitespace
    text = re.sub(r"[^a-z'\s]", " ", text)
    # Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenise(text: str) -> list[str]:
    """Split normalised text into word tokens.

    Parameters
    ----------
    text : str
        Already normalised (lowercase, no punctuation).

    Returns
    -------
    list[str] — one token per word.
    """
    return _WORD_RE.findall(text)


def tokenise_no_stopwords(text: str, stopwords: set[str] | None = None) -> list[str]:
    """Tokenise and remove stopwords.

    Used for TF-IDF and Word2Vec where stopwords add noise to bag-of-words.
    If no stopword list is provided, uses a minimal set of common English stops.

    Parameters
    ----------
    text : str
        Already normalised.
    stopwords : set[str] | None
        Custom stopword set. If None, uses the minimal built-in set.
    """
    if stopwords is None:
        stopwords = _MINIMAL_STOPWORDS
    return [t for t in tokenise(text) if t not in stopwords and len(t) > 1]


# Minimal English stopwords — intentionally small to preserve financial terms
# We do NOT use NLTK's full list because it would strip "uncertain", "risk", etc.
_MINIMAL_STOPWORDS: set[str] = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "shall",
    "can",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "into",
    "through",
    "during",
    "above",
    "below",
    "between",
    "i",
    "we",
    "you",
    "he",
    "she",
    "it",
    "they",
    "our",
    "your",
    "their",
    "this",
    "that",
    "these",
    "those",
    "so",
    "if",
    "then",
    "than",
    "not",
    "no",
    "yes",
    "very",
    "also",
    "more",
    "some",
    "any",
}


# ---------------------------------------------------------------------------
# High-level convenience function
# ---------------------------------------------------------------------------


def preprocess_transcript(
    raw_text: str,
    section: str = "full",
    remove_stopwords: bool = False,
) -> dict:
    """Full preprocessing pipeline for one transcript.

    Parameters
    ----------
    raw_text : str
        Raw content of a transcript file.
    section : str
        Which section to return: 'full', 'presentation', or 'qa'.
    remove_stopwords : bool
        If True, use tokenise_no_stopwords(). Default False (lexicon-safe).

    Returns
    -------
    dict with keys:
        'presentation_raw'  — presentation section, cleaned but not tokenised
        'qa_raw'            — Q&A section, cleaned but not tokenised
        'full_raw'          — full cleaned text
        'tokens'            — tokenised version of the requested section
    """
    # Step 1: strip header block
    cleaned = strip_header(raw_text)
    # Step 2: strip boilerplate lines
    cleaned = strip_boilerplate_lines(cleaned)
    # Step 3: split into sections
    presentation, qa = split_sections(cleaned)
    full_text = cleaned

    # Step 4: pick the requested section
    if section == "presentation":
        target = presentation
    elif section == "qa":
        target = qa
    else:
        target = full_text

    # Step 5: normalise and tokenise
    target_norm = normalise(target)
    if remove_stopwords:
        tokens = tokenise_no_stopwords(target_norm)
    else:
        tokens = tokenise(target_norm)

    return {
        "presentation_raw": normalise(presentation),
        "qa_raw": normalise(qa),
        "full_raw": normalise(full_text),
        "tokens": tokens,
    }
