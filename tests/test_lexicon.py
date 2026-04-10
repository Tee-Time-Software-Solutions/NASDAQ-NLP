import pytest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from nasdaq_nlp.features.lexicon import (
    load_lm_dictionary,
    compute_lexicon_features,
)


class TestLoadLmDictionary:
    def test_loads_from_csv(self, tmp_path):
        csv = tmp_path / "lm.csv"
        csv.write_text(
            "Word,Positive,Negative\n"
            "PROFIT,354,0\n"
            "LOSS,0,2355\n"
            "NEUTRAL,0,0\n"
        )
        pos, neg = load_lm_dictionary(csv)
        assert "profit" in pos
        assert "loss" in neg
        assert "neutral" not in pos
        assert "neutral" not in neg

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_lm_dictionary(tmp_path / "nonexistent.csv")

    def test_missing_columns_raises(self, tmp_path):
        csv = tmp_path / "bad.csv"
        csv.write_text("Word,SomeOtherCol\ntest,1\n")
        with pytest.raises(ValueError, match="Expected column"):
            load_lm_dictionary(csv)


class TestComputeLexiconFeatures:
    def test_basic_computation(self, tmp_path):
        transcript = tmp_path / "call.txt"
        transcript.write_text("Profit growth strong. Loss decline risk.")

        pos_words = {"profit", "growth", "strong"}
        neg_words = {"loss", "decline", "risk"}

        result = compute_lexicon_features(
            transcript, section="full",
            pos_words=pos_words, neg_words=neg_words,
        )
        assert result["pos_count"] == 3
        assert result["neg_count"] == 3
        assert result["total_tokens"] > 0
        assert result["neg_rate"] == pytest.approx(result["neg_count"] / result["total_tokens"])

    def test_empty_transcript(self, tmp_path):
        transcript = tmp_path / "empty.txt"
        transcript.write_text("")

        result = compute_lexicon_features(
            transcript, section="full",
            pos_words={"good"}, neg_words={"bad"},
        )
        assert result["neg_count"] == 0
        assert result["pos_count"] == 0

    def test_no_matching_words(self, tmp_path):
        transcript = tmp_path / "neutral.txt"
        transcript.write_text("The meeting discussed various topics today.")

        result = compute_lexicon_features(
            transcript, section="full",
            pos_words={"profit"}, neg_words={"loss"},
        )
        assert result["neg_count"] == 0
        assert result["pos_count"] == 0
        assert result["neg_rate"] == 0.0
        assert result["pos_rate"] == 0.0
