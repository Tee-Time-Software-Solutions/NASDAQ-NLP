import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch

from scripts.compute_lexicon_sentiment import (
    load_events,
    load_lm_dictionary,
    compute_lexicon_features,
)


class TestLoadEvents:
    def test_valid_csv(self, tmp_path):
        csv = tmp_path / "events.csv"
        csv.write_text(
            "ticker,file_name,event_trading_day_final,file_path\n"
            "AAPL,test.txt,2016-01-27,data/test.txt\n"
        )
        df = load_events(csv)
        assert len(df) == 1
        assert df.iloc[0]["ticker"] == "AAPL"

    def test_missing_columns_raises(self, tmp_path):
        csv = tmp_path / "bad.csv"
        csv.write_text("ticker,file_name\nAAPL,test.txt\n")
        with pytest.raises(ValueError, match="Missing required columns"):
            load_events(csv)


class TestLoadLmDictionary:
    def test_fallback_dictionary(self):
        pos, neg = load_lm_dictionary(None)
        assert isinstance(pos, set)
        assert isinstance(neg, set)
        assert len(pos) > 0
        assert len(neg) > 0
        assert "profit" in pos
        assert "loss" in neg

    def test_nonexistent_path_uses_fallback(self):
        pos, neg = load_lm_dictionary(Path("/nonexistent/file.csv"))
        assert "profit" in pos
        assert "loss" in neg

    def test_custom_csv(self, tmp_path):
        csv = tmp_path / "lm.csv"
        csv.write_text("Word,Positive,Negative\nbullish,1,0\nbearish,0,1\n")
        pos, neg = load_lm_dictionary(csv)
        assert "bullish" in pos
        assert "bearish" in neg


class TestComputeLexiconFeatures:
    def test_basic_computation(self, tmp_path):
        transcript = tmp_path / "transcript.txt"
        transcript.write_text("profit growth strong loss decline risk")

        events = pd.DataFrame([{
            "ticker": "TEST",
            "file_name": "transcript.txt",
            "event_trading_day_final": "2020-01-01",
            "file_path": str(transcript),
        }])

        result = compute_lexicon_features(events)
        assert len(result) == 1
        assert result.iloc[0]["pos_count_lm"] > 0
        assert result.iloc[0]["neg_count_lm"] > 0
        assert result.iloc[0]["total_tokens"] == 6

    def test_missing_file_skipped(self):
        events = pd.DataFrame([{
            "ticker": "TEST",
            "file_name": "missing.txt",
            "event_trading_day_final": "2020-01-01",
            "file_path": "/nonexistent/missing.txt",
        }])

        result = compute_lexicon_features(events)
        assert len(result) == 0

    def test_rates_sum_correctly(self, tmp_path):
        transcript = tmp_path / "t.txt"
        transcript.write_text("profit loss other other other")

        events = pd.DataFrame([{
            "ticker": "TEST",
            "file_name": "t.txt",
            "event_trading_day_final": "2020-01-01",
            "file_path": str(transcript),
        }])

        result = compute_lexicon_features(events)
        row = result.iloc[0]
        assert row["pos_rate_lm"] == pytest.approx(row["pos_count_lm"] / row["total_tokens"])
        assert row["neg_rate_lm"] == pytest.approx(row["neg_count_lm"] / row["total_tokens"])
