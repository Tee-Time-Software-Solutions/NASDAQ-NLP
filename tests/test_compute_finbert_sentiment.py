import pytest
from unittest.mock import patch, MagicMock

import pandas as pd

from scripts.compute_finbert_sentiment import (
    simple_sentence_split,
    batched,
    aggregate_probs,
    main,
    FinBertConfig,
    load_events,
    read_transcript,
)


class TestSimpleSentenceSplit:
    def test_basic_split(self):
        text = "First sentence. Second sentence. Third one."
        result = simple_sentence_split(text)
        assert len(result) == 3
        assert result[0] == "First sentence."

    def test_single_sentence(self):
        result = simple_sentence_split("Just one sentence.")
        assert len(result) == 1

    def test_empty_string(self):
        result = simple_sentence_split("")
        assert result == []

    def test_question_and_exclamation(self):
        text = "Is this working? Yes it is! Great."
        result = simple_sentence_split(text)
        assert len(result) == 3


class TestBatched:
    def test_exact_batches(self):
        items = ["a", "b", "c", "d"]
        result = list(batched(items, 2))
        assert result == [["a", "b"], ["c", "d"]]

    def test_remainder_batch(self):
        items = ["a", "b", "c", "d", "e"]
        result = list(batched(items, 2))
        assert result == [["a", "b"], ["c", "d"], ["e"]]

    def test_batch_larger_than_input(self):
        items = ["a", "b"]
        result = list(batched(items, 10))
        assert result == [["a", "b"]]

    def test_empty_input(self):
        result = list(batched([], 5))
        assert result == []

    def test_single_item_batches(self):
        items = ["a", "b", "c"]
        result = list(batched(items, 1))
        assert result == [["a"], ["b"], ["c"]]


class TestAggregateProbs:
    def test_standard_scores(self):
        scores = [
            [{"label": "positive", "score": 0.8},
             {"label": "negative", "score": 0.1},
             {"label": "neutral", "score": 0.1}],
            [{"label": "positive", "score": 0.2},
             {"label": "negative", "score": 0.7},
             {"label": "neutral", "score": 0.1}],
        ]
        result = aggregate_probs(scores)
        assert result["finbert_pos_mean"] == pytest.approx(0.5)
        assert result["finbert_neg_mean"] == pytest.approx(0.4)
        assert result["finbert_neu_mean"] == pytest.approx(0.1)

    def test_empty_scores(self):
        result = aggregate_probs([])
        assert result["finbert_pos_mean"] == 0.0
        assert result["finbert_neg_mean"] == 0.0
        assert result["finbert_neu_mean"] == 0.0

    def test_single_sentence(self):
        scores = [
            [{"label": "positive", "score": 0.6},
             {"label": "negative", "score": 0.3},
             {"label": "neutral", "score": 0.1}],
        ]
        result = aggregate_probs(scores)
        assert result["finbert_pos_mean"] == pytest.approx(0.6)
        assert result["finbert_neg_mean"] == pytest.approx(0.3)

    def test_dict_format(self):
        scores = [
            {"label": "positive", "score": 0.9},
            {"label": "negative", "score": 0.1},
        ]
        result = aggregate_probs(scores)
        assert result["finbert_pos_mean"] == pytest.approx(0.45)
        assert result["finbert_neg_mean"] == pytest.approx(0.05)

    def test_empty_label_ignored(self):
        scores = [
            [{"label": "", "score": 0.5}, {"label": "positive", "score": 0.5}],
        ]
        result = aggregate_probs(scores)
        assert result["finbert_pos_mean"] == pytest.approx(0.5)


class TestFinBertConfig:
    def test_defaults(self):
        cfg = FinBertConfig()
        assert cfg.model_name == "ProsusAI/finbert"
        assert cfg.max_length == 256
        assert cfg.batch_size == 16

    def test_custom_values(self):
        cfg = FinBertConfig(model_name="custom", max_length=128, batch_size=8)
        assert cfg.model_name == "custom"
        assert cfg.max_length == 128


class TestLoadEventsFinbert:
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


class TestReadTranscript:
    def test_reads_file(self, tmp_path):
        f = tmp_path / "transcript.txt"
        f.write_text("Hello world")
        assert read_transcript(f) == "Hello world"

    def test_handles_encoding(self, tmp_path):
        f = tmp_path / "transcript.txt"
        f.write_bytes(b"Hello \xff world")
        result = read_transcript(f)
        assert "Hello" in result


class TestMain:
    def _setup_data(self, tmp_path, transcripts=None):
        """Create events CSV and transcript files in tmp_path."""
        processed = tmp_path / "data" / "processed"
        processed.mkdir(parents=True)

        if transcripts is None:
            transcripts = [("AAPL", "t.txt", "2016-01-27",
                            "Revenue grew. Profits strong. Risk declined.")]

        rows = []
        for ticker, fname, date, text in transcripts:
            tf = tmp_path / fname
            tf.write_text(text)
            rows.append(f"{ticker},{fname},{date},{tf}")

        events_csv = processed / "event_study_dataset.csv"
        events_csv.write_text(
            "ticker,file_name,event_trading_day_final,file_path\n"
            + "\n".join(rows) + "\n"
        )
        return processed

    def test_processes_events_and_saves(self, tmp_path):
        processed = self._setup_data(tmp_path)

        fake_scores = [
            [{"label": "positive", "score": 0.7},
             {"label": "negative", "score": 0.2},
             {"label": "neutral", "score": 0.1}],
        ]
        mock_clf = MagicMock(return_value=fake_scores)

        with patch(
            "scripts.compute_finbert_sentiment.build_finbert_pipeline",
            return_value=mock_clf,
        ), patch(
            "scripts.compute_finbert_sentiment.Path.resolve",
            return_value=tmp_path / "scripts" / "compute_finbert_sentiment.py",
        ):
            main()

        result = pd.read_csv(processed / "finbert_sentiment_features.csv")
        assert len(result) == 1
        assert result.iloc[0]["ticker"] == "AAPL"
        assert "finbert_pos_mean" in result.columns
        assert "finbert_neg_mean" in result.columns

    def test_no_events_raises(self, tmp_path):
        processed = tmp_path / "data" / "processed"
        processed.mkdir(parents=True)
        events_csv = processed / "event_study_dataset.csv"
        events_csv.write_text(
            "ticker,file_name,event_trading_day_final,file_path\n"
        )

        with patch(
            "scripts.compute_finbert_sentiment.build_finbert_pipeline",
        ), patch(
            "scripts.compute_finbert_sentiment.Path.resolve",
            return_value=tmp_path / "scripts" / "compute_finbert_sentiment.py",
        ):
            with pytest.raises(RuntimeError, match="No events found"):
                main()

    def test_missing_transcript_skipped(self, tmp_path):
        processed = tmp_path / "data" / "processed"
        processed.mkdir(parents=True)
        events_csv = processed / "event_study_dataset.csv"
        events_csv.write_text(
            "ticker,file_name,event_trading_day_final,file_path\n"
            "AAPL,t.txt,2016-01-27,/nonexistent/path.txt\n"
        )

        mock_clf = MagicMock()

        with patch(
            "scripts.compute_finbert_sentiment.build_finbert_pipeline",
            return_value=mock_clf,
        ), patch(
            "scripts.compute_finbert_sentiment.Path.resolve",
            return_value=tmp_path / "scripts" / "compute_finbert_sentiment.py",
        ):
            main()

        output = processed / "finbert_sentiment_features.csv"
        assert output.exists()
        # Empty DataFrame produces a CSV with no data rows
        content = output.read_text()
        assert content.strip() == ""
