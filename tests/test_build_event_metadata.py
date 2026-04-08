import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch

from scripts.build_event_metadata import (
    quarter_from_month,
    parse_filename,
    parse_header_datetime,
    compute_event_fields,
    build_event_metadata,
)


class TestQuarterFromMonth:
    def test_q1(self):
        for m in [1, 2, 3]:
            assert quarter_from_month(m) == "Q1"

    def test_q2(self):
        for m in [4, 5, 6]:
            assert quarter_from_month(m) == "Q2"

    def test_q3(self):
        for m in [7, 8, 9]:
            assert quarter_from_month(m) == "Q3"

    def test_q4(self):
        for m in [10, 11, 12]:
            assert quarter_from_month(m) == "Q4"


class TestParseFilename:
    def test_valid_filename(self):
        result = parse_filename(Path("data/raw/Transcripts/AAPL/2016-Jan-26-AAPL.txt"))
        assert result["ticker"] == "AAPL"
        assert result["year"] == 2016
        assert result["quarter_label"] == "Q1"
        assert result["call_date_from_filename"] == "2016-01-26"
        assert result["file_name"] == "2016-Jan-26-AAPL.txt"

    def test_different_ticker_and_quarter(self):
        result = parse_filename(Path("2019-Oct-15-NVDA.txt"))
        assert result["ticker"] == "NVDA"
        assert result["year"] == 2019
        assert result["quarter_label"] == "Q4"

    def test_invalid_filename_raises(self):
        with pytest.raises(ValueError, match="does not match"):
            parse_filename(Path("bad_filename.txt"))


class TestParseHeaderDatetime:
    def test_valid_header(self):
        text = "JANUARY 26, 2016 / 10:00PM GMT\nSome content"
        raw, parsed = parse_header_datetime(text)
        assert raw == "JANUARY 26, 2016 / 10:00PM GMT"
        assert parsed is not None
        assert parsed.year == 2016
        assert parsed.month == 1
        assert parsed.day == 26

    def test_no_match(self):
        raw, parsed = parse_header_datetime("No datetime here")
        assert raw is None
        assert parsed is None

    def test_different_datetime(self):
        text = "JULY 15, 2019 / 5:30PM GMT"
        raw, parsed = parse_header_datetime(text)
        assert parsed.month == 7
        assert parsed.hour == 17
        assert parsed.minute == 30


class TestComputeEventFields:
    def test_after_market_close(self):
        # 10:00 PM GMT = 5:00 PM ET (after 4 PM close)
        dt = pd.Timestamp("2016-01-26 22:00:00", tz="UTC")
        result = compute_event_fields(dt, "2016-01-26")
        assert result["after_market_close_et"] is True
        assert result["event_trading_day_final"] == "2016-01-27"

    def test_before_market_close(self):
        # 2:00 PM GMT = 9:00 AM ET (before 4 PM close)
        dt = pd.Timestamp("2016-01-26 14:00:00", tz="UTC")
        result = compute_event_fields(dt, "2016-01-26")
        assert result["after_market_close_et"] is False
        assert result["event_trading_day_final"] == "2016-01-26"

    def test_none_timestamp_uses_fallback(self):
        result = compute_event_fields(None, "2016-01-26")
        assert result["event_trading_day_final"] == "2016-01-26"
        assert result["call_datetime_gmt"] is None
        assert result["after_market_close_et"] is None

    def test_friday_after_close_rolls_to_monday(self):
        # Friday 9 PM GMT = Friday 4 PM ET (after close) -> Monday
        dt = pd.Timestamp("2016-01-22 21:01:00", tz="UTC")
        result = compute_event_fields(dt, "2016-01-22")
        assert result["after_market_close_et"] is True
        assert result["event_trading_day_final"] == "2016-01-25"  # Monday

    def test_all_time_fields_populated(self):
        dt = pd.Timestamp("2016-01-26 14:00:00", tz="UTC")
        result = compute_event_fields(dt, "2016-01-26")
        assert result["call_datetime_gmt"] is not None
        assert result["call_datetime_et"] is not None
        assert result["call_time_gmt"] is not None
        assert result["call_time_et"] is not None


class TestBuildEventMetadata:
    def test_with_mock_transcripts(self, tmp_path):
        transcript_dir = tmp_path / "Transcripts" / "AAPL"
        transcript_dir.mkdir(parents=True)
        transcript = transcript_dir / "2016-Jan-26-AAPL.txt"
        transcript.write_text("JANUARY 26, 2016 / 10:00PM GMT\nSome earnings call content.")

        with patch("scripts.build_event_metadata.RAW_TRANSCRIPTS_DIR", tmp_path / "Transcripts"):
            df = build_event_metadata()

        assert len(df) == 1
        assert df.iloc[0]["ticker"] == "AAPL"
        assert df.iloc[0]["error"] is None

    def test_bad_filename_records_error(self, tmp_path):
        transcript_dir = tmp_path / "Transcripts"
        transcript_dir.mkdir(parents=True)
        transcript = transcript_dir / "bad_filename.txt"
        transcript.write_text("Some content")

        with patch("scripts.build_event_metadata.RAW_TRANSCRIPTS_DIR", tmp_path / "Transcripts"):
            df = build_event_metadata()

        assert len(df) == 1
        assert df.iloc[0]["error"] is not None
