from pathlib import Path

from nasdaq_nlp.config import (
    PROJECT_ROOT,
    DATASET_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
    ensure_output_dirs,
    EST_START,
    EST_END,
    EVENT_START,
    EVENT_END_SHORT,
    EVENT_END_LONG,
    TRAIN_YEARS,
    TEST_YEARS,
)


class TestConfig:
    def test_project_root_exists(self):
        assert PROJECT_ROOT.exists()

    def test_dataset_dir_is_under_root(self):
        assert str(DATASET_DIR).startswith(str(PROJECT_ROOT))

    def test_processed_dir_is_under_root(self):
        assert str(PROCESSED_DIR).startswith(str(PROJECT_ROOT))

    def test_results_dir_is_under_root(self):
        assert str(RESULTS_DIR).startswith(str(PROJECT_ROOT))

    def test_ensure_output_dirs_creates_dirs(self, tmp_path, monkeypatch):
        monkeypatch.setattr("nasdaq_nlp.config.PROCESSED_DIR", tmp_path / "proc")
        monkeypatch.setattr("nasdaq_nlp.config.RESULTS_DIR", tmp_path / "res")
        monkeypatch.setattr("nasdaq_nlp.config.LOGS_DIR", tmp_path / "logs")
        ensure_output_dirs()
        assert (tmp_path / "proc").exists()
        assert (tmp_path / "res").exists()
        assert (tmp_path / "logs").exists()

    def test_estimation_window_order(self):
        assert EST_START < EST_END < EVENT_START

    def test_event_window_order(self):
        assert EVENT_START <= EVENT_END_SHORT <= EVENT_END_LONG

    def test_train_test_years_no_overlap(self):
        assert set(TRAIN_YEARS).isdisjoint(set(TEST_YEARS))
