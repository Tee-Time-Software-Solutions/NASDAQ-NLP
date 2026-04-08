import pytest
import pandas as pd
import numpy as np

import sys
from pathlib import Path

# The modelling scripts live in notebooks/Data_Modelling/ and use numeric prefixes,
# so we import them by manipulating sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks" / "Data_Modelling"))

import importlib

model_mod = importlib.import_module("12_model_sentiment_vs_market")
time_split = model_mod.time_split
fit_ols = model_mod.fit_ols


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "year": [2017] * 25 + [2019] * 25,
        "CAR_01": np.random.normal(0, 0.05, n),
        "neg_rate_lm": np.random.uniform(0, 0.01, n),
        "pos_rate_lm": np.random.uniform(0, 0.01, n),
        "pre_volatility": np.random.uniform(0.01, 0.05, n),
    })


class TestTimeSplit:
    def test_default_split(self, sample_df):
        train, test = time_split(sample_df)
        assert len(train) == 25
        assert len(test) == 25
        assert train["year"].max() <= 2018
        assert test["year"].min() > 2018

    def test_custom_split_year(self, sample_df):
        train, test = time_split(sample_df, train_end_year=2016)
        assert len(train) == 0
        assert len(test) == 50

    def test_all_in_train(self, sample_df):
        train, test = time_split(sample_df, train_end_year=2020)
        assert len(train) == 50
        assert len(test) == 0


class TestFitOls:
    def test_returns_model_and_r2(self, sample_df):
        train, test = time_split(sample_df)
        model, train_r2, test_r2 = fit_ols(
            train, test, "CAR_01", ["neg_rate_lm", "pos_rate_lm"]
        )
        assert hasattr(model, "params")
        assert isinstance(train_r2, float)
        assert isinstance(test_r2, float)
        assert 0 <= train_r2 <= 1

    def test_single_feature(self, sample_df):
        train, test = time_split(sample_df)
        model, train_r2, test_r2 = fit_ols(
            train, test, "CAR_01", ["pre_volatility"]
        )
        assert "pre_volatility" in model.params.index

    def test_no_constant(self, sample_df):
        train, test = time_split(sample_df)
        model, _, _ = fit_ols(
            train, test, "CAR_01", ["pre_volatility"], add_const=False
        )
        assert "const" not in model.params.index
