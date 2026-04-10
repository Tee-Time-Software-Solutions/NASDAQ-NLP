import numpy as np
import pytest

from nasdaq_nlp.evaluation.metrics import (
    r_squared,
    oos_r_squared,
    classification_report_dict,
)


class TestRSquared:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert r_squared(y, y) == pytest.approx(1.0)

    def test_mean_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        y_pred = np.full_like(y, fill_value=y.mean())
        assert r_squared(y, y_pred) == pytest.approx(0.0)

    def test_worse_than_mean(self):
        y = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([3.0, 1.0, 2.0])
        assert r_squared(y, y_pred) < 0

    def test_constant_y(self):
        y = np.array([5.0, 5.0, 5.0])
        y_pred = np.array([5.0, 5.0, 5.0])
        assert r_squared(y, y_pred) == 1.0


class TestOosRSquared:
    def test_model_beats_mean(self):
        y_train = np.array([1.0, 2.0, 3.0])
        y_test = np.array([2.0, 3.0])
        y_pred = np.array([2.1, 2.9])
        assert oos_r_squared(y_train, y_test, y_pred) > 0

    def test_model_equals_mean(self):
        y_train = np.array([1.0, 2.0, 3.0])
        train_mean = y_train.mean()
        y_test = np.array([2.0, 3.0])
        y_pred = np.full_like(y_test, fill_value=train_mean)
        assert oos_r_squared(y_train, y_test, y_pred) == pytest.approx(0.0)

    def test_model_worse_than_mean(self):
        y_train = np.array([1.0, 2.0, 3.0])
        y_test = np.array([2.0, 3.0])
        y_pred = np.array([10.0, -5.0])
        assert oos_r_squared(y_train, y_test, y_pred) < 0


class TestClassificationReport:
    def test_perfect_classification(self):
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0])
        report = classification_report_dict(y_true, y_pred)
        assert report["accuracy"] == 1.0
        assert report["f1"] == 1.0

    def test_with_probabilities(self):
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.2])
        report = classification_report_dict(y_true, y_pred, y_prob=y_prob)
        assert "auc" in report
        assert report["auc"] == 1.0

    def test_all_wrong(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        report = classification_report_dict(y_true, y_pred)
        assert report["accuracy"] == 0.0
