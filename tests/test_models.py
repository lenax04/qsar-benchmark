"""Unit tests for the models module."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models import (
    get_models,
    evaluate_regression,
    evaluate_classification,
    cross_validate_model,
    train_and_evaluate,
)

# Synthetic data for testing
np.random.seed(42)
N = 200
X_dummy = np.random.randn(N, 50).astype(np.float32)
y_reg_dummy = (X_dummy[:, 0] * 2 + np.random.randn(N) * 0.5).astype(np.float32)
y_cls_dummy = (y_reg_dummy > y_reg_dummy.mean()).astype(np.int32)


class TestGetModels:
    def test_regression_models(self):
        models = get_models("regression")
        assert set(models.keys()) == {"Random Forest", "XGBoost", "SVM", "MLP"}

    def test_classification_models(self):
        models = get_models("classification")
        assert set(models.keys()) == {"Random Forest", "XGBoost", "SVM", "MLP"}


class TestEvaluateRegression:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        metrics = evaluate_regression(y, y)
        assert metrics["R2"] == 1.0
        assert metrics["RMSE"] == 0.0
        assert metrics["MAE"] == 0.0

    def test_metrics_range(self):
        y_true = np.random.randn(100)
        y_pred = np.random.randn(100)
        metrics = evaluate_regression(y_true, y_pred)
        assert "R2" in metrics
        assert "RMSE" in metrics
        assert "MAE" in metrics
        assert metrics["RMSE"] >= 0
        assert metrics["MAE"] >= 0


class TestEvaluateClassification:
    def test_perfect_prediction(self):
        y = np.array([0, 1, 0, 1, 1])
        metrics = evaluate_classification(y, y, y.astype(float))
        assert metrics["Accuracy"] == 1.0
        assert metrics["F1"] == 1.0
        assert metrics["ROC_AUC"] == 1.0

    def test_metrics_keys(self):
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        y_prob = np.array([0.1, 0.9, 0.6, 0.8, 0.3, 0.2])
        metrics = evaluate_classification(y_true, y_pred, y_prob)
        assert "Accuracy" in metrics
        assert "F1" in metrics
        assert "ROC_AUC" in metrics
        assert "PR_AUC" in metrics


class TestTrainAndEvaluate:
    def test_rf_regression(self):
        models = get_models("regression")
        model = models["Random Forest"]
        X_tr, X_te = X_dummy[:160], X_dummy[160:]
        y_tr, y_te = y_reg_dummy[:160], y_reg_dummy[160:]
        result, fitted, y_pred = train_and_evaluate(
            model, X_tr, y_tr, X_te, y_te, task="regression", model_name="RF"
        )
        assert "R2" in result
        assert "RMSE" in result
        assert result["R2"] > 0.5  # Should be reasonably good on synthetic data
        assert len(y_pred) == len(y_te)

    def test_xgb_classification(self):
        models = get_models("classification")
        model = models["XGBoost"]
        X_tr, X_te = X_dummy[:160], X_dummy[160:]
        y_tr, y_te = y_cls_dummy[:160], y_cls_dummy[160:]
        result, fitted, y_pred = train_and_evaluate(
            model, X_tr, y_tr, X_te, y_te, task="classification", model_name="XGB"
        )
        assert "ROC_AUC" in result
        assert result["ROC_AUC"] > 0.5
        assert len(y_pred) == len(y_te)


class TestCrossValidate:
    def test_rf_cv_regression(self):
        models = get_models("regression")
        model = models["Random Forest"]
        cv_metrics = cross_validate_model(
            model, X_dummy, y_reg_dummy, task="regression", n_splits=3
        )
        assert "R2_mean" in cv_metrics
        assert "RMSE_mean" in cv_metrics
        assert cv_metrics["R2_mean"] > 0.3

    def test_rf_cv_classification(self):
        models = get_models("classification")
        model = models["Random Forest"]
        cv_metrics = cross_validate_model(
            model, X_dummy, y_cls_dummy, task="classification", n_splits=3
        )
        assert "ROC_AUC_mean" in cv_metrics
        assert cv_metrics["ROC_AUC_mean"] > 0.5
