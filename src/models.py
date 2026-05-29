"""
models.py — ML model definitions and training/evaluation utilities for QSAR.

Supported models:
    - Random Forest (RF)
    - XGBoost (XGB)
    - Support Vector Machine (SVM)
    - Multi-Layer Perceptron (MLP)

Both regression (pIC50 prediction) and classification (active/inactive) modes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import time
import warnings
from typing import Literal, Optional

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    roc_auc_score, accuracy_score, f1_score,
    average_precision_score, matthews_corrcoef
)
import xgboost as xgb


TaskType = Literal["regression", "classification"]


def get_models(task: TaskType = "regression") -> dict:
    """
    Return a dictionary of model pipelines for benchmarking.

    Parameters
    ----------
    task : 'regression' or 'classification'

    Returns
    -------
    dict mapping model name -> sklearn Pipeline
    """
    if task == "regression":
        models = {
            "Random Forest": Pipeline([
                ("model", RandomForestRegressor(
                    n_estimators=200,
                    max_depth=None,
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=42,
                ))
            ]),
            "XGBoost": Pipeline([
                ("model", xgb.XGBRegressor(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    n_jobs=-1,
                    random_state=42,
                    verbosity=0,
                ))
            ]),
            "SVM": Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVR(
                    kernel="rbf",
                    C=10.0,
                    gamma="scale",
                    epsilon=0.1,
                ))
            ]),
            "MLP": Pipeline([
                ("scaler", StandardScaler()),
                ("model", MLPRegressor(
                    hidden_layer_sizes=(512, 256, 128),
                    activation="relu",
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=42,
                    learning_rate_init=0.001,
                ))
            ]),
        }
    else:  # classification
        models = {
            "Random Forest": Pipeline([
                ("model", RandomForestClassifier(
                    n_estimators=200,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=42,
                ))
            ]),
            "XGBoost": Pipeline([
                ("model", xgb.XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=1,
                    n_jobs=-1,
                    random_state=42,
                    verbosity=0,
                    eval_metric="logloss",
                ))
            ]),
            "SVM": Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVC(
                    kernel="rbf",
                    C=10.0,
                    gamma="scale",
                    probability=True,
                    class_weight="balanced",
                    random_state=42,
                ))
            ]),
            "MLP": Pipeline([
                ("scaler", StandardScaler()),
                ("model", MLPClassifier(
                    hidden_layer_sizes=(512, 256, 128),
                    activation="relu",
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=42,
                    learning_rate_init=0.001,
                ))
            ]),
        }
    return models


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics."""
    return {
        "R2":   round(r2_score(y_true, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "MAE":  round(mean_absolute_error(y_true, y_pred), 4),
    }


def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray,
                              y_prob: Optional[np.ndarray] = None) -> dict:
    """Compute classification metrics."""
    metrics = {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "F1":       round(f1_score(y_true, y_pred, zero_division=0), 4),
        "MCC":      round(matthews_corrcoef(y_true, y_pred), 4),
    }
    if y_prob is not None:
        metrics["ROC_AUC"] = round(roc_auc_score(y_true, y_prob), 4)
        metrics["PR_AUC"]  = round(average_precision_score(y_true, y_prob), 4)
    return metrics


def cross_validate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    task: TaskType = "regression",
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Perform n-fold cross-validation and return mean ± std metrics.

    Parameters
    ----------
    model : sklearn Pipeline
    X : np.ndarray
    y : np.ndarray
    task : 'regression' or 'classification'
    n_splits : int
    random_state : int

    Returns
    -------
    dict with mean and std for each metric
    """
    if task == "regression":
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scoring_r2   = cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=-1)
        scoring_rmse = cross_val_score(model, X, y, cv=cv,
                                        scoring="neg_root_mean_squared_error", n_jobs=-1)
        scoring_mae  = cross_val_score(model, X, y, cv=cv,
                                        scoring="neg_mean_absolute_error", n_jobs=-1)
        return {
            "R2_mean":   round(scoring_r2.mean(), 4),
            "R2_std":    round(scoring_r2.std(), 4),
            "RMSE_mean": round(-scoring_rmse.mean(), 4),
            "RMSE_std":  round(scoring_rmse.std(), 4),
            "MAE_mean":  round(-scoring_mae.mean(), 4),
            "MAE_std":   round(scoring_mae.std(), 4),
        }
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scoring_auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        scoring_f1  = cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=-1)
        scoring_acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        return {
            "ROC_AUC_mean": round(scoring_auc.mean(), 4),
            "ROC_AUC_std":  round(scoring_auc.std(), 4),
            "F1_mean":      round(scoring_f1.mean(), 4),
            "F1_std":       round(scoring_f1.std(), 4),
            "Accuracy_mean": round(scoring_acc.mean(), 4),
            "Accuracy_std":  round(scoring_acc.std(), 4),
        }


def train_and_evaluate(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task: TaskType = "regression",
    model_name: str = "model",
) -> dict:
    """
    Train model on training set and evaluate on test set.

    Returns dict with metrics and timing.
    """
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - t0

    result = {"model": model_name, "train_time_s": round(train_time, 2),
               "pred_time_s": round(pred_time, 4)}

    if task == "regression":
        result.update(evaluate_regression(y_test, y_pred))
    else:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = None
        result.update(evaluate_classification(y_test, y_pred, y_prob))

    return result, model, y_pred
