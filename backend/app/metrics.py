"""Consistent evaluation metrics for classification and regression."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_absolute_error, mean_squared_error)

def calculate_metrics(y_true, y_pred, task_type: str, *, average: str = "weighted") -> dict:
    """Calculate task-specific metrics with safe zero-division behavior."""
    if task_type == "classification":
        return {"accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0))}
    if task_type == "regression":
        mse = float(mean_squared_error(y_true, y_pred))
        return {"r2": float(r2_score(y_true, y_pred)), "mae": float(mean_absolute_error(y_true, y_pred)),
                "mse": mse, "rmse": float(np.sqrt(mse))}
    raise ValueError("task_type must be classification or regression")
