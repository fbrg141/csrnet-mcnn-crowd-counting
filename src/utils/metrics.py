"""Metrics for crowd counting experiments."""

from math import sqrt
from typing import Iterable


def mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    true_values = list(y_true)
    pred_values = list(y_pred)
    if len(true_values) != len(pred_values):
        raise ValueError("y_true and y_pred must have the same length")
    if not true_values:
        return 0.0
    return sum(abs(t - p) for t, p in zip(true_values, pred_values)) / len(true_values)


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    true_values = list(y_true)
    pred_values = list(y_pred)
    if len(true_values) != len(pred_values):
        raise ValueError("y_true and y_pred must have the same length")
    if not true_values:
        return 0.0
    mse = sum((t - p) ** 2 for t, p in zip(true_values, pred_values)) / len(true_values)
    return sqrt(mse)
