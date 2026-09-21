"""Error metrics. MAE is primary throughout."""
from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred, float) - np.asarray(y_true, float))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_pred, float) - np.asarray(y_true, float)) ** 2)))


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a, b = np.asarray(y_true, float), np.asarray(y_pred, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def bland_altman(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Bias and 95% limits of agreement (pred - true)."""
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    bias, sd = float(d.mean()), float(d.std(ddof=1))
    return bias, bias - 1.96 * sd, bias + 1.96 * sd
