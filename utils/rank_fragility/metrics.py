"""Metric helpers for rank-fragility leaderboard evaluation."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss as sklearn_log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


_HIGHER_IS_BETTER = {
    "auroc": True,
    "auprc": True,
    "accuracy": True,
    "balanced_accuracy": True,
    "log_loss": False,
    "brier": False,
    "rmse": False,
    "mae": False,
    "r2": True,
}


def higher_is_better(metric: str) -> bool:
    """Return whether larger values indicate better performance."""
    key = metric.lower()
    if key not in _HIGHER_IS_BETTER:
        raise ValueError(f"unsupported metric: {metric}")
    return _HIGHER_IS_BETTER[key]


def _as_float_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(yt) | np.isnan(yp))
    return yt[mask], yp[mask]


def _clipped_prob(y_pred) -> np.ndarray:
    return np.clip(np.asarray(y_pred, dtype=float), 1e-15, 1.0 - 1e-15)


def compute_metric(y_true, y_pred, task: str, metric: str) -> float:
    """Compute one supported classification or regression metric."""
    task = task.lower()
    metric = metric.lower()
    yt, yp = _as_float_arrays(y_true, y_pred)
    if len(yt) == 0:
        warnings.warn(f"{metric} is undefined on an empty subset", RuntimeWarning, stacklevel=2)
        return float("nan")

    if task == "classification":
        if metric == "auroc":
            if len(np.unique(yt)) < 2:
                warnings.warn("AUROC is undefined when a subset has one class", RuntimeWarning, stacklevel=2)
                return float("nan")
            return float(roc_auc_score(yt, yp))
        if metric == "auprc":
            if len(np.unique(yt)) < 2:
                warnings.warn("AUPRC is undefined when a subset has one class", RuntimeWarning, stacklevel=2)
                return float("nan")
            return float(average_precision_score(yt, yp))
        if metric == "accuracy":
            return float(accuracy_score(yt, yp >= 0.5))
        if metric == "balanced_accuracy":
            return float(balanced_accuracy_score(yt, yp >= 0.5))
        if metric == "log_loss":
            return float(sklearn_log_loss(yt, _clipped_prob(yp), labels=[0, 1]))
        if metric == "brier":
            return float(brier_score_loss(yt, _clipped_prob(yp)))
        raise ValueError(f"unsupported classification metric: {metric}")

    if task == "regression":
        if metric == "rmse":
            return float(np.sqrt(mean_squared_error(yt, yp)))
        if metric == "mae":
            return float(mean_absolute_error(yt, yp))
        if metric == "r2":
            if len(yt) < 2:
                warnings.warn("R2 is undefined for fewer than two samples", RuntimeWarning, stacklevel=2)
                return float("nan")
            return float(r2_score(yt, yp))
        raise ValueError(f"unsupported regression metric: {metric}")

    raise ValueError(f"task must be classification or regression, got {task!r}")


def per_sample_loss(y_true, y_pred, task: str, loss: str) -> np.ndarray:
    """Return per-example loss values for attribution summaries."""
    task = task.lower()
    loss = loss.lower()
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    if task == "classification":
        prob = _clipped_prob(yp)
        if loss == "log_loss":
            return -(yt * np.log(prob) + (1.0 - yt) * np.log(1.0 - prob))
        if loss == "brier":
            return (prob - yt) ** 2
        raise ValueError(f"unsupported classification per-sample loss: {loss}")

    if task == "regression":
        if loss == "absolute_error":
            return np.abs(yt - yp)
        if loss == "squared_error":
            return (yt - yp) ** 2
        raise ValueError(f"unsupported regression per-sample loss: {loss}")

    raise ValueError(f"task must be classification or regression, got {task!r}")
