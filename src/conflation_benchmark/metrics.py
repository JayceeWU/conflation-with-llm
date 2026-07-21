from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "balanced_accuracy", "mcc")


def classification_metrics(labels: Iterable[int], predictions: Iterable[int]) -> dict[str, float | int]:
    y_true = np.asarray(list(labels), dtype=int)
    y_pred = np.asarray(list(predictions), dtype=int)
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("labels and predictions must be non-empty and have equal length")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n": int(len(y_true)),
    }


def bootstrap_intervals(
    labels: Iterable[int], predictions: Iterable[int], iterations: int = 1000, seed: int = 42
) -> dict[str, dict[str, float]]:
    y_true = np.asarray(list(labels), dtype=int)
    y_pred = np.asarray(list(predictions), dtype=int)
    if len(y_true) == 0:
        raise ValueError("cannot bootstrap an empty sample")
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    strata = [np.flatnonzero(y_true == label) for label in np.unique(y_true)]
    for _ in range(iterations):
        # Preserve class support in every resample. This avoids undefined
        # one-class replicates and matches the stratified benchmark split.
        indices = np.concatenate([rng.choice(stratum, len(stratum), replace=True) for stratum in strata])
        values = classification_metrics(y_true[indices], y_pred[indices])
        for name in METRIC_NAMES:
            samples[name].append(float(values[name]))
    return {
        name: {"low": float(np.percentile(values, 2.5)), "high": float(np.percentile(values, 97.5))}
        for name, values in samples.items()
    }
