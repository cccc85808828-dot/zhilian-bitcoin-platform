from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def select_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    return float(thresholds[int(np.nanargmax(f1))])


def _safe_auc(metric, labels: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return float("nan")
    return float(metric(labels, scores))


def binary_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    top_fraction: float,
) -> Dict[str, float | int | list]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = (scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    top_count = max(1, int(math.ceil(labels.size * float(top_fraction))))
    top_indices = np.argsort(-scores)[:top_count]
    total_positive = max(int((labels == 1).sum()), 1)
    top_true_positive = int((labels[top_indices] == 1).sum())
    return {
        "count": int(labels.size),
        "positives": int((labels == 1).sum()),
        "threshold": float(threshold),
        "roc_auc": _safe_auc(roc_auc_score, labels, scores),
        "pr_auc": _safe_auc(average_precision_score, labels, scores),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(labels, predictions)),
        "brier": float(brier_score_loss(labels, scores)),
        "top_fraction": float(top_fraction),
        "top_count": int(top_count),
        "top_capture": float(top_true_positive / total_positive),
        "top_precision": float(top_true_positive / top_count),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def temporal_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    times: np.ndarray,
    threshold: float,
    top_fraction: float,
) -> Tuple[pd.DataFrame, dict]:
    rows = []
    for time_id in sorted(np.unique(times).tolist()):
        mask = times == time_id
        current = binary_metrics(
            labels[mask], scores[mask], threshold, top_fraction=top_fraction
        )
        current["time"] = int(time_id)
        rows.append(current)
    frame = pd.DataFrame(rows)
    finite_f1 = frame["f1"].to_numpy(dtype=float)
    summary = {
        "worst_time_f1": float(np.nanmin(finite_f1)),
        "mean_time_f1": float(np.nanmean(finite_f1)),
        "std_time_f1": float(np.nanstd(finite_f1)),
    }
    unique_times = sorted(np.unique(times).tolist())
    blocks = np.array_split(unique_times, 3)
    for name, block in zip(("near", "middle", "far"), blocks):
        mask = np.isin(times, block)
        block_metrics = binary_metrics(
            labels[mask], scores[mask], threshold, top_fraction=top_fraction
        )
        summary[f"{name}_f1"] = float(block_metrics["f1"])
        summary[f"{name}_pr_auc"] = float(block_metrics["pr_auc"])
    return frame, summary


def sanitize_metrics(value):
    """Convert NumPy values and non-finite floats before writing JSON."""
    if isinstance(value, dict):
        return {key: sanitize_metrics(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_metrics(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

