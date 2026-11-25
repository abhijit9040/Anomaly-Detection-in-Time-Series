"""
Metrics computation module — window-level and event-based anomaly metrics.

Supports:
- Window-level evaluation: Precision, Recall, F1, AUROC, AUPRC, Flagged Rate, Confusion Matrix.
- Event-based evaluation: Event Recall, False Alarm Events, Mean Detection Delay, Point-Adjusted F1.
- Comparison against Trivial Reference ("Flag Everything") and Baselines.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_contiguous_segments(labels: np.ndarray) -> list[tuple[int, int]]:
    """
    Extract contiguous segments where label == 1 as [start, end) index intervals.
    """
    labels_int = (labels > 0).astype(int)
    padded = np.pad(labels_int, (1, 1), mode="constant")
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts, ends))


def compute_window_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    anomaly_scores: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Compute standard window-level classification metrics.

    Args:
        y_true: Ground truth binary labels for windows, shape (N,).
        y_pred: Binary predictions (0 or 1), shape (N,).
        anomaly_scores: Continuous anomaly scores for AUROC/AUPRC, shape (N,).

    Returns:
        dict with precision, recall, f1, flagged_rate, anomaly_rate, confusion matrix,
        and optionally auc_roc, auc_pr.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    n_samples = len(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    p = float(precision_score(y_true, y_pred, zero_division=0))
    r = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    flagged_rate = float(np.mean(y_pred)) if n_samples > 0 else 0.0
    anomaly_rate = float(np.mean(y_true)) if n_samples > 0 else 0.0

    metrics: dict[str, Any] = {
        "unit": "window (size=30, stride=1, any-anomaly labeled)",
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "flagged_rate": round(flagged_rate, 4),
        "anomaly_rate": round(anomaly_rate, 4),
        "confusion_matrix": {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        },
    }

    if anomaly_scores is not None and len(np.unique(y_true)) > 1:
        try:
            auc_roc = float(roc_auc_score(y_true, anomaly_scores))
            auc_pr = float(average_precision_score(y_true, anomaly_scores))
            metrics["auc_roc"] = round(auc_roc, 4)
            metrics["auc_pr"] = round(auc_pr, 4)
        except Exception as e:
            logger.warning("Could not compute AUC metrics: %s", e)
            metrics["auc_roc"] = None
            metrics["auc_pr"] = None
    else:
        metrics["auc_roc"] = None
        metrics["auc_pr"] = None

    return metrics


def compute_event_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """
    Compute event-level metrics based on contiguous anomaly segments.

    An event is defined as a contiguous segment of ground-truth anomaly windows.
    - event_recall: fraction of anomaly segments with >= 1 flagged window.
    - false_alarm_events: contiguous runs of predictions where ground-truth is 0.
    - mean_detection_delay: average timesteps between event start and first flagged window.
    - point_adjusted_f1: standard point-adjustment (if any point in event is flagged, all are).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    truth_segments = compute_contiguous_segments(y_true)
    pred_segments = compute_contiguous_segments(y_pred)

    total_events = len(truth_segments)
    if total_events == 0:
        return {
            "total_events": 0,
            "detected_events": 0,
            "event_recall": 1.0,
            "false_alarm_events": len(pred_segments),
            "mean_detection_delay": 0.0,
            "point_adjusted_f1": 0.0,
        }

    detected_count = 0
    delays = []
    # Point-adjusted predictions array
    point_adjusted_preds = y_pred.copy()

    for start, end in truth_segments:
        segment_preds = y_pred[start:end]
        if np.any(segment_preds > 0):
            detected_count += 1
            first_hit = np.argmax(segment_preds > 0)
            delays.append(int(first_hit))
            point_adjusted_preds[start:end] = 1

    event_recall = float(detected_count / total_events)
    mean_delay = float(np.mean(delays)) if delays else 0.0

    # Count false alarm events: predicted segments with NO ground truth anomaly overlap
    false_alarm_events = 0
    for p_start, p_end in pred_segments:
        if not np.any(y_true[p_start:p_end] > 0):
            false_alarm_events += 1

    pt_adj_p = float(precision_score(y_true, point_adjusted_preds, zero_division=0))
    pt_adj_r = float(recall_score(y_true, point_adjusted_preds, zero_division=0))
    pt_adj_f1 = float(f1_score(y_true, point_adjusted_preds, zero_division=0))

    return {
        "unit": "event (contiguous anomaly segments in test data)",
        "total_events": total_events,
        "detected_events": detected_count,
        "event_recall": round(event_recall, 4),
        "false_alarm_events": false_alarm_events,
        "mean_detection_delay": round(mean_delay, 2),
        "point_adjusted": {
            "description": "point-adjust (optimistic, for comparison with literature)",
            "precision": round(pt_adj_p, 4),
            "recall": round(pt_adj_r, 4),
            "f1": round(pt_adj_f1, 4),
        },
    }


def compute_trivial_reference(y_true: np.ndarray) -> dict[str, Any]:
    """
    Compute reference metrics for the trivial strategy "Flag Everything" (predict all 1s).
    """
    y_true = np.asarray(y_true, dtype=int)
    r = float(np.mean(y_true))
    f1_trivial = (2 * r) / (1 + r) if (1 + r) > 0 else 0.0
    return {
        "method": "Flag_Everything_Trivial",
        "precision": round(r, 4),
        "recall": 1.0,
        "f1": round(f1_trivial, 4),
        "flagged_rate": 1.0,
        "note": "Trivial reference that flags every single window",
    }


def compute_improvements(
    model_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any] | None,
    trivial_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Compute both absolute and relative improvements versus baseline and trivial reference.
    """
    model_f1 = model_metrics.get("f1", 0.0)

    result: dict[str, Any] = {}

    if baseline_metrics is not None:
        base_f1 = baseline_metrics.get("f1", 0.0)
        abs_delta = round(model_f1 - base_f1, 4)
        rel_gain = round((abs_delta / max(base_f1, 1e-4)) * 100, 2)
        result["vs_baseline"] = {
            "baseline_method": baseline_metrics.get("method", "Baseline"),
            "f1_delta_abs": abs_delta,
            "f1_improvement_pct": rel_gain,
        }

    if trivial_metrics is not None:
        triv_f1 = trivial_metrics.get("f1", 0.0)
        abs_delta_triv = round(model_f1 - triv_f1, 4)
        rel_gain_triv = round((abs_delta_triv / max(triv_f1, 1e-4)) * 100, 2)
        result["vs_trivial"] = {
            "f1_delta_abs": abs_delta_triv,
            "f1_improvement_pct": rel_gain_triv,
        }

    return result
