"""
Causal Post-Processing Module for Anomaly Detection Alerts.

Provides:
- smooth_scores: Causal trailing window smoothing (mean / median).
- hysteresis_flags: Dual-threshold hysteresis (turn on > high, turn off < low).
- min_duration_filter: Suppress transient spikes (flag only after m consecutive windows).
- apply_alerting: End-to-end alerting pipeline combining smoothing, hysteresis, and duration filtering.
- rederive_threshold: Re-derive threshold on smoothed validation scores (p99).
- Evaluation utilities for full test set, sensitivity sweeps, and temporal calibration protocol.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.metrics import (
    compute_contiguous_segments,
    compute_event_metrics,
    compute_window_metrics,
)

logger = logging.getLogger(__name__)


def smooth_scores(
    scores: np.ndarray,
    k: int = 5,
    method: str = "mean",
) -> np.ndarray:
    """
    Causal trailing window smoothing.
    At timestep t, smoothed score is computed over scores[max(0, t - k + 1) : t + 1].

    Args:
        scores: 1D array of raw anomaly scores.
        k: Window length (number of trailing steps).
        method: "mean" or "median".

    Returns:
        1D array of smoothed scores with same shape and length as input.
    """
    scores_arr = np.asarray(scores, dtype=float)
    n = len(scores_arr)
    if k <= 1 or n == 0:
        return scores_arr.copy()

    out = np.empty(n, dtype=float)
    if method == "mean":
        cumsum = np.cumsum(scores_arr)
        for t in range(n):
            start = max(0, t - k + 1)
            total = cumsum[t] - (cumsum[start - 1] if start > 0 else 0.0)
            count = t - start + 1
            out[t] = total / count
    elif method == "median":
        for t in range(n):
            start = max(0, t - k + 1)
            out[t] = float(np.median(scores_arr[start : t + 1]))
    else:
        raise ValueError(f"Unsupported smoothing method: {method}. Choose 'mean' or 'median'.")

    return out


def hysteresis_flags(
    scores: np.ndarray,
    high: float,
    low: float,
) -> np.ndarray:
    """
    Stateful hysteresis thresholding.
    Alarm turns ON when score > high, and turns OFF when score < low.
    While low <= score <= high, previous alarm state is maintained.

    Args:
        scores: 1D array of anomaly scores.
        high: Upper threshold to trigger alarm.
        low: Lower threshold to clear alarm (low <= high).

    Returns:
        1D integer array (0 or 1) indicating active alarm state at each timestep.
    """
    scores_arr = np.asarray(scores, dtype=float)
    n = len(scores_arr)
    flags = np.zeros(n, dtype=int)
    state = False

    for t in range(n):
        val = scores_arr[t]
        if not state:
            if val > high:
                state = True
        else:
            if val < low:
                state = False
        flags[t] = 1 if state else 0

    return flags


def min_duration_filter(
    flags: np.ndarray,
    m: int = 3,
) -> np.ndarray:
    """
    Persistence / minimum duration filter.
    Flags a window only after m consecutive active windows.

    Args:
        flags: 1D array of binary flags (0 or 1).
        m: Minimum consecutive windows required before flagging.

    Returns:
        1D integer array (0 or 1).
    """
    flags_arr = np.asarray(flags, dtype=int)
    n = len(flags_arr)
    if m <= 1 or n == 0:
        return flags_arr.copy()

    out = np.zeros(n, dtype=int)
    consecutive = 0

    for t in range(n):
        if flags_arr[t] == 1:
            consecutive += 1
            if consecutive >= m:
                out[t] = 1
        else:
            consecutive = 0
            out[t] = 0

    return out


def apply_alerting(
    scores: np.ndarray,
    threshold: float,
    k: int = 5,
    m: int = 3,
    low_ratio: float = 0.8,
    method: str = "mean",
    return_smoothed: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Full causal post-processing alerting pipeline:
    1. Causal smoothing: smooth_scores(scores, k, method)
    2. Hysteresis thresholding: high=threshold, low=threshold * low_ratio
    3. Minimum duration filter: min_duration_filter(h_flags, m)

    Args:
        scores: 1D array of raw anomaly scores.
        threshold: Decision threshold for high alarm.
        k: Smoothing window length.
        m: Minimum consecutive duration.
        low_ratio: Multiplier for low hysteresis threshold (e.g. 0.8).
        method: Smoothing method ("mean" or "median").
        return_smoothed: If True, returns tuple (flags, smoothed_scores).

    Returns:
        Binary flags array (0 or 1), or tuple (flags, smoothed_scores).
    """
    smoothed = smooth_scores(scores, k=k, method=method)
    high = float(threshold)
    low = float(threshold * low_ratio)
    h_flags = hysteresis_flags(smoothed, high=high, low=low)
    final_flags = min_duration_filter(h_flags, m=m)

    if return_smoothed:
        return final_flags, smoothed
    return final_flags


def rederive_threshold(
    val_scores: np.ndarray,
    k: int = 5,
    percentile: float = 99.0,
    method: str = "mean",
) -> float:
    """
    Re-derive threshold on smoothed validation scores.
    Since smoothing reduces variance and shifts the score distribution,
    the threshold must be re-calibrated on normal validation scores.
    """
    smoothed_val = smooth_scores(val_scores, k=k, method=method)
    return float(np.percentile(smoothed_val, percentile))


def evaluate_alert_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    anomaly_scores: np.ndarray | None = None,
    threshold: float = 0.0,
    method_name: str = "Alerting",
) -> dict[str, Any]:
    """
    Compute comprehensive metrics for an alert prediction vector:
    - Window-level: precision, recall, f1, flagged_rate, auroc, auprc
    - Event-level: event_recall, detected_events, false_alarm_events, mean_detection_delay
    - Operational: alerts_per_1000_windows (rate of contiguous alert bursts per 1000 windows)
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    w_metrics = compute_window_metrics(y_true_arr, y_pred_arr, anomaly_scores=anomaly_scores)
    e_metrics = compute_event_metrics(y_true_arr, y_pred_arr)

    pred_segments = compute_contiguous_segments(y_pred_arr)
    n_windows = len(y_pred_arr)
    alerts_per_1000 = (
        round((len(pred_segments) / n_windows) * 1000, 2) if n_windows > 0 else 0.0
    )

    return {
        "method": method_name,
        "threshold": round(float(threshold), 6),
        "precision": w_metrics["precision"],
        "recall": w_metrics["recall"],
        "f1": w_metrics["f1"],
        "flagged_rate": w_metrics["flagged_rate"],
        "auc_roc": w_metrics.get("auc_roc"),
        "auc_pr": w_metrics.get("auc_pr"),
        "event_recall": e_metrics["event_recall"],
        "detected_events": f"{e_metrics['detected_events']}/{e_metrics['total_events']}",
        "false_alarm_events": e_metrics["false_alarm_events"],
        "mean_detection_delay": e_metrics["mean_detection_delay"],
        "alerts_per_1000_windows": alerts_per_1000,
    }


def compute_sensitivity_grid(
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    k_values: list[int] | None = None,
    m_values: list[int] | None = None,
    low_ratio: float = 0.8,
    percentile: float = 99.0,
) -> list[dict[str, Any]]:
    """
    Compute sensitivity grid over k in {1, 3, 5, 10, 20} and m in {1, 3, 5, 10}.
    Clearly labelled: 'transparency only, not used for selection'.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10, 20]
    if m_values is None:
        m_values = [1, 3, 5, 10]

    rows: list[dict[str, Any]] = []

    for k in k_values:
        th = rederive_threshold(val_scores, k=k, percentile=percentile, method="mean")
        for m in m_values:
            preds = apply_alerting(test_scores, th, k=k, m=m, low_ratio=low_ratio, method="mean")
            m_res = evaluate_alert_metrics(
                test_labels, preds, anomaly_scores=test_scores, threshold=th, method_name=f"k={k},m={m}"
            )
            rows.append({
                "k": k,
                "m": m,
                "threshold": m_res["threshold"],
                "precision": m_res["precision"],
                "recall": m_res["recall"],
                "f1": m_res["f1"],
                "flagged_rate": m_res["flagged_rate"],
                "event_recall": m_res["event_recall"],
                "detected_events": m_res["detected_events"],
                "false_alarm_events": m_res["false_alarm_events"],
                "mean_detection_delay": m_res["mean_detection_delay"],
                "alerts_per_1000_windows": m_res["alerts_per_1000_windows"],
                "note": "Fixed default (decided in advance)" if (k == 5 and m == 3) else "Transparency grid only",
            })

    return rows


def compute_temporal_calibration_postprocessing(
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    split_ratio: float = 0.3,
    low_ratio: float = 0.8,
    percentile: float = 99.0,
) -> dict[str, Any]:
    """
    Temporal-calibration protocol:
    - First 30% calibration slice: parameter selection across (k, m) grid.
    - Last 70% held-out test slice: honest headline evaluation.
    Both slices and results are clearly separated and labelled.
    """
    n_total = len(test_scores)
    split_idx = int(n_total * split_ratio)

    cal_scores = test_scores[:split_idx]
    cal_labels = test_labels[:split_idx]

    held_scores = test_scores[split_idx:]
    held_labels = test_labels[split_idx:]

    k_grid = [1, 3, 5, 10, 20]
    m_grid = [1, 3, 5, 10]

    # Evaluate grid on calibration slice (purely normal data in SMD machine-1-1)
    best_combo = (5, 3)
    min_false_alarms = float("inf")
    cal_grid_rows = []

    for k in k_grid:
        th = rederive_threshold(val_scores, k=k, percentile=percentile, method="mean")
        for m in m_grid:
            cal_preds = apply_alerting(cal_scores, th, k=k, m=m, low_ratio=low_ratio, method="mean")
            fa_events = compute_event_metrics(cal_labels, cal_preds)["false_alarm_events"]
            flagged = float(np.mean(cal_preds))
            cal_grid_rows.append({
                "k": k,
                "m": m,
                "calibration_false_alarms": fa_events,
                "calibration_flagged_rate": round(flagged, 4),
            })
            # Selection criterion: minimize false alarm events on calibration slice;
            # tie-break with lower k (less delay)
            if fa_events < min_false_alarms:
                min_false_alarms = fa_events
                best_combo = (k, m)

    # Evaluate on held-out 70% slice:
    # 1. Raw baseline (threshold from validation p99)
    raw_th = float(np.percentile(val_scores, percentile))
    raw_held_preds = (held_scores > raw_th).astype(int)
    raw_held_metrics = evaluate_alert_metrics(
        held_labels, raw_held_preds, anomaly_scores=held_scores, threshold=raw_th, method_name="Raw Held-out"
    )

    # 2. Fixed defaults post-processed (k=5, m=3)
    def_th = rederive_threshold(val_scores, k=5, percentile=percentile, method="mean")
    def_held_preds = apply_alerting(held_scores, def_th, k=5, m=3, low_ratio=low_ratio, method="mean")
    def_held_metrics = evaluate_alert_metrics(
        held_labels, def_held_preds, anomaly_scores=held_scores, threshold=def_th, method_name="Post-processed (Fixed Defaults k=5, m=3)"
    )

    # 3. Calibration-slice selected post-processed
    sel_k, sel_m = best_combo
    sel_th = rederive_threshold(val_scores, k=sel_k, percentile=percentile, method="mean")
    sel_held_preds = apply_alerting(held_scores, sel_th, k=sel_k, m=sel_m, low_ratio=low_ratio, method="mean")
    sel_held_metrics = evaluate_alert_metrics(
        held_labels, sel_held_preds, anomaly_scores=held_scores, threshold=sel_th, method_name=f"Post-processed (Calibration-selected k={sel_k}, m={sel_m})"
    )

    return {
        "description": "Temporal calibration protocol (30% calibration, 70% held-out test)",
        "calibration_samples": split_idx,
        "heldout_samples": len(held_scores),
        "calibration_selected_params": {"k": sel_k, "m": sel_m},
        "calibration_grid": cal_grid_rows,
        "raw_heldout": raw_held_metrics,
        "fixed_defaults_heldout": def_held_metrics,
        "calibration_selected_heldout": sel_held_metrics,
    }
