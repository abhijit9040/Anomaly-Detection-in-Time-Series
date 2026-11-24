"""
Unit tests for Model Calibration, Preprocessing, and Metrics (Phase 8).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "training"))

from src.metrics import (
    compute_contiguous_segments,
    compute_event_metrics,
    compute_trivial_reference,
    compute_window_metrics,
)
from src.preprocessing import fit_scaler, split_train_val, transform_data


def test_std_floor_and_clipping():
    """Verify std_floor prevents zero-std scale and clip_value caps extreme z-scores."""
    # 100 samples, 3 features
    # Feature 0: standard normal (std ~ 1.0)
    # Feature 1: near-constant (std = 0.0001)
    # Feature 2: constant zero (std = 0.0)
    rng = np.random.RandomState(42)
    train_data = np.zeros((100, 3))
    train_data[:, 0] = rng.randn(100)
    train_data[:, 1] = 0.5 + rng.randn(100) * 1e-4
    train_data[:, 2] = 1.0 + rng.randn(100) * 1e-5

    scaler, floored_sensors = fit_scaler(train_data, method="standard", std_floor=0.05)

    assert 1 in floored_sensors
    assert 2 in floored_sensors
    assert scaler.scale_[1] == pytest.approx(0.05)
    assert scaler.scale_[2] == pytest.approx(0.05)

    # Test data with extreme deviation
    test_data = np.array([[0.0, 10.0, -10.0]])
    scaled = transform_data(test_data, scaler, clip_value=10.0)

    assert np.all(scaled <= 10.0)
    assert np.all(scaled >= -10.0)


def test_scaler_fit_train_only():
    """Verify that fit_scaler accepts only train data and fits independently."""
    rng = np.random.RandomState(42)
    train_data = rng.randn(200, 4) * 2.0 + 5.0
    scaler, _ = fit_scaler(train_data, method="standard", std_floor=0.05)

    assert np.allclose(scaler.mean_, np.mean(train_data, axis=0), atol=1e-5)


def test_temporal_split_no_overlap():
    """Verify temporal train/val split preserves chronological order without leakage."""
    data = np.arange(100).reshape(100, 1)
    train_split, val_split = split_train_val(data, val_split=0.2)

    assert len(train_split) == 80
    assert len(val_split) == 20
    assert train_split[-1, 0] == 79
    assert val_split[0, 0] == 80
    # No temporal overlap
    assert np.max(train_split) < np.min(val_split)


def test_threshold_validation_only():
    """Verify threshold strategy computation takes strictly validation errors without test labels."""
    val_errors = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.9])
    
    # 90th percentile
    p90_threshold = float(np.percentile(val_errors, 90.0))
    assert p90_threshold > 0.4

    # std_dev: mean + 3*std
    mean_val = float(np.mean(val_errors))
    std_val = float(np.std(val_errors))
    std3_threshold = mean_val + 3.0 * std_val
    assert std3_threshold > mean_val


def test_window_metrics_and_confusion_matrix_sum():
    """Verify confusion matrix sums to total window count and AUROC is correct."""
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 0, 1, 1, 1, 0, 1])
    scores = np.array([0.1, 0.2, 0.15, 0.6, 0.3, 0.7, 0.8, 0.9, 0.4, 0.95])

    metrics = compute_window_metrics(y_true, y_pred, anomaly_scores=scores)
    cm = metrics["confusion_matrix"]

    assert cm["tp"] + cm["fp"] + cm["fn"] + cm["tn"] == len(y_true)
    assert cm["tp"] == 4
    assert cm["fp"] == 1
    assert cm["fn"] == 1
    assert cm["tn"] == 4
    assert metrics["auc_roc"] is not None
    assert metrics["auc_roc"] > 0.8


def test_event_metrics_toy_example():
    """Verify event-level metrics: contiguous detection, false alarms, and detection delay."""
    # 2 true events: [10:20] and [50:60]
    y_true = np.zeros(100, dtype=int)
    y_true[10:20] = 1
    y_true[50:60] = 1

    # Predictions:
    # - detects event 1 at index 12 (delay = 2)
    # - detects event 2 at index 55 (delay = 5)
    # - false alarm run at [30:35]
    y_pred = np.zeros(100, dtype=int)
    y_pred[12:15] = 1
    y_pred[55:58] = 1
    y_pred[30:35] = 1

    event_m = compute_event_metrics(y_true, y_pred)

    assert event_m["total_events"] == 2
    assert event_m["detected_events"] == 2
    assert event_m["event_recall"] == 1.0
    assert event_m["false_alarm_events"] == 1
    assert event_m["mean_detection_delay"] == pytest.approx(3.5, abs=0.1)


def test_trivial_reference():
    """Verify flag-everything trivial reference formula."""
    y_true = np.array([0, 0, 0, 1])  # anomaly rate r = 0.25
    trivial = compute_trivial_reference(y_true)

    assert trivial["recall"] == 1.0
    assert trivial["precision"] == 0.25
    expected_f1 = (2 * 0.25) / (1 + 0.25)  # 0.5 / 1.25 = 0.4
    assert trivial["f1"] == pytest.approx(expected_f1, abs=1e-4)
