"""
Preprocessing pipeline — leakage-safe normalization and windowing.

CRITICAL: Per PRD §18 (Distinction Criteria) and §10:
- Scaler is fitted on TRAINING DATA ONLY.
- The scaler is then applied (transform only) to validation and test data.
- The anomaly threshold is selected on the VALIDATION set.
- The test set remains untouched until final evaluation.

This module provides:
1. StandardScaler / MinMaxScaler fitting (train only)
2. Sliding window creation for LSTM input
3. Train/validation splitting
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

logger = logging.getLogger(__name__)


def fit_scaler(
    train_data: np.ndarray,
    method: str = "standard",
) -> StandardScaler | MinMaxScaler:
    """
    Fit a scaler on TRAINING DATA ONLY.

    This is the critical leakage-prevention step. The scaler
    must never see validation or test data during fitting.

    Args:
        train_data: Training data, shape (n_samples, n_features).
        method: "standard" (zero mean, unit variance) or "minmax" (0-1 range).

    Returns:
        Fitted scaler object.
    """
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler method: {method}. Use 'standard' or 'minmax'.")

    scaler.fit(train_data)
    logger.info(
        "Scaler fitted on training data: method=%s, shape=%s",
        method,
        train_data.shape,
    )
    return scaler


def transform_data(
    data: np.ndarray,
    scaler: StandardScaler | MinMaxScaler,
) -> np.ndarray:
    """
    Transform data using a PRE-FITTED scaler.

    The scaler was fitted on training data only. This function
    applies the same transformation to any split (train, val, test).

    Args:
        data: Data to transform, shape (n_samples, n_features).
        scaler: Pre-fitted scaler from fit_scaler().

    Returns:
        Transformed data with same shape.
    """
    return scaler.transform(data)


def create_windows(
    data: np.ndarray,
    window_size: int = 30,
    step_size: int = 1,
    labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Create sliding windows for LSTM input.

    The LSTM autoencoder expects input of shape:
        (batch_size, window_size, num_features)

    This function slides a window across the time series to
    generate training/inference samples.

    Args:
        data: Time-series data, shape (n_timesteps, n_features).
        window_size: Number of timesteps per window.
        step_size: Stride between consecutive windows.
        labels: Optional labels, shape (n_timesteps,). If provided,
                a window is labeled anomalous if ANY timestep in the
                window is anomalous (conservative approach).

    Returns:
        Tuple of:
            - windows: np.ndarray, shape (n_windows, window_size, n_features)
            - window_labels: np.ndarray or None, shape (n_windows,)
    """
    n_timesteps, n_features = data.shape
    n_windows = (n_timesteps - window_size) // step_size + 1

    if n_windows <= 0:
        raise ValueError(
            f"Cannot create windows: data has {n_timesteps} timesteps "
            f"but window_size={window_size}."
        )

    windows = np.zeros((n_windows, window_size, n_features), dtype=np.float32)
    window_labels = None

    if labels is not None:
        window_labels = np.zeros(n_windows, dtype=np.int32)

    for i in range(n_windows):
        start = i * step_size
        end = start + window_size
        windows[i] = data[start:end]

        if labels is not None:
            # Window is anomalous if ANY timestep in the window is anomalous
            window_labels[i] = int(labels[start:end].max())

    logger.info(
        "Created %d windows: window_size=%d, step_size=%d, features=%d",
        n_windows,
        window_size,
        step_size,
        n_features,
    )

    if window_labels is not None:
        anomaly_ratio = window_labels.sum() / len(window_labels)
        logger.info("Window anomaly ratio: %.3f (%d / %d)", anomaly_ratio, window_labels.sum(), len(window_labels))

    return windows, window_labels


def split_train_val(
    train_data: np.ndarray,
    val_split: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split training data into train and validation sets.

    For time-series data, we use a TEMPORAL split (not random)
    to respect temporal ordering and avoid look-ahead bias.

    Args:
        train_data: Training data, shape (n_samples, n_features).
        val_split: Fraction of data to use for validation.
        seed: Not used for temporal split, but included for API consistency.

    Returns:
        Tuple of (train_split, val_split), both np.ndarray.
    """
    n = len(train_data)
    split_idx = int(n * (1 - val_split))

    train_split = train_data[:split_idx]
    val_split_data = train_data[split_idx:]

    logger.info(
        "Temporal train/val split: train=%d, val=%d (%.1f%%)",
        len(train_split),
        len(val_split_data),
        val_split * 100,
    )

    return train_split, val_split_data


def preprocess_pipeline(
    train_raw: np.ndarray,
    test_raw: np.ndarray,
    test_labels: np.ndarray,
    window_size: int = 30,
    step_size: int = 1,
    val_split: float = 0.2,
    scaler_method: str = "standard",
    seed: int = 42,
) -> dict:
    """
    Full preprocessing pipeline: normalize → split → window.

    This is the main entry point for data preparation.
    Ensures leakage-free processing per PRD requirements.

    Pipeline:
        1. Split raw train into train/val (temporal split)
        2. Fit scaler on TRAIN SPLIT ONLY
        3. Transform train, val, and test using fitted scaler
        4. Create sliding windows for all splits

    Args:
        train_raw: Raw training data, shape (n_train, n_features).
        test_raw: Raw test data, shape (n_test, n_features).
        test_labels: Test labels, shape (n_test,).
        window_size: Sliding window size.
        step_size: Sliding window stride.
        val_split: Fraction for validation.
        scaler_method: "standard" or "minmax".
        seed: Random seed.

    Returns:
        Dictionary with:
            - "train_windows": np.ndarray
            - "val_windows": np.ndarray
            - "test_windows": np.ndarray
            - "test_window_labels": np.ndarray
            - "scaler": fitted scaler object
            - "config": dict of preprocessing parameters
    """
    logger.info("=" * 50)
    logger.info("Starting preprocessing pipeline")
    logger.info("=" * 50)

    # Step 1: Temporal train/val split
    train_split, val_split_data = split_train_val(train_raw, val_split, seed)

    # Step 2: Fit scaler on TRAINING DATA ONLY (leakage-safe)
    scaler = fit_scaler(train_split, method=scaler_method)

    # Step 3: Transform all splits
    train_scaled = transform_data(train_split, scaler)
    val_scaled = transform_data(val_split_data, scaler)
    test_scaled = transform_data(test_raw, scaler)

    # Step 4: Create sliding windows
    train_windows, _ = create_windows(train_scaled, window_size, step_size)
    val_windows, _ = create_windows(val_scaled, window_size, step_size)
    test_windows, test_window_labels = create_windows(
        test_scaled, window_size, step_size, labels=test_labels
    )

    logger.info("Preprocessing complete:")
    logger.info("  Train windows: %s", train_windows.shape)
    logger.info("  Val windows: %s", val_windows.shape)
    logger.info("  Test windows: %s", test_windows.shape)

    return {
        "train_windows": train_windows,
        "val_windows": val_windows,
        "test_windows": test_windows,
        "test_window_labels": test_window_labels,
        "scaler": scaler,
        "config": {
            "window_size": window_size,
            "step_size": step_size,
            "val_split": val_split,
            "scaler_method": scaler_method,
            "seed": seed,
            "n_train": len(train_split),
            "n_val": len(val_split_data),
            "n_test": len(test_raw),
        },
    }
