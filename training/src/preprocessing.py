"""
Preprocessing pipeline — leakage-safe normalization and windowing.

CRITICAL: Per PRD §18 (Distinction Criteria) and §10:
- Scaler is fitted on TRAINING DATA ONLY.
- The scaler is then applied (transform only) to validation and test data.
- The anomaly threshold is selected on the VALIDATION set.
- The test set remains untouched until final evaluation.

Phase 1 Additions:
- std_floor: prevents near-constant sensors from producing astronomical z-scores.
- clip_value: caps extreme outlier z-scores after scaling.
- preprocessing_{machine_id}.json saved to artifacts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

logger = logging.getLogger(__name__)


def fit_scaler(
    train_data: np.ndarray,
    method: str = "standard",
    std_floor: float | None = 0.05,
) -> tuple[StandardScaler | MinMaxScaler, list[int]]:
    """
    Fit a scaler on TRAINING DATA ONLY with optional std_floor.

    Args:
        train_data: Training data, shape (n_samples, n_features).
        method: "standard" or "minmax".
        std_floor: Minimum scale allowed for features.

    Returns:
        (fitted_scaler, floored_sensors_indices)
    """
    floored_sensors: list[int] = []
    if method == "standard":
        scaler = StandardScaler()
        scaler.fit(train_data)
        if std_floor is not None and std_floor > 0:
            floored_mask = scaler.scale_ < std_floor
            floored_sensors = [int(i) for i in np.where(floored_mask)[0]]
            scaler.scale_ = np.maximum(scaler.scale_, std_floor)
            logger.info(
                "Applied std_floor=%.4f: %d / %d sensors floored: %s",
                std_floor,
                len(floored_sensors),
                len(scaler.scale_),
                floored_sensors,
            )
    elif method == "minmax":
        scaler = MinMaxScaler()
        scaler.fit(train_data)
    else:
        raise ValueError(f"Unknown scaler method: {method}. Use 'standard' or 'minmax'.")

    logger.info("Scaler fitted on training data: method=%s, shape=%s", method, train_data.shape)
    return scaler, floored_sensors


def transform_data(
    data: np.ndarray,
    scaler: StandardScaler | MinMaxScaler,
    clip_value: float | None = 10.0,
) -> np.ndarray:
    """
    Transform data using a PRE-FITTED scaler with optional z-score clipping.
    """
    scaled = scaler.transform(data)
    if clip_value is not None:
        scaled = np.clip(scaled, -clip_value, clip_value)
    return scaled


def create_windows(
    data: np.ndarray,
    window_size: int = 30,
    step_size: int = 1,
    labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Create sliding windows for LSTM input.
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
        logger.info(
            "Window anomaly ratio: %.3f (%d / %d)",
            anomaly_ratio,
            window_labels.sum(),
            len(window_labels),
        )

    return windows, window_labels


def split_train_val(
    train_data: np.ndarray,
    val_split: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split training data into train and validation sets (TEMPORAL split).
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
    std_floor: float | None = 0.05,
    clip_value: float | None = 10.0,
    seed: int = 42,
    machine_id: str = "machine-1-1",
    artifacts_dir: str | Path | None = None,
) -> dict:
    """
    Full preprocessing pipeline: normalize (with std_floor & clipping) -> split -> window.
    """
    logger.info("=" * 50)
    logger.info("Starting preprocessing pipeline (std_floor=%s, clip=%s)", std_floor, clip_value)
    logger.info("=" * 50)

    # Step 1: Temporal train/val split
    train_split, val_split_data = split_train_val(train_raw, val_split, seed)

    # Step 2: Fit scaler on TRAINING DATA ONLY (with std_floor)
    scaler, floored_sensors = fit_scaler(train_split, method=scaler_method, std_floor=std_floor)

    # Evaluate clipping statistics before/after on test set
    raw_test_scaled = scaler.transform(test_raw)
    test_clipped_count = 0
    test_max_unclipped_zscore = float(np.abs(raw_test_scaled).max())
    if clip_value is not None:
        test_clipped_count = int((np.abs(raw_test_scaled) > clip_value).sum())
        logger.info(
            "Test clipping stats: max raw |z|=%.2f, elements with |z| > %.1f: %d (%.2f%%)",
            test_max_unclipped_zscore,
            clip_value,
            test_clipped_count,
            (test_clipped_count / raw_test_scaled.size) * 100,
        )

    # Step 3: Transform all splits
    train_scaled = transform_data(train_split, scaler, clip_value=clip_value)
    val_scaled = transform_data(val_split_data, scaler, clip_value=clip_value)
    test_scaled = transform_data(test_raw, scaler, clip_value=clip_value)

    # Step 4: Create sliding windows
    train_windows, _ = create_windows(train_scaled, window_size, step_size)
    val_windows, _ = create_windows(val_scaled, window_size, step_size)
    test_windows, test_window_labels = create_windows(
        test_scaled, window_size, step_size, labels=test_labels
    )

    preprocessing_info = {
        "machine_id": machine_id,
        "std_floor": float(std_floor) if std_floor is not None else None,
        "clip_value": float(clip_value) if clip_value is not None else None,
        "window_size": int(window_size),
        "step_size": int(step_size),
        "val_split": float(val_split),
        "scaler_method": scaler_method,
        "floored_sensors": floored_sensors,
        "n_floored_sensors": len(floored_sensors),
        "test_clipped_values_count": test_clipped_count,
        "test_max_unclipped_zscore": test_max_unclipped_zscore,
    }

    if artifacts_dir is not None:
        artifacts_path = Path(artifacts_dir)
        artifacts_path.mkdir(parents=True, exist_ok=True)
        out_file = artifacts_path / f"preprocessing_{machine_id}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(preprocessing_info, f, indent=2)
        logger.info("Saved preprocessing metadata to %s", out_file)

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
        "preprocessing_info": preprocessing_info,
        "config": {
            "window_size": window_size,
            "step_size": step_size,
            "val_split": val_split,
            "scaler_method": scaler_method,
            "std_floor": std_floor,
            "clip_value": clip_value,
            "seed": seed,
            "n_train": len(train_split),
            "n_val": len(val_split_data),
            "n_test": len(test_raw),
        },
    }
