"""
Evaluation script for the LSTM Autoencoder.

Features:
- Loads the best model weights and the fitted scaler.
- Applies identical preprocessing (std_floor, clipping) as training.
- Selects the anomaly threshold strictly on the VALIDATION SET (no leakage).
  - Default rule decided in advance: 99.0th percentile of validation errors.
  - Configurable: percentile or std_dev.
- Evaluates on the unseen TEST SET using window-level and event-based metrics.
- Computes sensitivity sweep table (reported for transparency, not used for selection).
- Computes oracle best-F1 threshold (clearly labeled "oracle / diagnostic only").
- Computes temporal calibration slice (first 30% calibration, last 70% held-out test).
- Computes per-sensor validation p99 errors for optional error normalization.
- Compares against fair baselines and trivial flag-everything reference.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from src.config import load_config
from src.data_loader import load_smd_machine
from src.dataset import TimeSeriesDataset
from src.metrics import (
    compute_event_metrics,
    compute_improvements,
    compute_trivial_reference,
    compute_window_metrics,
)
from src.model import LSTMAutoencoder
from src.preprocessing import preprocess_pipeline

logger = logging.getLogger(__name__)


def get_reconstruction_and_sensor_errors(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes per-window reconstruction error (mean over time and features)
    and per-sensor MSE (mean over time for each sensor, shape (n_samples, n_features)).
    """
    model.eval()
    window_errors: list[float] = []
    sensor_errors_list: list[np.ndarray] = []

    criterion.reduction = "none"

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            reconstruction = model(batch)

            # Loss per element: (batch_size, window_size, n_features)
            loss_per_element = criterion(reconstruction, batch)

            # Per-window loss (mean over time and features)
            loss_per_window = torch.mean(loss_per_element, dim=[1, 2])
            window_errors.extend(loss_per_window.cpu().numpy().tolist())

            # Per-sensor loss (mean over time dimension) -> shape (batch_size, n_features)
            sensor_loss_batch = torch.mean(loss_per_element, dim=1)
            sensor_errors_list.append(sensor_loss_batch.cpu().numpy())

    sensor_errors = np.vstack(sensor_errors_list) if sensor_errors_list else np.empty((0, 0))
    return np.array(window_errors), sensor_errors


def find_oracle_best_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    num_steps: int = 300,
) -> dict[str, Any]:
    """
    Diagnostic oracle: finds the threshold on test errors that maximizes F1.
    MUST NEVER be used for headline selection.
    """
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    thresholds = np.linspace(min_score, max_score, num_steps)

    best_f1 = 0.0
    best_thresh = min_score
    best_p = 0.0
    best_r = 0.0
    best_flagged = 0.0

    for thresh in thresholds:
        preds = (scores > thresh).astype(int)
        f1 = float(f1_score(labels, preds, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(thresh)
            best_flagged = float(np.mean(preds))

    oracle_preds = (scores > best_thresh).astype(int)
    oracle_metrics = compute_window_metrics(labels, oracle_preds, anomaly_scores=scores)

    return {
        "status": "oracle / diagnostic only (uses test labels to tune threshold; NEVER use for deployment)",
        "threshold": round(best_thresh, 6),
        "precision": oracle_metrics["precision"],
        "recall": oracle_metrics["recall"],
        "f1": oracle_metrics["f1"],
        "flagged_rate": oracle_metrics["flagged_rate"],
    }


def compute_sensitivity_table(
    val_errors: np.ndarray,
    test_errors: np.ndarray,
    test_labels: np.ndarray,
) -> list[dict[str, Any]]:
    """
    Sweeps percentile and std_dev thresholds on validation errors,
    and evaluates each on test data.
    Clearly labeled: reported for transparency, not used for selection.
    """
    rows: list[dict[str, Any]] = []
    val_mean = float(np.mean(val_errors))
    val_std = float(np.std(val_errors))

    # Percentile sweep
    for p in [95.0, 99.0, 99.5, 99.9]:
        th = float(np.percentile(val_errors, p))
        preds = (test_errors > th).astype(int)
        w_m = compute_window_metrics(test_labels, preds)
        rows.append({
            "method": "percentile",
            "parameter": f"p{p}",
            "threshold": round(th, 6),
            "flagged_rate": w_m["flagged_rate"],
            "precision": w_m["precision"],
            "recall": w_m["recall"],
            "f1": w_m["f1"],
            "selection_note": "Default selection rule is p99.0 (decided in advance)" if p == 99.0 else "Sensitivity sweep only",
        })

    # Std dev sweep
    for k in [2.0, 3.0, 4.0, 5.0]:
        th = val_mean + (k * val_std)
        preds = (test_errors > th).astype(int)
        w_m = compute_window_metrics(test_labels, preds)
        rows.append({
            "method": "std_dev",
            "parameter": f"mean+{k}*std",
            "threshold": round(th, 6),
            "flagged_rate": w_m["flagged_rate"],
            "precision": w_m["precision"],
            "recall": w_m["recall"],
            "f1": w_m["f1"],
            "selection_note": "Sensitivity sweep only",
        })

    return rows


def compute_temporal_calibration(
    test_errors: np.ndarray,
    test_labels: np.ndarray,
    percentile: float = 99.0,
) -> dict[str, Any]:
    """
    Secondary evaluation: split test set temporally (first 30% calibration, last 70% held-out test).
    Threshold is chosen strictly on the calibration slice.
    """
    n_total = len(test_errors)
    split_idx = int(n_total * 0.3)

    cal_errors = test_errors[:split_idx]
    heldout_errors = test_errors[split_idx:]
    heldout_labels = test_labels[split_idx:]

    cal_threshold = float(np.percentile(cal_errors, percentile))
    heldout_preds = (heldout_errors > cal_threshold).astype(int)

    w_metrics = compute_window_metrics(heldout_labels, heldout_preds, anomaly_scores=heldout_errors)
    e_metrics = compute_event_metrics(heldout_labels, heldout_preds)

    return {
        "description": "Temporal calibration slice (first 30% calibration, last 70% held-out test)",
        "calibration_samples": split_idx,
        "heldout_samples": len(heldout_errors),
        "calibration_threshold": round(cal_threshold, 6),
        "threshold_strategy": f"Calibration 30% slice {percentile}th percentile",
        "precision": w_metrics["precision"],
        "recall": w_metrics["recall"],
        "f1": w_metrics["f1"],
        "flagged_rate": w_metrics["flagged_rate"],
        "auc_roc": w_metrics["auc_roc"],
        "auc_pr": w_metrics["auc_pr"],
        "event_metrics": e_metrics,
    }


def evaluate_model() -> dict[str, Any]:
    """Main evaluation orchestrator."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    # 1. Setup & Config
    config = load_config()
    machine_id = config.dataset.machine_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    artifacts_dir = Path(__file__).parent.parent / "artifacts"
    model_path = artifacts_dir / f"lstm_autoencoder_{machine_id}.pt"

    if not model_path.exists():
        logger.error(f"Model not found at {model_path}. Run train.py first.")
        raise FileNotFoundError(f"Model not found at {model_path}")

    # 2. Data Loading & Preprocessing
    logger.info(f"Loading data for machine: {machine_id}")
    raw_data = load_smd_machine(machine_id)

    prep_result = preprocess_pipeline(
        train_raw=raw_data["train"],
        test_raw=raw_data["test"],
        test_labels=raw_data["test_labels"],
        window_size=config.preprocessing.window_size,
        step_size=config.preprocessing.step_size,
        val_split=config.preprocessing.val_split,
        scaler_method=config.preprocessing.scaler,
        std_floor=config.preprocessing.std_floor,
        clip_value=config.preprocessing.clip_value,
        machine_id=machine_id,
        artifacts_dir=artifacts_dir,
        seed=config.training.seed,
    )

    val_dataset = TimeSeriesDataset(prep_result["val_windows"])
    test_dataset = TimeSeriesDataset(prep_result["test_windows"])
    test_labels = prep_result["test_window_labels"]

    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False)

    # 3. Model Initialization & Loading
    model = LSTMAutoencoder(
        window_size=config.preprocessing.window_size,
        n_features=raw_data["num_features"],
        hidden_dim=config.model.hidden_size,
        n_layers=config.model.num_layers,
        dropout=config.model.dropout,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    logger.info("Loaded pre-trained model weights from %s", model_path)

    criterion = nn.MSELoss()

    # 4. Validation Phase (Threshold Selection)
    logger.info("Calculating reconstruction errors for Validation set...")
    val_window_errors, val_sensor_errors = get_reconstruction_and_sensor_errors(
        model, val_loader, criterion, device
    )

    # Compute per-sensor validation p99 for optional Phase 6 error normalization
    per_sensor_val_p99 = np.percentile(val_sensor_errors, 99.0, axis=0).tolist()
    prep_json_path = artifacts_dir / f"preprocessing_{machine_id}.json"
    if prep_json_path.exists():
        try:
            with open(prep_json_path, "r", encoding="utf-8") as f:
                prep_meta = json.load(f)
            prep_meta["per_sensor_val_p99"] = [round(float(v), 6) for v in per_sensor_val_p99]
            prep_meta["error_normalization"] = getattr(config.model, "error_normalization", "none")
            prep_meta["ignored_sensors"] = getattr(config.model, "ignored_sensors", [])
            with open(prep_json_path, "w", encoding="utf-8") as f:
                json.dump(prep_meta, f, indent=2)
            logger.info("Updated %s with per_sensor_val_p99", prep_json_path)
        except Exception as e:
            logger.warning("Could not update preprocessing JSON with per-sensor p99: %s", e)

    # Threshold selection on VALIDATION set only
    val_mean = float(np.mean(val_window_errors))
    val_std = float(np.std(val_window_errors))

    threshold_method = getattr(config.threshold, "method", "percentile")
    if threshold_method == "percentile":
        pct = float(config.threshold.percentile)
        threshold = float(np.percentile(val_window_errors, pct))
        threshold_strategy = f"Val {pct}th Percentile"
    elif threshold_method == "std_dev":
        z_multiplier = float(config.threshold.std_multiplier)
        threshold = val_mean + (z_multiplier * val_std)
        threshold_strategy = f"Val Mean + {z_multiplier}*Std"
    else:
        # Default fallback
        threshold = float(np.percentile(val_window_errors, 99.0))
        threshold_strategy = "Val 99.0th Percentile"

    logger.info("Validation Errors - Mean: %.6f, Std: %.6f", val_mean, val_std)
    logger.info("Selected Anomaly Threshold (%s): %.6f", threshold_strategy, threshold)

    # 5. Test Phase (Evaluation)
    logger.info("Calculating reconstruction errors for Test set...")
    test_window_errors, _ = get_reconstruction_and_sensor_errors(
        model, test_loader, criterion, device
    )

    predictions = (test_window_errors > threshold).astype(int)

    # Window-level metrics (primary)
    window_metrics = compute_window_metrics(
        test_labels, predictions, anomaly_scores=test_window_errors
    )

    # Event-based metrics (secondary)
    event_metrics = compute_event_metrics(test_labels, predictions)

    # Sensitivity table (for transparency)
    sensitivity_table = compute_sensitivity_table(
        val_window_errors, test_window_errors, test_labels
    )

    # Oracle best-F1 (diagnostic only)
    oracle_diagnostic = find_oracle_best_threshold(test_window_errors, test_labels)

    # Temporal calibration slice (secondary)
    temporal_calibration = compute_temporal_calibration(
        test_window_errors, test_labels, percentile=99.0
    )

    # Load baseline results if present and compute improvements
    baseline_file = artifacts_dir / "baseline_results.json"
    baseline_data = None
    best_baseline = None
    trivial_ref = compute_trivial_reference(test_labels)

    if baseline_file.exists():
        try:
            with open(baseline_file, "r", encoding="utf-8") as f:
                baseline_data = json.load(f)
            baselines_list = baseline_data.get("baselines", [])
            if baselines_list:
                best_baseline = max(baselines_list, key=lambda b: b.get("f1", 0.0))
        except Exception as e:
            logger.warning("Could not read baseline_results.json: %s", e)

    improvements = compute_improvements(
        model_metrics=window_metrics,
        baseline_metrics=best_baseline,
        trivial_metrics=trivial_ref,
    )

    logger.info("=== Test Set Evaluation Results ===")
    logger.info(f"Precision:    {window_metrics['precision']:.4f}")
    logger.info(f"Recall:       {window_metrics['recall']:.4f}")
    logger.info(f"F1 Score:     {window_metrics['f1']:.4f}")
    logger.info(f"AUROC:        {window_metrics['auc_roc']}")
    logger.info(f"AUPRC:        {window_metrics['auc_pr']}")
    logger.info(f"Flagged Rate: {window_metrics['flagged_rate']:.4f}")
    logger.info(f"Event Recall: {event_metrics['event_recall']:.4f} ({event_metrics['detected_events']}/{event_metrics['total_events']})")
    logger.info(f"False Alarms: {event_metrics['false_alarm_events']}")
    logger.info(f"Confusion Matrix: {window_metrics['confusion_matrix']}")

    # 6. Save Report
    report: dict[str, Any] = {
        "machine_id": machine_id,
        "method": "LSTM_Autoencoder",
        "threshold": float(round(threshold, 6)),
        "threshold_strategy": threshold_strategy,
        "precision": window_metrics["precision"],
        "recall": window_metrics["recall"],
        "f1": window_metrics["f1"],
        "flagged_rate": window_metrics["flagged_rate"],
        "anomaly_rate": window_metrics["anomaly_rate"],
        "auc_roc": window_metrics["auc_roc"],
        "auc_pr": window_metrics["auc_pr"],
        "confusion_matrix": window_metrics["confusion_matrix"],
        "unit": window_metrics["unit"],
        "event_metrics": event_metrics,
        "trivial_reference": trivial_ref,
        "best_baseline": best_baseline,
        "improvements": improvements,
        "sensitivity_table": sensitivity_table,
        "oracle_diagnostic": oracle_diagnostic,
        "temporal_calibration": temporal_calibration,
    }

    out_path = artifacts_dir / "evaluation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved evaluation results to {out_path}")

    return report


if __name__ == "__main__":
    evaluate_model()
