"""
Run full post-processing evaluation for LSTM and PCA.
Outputs:
- Full test set results (raw vs post-processed with fixed defaults k=5, m=3, low_ratio=0.8)
- Sensitivity grid (k in {1,3,5,10,20}, m in {1,3,5,10})
- Temporal calibration protocol (30% calibration, 70% held-out test)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str((project_root / "training").resolve()))

from src.baseline import PCABaseline
from src.config import load_config
from src.data_loader import load_smd_machine
from src.dataset import TimeSeriesDataset
from src.evaluate import get_reconstruction_and_sensor_errors
from src.model import LSTMAutoencoder
from src.postprocess import (
    apply_alerting,
    compute_sensitivity_grid,
    compute_temporal_calibration_postprocessing,
    evaluate_alert_metrics,
    rederive_threshold,
)
from src.preprocessing import preprocess_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def run():
    config = load_config()
    machine_id = config.dataset.machine_id
    artifacts_dir = project_root / "training" / "artifacts"
    model_path = artifacts_dir / f"lstm_autoencoder_{machine_id}.pt"

    logger.info("Loading SMD data for machine: %s", machine_id)
    raw_data = load_smd_machine(machine_id)
    prep = preprocess_pipeline(
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
        artifacts_dir=None,
        seed=config.training.seed,
    )

    val_dataset = TimeSeriesDataset(prep["val_windows"])
    test_dataset = TimeSeriesDataset(prep["test_windows"])
    test_labels = prep["test_window_labels"]

    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMAutoencoder(
        window_size=config.preprocessing.window_size,
        n_features=raw_data["num_features"],
        hidden_dim=config.model.hidden_size,
        n_layers=config.model.num_layers,
        dropout=config.model.dropout,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    criterion = nn.MSELoss()

    logger.info("Computing LSTM reconstruction errors on val and test...")
    lstm_val, _ = get_reconstruction_and_sensor_errors(model, val_loader, criterion, device)
    lstm_test, _ = get_reconstruction_and_sensor_errors(model, test_loader, criterion, device)

    logger.info("Fitting and computing PCA reconstruction errors on val and test...")
    pca = PCABaseline(n_components=10, random_state=config.training.seed)
    pca.fit(prep["train_windows"])
    pca_val = pca.compute_scores(prep["val_windows"])
    pca_test = pca.compute_scores(prep["test_windows"])

    percentile = config.threshold.percentile

    # 1. Full test set evaluation with fixed defaults (k=5, m=3, low_ratio=0.8)
    # Raw LSTM
    lstm_raw_th = float(np.percentile(lstm_val, percentile))
    lstm_raw_preds = (lstm_test > lstm_raw_th).astype(int)
    lstm_raw_res = evaluate_alert_metrics(
        test_labels, lstm_raw_preds, anomaly_scores=lstm_test, threshold=lstm_raw_th, method_name="LSTM_Autoencoder_Raw"
    )

    # Post-processed LSTM
    lstm_post_th = rederive_threshold(lstm_val, k=5, percentile=percentile, method="mean")
    lstm_post_preds = apply_alerting(lstm_test, lstm_post_th, k=5, m=3, low_ratio=0.8, method="mean")
    lstm_post_res = evaluate_alert_metrics(
        test_labels, lstm_post_preds, anomaly_scores=lstm_test, threshold=lstm_post_th, method_name="LSTM_Autoencoder_Postprocessed"
    )

    # Raw PCA
    pca_raw_th = float(np.percentile(pca_val, percentile))
    pca_raw_preds = (pca_test > pca_raw_th).astype(int)
    pca_raw_res = evaluate_alert_metrics(
        test_labels, pca_raw_preds, anomaly_scores=pca_test, threshold=pca_raw_th, method_name="PCA_Reconstruction_Raw"
    )

    # Post-processed PCA
    pca_post_th = rederive_threshold(pca_val, k=5, percentile=percentile, method="mean")
    pca_post_preds = apply_alerting(pca_test, pca_post_th, k=5, m=3, low_ratio=0.8, method="mean")
    pca_post_res = evaluate_alert_metrics(
        test_labels, pca_post_preds, anomaly_scores=pca_test, threshold=pca_post_th, method_name="PCA_Reconstruction_Postprocessed"
    )

    # 2. Sensitivity grids
    logger.info("Computing sensitivity grids...")
    lstm_grid = compute_sensitivity_grid(lstm_val, lstm_test, test_labels, percentile=percentile)
    pca_grid = compute_sensitivity_grid(pca_val, pca_test, test_labels, percentile=percentile)

    # 3. Temporal calibration protocol (30% cal, 70% held-out)
    logger.info("Computing temporal calibration protocol...")
    lstm_cal = compute_temporal_calibration_postprocessing(lstm_val, lstm_test, test_labels, percentile=percentile)
    pca_cal = compute_temporal_calibration_postprocessing(pca_val, pca_test, test_labels, percentile=percentile)

    results = {
        "full_test_fixed_defaults": {
            "lstm_raw": lstm_raw_res,
            "lstm_postprocessed": lstm_post_res,
            "pca_raw": pca_raw_res,
            "pca_postprocessed": pca_post_res,
        },
        "temporal_calibration_70pct_heldout": {
            "lstm": lstm_cal,
            "pca": pca_cal,
        },
        "sensitivity_grids": {
            "lstm": lstm_grid,
            "pca": pca_grid,
        },
    }

    out_file = project_root / "training" / "reports" / "postprocess_data.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Saved postprocess evaluation data to %s", out_file)
    return results


if __name__ == "__main__":
    res = run()
    print("\n" + "=" * 90)
    print("=== FULL TEST SET: RAW VS POST-PROCESSED (FIXED DEFAULTS k=5, m=3, low_ratio=0.8) ===")
    print("=" * 90)
    headers = ["Method", "Thresh", "Prec", "Recall", "F1", "Flagged", "EvRecall", "DetEv", "FalseAlarms", "Delay", "Alerts/1k"]
    print(f"{headers[0]:<35} | {headers[1]:<8} | {headers[2]:<6} | {headers[3]:<6} | {headers[4]:<6} | {headers[5]:<7} | {headers[6]:<8} | {headers[7]:<6} | {headers[8]:<11} | {headers[9]:<5} | {headers[10]:<8}")
    print("-" * 125)
    for k, v in res["full_test_fixed_defaults"].items():
        print(f"{v['method']:<35} | {v['threshold']:<8.6f} | {v['precision']:<6.4f} | {v['recall']:<6.4f} | {v['f1']:<6.4f} | {v['flagged_rate']:<7.4f} | {v['event_recall']:<8.4f} | {v['detected_events']:<6} | {v['false_alarm_events']:<11} | {v['mean_detection_delay']:<5.1f} | {v['alerts_per_1000_windows']:<8.2f}")

    print("\n" + "=" * 90)
    print("=== HELD-OUT 70% TEST SET (CALIBRATION PROTOCOL) ===")
    print("=" * 90)
    for model_name in ["lstm", "pca"]:
        cal_res = res["temporal_calibration_70pct_heldout"][model_name]
        print(f"\nModel: {model_name.upper()}")
        print(f"Calibration selected params: {cal_res['calibration_selected_params']}")
        for key in ["raw_heldout", "fixed_defaults_heldout", "calibration_selected_heldout"]:
            v = cal_res[key]
            print(f"  {v['method']:<45} | F1: {v['f1']:.4f} | Prec: {v['precision']:.4f} | Rec: {v['recall']:.4f} | EvRec: {v['event_recall']:.4f} | FA: {v['false_alarm_events']:<3} | Delay: {v['mean_detection_delay']:<4.1f} | Alerts/1k: {v['alerts_per_1000_windows']}")
