"""
Evaluate offline LSTM+PCA ensemble.
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

    lstm_val, _ = get_reconstruction_and_sensor_errors(model, val_loader, criterion, device)
    lstm_test, _ = get_reconstruction_and_sensor_errors(model, test_loader, criterion, device)

    pca = PCABaseline(n_components=10, random_state=config.training.seed)
    pca.fit(prep["train_windows"])
    pca_val = pca.compute_scores(prep["val_windows"])
    pca_test = pca.compute_scores(prep["test_windows"])

    val_p99_lstm = float(np.percentile(lstm_val, 99.0))
    val_p99_pca = float(np.percentile(pca_val, 99.0))

    split_idx = int(len(test_labels) * 0.3)
    held_labels = test_labels[split_idx:]
    lstm_held = lstm_test[split_idx:]
    pca_held = pca_test[split_idx:]

    print(f"val_p99_lstm: {val_p99_lstm:.6f}")
    print(f"val_p99_pca:  {val_p99_pca:.6f}")

    weights = [0.25, 0.5, 0.75]
    results = {}

    for w in weights:
        w_pca = round(1.0 - w, 2)
        ens_val = w * (lstm_val / val_p99_lstm) + w_pca * (pca_val / val_p99_pca)
        ens_test = w * (lstm_test / val_p99_lstm) + w_pca * (pca_test / val_p99_pca)
        ens_held = w * (lstm_held / val_p99_lstm) + w_pca * (pca_held / val_p99_pca)

        # Raw (threshold = val p99 of ensemble score)
        th_raw = float(np.percentile(ens_val, 99.0))
        m_raw_full = evaluate_alert_metrics(
            test_labels, (ens_test > th_raw).astype(int), ens_test, th_raw, f"Ensemble_w={w}_Raw_Full"
        )
        m_raw_held = evaluate_alert_metrics(
            held_labels, (ens_held > th_raw).astype(int), ens_held, th_raw, f"Ensemble_w={w}_Raw_Heldout"
        )

        # Post-processed (threshold = rederived val p99 of smoothed ensemble score)
        th_post = rederive_threshold(ens_val, k=5, percentile=99.0, method="mean")
        preds_post_full = apply_alerting(ens_test, th_post, k=5, m=3, low_ratio=0.8, method="mean")
        preds_post_held = apply_alerting(ens_held, th_post, k=5, m=3, low_ratio=0.8, method="mean")
        m_post_full = evaluate_alert_metrics(
            test_labels, preds_post_full, ens_test, th_post, f"Ensemble_w={w}_Post_Full"
        )
        m_post_held = evaluate_alert_metrics(
            held_labels, preds_post_held, ens_held, th_post, f"Ensemble_w={w}_Post_Heldout"
        )

        results[f"w_{w}"] = {
            "weight_lstm": w,
            "weight_pca": w_pca,
            "raw_full": m_raw_full,
            "raw_heldout": m_raw_held,
            "post_full": m_post_full,
            "post_heldout": m_post_held,
        }

        print(f"\n=== ENSEMBLE w={w} (w_lstm={w}, w_pca={w_pca}) ===")
        print(f"Raw Full:     F1={m_raw_full['f1']:.4f} | Prec={m_raw_full['precision']:.4f} | Rec={m_raw_full['recall']:.4f} | FA={m_raw_full['false_alarm_events']} | EvRec={m_raw_full['event_recall']:.4f}")
        print(f"Raw Heldout:  F1={m_raw_held['f1']:.4f} | Prec={m_raw_held['precision']:.4f} | Rec={m_raw_held['recall']:.4f} | FA={m_raw_held['false_alarm_events']} | EvRec={m_raw_held['event_recall']:.4f}")
        print(f"Post Full:    F1={m_post_full['f1']:.4f} | Prec={m_post_full['precision']:.4f} | Rec={m_post_full['recall']:.4f} | FA={m_post_full['false_alarm_events']} | EvRec={m_post_full['event_recall']:.4f} | Delay={m_post_full['mean_detection_delay']:.1f}")
        print(f"Post Heldout: F1={m_post_held['f1']:.4f} | Prec={m_post_held['precision']:.4f} | Rec={m_post_held['recall']:.4f} | FA={m_post_held['false_alarm_events']} | EvRec={m_post_held['event_recall']:.4f} | Delay={m_post_held['mean_detection_delay']:.1f}")

    out_file = project_root / "training" / "reports" / "ensemble_data.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved ensemble evaluation data to %s", out_file)
    return results


if __name__ == "__main__":
    run()
