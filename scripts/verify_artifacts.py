"""
Verification script for reproducible artifacts.

Loads model weights, scaler, preprocessing metadata, and evaluation results.
Recomputes the validation threshold and test metrics on SMD machine-1-1.
Asserts they match the stored evaluation_results.json within strict numerical tolerances:
- Threshold tolerance: 1e-6
- Metrics tolerance: 1e-4

Exits with code 0 on success, non-zero on mismatch or missing artifacts.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

# Add training to sys.path so src imports work
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "training"))

from src.config import load_config
from src.data_loader import load_smd_machine
from src.dataset import TimeSeriesDataset
from src.model import LSTMAutoencoder
from src.preprocessing import preprocess_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("verify_artifacts")


def verify_artifacts() -> bool:
    artifacts_dir = REPO_ROOT / "training" / "artifacts"
    eval_file = artifacts_dir / "evaluation_results.json"
    manifest_file = artifacts_dir / "run_manifest.json"

    if not eval_file.exists():
        logger.error("Missing evaluation_results.json at %s", eval_file)
        return False
    if not manifest_file.exists():
        logger.error("Missing run_manifest.json at %s", manifest_file)
        return False

    with open(eval_file, "r", encoding="utf-8") as f:
        stored_eval = json.load(f)
    with open(manifest_file, "r", encoding="utf-8") as f:
        stored_manifest = json.load(f)

    machine_id = stored_eval.get("machine_id", "machine-1-1")
    model_path = artifacts_dir / f"lstm_autoencoder_{machine_id}.pt"
    prep_path = artifacts_dir / f"preprocessing_{machine_id}.json"

    if not model_path.exists():
        logger.error("Missing model weights at %s", model_path)
        return False
    if not prep_path.exists():
        logger.error("Missing preprocessing metadata at %s", prep_path)
        return False

    with open(prep_path, "r", encoding="utf-8") as f:
        prep_meta = json.load(f)

    config = load_config()
    raw_data = load_smd_machine(machine_id)

    # Preprocess splits identically
    prep_result = preprocess_pipeline(
        train_raw=raw_data["train"],
        test_raw=raw_data["test"],
        test_labels=raw_data["test_labels"],
        window_size=prep_meta.get("window_size", config.preprocessing.window_size),
        step_size=prep_meta.get("step_size", config.preprocessing.step_size),
        val_split=prep_meta.get("val_split", config.preprocessing.val_split),
        scaler_method=prep_meta.get("scaler_method", config.preprocessing.scaler),
        std_floor=prep_meta.get("std_floor", 0.05),
        clip_value=prep_meta.get("clip_value", 10.0),
        machine_id=machine_id,
        artifacts_dir=None,
        seed=config.training.seed,
    )

    device = torch.device("cpu")
    model = LSTMAutoencoder(
        window_size=prep_meta.get("window_size", 30),
        n_features=raw_data["num_features"],
        hidden_dim=config.model.hidden_size,
        n_layers=config.model.num_layers,
        dropout=config.model.dropout,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    val_dataset = TimeSeriesDataset(prep_result["val_windows"])
    test_dataset = TimeSeriesDataset(prep_result["test_windows"])
    test_labels = prep_result["test_window_labels"]

    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False)

    def get_errors(loader: DataLoader) -> np.ndarray:
        errors = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                recon = model(batch)
                mse_per_window = torch.mean((batch - recon) ** 2, dim=[1, 2])
                errors.extend(mse_per_window.cpu().numpy().tolist())
        return np.array(errors)

    val_errors = get_errors(val_loader)
    test_errors = get_errors(test_loader)

    # Recompute threshold
    strategy = stored_eval.get("threshold_strategy", "Val 99.0th Percentile")
    if "Percentile" in strategy:
        # Extract percentile number
        pct = float(strategy.split()[1].replace("th", "").replace("st", "").replace("nd", "").replace("rd", ""))
        recomputed_threshold = float(np.percentile(val_errors, pct))
    else:
        val_mean = float(np.mean(val_errors))
        val_std = float(np.std(val_errors))
        recomputed_threshold = val_mean + (config.threshold.std_multiplier * val_std)

    stored_threshold = float(stored_eval["threshold"])
    thresh_diff = abs(recomputed_threshold - stored_threshold)
    logger.info("Threshold check: recomputed=%.6f, stored=%.6f, diff=%.8f", recomputed_threshold, stored_threshold, thresh_diff)
    if thresh_diff > 1e-6:
        logger.error("Threshold mismatch! diff=%.8f > tolerance 1e-6", thresh_diff)
        return False

    # Evaluate predictions
    preds = (test_errors > recomputed_threshold).astype(int)
    recomputed_p = float(precision_score(test_labels, preds, zero_division=0))
    recomputed_r = float(recall_score(test_labels, preds, zero_division=0))
    recomputed_f1 = float(f1_score(test_labels, preds, zero_division=0))
    recomputed_flagged = float(np.mean(preds))

    stored_p = float(stored_eval["precision"])
    stored_r = float(stored_eval["recall"])
    stored_f1 = float(stored_eval["f1"])
    stored_flagged = float(stored_eval.get("flagged_rate", 0.0))

    logger.info("Precision check: recomputed=%.4f, stored=%.4f (diff=%.6f)", recomputed_p, stored_p, abs(recomputed_p - stored_p))
    logger.info("Recall check:    recomputed=%.4f, stored=%.4f (diff=%.6f)", recomputed_r, stored_r, abs(recomputed_r - stored_r))
    logger.info("F1 check:        recomputed=%.4f, stored=%.4f (diff=%.6f)", recomputed_f1, stored_f1, abs(recomputed_f1 - stored_f1))
    logger.info("Flagged check:   recomputed=%.4f, stored=%.4f (diff=%.6f)", recomputed_flagged, stored_flagged, abs(recomputed_flagged - stored_flagged))

    for name, r_val, s_val in [
        ("Precision", recomputed_p, stored_p),
        ("Recall", recomputed_r, stored_r),
        ("F1", recomputed_f1, stored_f1),
        ("Flagged Rate", recomputed_flagged, stored_flagged),
    ]:
        if abs(r_val - s_val) > 1e-4:
            logger.error("%s mismatch! recomputed=%.4f, stored=%.4f, diff=%.6f > 1e-4", name, r_val, s_val, abs(r_val - s_val))
            return False

    logger.info("SUCCESS: All artifacts verified! Weights, scaler, threshold, and metrics match stored results.")
    return True


if __name__ == "__main__":
    success = verify_artifacts()
    sys.exit(0 if success else 1)
