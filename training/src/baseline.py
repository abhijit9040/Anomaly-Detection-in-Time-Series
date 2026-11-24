"""
Fair Statistical & Classical ML Baselines for Time-Series Anomaly Detection.

Protocol guarantees (Phase 4):
1. Same unit: Test windows (shape: window_size x 38), labeled 1 if any timestep in the window is anomalous.
2. Same threshold protocol: Threshold selected strictly on VALIDATION windows (normal only), using the same percentile rule (default: 99.0th percentile).
3. Evaluates:
   (a) Moving-Average Z-Score
   (b) Isolation Forest
   (c) PCA Reconstruction Error
   (d) Trivial "Flag Everything" reference
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from src.config import load_config
from src.data_loader import load_smd_machine
from src.metrics import (
    compute_event_metrics,
    compute_trivial_reference,
    compute_window_metrics,
)
from src.preprocessing import preprocess_pipeline

logger = logging.getLogger(__name__)


class MovingAverageZScoreBaseline:
    """Window-level Moving-Average Z-Score anomaly detector."""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.feature_means: np.ndarray | None = None
        self.feature_stds: np.ndarray | None = None

    def fit(self, train_windows: np.ndarray) -> None:
        """Fit global mean and std from training windows."""
        flattened = train_windows.reshape(-1, train_windows.shape[-1])
        self.feature_means = np.mean(flattened, axis=0)
        self.feature_stds = np.std(flattened, axis=0) + 1e-4

    def compute_scores(self, windows: np.ndarray) -> np.ndarray:
        """Compute window anomaly score as the max sensor deviation across time."""
        # windows: (N, W, D)
        devs = np.abs(windows - self.feature_means) / self.feature_stds
        # Mean deviation over time for each sensor, then max over sensors
        sensor_scores = np.mean(devs, axis=1)  # (N, D)
        window_scores = np.max(sensor_scores, axis=1)  # (N,)
        return window_scores


class PCABaseline:
    """PCA Reconstruction Error baseline."""

    def __init__(self, n_components: int = 10, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.pca = PCA(n_components=n_components, random_state=random_state)

    def fit(self, train_windows: np.ndarray) -> None:
        """Fit PCA on flattened training windows."""
        n_samples = len(train_windows)
        flattened = train_windows.reshape(n_samples, -1)
        # Subsample if dataset is large for CPU efficiency
        if n_samples > 10000:
            idx = np.random.RandomState(self.random_state).choice(n_samples, 10000, replace=False)
            flattened = flattened[idx]
        self.pca.fit(flattened)

    def compute_scores(self, windows: np.ndarray) -> np.ndarray:
        """Compute MSE reconstruction error for each window."""
        n_samples = len(windows)
        flattened = windows.reshape(n_samples, -1)
        recon = self.pca.inverse_transform(self.pca.transform(flattened))
        mse = np.mean((flattened - recon) ** 2, axis=1)
        return mse


class IsolationForestBaseline:
    """Isolation Forest baseline trained on normal window summary statistics."""

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.clf = IsolationForest(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        self.random_state = random_state

    def _extract_features(self, windows: np.ndarray) -> np.ndarray:
        # Features: mean and std per sensor across the window (shape: N, 2 * D = 76)
        means = np.mean(windows, axis=1)
        stds = np.std(windows, axis=1)
        return np.hstack([means, stds])

    def fit(self, train_windows: np.ndarray) -> None:
        feats = self._extract_features(train_windows)
        n_samples = len(feats)
        if n_samples > 8000:
            idx = np.random.RandomState(self.random_state).choice(n_samples, 8000, replace=False)
            feats = feats[idx]
        self.clf.fit(feats)

    def compute_scores(self, windows: np.ndarray) -> np.ndarray:
        feats = self._extract_features(windows)
        # Invert score_samples so higher score = more anomalous
        scores = -self.clf.score_samples(feats)
        return scores


def evaluate_baseline_method(
    name: str,
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    percentile: float = 99.0,
) -> dict[str, Any]:
    """
    Select threshold on validation scores, evaluate on test scores.
    """
    threshold = float(np.percentile(val_scores, percentile))
    preds = (test_scores > threshold).astype(int)

    w_metrics = compute_window_metrics(test_labels, preds, anomaly_scores=test_scores)
    e_metrics = compute_event_metrics(test_labels, preds)

    return {
        "method": name,
        "threshold": round(threshold, 6),
        "threshold_strategy": f"Val {percentile}th Percentile",
        "precision": w_metrics["precision"],
        "recall": w_metrics["recall"],
        "f1": w_metrics["f1"],
        "flagged_rate": w_metrics["flagged_rate"],
        "auc_roc": w_metrics["auc_roc"],
        "auc_pr": w_metrics["auc_pr"],
        "confusion_matrix": w_metrics["confusion_matrix"],
        "event_metrics": e_metrics,
    }


def run_all_baselines(save_artifacts: bool = True) -> dict[str, Any]:
    """Execute all baselines and save results."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger.info("Running Fair Baselines Suite (Window-Level, Validation-Threshold Protocol)...")

    config = load_config()
    machine_id = config.dataset.machine_id

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
        seed=config.training.seed,
        machine_id=machine_id,
    )

    train_w = prep["train_windows"]
    val_w = prep["val_windows"]
    test_w = prep["test_windows"]
    test_labels = prep["test_window_labels"]

    percentile = config.threshold.percentile

    # 1. Moving Average Z-score
    logger.info("Evaluating (1/3) Moving-Average Z-Score Baseline...")
    ma = MovingAverageZScoreBaseline(window_size=config.preprocessing.window_size)
    ma.fit(train_w)
    ma_val = ma.compute_scores(val_w)
    ma_test = ma.compute_scores(test_w)
    res_ma = evaluate_baseline_method("MovingAverage_ZScore", ma_val, ma_test, test_labels, percentile)

    # 2. PCA Reconstruction Error
    logger.info("Evaluating (2/3) PCA Reconstruction Error Baseline...")
    pca = PCABaseline(n_components=10, random_state=config.training.seed)
    pca.fit(train_w)
    pca_val = pca.compute_scores(val_w)
    pca_test = pca.compute_scores(test_w)
    res_pca = evaluate_baseline_method("PCA_Reconstruction", pca_val, pca_test, test_labels, percentile)

    # 3. Isolation Forest
    logger.info("Evaluating (3/3) Isolation Forest Baseline...")
    iso = IsolationForestBaseline(n_estimators=100, random_state=config.training.seed)
    iso.fit(train_w)
    iso_val = iso.compute_scores(val_w)
    iso_test = iso.compute_scores(test_w)
    res_iso = evaluate_baseline_method("IsolationForest", iso_val, iso_test, test_labels, percentile)

    # 4. Trivial Reference: Flag Everything
    trivial = compute_trivial_reference(test_labels)

    baselines_list = [res_ma, res_pca, res_iso]

    out_data = {
        "machine_id": machine_id,
        "method": res_ma["method"],
        "threshold": res_ma["threshold"],
        "threshold_strategy": res_ma["threshold_strategy"],
        "precision": res_ma["precision"],
        "recall": res_ma["recall"],
        "f1": res_ma["f1"],
        "flagged_rate": res_ma["flagged_rate"],
        "auc_roc": res_ma["auc_roc"],
        "auc_pr": res_ma["auc_pr"],
        "baselines": baselines_list,
        "trivial_reference": trivial,
    }

    logger.info("=== Baseline Results Summary ===")
    logger.info("%-25s | %-8s | %-8s | %-8s | %-8s | %-8s", "Method", "Prec", "Recall", "F1", "AUROC", "Flagged")
    logger.info("-" * 75)
    for b in baselines_list:
        logger.info(
            "%-25s | %-8.4f | %-8.4f | %-8.4f | %-8s | %-8.4f",
            b["method"],
            b["precision"],
            b["recall"],
            b["f1"],
            str(b["auc_roc"]),
            b["flagged_rate"],
        )
    logger.info(
        "%-25s | %-8.4f | %-8.4f | %-8.4f | %-8s | %-8.4f",
        "Flag_Everything_Trivial",
        trivial["precision"],
        trivial["recall"],
        trivial["f1"],
        "0.5000",
        trivial["flagged_rate"],
    )

    if save_artifacts:
        artifacts_dir = Path(__file__).parent.parent / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        out_file = artifacts_dir / "baseline_results.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2)
        logger.info("Saved baseline results to %s", out_file)

    return out_data


if __name__ == "__main__":
    run_all_baselines()
