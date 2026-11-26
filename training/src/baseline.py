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

import joblib
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

    def save(self, path: Path | str) -> None:
        """Save fitted PCA model with joblib."""
        joblib.dump(self.pca, path)

    @classmethod
    def load(cls, path: Path | str, n_components: int = 10, random_state: int = 42) -> PCABaseline:
        """Load fitted PCA model from disk."""
        instance = cls(n_components=n_components, random_state=random_state)
        instance.pca = joblib.load(path)
        return instance


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

        # 1. Verify refit PCA against baseline_results.json within 1e-6 before saving
        baseline_file = artifacts_dir / "baseline_results.json"
        if baseline_file.exists():
            try:
                with open(baseline_file, "r", encoding="utf-8") as f:
                    existing_base = json.load(f)
                for b in existing_base.get("baselines", []):
                    if b.get("method") == "PCA_Reconstruction":
                        for key in ["f1", "threshold", "precision", "recall"]:
                            diff = abs(float(b.get(key, 0.0)) - float(res_pca.get(key, 0.0)))
                            if diff > 1e-6:
                                raise ValueError(
                                    f"Refit PCA {key} ({res_pca.get(key)}) differs from baseline_results.json "
                                    f"({b.get(key)}) by {diff:.6e} > 1e-6!"
                                )
                        logger.info("Refit PCA verified against baseline_results.json within 1e-6.")
                        break
            except Exception as e:
                logger.error("PCA verification failed: %s", e)
                raise

        # 2. Save fitted PCA artifact
        pca_path = artifacts_dir / f"pca_{machine_id}.pkl"
        pca.save(pca_path)
        logger.info("Saved fitted PCA model to %s", pca_path)

        # 3. Read LSTM validation p99 for normalization and compute offline ensemble metadata
        val_p99_lstm = None
        eval_file = artifacts_dir / "evaluation_results.json"
        if eval_file.exists():
            try:
                with open(eval_file, "r", encoding="utf-8") as f:
                    eval_data = json.load(f)
                val_p99_lstm = float(eval_data.get("threshold", 0.0))
            except Exception as e:
                logger.warning("Could not load threshold from evaluation_results.json: %s", e)

        val_p99_pca = float(res_pca["threshold"])

        # Compute equal-weight ensemble scores if val_p99_lstm is available
        ensemble_meta: dict[str, Any] = {
            "machine_id": machine_id,
            "val_p99_lstm": val_p99_lstm,
            "val_p99_pca": val_p99_pca,
            "weight_lstm": 0.5,
            "weight_pca": 0.5,
            "formula": "score_ens = 0.5 * (lstm / val_p99_lstm) + 0.5 * (pca / val_p99_pca)",
            "pca_components": 10,
            "note": "Offline LSTM+PCA ensemble. Equal weights fixed in advance.",
        }

        if val_p99_lstm is not None:
            # We can also load evaluation errors to compute ensemble threshold & metrics
            try:
                # Load pre-saved ensemble evaluation data if available or compute
                ens_report_file = Path(__file__).parent.parent / "reports" / "ensemble_data.json"
                if ens_report_file.exists():
                    with open(ens_report_file, "r", encoding="utf-8") as f:
                        ens_data = json.load(f)
                    if "w_0.5" in ens_data:
                        ensemble_meta["threshold"] = ens_data["w_0.5"]["raw_full"]["threshold"]
                        ensemble_meta["metrics"] = ens_data["w_0.5"]["raw_full"]
                        ensemble_meta["heldout_metrics"] = ens_data["w_0.5"]["raw_heldout"]
                        ensemble_meta["postprocessed_metrics"] = ens_data["w_0.5"]["post_full"]
                        ensemble_meta["postprocessed_heldout"] = ens_data["w_0.5"]["post_heldout"]
                        ensemble_meta["weight_sensitivity"] = {
                            "w_0.25": ens_data["w_0.25"]["raw_full"],
                            "w_0.5": ens_data["w_0.5"]["raw_full"],
                            "w_0.75": ens_data["w_0.75"]["raw_full"],
                        }
            except Exception as e:
                logger.warning("Could not attach full metrics to ensemble_meta: %s", e)

        ensemble_file = artifacts_dir / f"ensemble_{machine_id}.json"
        with open(ensemble_file, "w", encoding="utf-8") as f:
            json.dump(ensemble_meta, f, indent=2)
        logger.info("Saved ensemble metadata to %s", ensemble_file)

        # 4. Save baseline_results.json with additive ensemble section
        if "metrics" in ensemble_meta:
            out_data["ensemble"] = ensemble_meta

        out_file = artifacts_dir / "baseline_results.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2)
        logger.info("Saved baseline results to %s", out_file)

    return out_data


if __name__ == "__main__":
    run_all_baselines()
