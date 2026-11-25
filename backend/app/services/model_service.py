"""
Model Inference Service.

Loads the trained LSTM Autoencoder and scaler on startup,
provides thread-safe inference for the API endpoints.
"""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from app.config import get_settings

logger = logging.getLogger(__name__)


class ModelService:
    """
    Singleton service that holds the loaded model, scaler, and threshold.
    Thread-safe for read-only inference.
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.threshold: float = 0.0
        self.threshold_strategy: str = ""
        self.model_version: str = "1.0.0"
        self.training_date: str = ""
        self.machine_id: str = ""
        self.n_features: int = 38
        self.evaluation_metrics: dict = {}
        self.is_loaded: bool = False
        self._device = torch.device("cpu")
        self.clip_value: float | None = 10.0
        self.preprocessing_info: dict = {}
        self.run_manifest: dict | None = None
        self.per_sensor_val_p99: list[float] | None = None
        self.error_normalization: str = "none"
        self.ignored_sensors: list[int] = []

    def load(self, artifacts_dir: Path) -> bool:
        """
        Load model weights, scaler and evaluation metadata from the artifacts dir.
        Returns True if successful.
        """
        try:
            # Find the model file - supports any machine_id
            pt_files = list(artifacts_dir.glob("lstm_autoencoder_*.pt"))
            if not pt_files:
                logger.warning("No trained model weights found in %s", artifacts_dir)
                return False

            model_path = pt_files[0]
            self.machine_id = model_path.stem.replace("lstm_autoencoder_", "")

            # Load scaler
            scaler_path = artifacts_dir / f"scaler_{self.machine_id}.pkl"
            if not scaler_path.exists():
                logger.warning("Scaler not found at %s", scaler_path)
                return False

            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            if hasattr(self.scaler, "n_features_in_"):
                self.n_features = int(self.scaler.n_features_in_)

            # Load evaluation results for threshold + metrics
            eval_path = artifacts_dir / "evaluation_results.json"
            if eval_path.exists():
                with open(eval_path) as f:
                    eval_data = json.load(f)
                self.threshold = eval_data.get("threshold", 0.5)
                self.threshold_strategy = eval_data.get("threshold_strategy", "")
                self.evaluation_metrics = {
                    "f1": eval_data.get("f1"),
                    "precision": eval_data.get("precision"),
                    "recall": eval_data.get("recall"),
                }

            # Load preprocessing metadata if present
            prep_path = artifacts_dir / f"preprocessing_{self.machine_id}.json"
            if prep_path.exists():
                with open(prep_path) as f:
                    self.preprocessing_info = json.load(f)
                self.clip_value = self.preprocessing_info.get("clip_value")
                self.per_sensor_val_p99 = self.preprocessing_info.get("per_sensor_val_p99")
                self.error_normalization = self.preprocessing_info.get("error_normalization", "none")
                self.ignored_sensors = self.preprocessing_info.get("ignored_sensors", [])

            # Load run manifest if present
            manifest_path = artifacts_dir / "run_manifest.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    self.run_manifest = json.load(f)

            # Load PyTorch model
            from app.services.lstm_loader import build_lstm_autoencoder
            self.model = build_lstm_autoencoder()
            state = torch.load(model_path, map_location=self._device, weights_only=True)
            self.model.load_state_dict(state)
            self.model.eval()

            self.training_date = datetime.fromtimestamp(
                model_path.stat().st_mtime, tz=timezone.utc
            ).strftime("%Y-%m-%d")

            self.is_loaded = True
            logger.info(
                "Model loaded: machine=%s, threshold=%.4f, F1=%.4f",
                self.machine_id, self.threshold,
                self.evaluation_metrics.get("f1", 0)
            )
            return True

        except Exception as e:
            logger.error("Failed to load model: %s", e, exc_info=True)
            return False

    def predict(self, window: np.ndarray) -> dict:
        """
        Run inference on a single window.
        Args:
            window: numpy array of shape (window_size, n_features)
        Returns:
            dict with reconstruction_error, is_anomaly, per_sensor_errors
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded")

        # Scale the input
        scaled = self.scaler.transform(window)
        if self.clip_value is not None:
            scaled = np.clip(scaled, -self.clip_value, self.clip_value)

        # Convert to tensor (1, window_size, n_features)
        x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            reconstruction = self.model(x)

        # MSE per timestep per feature -> mean over both
        error_tensor = (x - reconstruction) ** 2

        # Optional Phase 6 error normalization
        if self.error_normalization == "per_sensor_val_p99" and self.per_sensor_val_p99:
            p99_tensor = torch.tensor(self.per_sensor_val_p99, dtype=torch.float32).view(1, 1, -1)
            error_tensor = error_tensor / torch.clamp(p99_tensor, min=1e-4)

        if self.ignored_sensors:
            for idx in self.ignored_sensors:
                error_tensor[:, :, idx] = 0.0

        # Per-sensor error: mean over timesteps
        per_sensor = error_tensor[0].mean(dim=0).cpu().numpy()
        total_error = float(per_sensor.mean())

        return {
            "reconstruction_error": total_error,
            "is_anomaly": total_error > self.threshold,
            "per_sensor_errors": per_sensor.tolist(),
        }

    def get_stream_window(self, test_data: np.ndarray, idx: int, window_size: int = 30) -> dict:
        """
        Get inference result for a specific window index in the test set.
        """
        if idx + window_size > len(test_data):
            idx = 0
        window = test_data[idx: idx + window_size]
        result = self.predict(window)
        result["sensor_values"] = test_data[idx + window_size - 1].tolist()
        result["window_index"] = idx
        return result


# Global singleton
model_service = ModelService()
