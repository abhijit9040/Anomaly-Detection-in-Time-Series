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
        self.evaluation_metrics: dict = {}
        self.is_loaded: bool = False
        self._device = torch.device("cpu")

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

        # Convert to tensor (1, window_size, n_features)
        x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            reconstruction = self.model(x)

        # MSE per timestep per feature -> mean over both
        error_tensor = (x - reconstruction) ** 2
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
