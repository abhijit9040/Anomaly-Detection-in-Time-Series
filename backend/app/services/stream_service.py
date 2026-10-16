"""
Data replay service for the simulated real-time stream endpoint.
Loads the SMD test data and replays it window by window.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class StreamService:
    """Holds test data in memory and returns sequential windows."""

    def __init__(self):
        self.test_data: np.ndarray | None = None
        self.test_labels: np.ndarray | None = None
        self.current_index: int = 0
        self.machine_id: str = ""
        self.window_size: int = 30

    def load(self, data_dir: Path, machine_id: str = "machine-1-1") -> bool:
        """Load the test split for replay."""
        try:
            test_path = data_dir / "ServerMachineDataset" / "test" / f"{machine_id}.txt"
            label_path = data_dir / "ServerMachineDataset" / "test_label" / f"{machine_id}.txt"

            if not test_path.exists():
                logger.warning("Test data not found at %s", test_path)
                return False

            self.test_data = np.loadtxt(test_path, delimiter=",", dtype=np.float32)
            self.machine_id = machine_id
            self.current_index = 0

            if label_path.exists():
                self.test_labels = np.loadtxt(label_path, delimiter=",", dtype=np.int32)

            logger.info("Stream service loaded: %s, %d timesteps", machine_id, len(self.test_data))
            return True
        except Exception as e:
            logger.error("Failed to load stream data: %s", e)
            return False

    def get_next(self) -> dict | None:
        """Return the next raw timestep and its ground truth label."""
        if self.test_data is None:
            return None

        idx = self.current_index
        if idx >= len(self.test_data):
            self.current_index = 0
            idx = 0

        row = self.test_data[idx].tolist()
        label = int(self.test_labels[idx]) if self.test_labels is not None else None

        self.current_index += 1

        return {
            "timestep": idx,
            "sensor_values": row,
            "ground_truth_label": label,
            "total_timesteps": len(self.test_data),
        }


stream_service = StreamService()
