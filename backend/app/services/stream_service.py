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
        self.first_anomaly_index: int = 15849
        self.injected_anomaly_steps: int = 0

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

            if label_path.exists():
                self.test_labels = np.loadtxt(label_path, delimiter=",", dtype=np.int32)
                ano_idx = np.where(self.test_labels == 1)[0]
                if len(ano_idx) > 0:
                    self.first_anomaly_index = int(ano_idx[0])
                else:
                    self.first_anomaly_index = 15849

            # Start near the anomaly with 15 normal timesteps lead-in
            # so live demo immediately demonstrates normal-to-anomaly transition
            self.current_index = max(0, self.first_anomaly_index - 15)

            logger.info(
                "Stream service loaded: %s, %d timesteps, first_anomaly=%d, starting_at=%d",
                machine_id,
                len(self.test_data),
                self.first_anomaly_index,
                self.current_index,
            )
            return True
        except Exception as e:
            logger.error("Failed to load stream data: %s", e)
            return False

    def seek(self, index: int) -> int:
        """Seek replay to a specific timestep."""
        if self.test_data is not None:
            self.current_index = max(0, min(index, len(self.test_data) - 1))
        else:
            self.current_index = max(0, index)
        return self.current_index

    def jump_to_anomaly(self, lead_in: int = 15) -> int:
        """Jump to just before the first real anomaly event."""
        target = max(30, self.first_anomaly_index - lead_in)
        return self.seek(target)

    def inject_anomaly(self, duration: int = 15) -> int:
        """Inject synthetic anomaly spikes for the next N timesteps."""
        self.injected_anomaly_steps = max(self.injected_anomaly_steps, duration)
        return self.injected_anomaly_steps

    def get_next(self) -> dict | None:
        """Return the next raw timestep and its ground truth label."""
        if self.test_data is None:
            return None

        idx = self.current_index
        if idx >= len(self.test_data):
            self.current_index = 0
            idx = 0

        row = self.test_data[idx].copy().tolist()
        label = int(self.test_labels[idx]) if self.test_labels is not None else None
        was_injected = False

        if self.injected_anomaly_steps > 0:
            self.injected_anomaly_steps -= 1
            was_injected = True
            # Add synthetic deviations to top sensors
            row[0] = round(row[0] + 2.5, 4)
            row[1] = round(row[1] + 3.0, 4)
            row[2] = round(row[2] + 1.8, 4)
            label = 1

        self.current_index += 1

        return {
            "timestep": idx,
            "sensor_values": row,
            "ground_truth_label": label,
            "total_timesteps": len(self.test_data),
            "was_injected": was_injected,
        }


stream_service = StreamService()
