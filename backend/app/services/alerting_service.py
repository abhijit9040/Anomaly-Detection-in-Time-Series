"""
Stateful Causal Alerting Service for streaming anomaly detection.

Implements online rolling buffer causal smoothing, dual-threshold hysteresis,
and persistence (minimum duration) filtering.
Parity-guaranteed against offline training.src.postprocess.apply_alerting.
"""

from __future__ import annotations

from collections import deque
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class AlertingService:
    """
    Online causal alerting pipeline:
    1. Causal trailing window mean smoothing of size k (default 5).
    2. Hysteresis thresholding with high=threshold, low=threshold * low_ratio (default 0.8).
    3. Minimum duration filter: flags alert_active only after m consecutive windows (default 3).
    """

    def __init__(
        self,
        k: int = 5,
        m: int = 3,
        low_ratio: float = 0.8,
    ):
        self.k = k
        self.m = m
        self.low_ratio = low_ratio
        self.buffer: deque[float] = deque(maxlen=k)
        self.hysteresis_state: bool = False
        self.consecutive_active: int = 0

    def reset(self) -> None:
        """Reset internal buffer and alarm state (e.g. on stream seek/restart)."""
        self.buffer.clear()
        self.hysteresis_state = False
        self.consecutive_active = 0

    def process(
        self,
        error: float,
        threshold: float,
    ) -> dict[str, Any]:
        """
        Process a single timestep's raw reconstruction error.

        Args:
            error: Raw MSE reconstruction error for the current window.
            threshold: Current upper anomaly threshold.

        Returns:
            dict with:
                - smoothed_error: trailing mean error over window k.
                - alert_active: boolean flag indicating whether causal alarm is active.
                - hysteresis_state: raw hysteresis state before persistence filtering.
        """
        self.buffer.append(float(error))
        smoothed_error = float(np.mean(self.buffer))

        high = float(threshold)
        low = float(threshold * self.low_ratio)

        # 1. Stateful Hysteresis
        if not self.hysteresis_state:
            if smoothed_error > high:
                self.hysteresis_state = True
        else:
            if smoothed_error < low:
                self.hysteresis_state = False

        # 2. Minimum Duration Persistence Filter
        if self.hysteresis_state:
            self.consecutive_active += 1
            alert_active = self.consecutive_active >= self.m
        else:
            self.consecutive_active = 0
            alert_active = False

        return {
            "smoothed_error": round(smoothed_error, 6),
            "alert_active": alert_active,
            "hysteresis_state": self.hysteresis_state,
        }
