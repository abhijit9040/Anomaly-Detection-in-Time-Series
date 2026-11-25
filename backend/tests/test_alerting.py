"""
Tests for stateful online AlertingService and parity with offline postprocess functions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

# Add project root and training to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str((project_root / "training").resolve()))

from app.main import app
from app.services.alerting_service import AlertingService
from src.postprocess import apply_alerting, smooth_scores


def test_alerting_service_offline_parity():
    """
    Assert that AlertingService processed point-by-point yields
    exact numerical parity with offline apply_alerting on the whole sequence.
    """
    np.random.seed(42)
    scores = np.random.uniform(0.05, 0.45, size=200)
    # Add a couple of synthetic anomaly bursts
    scores[50:65] = np.random.uniform(0.5, 0.9, size=15)
    scores[120:122] = np.random.uniform(0.5, 0.8, size=2)  # short burst < 3 steps
    scores[150:170] = np.random.uniform(0.6, 1.2, size=20)

    threshold = 0.35
    k = 5
    m = 3
    low_ratio = 0.8

    # 1. Offline batch processing
    offline_flags, offline_smoothed = apply_alerting(
        scores, threshold=threshold, k=k, m=m, low_ratio=low_ratio, method="mean", return_smoothed=True
    )

    # 2. Online streaming processing with AlertingService
    online_service = AlertingService(k=k, m=m, low_ratio=low_ratio)
    online_smoothed = []
    online_flags = []

    for s in scores:
        res = online_service.process(error=s, threshold=threshold)
        online_smoothed.append(res["smoothed_error"])
        online_flags.append(int(res["alert_active"]))

    online_smoothed_arr = np.array(online_smoothed)
    online_flags_arr = np.array(online_flags)

    # Assert exact match
    np.testing.assert_allclose(online_smoothed_arr, offline_smoothed, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(online_flags_arr, offline_flags)


def test_alerting_service_reset():
    """AlertingService reset clears buffer and state."""
    service = AlertingService(k=5, m=3, low_ratio=0.8)
    for _ in range(10):
        service.process(error=1.5, threshold=0.5)

    assert service.hysteresis_state is True
    assert service.consecutive_active >= 3
    assert len(service.buffer) == 5

    service.reset()
    assert service.hysteresis_state is False
    assert service.consecutive_active == 0
    assert len(service.buffer) == 0


@pytest.mark.asyncio
async def test_stream_endpoint_additive_fields():
    """Test /api/v1/stream schema includes smoothed_error and alert_active additive fields."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Load stream test data if not loaded
        from app.services.stream_service import stream_service
        if stream_service.test_data is None:
            data_dir = project_root / "data" / "raw"
            stream_service.load(data_dir, machine_id="machine-1-1")

        response = await client.get("/api/v1/stream")
        assert response.status_code == 200
        data = response.json()
        assert "smoothed_error" in data
        assert "alert_active" in data
        assert "reconstruction_error" in data
        assert "is_anomaly" in data
        assert "sensor_values" in data
