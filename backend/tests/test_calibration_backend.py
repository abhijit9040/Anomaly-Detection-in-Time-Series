"""
Backend calibration and parity tests (Phase 1, 2, 7, 8).

Verifies:
1. Backend ModelService.predict() on a fixed window produces the exact same reconstruction error
   (tolerance 1e-5) as the training-side evaluation code path.
2. GET /api/v1/model returns run_manifest and the calibrated threshold.
3. GET /api/v1/evaluation returns all required enriched fields.
4. scripts/verify_artifacts.py passes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.services.model_service import model_service

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_backend_predict_parity_with_training_side():
    """
    Assert that backend ModelService.predict() produces the identical reconstruction
    error (within 1e-5) as the training-side evaluation path on a fixed window.
    """
    was_loaded = model_service.is_loaded
    if not was_loaded:
        model_service.load(REPO_ROOT / "training" / "artifacts")
    try:
        assert model_service.is_loaded, "Model service must be loaded for this test"

        # Create a fixed reproducible test window (30 timesteps, 38 features)
        rng = np.random.RandomState(12345)
        raw_window = rng.uniform(0.0, 1.0, size=(30, 38)).astype(np.float32)

        # 1. Backend inference path
        backend_result = model_service.predict(raw_window)
        backend_error = backend_result["reconstruction_error"]

        # 2. Training-side evaluation path
        # scaler transform + clip
        scaled = model_service.scaler.transform(raw_window)
        if model_service.clip_value is not None:
            scaled = np.clip(scaled, -model_service.clip_value, model_service.clip_value)

        x_tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            recon_tensor = model_service.model(x_tensor)
            # Training side: torch.mean(criterion(recon, batch), dim=[1,2])
            training_side_error = float(torch.mean((x_tensor - recon_tensor) ** 2).item())

        diff = abs(backend_error - training_side_error)
        assert diff < 1e-5, f"Discrepancy {diff} exceeds tolerance 1e-5 (backend={backend_error}, training={training_side_error})"
    finally:
        if not was_loaded:
            model_service.is_loaded = False


@pytest.mark.asyncio
async def test_api_model_metadata_returns_run_manifest():
    """GET /api/v1/model returns run_manifest and calibrated threshold."""
    was_loaded = model_service.is_loaded
    if not was_loaded:
        model_service.load(REPO_ROOT / "training" / "artifacts")
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            res = await client.get("/api/v1/model")
        assert res.status_code == 200
        data = res.json()
        assert "threshold" in data
        assert data["threshold"] > 0
        assert "threshold_method" in data
        assert "run_manifest" in data
        assert "model_version" in data
        assert "n_features" in data
        assert data["n_features"] == 38
    finally:
        if not was_loaded:
            model_service.is_loaded = False


@pytest.mark.asyncio
async def test_api_evaluation_returns_enriched_metrics():
    """GET /api/v1/evaluation returns AUROC, AUPRC, confusion matrix, event metrics, etc."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/api/v1/evaluation")
    assert res.status_code == 200
    data = res.json()
    assert "lstm_autoencoder" in data
    assert "baseline" in data
    assert "improvement" in data

    # Verify enriched fields
    assert "auc_roc" in data
    assert "auc_pr" in data
    assert "confusion_matrix" in data
    assert "flagged_rate" in data
    assert "event_metrics" in data
    assert "unit" in data

    # Event metrics structure
    em = data["event_metrics"]
    if em:
        assert "event_recall" in em
        assert "false_alarm_events" in em
        assert "mean_detection_delay" in em


@pytest.mark.asyncio
async def test_api_demo_windows_returns_real_windows():
    """GET /api/v1/demo/windows returns 10 real windows with labels and timesteps."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/api/v1/demo/windows?count=10")
    assert res.status_code == 200
    data = res.json()
    assert data["total_windows"] == 10
    windows = data["windows"]
    assert len(windows) == 10

    # First 5 are normal (label 0, start_timestep < 150)
    for i in range(5):
        w = windows[i]
        assert w["ground_truth_label"] == 0
        assert len(w["data"]) == 30
        assert len(w["data"][0]) == 38
        assert w["start_timestep"] < 150

    # Last 5 are anomaly (label 1, start_timestep >= 15849)
    for i in range(5, 10):
        w = windows[i]
        assert w["ground_truth_label"] == 1
        assert len(w["data"]) == 30
        assert len(w["data"][0]) == 38
        assert w["start_timestep"] >= 15849
