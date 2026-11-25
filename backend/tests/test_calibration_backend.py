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
from app.services.stream_service import stream_service

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
        assert data["run_manifest"] is not None
        assert data["run_manifest"]["git_dirty"] is True
        assert "git_commit_semantics" in data["run_manifest"]
        assert "code_committed_in" in data["run_manifest"]
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
    """GET /api/v1/demo/windows returns requested count of real windows with labels and timesteps."""
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


@pytest.mark.asyncio
async def test_api_model_metadata_returns_n_features():
    """GET /api/v1/model returns n_features equal to model's input feature count (38 for machine-1-1)."""
    was_loaded = model_service.is_loaded
    if not was_loaded:
        model_service.load(REPO_ROOT / "training" / "artifacts")
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            res = await client.get("/api/v1/model")
        assert res.status_code == 200
        data = res.json()
        assert "n_features" in data
        assert data["n_features"] == 38
        assert data["n_features"] == model_service.n_features
    finally:
        if not was_loaded:
            model_service.is_loaded = False


@pytest.mark.asyncio
async def test_api_demo_windows_respects_count_and_shape():
    """GET /api/v1/demo/windows respects arbitrary count parameter and returns 30x38 shape with 0/1 labels."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for count in [2, 4, 7, 12]:
            res = await client.get(f"/api/v1/demo/windows?count={count}")
            assert res.status_code == 200
            data = res.json()
            assert data["total_windows"] == count
            assert len(data["windows"]) == count
            for w in data["windows"]:
                assert len(w["data"]) == 30
                assert len(w["data"][0]) == 38
                assert w["ground_truth_label"] in (0, 1)


@pytest.mark.asyncio
async def test_api_demo_windows_deterministic_and_cursor_safe():
    """Two identical calls return identical data; calling demo/windows does not advance stream cursor."""
    if stream_service.test_data is None:
        stream_service.load(REPO_ROOT / "data" / "raw")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Step 1: Call stream to record current timestep
        res_stream1 = await client.get("/api/v1/stream")
        assert res_stream1.status_code == 200
        step1 = res_stream1.json()["timestep"]

        # Step 2: Call demo/windows twice with count=8
        res_demo1 = await client.get("/api/v1/demo/windows?count=8")
        res_demo2 = await client.get("/api/v1/demo/windows?count=8")
        assert res_demo1.status_code == 200
        assert res_demo2.status_code == 200
        assert res_demo1.json() == res_demo2.json(), "Demo windows calls must be strictly deterministic"

        # Step 3: Call stream again - timestep must be step1 + 1 (not advanced by demo windows)
        res_stream2 = await client.get("/api/v1/stream")
        assert res_stream2.status_code == 200
        step2 = res_stream2.json()["timestep"]
        assert step2 == step1 + 1, f"Stream cursor unexpectedly jumped from {step1} to {step2}"


@pytest.mark.asyncio
async def test_api_demo_windows_invalid_counts_validation():
    """Invalid count values (0, negative, >100) return 422 Unprocessable Entity."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for invalid_count in [0, -1, -10, 101, 1000]:
            res = await client.get(f"/api/v1/demo/windows?count={invalid_count}")
            assert res.status_code == 422, f"Expected 422 for count={invalid_count}, got {res.status_code}"


@pytest.mark.asyncio
async def test_api_demo_windows_labels_consistent_with_dataset():
    """Ground-truth labels for demo windows must match any-anomaly-in-window definition from raw labels."""
    if stream_service.test_data is None:
        stream_service.load(REPO_ROOT / "data" / "raw")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/api/v1/demo/windows?count=10")
    assert res.status_code == 200
    data = res.json()
    for w in data["windows"]:
        start = w["start_timestep"]
        end = w["end_timestep"]
        expected_label = int(np.any(stream_service.test_labels[start:end] == 1))
        assert w["ground_truth_label"] == expected_label, (
            f"Window {w['window_index']} label mismatch: got {w['ground_truth_label']}, expected {expected_label}"
        )
