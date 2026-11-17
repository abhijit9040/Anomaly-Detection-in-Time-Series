"""
Tests for API v1 endpoints — Authentication, Evaluation, and Stream.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_auth_login_success():
    """Valid credentials return 200 with bearer token."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "admin"},
        )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["expires_in"] == 28800


@pytest.mark.asyncio
async def test_auth_login_invalid():
    """Invalid credentials return 401."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/auth/login",
            json={"username": "wrong_user", "password": "wrong_password"},
        )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_evaluation_endpoint():
    """Evaluation endpoint returns LSTM and baseline metrics."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/evaluation")
    assert response.status_code == 200
    data = response.json()
    assert "lstm_autoencoder" in data
    assert "baseline" in data
    assert "improvement" in data
    assert data["lstm_autoencoder"]["f1"] > 0


@pytest.mark.asyncio
async def test_stream_controls():
    """Test seeking and jumping to anomaly in stream."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res_seek = await client.post("/api/v1/stream/seek", json={"index": 100})
        assert res_seek.status_code == 200
        assert res_seek.json()["current_index"] == 100

        res_ano = await client.post("/api/v1/stream/anomaly")
        assert res_ano.status_code == 200
        assert res_ano.json()["current_index"] > 1000

        res_inj = await client.post("/api/v1/stream/inject?duration=5")
        assert res_inj.status_code == 200
        assert res_inj.json()["injected"] is True
