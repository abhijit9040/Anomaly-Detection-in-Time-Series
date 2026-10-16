"""
Tests for the health check endpoint — FR-008.

Uses httpx + pytest-asyncio for async FastAPI testing.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_health_returns_200():
    """Health endpoint should return 200 with expected fields."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is False  # No model in Week 1
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_health_response_schema():
    """Health response should match the HealthResponse schema."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    data = response.json()
    required_fields = {"status", "model_loaded", "model_version", "timestamp"}
    assert required_fields.issubset(set(data.keys()))


@pytest.mark.asyncio
async def test_root_returns_info():
    """Root endpoint should return service info."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert "docs" in data
    assert "health" in data


@pytest.mark.asyncio
async def test_detect_returns_503_when_model_not_loaded():
    """Detect endpoint should return 503 when model is not loaded in test env."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/detect",
            json={"data": [[1.0, 2.0, 3.0]], "machine_id": "test"},
        )

    assert response.status_code == 503


@pytest.mark.asyncio
async def test_model_metadata_returns_503_when_model_not_loaded():
    """Model metadata endpoint should return 503 when model is not loaded in test env."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/model")

    assert response.status_code == 503
