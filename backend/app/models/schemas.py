"""
Pydantic schemas for API request/response models.
These define the API contract for the anomaly detection system.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════
# Health Check
# ═══════════════════════════════════════════════
class HealthResponse(BaseModel):
    """Response for GET /health — FR-008."""

    status: str = Field(description="Service status: 'healthy' or 'unhealthy'")
    model_loaded: bool = Field(description="Whether the ML model is loaded and ready")
    model_version: str | None = Field(default=None, description="Loaded model version")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ═══════════════════════════════════════════════
# Anomaly Detection — FR-003
# ═══════════════════════════════════════════════
class SensorWindow(BaseModel):
    """Input: a window of multi-sensor time-series data."""

    data: list[list[float]] = Field(
        description="2D array: [timesteps x sensors]. Each inner list is one timestep."
    )
    machine_id: str | None = Field(default=None, description="Optional machine identifier")


class SensorErrorBreakdown(BaseModel):
    """Per-sensor reconstruction error contribution."""

    sensor_index: int
    sensor_name: str | None = None
    reconstruction_error: float


class AnomalyDetectionResponse(BaseModel):
    """Response for POST /api/v1/detect — FR-003."""

    is_anomaly: bool = Field(description="Whether the window is flagged as anomalous")
    reconstruction_error: float = Field(description="Total reconstruction error for the window")
    threshold: float = Field(description="Current anomaly threshold")
    sensor_errors: list[SensorErrorBreakdown] = Field(
        description="Per-sensor error breakdown for root cause analysis (FR-009)"
    )
    model_version: str = Field(description="Version of the model used for inference")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BatchDetectionRequest(BaseModel):
    """Batch inference request containing multiple windows — FR-012."""
    windows: list[SensorWindow] = Field(description="List of sensor windows to analyze")


class WindowDetectionResult(BaseModel):
    window_index: int
    is_anomaly: bool
    reconstruction_error: float


class BatchDetectionResponse(BaseModel):
    """Response for POST /api/v1/detect/batch — FR-012."""
    total_windows: int
    anomaly_count: int
    anomaly_rate_pct: float
    threshold: float
    model_version: str
    results: list[WindowDetectionResult]
    top_anomalous_sensors: list[SensorErrorBreakdown]


# ═══════════════════════════════════════════════
# Model Metadata — FR-006
# ═══════════════════════════════════════════════
class ModelMetadataResponse(BaseModel):
    """Response for GET /api/v1/model — FR-006."""

    model_version: str
    model_type: str = "LSTM Autoencoder"
    training_date: str | None = None
    dataset_name: str | None = None
    dataset_version: str | None = None
    hyperparameters: dict | None = None
    threshold: float | None = None
    threshold_method: str | None = None
    evaluation_metrics: dict | None = None


# ═══════════════════════════════════════════════
# Streaming / Replay — FR-004
# ═══════════════════════════════════════════════
class StreamDataPoint(BaseModel):
    """A single data point from the simulated real-time stream."""

    timestamp: datetime
    sensor_values: list[float]
    is_anomaly: bool | None = None
    reconstruction_error: float | None = None
    sensor_errors: list[SensorErrorBreakdown] | None = None


# ═══════════════════════════════════════════════
# Auth — FR-007
# ═══════════════════════════════════════════════
class LoginRequest(BaseModel):
    """Login credentials."""

    username: str
    password: str


class TokenResponse(BaseModel):
    """JWT token response."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="Token expiry in seconds")


# ═══════════════════════════════════════════════
# Error Response
# ═══════════════════════════════════════════════
class ErrorResponse(BaseModel):
    """Structured error response."""

    error: str
    message: str
    request_id: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
