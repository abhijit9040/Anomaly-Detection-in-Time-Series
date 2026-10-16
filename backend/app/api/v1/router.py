"""
API v1 Router — Live endpoints for the anomaly detection system.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from app.models.schemas import (
    AnomalyDetectionResponse,
    BatchDetectionRequest,
    BatchDetectionResponse,
    ErrorResponse,
    LoginRequest,
    ModelMetadataResponse,
    SensorErrorBreakdown,
    SensorWindow,
    StreamDataPoint,
    TokenResponse,
    WindowDetectionResult,
)
from app.services.model_service import model_service
from app.services.stream_service import stream_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["v1"])


def _require_model():
    """Raises 503 if model is not loaded."""
    if not model_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "model_unavailable",
                "message": "The LSTM model is not loaded. Check server startup logs.",
            },
        )


# ═══════════════════════════════════════════════
# POST /api/v1/detect — FR-003
# ═══════════════════════════════════════════════
@router.post(
    "/detect",
    response_model=AnomalyDetectionResponse,
    responses={503: {"model": ErrorResponse}},
    summary="Detect anomalies in a sensor data window",
)
async def detect_anomaly(window: SensorWindow):
    """
    FR-003: Run the LSTM autoencoder on a window of sensor data.
    Returns an anomaly flag, reconstruction error, and per-sensor breakdown.
    """
    _require_model()

    import numpy as np
    data = np.array(window.data, dtype=np.float32)

    expected_window = 30
    expected_features = 38

    if data.shape[1] != expected_features:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Expected {expected_features} features per timestep, got {data.shape[1]}",
        )
    if data.shape[0] != expected_window:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Expected window size {expected_window}, got {data.shape[0]}",
        )

    result = model_service.predict(data)

    sensor_errors = [
        SensorErrorBreakdown(
            sensor_index=i,
            sensor_name=f"sensor_{i:02d}",
            reconstruction_error=round(e, 6),
        )
        for i, e in enumerate(result["per_sensor_errors"])
    ]

    return AnomalyDetectionResponse(
        is_anomaly=result["is_anomaly"],
        reconstruction_error=round(result["reconstruction_error"], 6),
        threshold=round(model_service.threshold, 6),
        sensor_errors=sensor_errors,
        model_version=model_service.model_version,
        timestamp=datetime.now(timezone.utc),
    )


# ═══════════════════════════════════════════════
# POST /api/v1/detect/batch — FR-012
# ═══════════════════════════════════════════════
@router.post(
    "/detect/batch",
    response_model=BatchDetectionResponse,
    responses={503: {"model": ErrorResponse}},
    summary="Run batch anomaly detection on multiple time-series windows",
)
async def detect_batch(batch: BatchDetectionRequest):
    """
    FR-012: Batch Inference.
    Processes a collection of sensor windows and returns aggregate metrics,
    individual window flags, and overall top contributing sensors.
    """
    _require_model()
    import numpy as np

    if not batch.windows:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Batch contains no windows",
        )

    results = []
    total_sensor_errors = np.zeros(38, dtype=np.float32)
    anomaly_count = 0

    for idx, window in enumerate(batch.windows):
        data = np.array(window.data, dtype=np.float32)
        if data.shape != (30, 38):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Window {idx} shape is {data.shape}, expected (30, 38)",
            )
        res = model_service.predict(data)
        is_ano = bool(res["is_anomaly"])
        recon_err = round(float(res["reconstruction_error"]), 6)
        if is_ano:
            anomaly_count += 1
        results.append(
            WindowDetectionResult(
                window_index=idx,
                is_anomaly=is_ano,
                reconstruction_error=recon_err,
            )
        )
        total_sensor_errors += np.array(res["per_sensor_errors"], dtype=np.float32)

    avg_sensor_errors = total_sensor_errors / len(batch.windows)
    top_indices = np.argsort(avg_sensor_errors)[::-1][:5]
    top_sensors = [
        SensorErrorBreakdown(
            sensor_index=int(i),
            sensor_name=f"sensor_{int(i):02d}",
            reconstruction_error=round(float(avg_sensor_errors[i]), 6),
        )
        for i in top_indices
    ]

    total_win = len(batch.windows)
    rate = round((anomaly_count / total_win) * 100, 2) if total_win > 0 else 0.0

    return BatchDetectionResponse(
        total_windows=total_win,
        anomaly_count=anomaly_count,
        anomaly_rate_pct=rate,
        threshold=round(model_service.threshold, 6),
        model_version=model_service.model_version,
        results=results,
        top_anomalous_sensors=top_sensors,
    )


# ═══════════════════════════════════════════════
# GET /api/v1/model — FR-006
# ═══════════════════════════════════════════════
@router.get(
    "/model",
    response_model=ModelMetadataResponse,
    responses={503: {"model": ErrorResponse}},
    summary="Get model metadata and evaluation metrics",
)
async def get_model_metadata():
    """FR-006: Returns model version, hyperparameters, and evaluation metrics."""
    _require_model()

    return ModelMetadataResponse(
        model_version=model_service.model_version,
        model_type="LSTM Autoencoder",
        training_date=model_service.training_date,
        dataset_name="Server Machine Dataset (SMD)",
        dataset_version="1.0",
        hyperparameters={
            "window_size": 30,
            "hidden_dim": 64,
            "n_layers": 2,
            "dropout": 0.1,
            "batch_size": 64,
            "learning_rate": 0.001,
            "optimizer": "Adam",
        },
        threshold=round(model_service.threshold, 6),
        threshold_method=model_service.threshold_strategy,
        evaluation_metrics=model_service.evaluation_metrics,
    )


# ═══════════════════════════════════════════════
# GET /api/v1/evaluation — FR-005
# ═══════════════════════════════════════════════
@router.get(
    "/evaluation",
    summary="Get full evaluation report (F1, precision, recall)",
)
async def get_evaluation_report():
    """
    FR-005: Returns the full evaluation report generated on the test set.
    Reads from the saved evaluation_results.json artifact.
    """
    # __file__ = backend/app/api/v1/router.py → 4 parents = project root
    project_root = Path(__file__).parent.parent.parent.parent.parent
    eval_path = project_root / "training" / "artifacts" / "evaluation_results.json"
    baseline_path = project_root / "training" / "artifacts" / "baseline_results.json"

    if not eval_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation results not found. Run training/evaluate.py first.",
        )

    with open(eval_path) as f:
        lstm_results = json.load(f)

    baseline = None
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)

    return {
        "lstm_autoencoder": lstm_results,
        "baseline": baseline,
        "improvement": {
            "f1_delta": round(
                lstm_results.get("f1", 0) - (baseline.get("f1", 0) if baseline else 0), 4
            ),
            "f1_improvement_pct": round(
                (lstm_results.get("f1", 0) - (baseline.get("f1", 0) if baseline else 0))
                / max(baseline.get("f1", 0.001) if baseline else 0.001, 0.001)
                * 100,
                1,
            ),
        },
    }


# ═══════════════════════════════════════════════
# GET /api/v1/stream — FR-004
# ═══════════════════════════════════════════════
@router.get(
    "/stream",
    response_model=StreamDataPoint,
    responses={503: {"model": ErrorResponse}},
    summary="Get next data point from the simulated real-time stream",
)
async def get_stream_data():
    """
    FR-004: Returns the next sensor data point from the SMD test set replay.
    If the model is loaded, also runs inference and returns anomaly detection result.
    """
    point = stream_service.get_next()

    if point is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "stream_unavailable", "message": "Stream data not loaded."},
        )

    # Run anomaly detection if model is available
    reconstruction_error = None
    is_anomaly = None
    sensor_errors = None

    if model_service.is_loaded and stream_service.test_data is not None:
        try:
            import numpy as np
            end_idx = max(30, stream_service.current_index)
            start_idx = end_idx - 30
            window = stream_service.test_data[start_idx:end_idx]
            result = model_service.predict(window)
            reconstruction_error = round(result["reconstruction_error"], 6)
            is_anomaly = result["is_anomaly"]
            sensor_errors = [
                SensorErrorBreakdown(
                    sensor_index=i,
                    sensor_name=f"sensor_{i:02d}",
                    reconstruction_error=round(e, 6),
                )
                for i, e in enumerate(result.get("per_sensor_errors", []))
            ]
        except Exception as e:
            logger.warning("Inference failed for stream point: %s", e)

    return StreamDataPoint(
        timestamp=datetime.now(timezone.utc),
        sensor_values=point["sensor_values"],
        is_anomaly=is_anomaly,
        reconstruction_error=reconstruction_error,
        sensor_errors=sensor_errors,
    )


# ═══════════════════════════════════════════════
# POST /api/v1/auth/login — FR-007
# ═══════════════════════════════════════════════
@router.post(
    "/auth/login",
    response_model=TokenResponse,
    responses={401: {"model": ErrorResponse}},
    summary="Login and receive session token",
)
async def login(credentials: LoginRequest):
    """
    FR-007: Basic Authentication & Session Handling.
    Validates username and password, returns access token with expiry.
    """
    valid_users = {
        "admin": "admin",
        "operator": "operator",
        "engineer": "engineer",
    }
    if credentials.username not in valid_users or valid_users[credentials.username] != credentials.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "invalid_credentials", "message": "Invalid username or password."},
        )

    import base64
    payload = json.dumps({
        "sub": credentials.username,
        "role": "admin" if credentials.username == "admin" else "operator",
        "exp": int(datetime.now(timezone.utc).timestamp()) + 3600 * 8,
    })
    token = f"jwt_{base64.urlsafe_b64encode(payload.encode()).decode()}"

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=28800,
    )
