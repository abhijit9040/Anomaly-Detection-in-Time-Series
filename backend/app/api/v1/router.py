"""
API v1 Router — Live endpoints for the anomaly detection system.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.security import create_access_token, get_password_hash, verify_password
from app.db.models import User
from app.db.session import get_db
from app.models.schemas import (
    AnomalyDetectionResponse,
    BatchDetectionRequest,
    BatchDetectionResponse,
    DemoWindowItem,
    DemoWindowsResponse,
    ErrorResponse,
    LoginRequest,
    ModelMetadataResponse,
    RegisterRequest,
    SensorErrorBreakdown,
    SensorWindow,
    StreamDataPoint,
    StreamSeekRequest,
    TokenResponse,
    UserResponse,
    WindowDetectionResult,
)
from app.services.alerting_service import AlertingService
from app.services.model_service import model_service
from app.services.stream_service import stream_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["v1"])
alerting_service = AlertingService(k=5, m=3, low_ratio=0.8)


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
# GET /api/v1/demo/windows — Real test windows for batch demo
# ═══════════════════════════════════════════════
@router.get(
    "/demo/windows",
    response_model=DemoWindowsResponse,
    summary="Get real test windows (normal and anomaly) for batch demonstration",
)
async def get_demo_windows(
    count: int = Query(
        default=10,
        ge=1,
        le=100,
        description="Number of real demo windows (between 1 and 100). Invalid counts return 422.",
    )
):
    """
    Returns real windows from the SMD test dataset (split evenly between normal windows
    from timesteps 0-150 and anomaly windows from the first failure segment starting at 15,849).
    Deterministic, no randomness. Does not advance the stream replay cursor.

    Query parameters:
        count (int): Number of demo windows (1 <= count <= 100, default 10).
                     Values outside [1, 100] return 422 Unprocessable Entity.
    """
    project_root = Path(__file__).parent.parent.parent.parent.parent
    if stream_service.test_data is None:
        data_dir = project_root / "data" / "raw"
        stream_service.load(data_dir, machine_id=model_service.machine_id or "machine-1-1")

    if stream_service.test_data is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "stream_unavailable", "message": "Test data not loaded."},
        )

    n_normal = count // 2
    n_anomaly = count - n_normal

    # 5 normal windows from index 0 to 150
    normal_starts = [0, 25, 50, 75, 100]
    if n_normal != 5:
        normal_starts = [int(i * (120 / max(1, n_normal - 1))) for i in range(n_normal)]

    # 5 anomaly windows starting at first_anomaly_index (15849)
    ano_start = stream_service.first_anomaly_index
    anomaly_starts = [ano_start, ano_start + 11, ano_start + 26, ano_start + 41, ano_start + 61]
    if n_anomaly != 5:
        anomaly_starts = [ano_start + int(i * 15) for i in range(n_anomaly)]

    selected_starts = [(s, 0) for s in normal_starts] + [(s, 1) for s in anomaly_starts]
    windows = []

    for idx, (start_idx, default_label) in enumerate(selected_starts):
        end_idx = start_idx + 30
        window_arr = stream_service.test_data[start_idx:end_idx]
        gt_label = default_label
        if stream_service.test_labels is not None and len(stream_service.test_labels) >= end_idx:
            gt_label = int(np.any(stream_service.test_labels[start_idx:end_idx] == 1))

        windows.append(
            DemoWindowItem(
                window_index=idx,
                data=window_arr.tolist(),
                ground_truth_label=gt_label,
                start_timestep=start_idx,
                end_timestep=end_idx,
            )
        )

    return DemoWindowsResponse(
        total_windows=len(windows),
        windows=windows,
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

    n_feat = 38
    if model_service.is_loaded:
        if model_service.scaler is not None and hasattr(model_service.scaler, "n_features_in_"):
            n_feat = int(model_service.scaler.n_features_in_)
        elif hasattr(model_service, "n_features"):
            n_feat = int(model_service.n_features)

    return ModelMetadataResponse(
        model_version=model_service.model_version,
        model_type="LSTM Autoencoder",
        n_features=n_feat,
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
        run_manifest=model_service.run_manifest,
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

    # Legacy fallback calculation if improvements dictionary not present
    improvement = lstm_results.get("improvements", {})
    if not improvement:
        f1_delta = round(
            lstm_results.get("f1", 0) - (baseline.get("f1", 0) if baseline else 0), 4
        )
        improvement = {
            "f1_delta": f1_delta,
            "f1_improvement_pct": round(
                f1_delta / max(baseline.get("f1", 0.001) if baseline else 0.001, 0.001) * 100,
                1,
            ),
        }

    postprocess_data = None
    postprocess_path = project_root / "training" / "reports" / "postprocess_data.json"
    if postprocess_path.exists():
        try:
            with open(postprocess_path, "r", encoding="utf-8") as f:
                postprocess_data = json.load(f)
        except Exception as e:
            logger.warning("Could not read postprocess_data.json: %s", e)

    return {
        "lstm_autoencoder": lstm_results,
        "baseline": baseline,
        "improvement": improvement,
        "postprocessing": postprocess_data,
        "f1": lstm_results.get("f1"),
        "precision": lstm_results.get("precision"),
        "recall": lstm_results.get("recall"),
        "auc_roc": lstm_results.get("auc_roc"),
        "auc_pr": lstm_results.get("auc_pr"),
        "threshold": lstm_results.get("threshold"),
        "threshold_strategy": lstm_results.get("threshold_strategy"),
        "flagged_rate": lstm_results.get("flagged_rate"),
        "anomaly_rate": lstm_results.get("anomaly_rate"),
        "confusion_matrix": lstm_results.get("confusion_matrix"),
        "event_metrics": lstm_results.get("event_metrics"),
        "sensitivity_table": lstm_results.get("sensitivity_table"),
        "oracle_diagnostic": lstm_results.get("oracle_diagnostic"),
        "temporal_calibration": lstm_results.get("temporal_calibration"),
        "unit": lstm_results.get("unit"),
        "trivial_reference": lstm_results.get("trivial_reference"),
        "best_baseline": lstm_results.get("best_baseline"),
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
    smoothed_error = None
    alert_active = None
    is_anomaly = None
    sensor_errors = None

    if model_service.is_loaded and stream_service.test_data is not None:
        try:
            import numpy as np
            end_idx = max(30, stream_service.current_index)
            start_idx = end_idx - 30
            window = stream_service.test_data[start_idx:end_idx].copy()
            if point.get("was_injected"):
                window[-1] = point["sensor_values"]
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

            # Apply stateful causal post-processing alerting pipeline
            alert_res = alerting_service.process(
                error=reconstruction_error,
                threshold=model_service.threshold,
            )
            smoothed_error = alert_res["smoothed_error"]
            alert_active = alert_res["alert_active"]
        except Exception as e:
            logger.warning("Inference failed for stream point: %s", e)

    return StreamDataPoint(
        timestamp=datetime.now(timezone.utc),
        sensor_values=point["sensor_values"],
        is_anomaly=is_anomaly,
        reconstruction_error=reconstruction_error,
        smoothed_error=smoothed_error,
        alert_active=alert_active,
        sensor_errors=sensor_errors,
        timestep=point.get("timestep"),
        ground_truth_label=point.get("ground_truth_label"),
    )


# ═══════════════════════════════════════════════
# Stream Controls — Seek, Jump to Anomaly, Inject
# ═══════════════════════════════════════════════
@router.post(
    "/stream/seek",
    summary="Seek replay stream to a specific timestep",
)
async def seek_stream(req: StreamSeekRequest):
    alerting_service.reset()
    new_idx = stream_service.seek(req.index)
    return {"current_index": new_idx, "message": f"Stream moved to timestep {new_idx}"}


@router.post(
    "/stream/anomaly",
    summary="Jump replay stream to just before the first real anomaly event",
)
async def jump_to_anomaly():
    alerting_service.reset()
    new_idx = stream_service.jump_to_anomaly(lead_in=15)
    return {
        "current_index": new_idx,
        "first_anomaly_index": stream_service.first_anomaly_index,
        "message": f"Stream jumped to timestep {new_idx} (15 steps before true anomaly event at {stream_service.first_anomaly_index})",
    }


@router.post(
    "/stream/inject",
    summary="Inject a synthetic anomaly spike into the live stream",
)
async def inject_stream_anomaly(duration: int = 15):
    steps = stream_service.inject_anomaly(duration=duration)
    return {"injected": True, "duration": steps, "message": f"Injected anomaly spike for next {steps} timesteps"}


@router.post(
    "/stream/reset",
    summary="Reset stream replay back to timestep 0",
)
async def reset_stream():
    new_idx = stream_service.seek(0)
    return {"current_index": new_idx, "message": "Stream reset to timestep 0"}


# ═══════════════════════════════════════════════
# Authentication & User Management — FR-007
# ═══════════════════════════════════════════════

@router.post(
    "/auth/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        409: {"model": ErrorResponse},
    },
    summary="Register a new user account",
)
async def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    """
    Register a new user in the database with salted bcrypt password hash
    and return an immediate signed JWT access token.
    """
    username = payload.username.strip().lower()
    email = payload.email.strip().lower()

    if db.query(User).filter(User.username == username).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"error": "username_taken", "message": f"Username '{username}' is already in use."},
        )

    if db.query(User).filter(User.email == email).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"error": "email_taken", "message": f"Email '{email}' is already in use."},
        )

    user = User(
        username=username,
        email=email,
        hashed_password=get_password_hash(payload.password),
        role=payload.role,
        is_active=True,
        created_at=datetime.now(timezone.utc),
        last_login=datetime.now(timezone.utc),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(subject=user.username, role=user.role, user_id=user.id)
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=28800,
        user=UserResponse.model_validate(user),
    )


@router.post(
    "/auth/login",
    response_model=TokenResponse,
    responses={401: {"model": ErrorResponse}},
    summary="Login and receive session token",
)
async def login(credentials: LoginRequest, db: Session = Depends(get_db)):
    """
    FR-007: Database Authentication & Session Handling.
    Validates username or email with salted bcrypt verification.
    Returns signed HMAC-SHA256 JWT access token.
    """
    login_id = credentials.username.strip().lower()

    user = db.query(User).filter(
        (User.username == login_id) | (User.email == login_id)
    ).first()

    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "invalid_credentials", "message": "Invalid username or password."},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": "account_disabled", "message": "Account has been deactivated."},
        )

    user.last_login = datetime.now(timezone.utc)
    db.commit()

    token = create_access_token(subject=user.username, role=user.role, user_id=user.id)
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=28800,
        user=UserResponse.model_validate(user),
    )


@router.get(
    "/auth/me",
    response_model=UserResponse,
    responses={401: {"model": ErrorResponse}},
    summary="Get current authenticated user profile",
)
async def get_me(current_user: User = Depends(get_current_user)):
    """Returns the profile and role of the currently logged-in user."""
    return UserResponse.model_validate(current_user)
