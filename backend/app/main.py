"""
Anomaly Detection in Sensor / IoT Time Series Data — FastAPI Backend

Main application entry point.
- FR-008: Health check endpoint
- CORS middleware for frontend communication
- Structured error handling
- Model loading on startup
- API v1 router with live endpoints
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import router as v1_router
from app.config import get_settings
from app.models.schemas import HealthResponse
from app.services.model_service import model_service
from app.services.stream_service import stream_service

# ── Logging ──────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("anomaly_detection")


# ── Application State ────────────────────────────
class AppState:
    """Mutable application state for model lifecycle."""
    model_loaded: bool = False
    model_version: str | None = None


app_state = AppState()


# ── Lifespan (startup / shutdown) ────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("=" * 60)
    logger.info("  Anomaly Detection Backend — Starting")
    logger.info("  Environment: %s", settings.app_env)
    logger.info("  API Version: %s", settings.api_version)
    logger.info("=" * 60)

    # Resolve artifacts directory (training/artifacts relative to project root)
    backend_dir = Path(__file__).parent.parent
    project_root = backend_dir.parent
    artifacts_dir = project_root / "training" / "artifacts"
    data_dir = project_root / "data" / "raw"

    # Load model
    if model_service.load(artifacts_dir):
        app_state.model_loaded = True
        app_state.model_version = model_service.model_version
        logger.info("✅ Model loaded successfully (machine=%s)", model_service.machine_id)
    else:
        logger.warning("⚠️  Model not loaded — inference endpoints will return 503")

    # Load stream data
    if stream_service.load(data_dir, machine_id=model_service.machine_id or "machine-1-1"):
        logger.info("✅ Stream data loaded for replay")
    else:
        logger.warning("⚠️  Stream data not found — stream endpoint will return empty data")

    yield  # application runs here

    logger.info("Anomaly Detection Backend — Shutting down")


# ── FastAPI App ──────────────────────────────────
app = FastAPI(
    title="AnomalyGuard API",
    description=(
        "Full-stack anomaly detection system for multi-sensor IoT time-series data. "
        "Uses an LSTM autoencoder (ICML 2016) to detect subtle deviations in sensor "
        "behavior patterns that are precursors to equipment failure. "
        "\n\n**⚠ Decision-support only** — anomaly flags are for human review. "
        "This system does not autonomously trigger equipment shutdown or maintenance orders."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS Middleware ──────────────────────────────
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global Exception Handler ────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred. Please try again later.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


# ═══════════════════════════════════════════════
# FR-008: Health Check Endpoint
# ═══════════════════════════════════════════════
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Service and model health check",
)
async def health_check():
    """FR-008: Returns service status and model readiness."""
    return HealthResponse(
        status="healthy",
        model_loaded=app_state.model_loaded,
        model_version=app_state.model_version,
        timestamp=datetime.now(timezone.utc),
    )


# ── Include API Routers ─────────────────────────
app.include_router(v1_router)


# ── Root ────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "AnomalyGuard API",
        "docs": "/docs",
        "health": "/health",
        "version": "1.0.0",
    }
