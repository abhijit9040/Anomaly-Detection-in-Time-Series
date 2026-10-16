"""
Configuration — loads settings from environment variables.
Uses pydantic-settings for typed, validated configuration.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings sourced from .env / environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Application ──────────────────────────────
    app_env: str = "development"
    debug: bool = True
    api_version: str = "v1"

    # ── CORS ─────────────────────────────────────
    cors_origins: str = "http://localhost:5173,http://localhost:5174,http://localhost:5175,http://localhost:3000"

    # ── Model Paths ──────────────────────────────
    model_path: str = "../training/artifacts/lstm_autoencoder.pt"
    scaler_path: str = "../training/artifacts/scaler.pkl"
    threshold_path: str = "../training/artifacts/threshold.json"

    # ── Auth ─────────────────────────────────────
    secret_key: str = "change-me-in-production-use-a-real-secret"
    access_token_expire_minutes: int = 30

    # ── Server ───────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    @property
    def cors_origin_list(self) -> list[str]:
        """Parse comma-separated CORS origins into a list."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def model_artifact_path(self) -> Path:
        return Path(self.model_path)

    @property
    def scaler_artifact_path(self) -> Path:
        return Path(self.scaler_path)


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
