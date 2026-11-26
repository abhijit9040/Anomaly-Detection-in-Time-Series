"""
SQLAlchemy ORM models for AnomalyGuard.
- User: Authentication, credentials, role (admin, operator, viewer).
- AlertLog: Persistent anomaly alert history with operator acknowledgment.
"""

from __future__ import annotations

from datetime import datetime, timezone
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String
from app.db.session import Base


class User(Base):
    """Registered application user with hashed credentials and RBAC role."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), default="operator", nullable=False)  # admin, operator, viewer
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    last_login = Column(DateTime(timezone=True), nullable=True)

    def __repr__(self) -> str:
        return f"<User id={self.id} username='{self.username}' role='{self.role}'>"


class AlertLog(Base):
    """Persistent audit log of detected anomaly events."""
    __tablename__ = "anomaly_alerts"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
        nullable=False,
    )
    reconstruction_error = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)
    top_sensor = Column(String(50), nullable=True)
    acknowledged = Column(Boolean, default=False, nullable=False)
    acknowledged_by = Column(String(50), nullable=True)

    def __repr__(self) -> str:
        return f"<AlertLog id={self.id} error={self.reconstruction_error} acknowledged={self.acknowledged}>"
