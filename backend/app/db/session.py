"""
Database session management & connection setup.
Supports both PostgreSQL (Neon Serverless) and local SQLite fallback.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Generator
from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import get_settings

logger = logging.getLogger("anomaly_detection.db")

Base = declarative_base()


def get_engine():
    """Create and configure SQLAlchemy engine with connection resilience."""
    settings = get_settings()
    raw_url = settings.sync_database_url

    connect_args = {}
    engine_kwargs = {}

    if "sqlite" in raw_url:
        connect_args["check_same_thread"] = False
        engine_kwargs["connect_args"] = connect_args
    else:
        # PostgreSQL / Neon configuration
        # Remove unsupported parameter combinations if needed
        clean_url = re.sub(r"[?&]channel_binding=[^&]+", "", raw_url)
        # Ensure sslmode=require for Neon if not specified
        if "sslmode" not in clean_url and "neon.tech" in clean_url:
            separator = "&" if "?" in clean_url else "?"
            clean_url = f"{clean_url}{separator}sslmode=require"

        # SQLAlchemy 2.1+ changed the default PostgreSQL driver from psycopg2 to
        # psycopg (v3). We ship psycopg2-binary, so explicitly force +psycopg2
        # dialect to avoid "No module named 'psycopg'" on deploy.
        clean_url = re.sub(
            r"^postgresql(\+[^:]+)?://",
            "postgresql+psycopg2://",
            clean_url,
        )

        raw_url = clean_url
        engine_kwargs["pool_pre_ping"] = True   # Automatically test connections on checkout (vital for Neon)
        engine_kwargs["pool_recycle"] = 300     # Recycle connections every 5 minutes

    return create_engine(raw_url, **engine_kwargs)


engine = get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

_db_initialized = False


def init_db():
    """
    Initialize database schema and seed default users if table is empty.
    Runs automatically during FastAPI lifespan startup or first database request.
    """
    global _db_initialized
    from app.core.security import get_password_hash
    from app.db.models import User

    try:
        Base.metadata.create_all(bind=engine)

        db = SessionLocal()
        try:
            existing_count = db.query(User).count()
            if existing_count == 0:
                logger.info("Seeding initial users into database...")
                initial_users = [
                    User(
                        username="admin",
                        email="admin@anomalyguard.io",
                        hashed_password=get_password_hash("admin"),
                        role="admin",
                        is_active=True,
                        created_at=datetime.now(timezone.utc),
                    ),
                    User(
                        username="operator",
                        email="operator@anomalyguard.io",
                        hashed_password=get_password_hash("operator"),
                        role="operator",
                        is_active=True,
                        created_at=datetime.now(timezone.utc),
                    ),
                    User(
                        username="engineer",
                        email="engineer@anomalyguard.io",
                        hashed_password=get_password_hash("engineer"),
                        role="operator",
                        is_active=True,
                        created_at=datetime.now(timezone.utc),
                    ),
                ]
                db.add_all(initial_users)
                db.commit()
                logger.info("✅ Database seeded with default users (admin, operator, engineer).")
        except Exception as e:
            logger.warning("Database seeding check failed: %s", e)
            db.rollback()
        finally:
            db.close()

        _db_initialized = True
    except Exception as e:
        logger.error("Failed to initialize database: %s", e)


def get_db() -> Generator:
    """FastAPI dependency that provides a transactional database session."""
    global _db_initialized
    if not _db_initialized:
        init_db()

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
