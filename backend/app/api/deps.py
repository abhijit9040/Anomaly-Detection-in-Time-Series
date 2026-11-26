"""
FastAPI authentication & authorization dependencies.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.core.security import decode_access_token
from app.db.models import User
from app.db.session import get_db

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/login",
    auto_error=False,
)


async def get_current_user(
    db: Session = Depends(get_db),
    token: str | None = Depends(oauth2_scheme),
) -> User:
    """
    Dependency that decodes the bearer JWT token, validates claims,
    and retrieves the active user from the database.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"error": "unauthorized", "message": "Could not validate credentials or session expired."},
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not token:
        raise credentials_exception

    # Handle 'Bearer <token>' or raw token
    if token.startswith("Bearer "):
        token = token[7:]

    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception

    username: str | None = payload.get("sub")
    if username is None:
        raise credentials_exception

    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": "inactive_user", "message": "User account is disabled."},
        )

    return user


async def get_current_active_admin(
    current_user: User = Depends(get_current_user),
) -> User:
    """Dependency that restricts access to admin-only operations."""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": "insufficient_permissions", "message": "Admin privileges required."},
        )
    return current_user
