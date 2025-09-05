# services/api/auth.py
import os
from typing import Optional

from fastapi import Header, HTTPException, status, Depends

ROLE_KEYS = {
    os.getenv("ADMIN_API_KEY", "admin-key-123"): "admin",
    os.getenv("ENGINEER_API_KEY", "engineer-key-123"): "engineer",
    os.getenv("VIEWER_API_KEY", "viewer-key-123"): "viewer",
}


class Role:
    ADMIN = "admin"
    ENGINEER = "engineer"
    VIEWER = "viewer"


def require_roles(*allowed: str):
    async def _check(x_api_key: Optional[str] = Header(None)):
        if not x_api_key or x_api_key not in ROLE_KEYS:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
        role = ROLE_KEYS[x_api_key]
        if allowed and role not in allowed:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
        return role

    return Depends(_check)
