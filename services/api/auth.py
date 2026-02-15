# services/api/auth.py
"""Authentication module with security best practices.

Security features:
- Timing-safe comparison to prevent timing attacks
- Rate limiting on failed authentication attempts
- No hardcoded fallback credentials
- Proper logging of auth failures
"""
import hmac
import logging
import os
import time
from typing import Optional, Dict, Tuple

from fastapi import Header, HTTPException, Request, status, Depends

logger = logging.getLogger(__name__)

# Track failed authentication attempts for rate limiting
_auth_failures: Dict[str, Tuple[int, float]] = {}
AUTH_FAILURE_WINDOW = 300  # 5 minutes
AUTH_FAILURE_MAX = 10  # Max failures before lockout


def _get_role_keys() -> Dict[str, str]:
    """Get role keys from environment variables.

    Security: No fallback values - keys must be explicitly configured.
    """
    keys = {}

    admin_key = os.getenv("ADMIN_API_KEY")
    if admin_key:
        keys[admin_key] = "admin"

    engineer_key = os.getenv("ENGINEER_API_KEY")
    if engineer_key:
        keys[engineer_key] = "engineer"

    viewer_key = os.getenv("VIEWER_API_KEY")
    if viewer_key:
        keys[viewer_key] = "viewer"

    return keys


def _secure_compare(a: str, b: str) -> bool:
    """Timing-safe string comparison to prevent timing attacks."""
    return hmac.compare_digest(a.encode(), b.encode())


def _get_client_ip(request: Request) -> str:
    """Extract client IP from request."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def _check_auth_rate_limit(client_ip: str) -> bool:
    """Check if client is rate limited due to auth failures."""
    now = time.time()

    # Clean old entries periodically
    if len(_auth_failures) > 1000:
        cutoff = now - AUTH_FAILURE_WINDOW
        to_delete = [k for k, (_, t) in _auth_failures.items() if t < cutoff]
        for k in to_delete:
            del _auth_failures[k]

    if client_ip in _auth_failures:
        failures, first_failure = _auth_failures[client_ip]
        if now - first_failure > AUTH_FAILURE_WINDOW:
            del _auth_failures[client_ip]
            return True
        if failures >= AUTH_FAILURE_MAX:
            return False
    return True


def _record_auth_failure(client_ip: str):
    """Record an authentication failure."""
    now = time.time()
    if client_ip in _auth_failures:
        failures, first_failure = _auth_failures[client_ip]
        _auth_failures[client_ip] = (failures + 1, first_failure)
    else:
        _auth_failures[client_ip] = (1, now)


class Role:
    ADMIN = "admin"
    ENGINEER = "engineer"
    VIEWER = "viewer"


def require_roles(*allowed: str):
    """Dependency that requires specific roles for access.

    Security features:
    - Timing-safe key comparison
    - Rate limiting on auth failures
    - Proper logging without exposing keys
    """
    async def _check(
        request: Request,
        x_api_key: Optional[str] = Header(None),
    ):
        client_ip = _get_client_ip(request)

        # Check rate limit
        if not _check_auth_rate_limit(client_ip):
            logger.warning(f"Auth rate limit exceeded for {client_ip}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many authentication failures. Try again later.",
            )

        if not x_api_key:
            _record_auth_failure(client_ip)
            logger.warning(f"Missing API key from {client_ip} for {request.url.path}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing API key. Provide X-API-Key header.",
            )

        role_keys = _get_role_keys()

        if not role_keys:
            logger.error("No API keys configured in environment")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Server configuration error.",
            )

        # Use timing-safe comparison for each key
        matched_role = None
        for key, role in role_keys.items():
            if _secure_compare(x_api_key, key):
                matched_role = role
                break

        if matched_role is None:
            _record_auth_failure(client_ip)
            logger.warning(f"Invalid API key from {client_ip} for {request.url.path}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key.",
            )

        if allowed and matched_role not in allowed:
            logger.warning(f"Forbidden access attempt by {matched_role} from {client_ip} to {request.url.path}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden. Insufficient permissions.",
            )

        return matched_role

    return Depends(_check)
