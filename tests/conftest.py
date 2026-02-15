"""Pytest configuration and fixtures."""
import os
import sys

import pytest

# Add services to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "api"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "inference"))

# Set test environment variables
os.environ.setdefault("ADMIN_API_KEY", "test-admin-key")
os.environ.setdefault("ENGINEER_API_KEY", "test-engineer-key")
os.environ.setdefault("VIEWER_API_KEY", "test-viewer-key")
os.environ.setdefault("MINIO_ENDPOINT", "http://localhost:9000")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "minio")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minio123")


@pytest.fixture
def admin_headers():
    """Headers with admin API key."""
    return {"X-API-Key": os.environ["ADMIN_API_KEY"]}


@pytest.fixture
def engineer_headers():
    """Headers with engineer API key."""
    return {"X-API-Key": os.environ["ENGINEER_API_KEY"]}


@pytest.fixture
def viewer_headers():
    """Headers with viewer API key."""
    return {"X-API-Key": os.environ["VIEWER_API_KEY"]}


@pytest.fixture
def invalid_headers():
    """Headers with invalid API key."""
    return {"X-API-Key": "invalid-key"}
