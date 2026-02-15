"""Tests for authentication module."""
import os
import pytest
from unittest.mock import MagicMock

# Set environment variables before importing auth
os.environ["ADMIN_API_KEY"] = "test-admin-key"
os.environ["ENGINEER_API_KEY"] = "test-engineer-key"
os.environ["VIEWER_API_KEY"] = "test-viewer-key"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "api"))

from auth import (
    _get_role_keys,
    _secure_compare,
    _get_client_ip,
    _check_auth_rate_limit,
    _record_auth_failure,
    _auth_failures,
    Role,
)


class TestGetRoleKeys:
    def test_returns_configured_keys(self):
        keys = _get_role_keys()
        assert "test-admin-key" in keys
        assert keys["test-admin-key"] == "admin"
        assert "test-engineer-key" in keys
        assert keys["test-engineer-key"] == "engineer"
        assert "test-viewer-key" in keys
        assert keys["test-viewer-key"] == "viewer"

    def test_no_hardcoded_fallbacks(self, monkeypatch):
        """Ensure no default keys if env vars not set."""
        monkeypatch.delenv("ADMIN_API_KEY", raising=False)
        monkeypatch.delenv("ENGINEER_API_KEY", raising=False)
        monkeypatch.delenv("VIEWER_API_KEY", raising=False)

        keys = _get_role_keys()
        # Should not contain the old hardcoded values
        assert "admin-key-123" not in keys
        assert "engineer-key-123" not in keys
        assert "viewer-key-123" not in keys


class TestSecureCompare:
    def test_equal_strings_return_true(self):
        assert _secure_compare("test", "test") is True
        assert _secure_compare("", "") is True
        assert _secure_compare("a" * 100, "a" * 100) is True

    def test_different_strings_return_false(self):
        assert _secure_compare("test", "test1") is False
        assert _secure_compare("test", "Test") is False
        assert _secure_compare("a", "b") is False


class TestGetClientIp:
    def test_returns_forwarded_ip(self):
        request = MagicMock()
        request.headers = {"X-Forwarded-For": "1.2.3.4, 5.6.7.8"}
        request.client = None

        assert _get_client_ip(request) == "1.2.3.4"

    def test_strips_whitespace(self):
        request = MagicMock()
        request.headers = {"X-Forwarded-For": "  1.2.3.4  , 5.6.7.8"}
        request.client = None

        assert _get_client_ip(request) == "1.2.3.4"

    def test_returns_client_host(self):
        request = MagicMock()
        request.headers = {}
        request.client = MagicMock()
        request.client.host = "192.168.1.1"

        assert _get_client_ip(request) == "192.168.1.1"

    def test_returns_unknown_if_no_client(self):
        request = MagicMock()
        request.headers = {}
        request.client = None

        assert _get_client_ip(request) == "unknown"


class TestAuthRateLimit:
    def setup_method(self):
        """Clear rate limit state before each test."""
        _auth_failures.clear()

    def test_allows_first_attempt(self):
        assert _check_auth_rate_limit("test-client") is True

    def test_blocks_after_max_failures(self):
        # Record max failures
        for _ in range(10):
            _record_auth_failure("test-client-2")

        # Should be blocked
        assert _check_auth_rate_limit("test-client-2") is False

    def test_different_clients_separate(self):
        # Exhaust one client
        for _ in range(10):
            _record_auth_failure("client-a")

        # Other client should still be allowed
        assert _check_auth_rate_limit("client-b") is True


class TestRole:
    def test_role_constants(self):
        assert Role.ADMIN == "admin"
        assert Role.ENGINEER == "engineer"
        assert Role.VIEWER == "viewer"
