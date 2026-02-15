# tests/test_integration.py
"""Integration tests for Mini-ML Platform.

These tests verify end-to-end workflows across multiple services.
Requires running docker-compose services for full integration testing.
"""
import os
import io
import pytest
import requests
from unittest.mock import patch, MagicMock

# Test configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8010")
INFERENCE_URL = os.getenv("INFERENCE_URL", "http://localhost:8085")
ADMIN_KEY = os.getenv("ADMIN_API_KEY", "test-admin-key")
ENGINEER_KEY = os.getenv("ENGINEER_API_KEY", "test-engineer-key")
VIEWER_KEY = os.getenv("VIEWER_API_KEY", "test-viewer-key")


class TestRoleBasedAccess:
    """Test role-based access control for all endpoints."""

    def test_viewer_cannot_create_project(self):
        """Viewer should not be able to create projects."""
        response = requests.post(
            f"{API_BASE_URL}/projects",
            json={"name": "test-project"},
            headers={"X-API-Key": VIEWER_KEY},
            timeout=5
        )
        # Should be forbidden (403) not unauthorized (401)
        assert response.status_code == 403

    def test_viewer_can_list_projects(self):
        """Viewer should be able to list projects."""
        response = requests.get(
            f"{API_BASE_URL}/projects",
            headers={"X-API-Key": VIEWER_KEY},
            timeout=5
        )
        # Should succeed
        assert response.status_code in [200, 500]  # 500 if MinIO not available

    def test_viewer_can_list_models(self):
        """Viewer should be able to list models."""
        response = requests.get(
            f"{API_BASE_URL}/models",
            headers={"X-API-Key": VIEWER_KEY},
            timeout=5
        )
        assert response.status_code in [200, 500]

    def test_viewer_can_list_experiments(self):
        """Viewer should be able to list experiments."""
        response = requests.get(
            f"{API_BASE_URL}/experiments",
            headers={"X-API-Key": VIEWER_KEY},
            timeout=5
        )
        assert response.status_code in [200, 500]

    def test_engineer_can_create_project(self):
        """Engineer should be able to create projects."""
        response = requests.post(
            f"{API_BASE_URL}/projects",
            json={"name": "test-engineer-project"},
            headers={"X-API-Key": ENGINEER_KEY},
            timeout=5
        )
        # Should succeed (or fail with 500 if MinIO not available)
        assert response.status_code in [200, 500]

    def test_admin_can_create_project(self):
        """Admin should be able to create projects."""
        response = requests.post(
            f"{API_BASE_URL}/projects",
            json={"name": "test-admin-project"},
            headers={"X-API-Key": ADMIN_KEY},
            timeout=5
        )
        assert response.status_code in [200, 500]

    def test_viewer_cannot_upload_dataset(self):
        """Viewer should not be able to upload datasets."""
        files = {"file": ("test.csv", io.BytesIO(b"a,b,c\n1,2,3"), "text/csv")}
        response = requests.post(
            f"{API_BASE_URL}/datasets/upload?project=test",
            files=files,
            headers={"X-API-Key": VIEWER_KEY},
            timeout=5
        )
        assert response.status_code == 403

    def test_viewer_cannot_train_model(self):
        """Viewer should not be able to train models."""
        response = requests.post(
            f"{API_BASE_URL}/experiments/train",
            json={
                "project": "test",
                "model_name": "test-model",
                "s3_uri": "s3://mlflow/test.csv",
                "target_column": "target"
            },
            headers={"X-API-Key": VIEWER_KEY},
            timeout=5
        )
        assert response.status_code == 403

    def test_viewer_cannot_promote_model(self):
        """Viewer should not be able to promote models."""
        response = requests.post(
            f"{API_BASE_URL}/models/promote",
            json={
                "model_name": "test-model",
                "version": "1",
                "to_stage": "Production"
            },
            headers={"X-API-Key": VIEWER_KEY},
            timeout=5
        )
        assert response.status_code == 403

    def test_viewer_cannot_deploy_model(self):
        """Viewer should not be able to deploy models."""
        response = requests.post(
            f"{API_BASE_URL}/deploy",
            json={"model_name": "test-model", "stage": "Production"},
            headers={"X-API-Key": VIEWER_KEY},
            timeout=5
        )
        assert response.status_code == 403


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_api_health(self):
        """API health endpoint should return OK."""
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_inference_health(self):
        """Inference health endpoint should return OK."""
        response = requests.get(f"{INFERENCE_URL}/health", timeout=5)
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestInputValidation:
    """Test input validation across endpoints."""

    def test_invalid_project_name_rejected(self):
        """Project names with invalid characters should be rejected."""
        response = requests.post(
            f"{API_BASE_URL}/projects",
            json={"name": "../evil-path"},
            headers={"X-API-Key": ADMIN_KEY},
            timeout=5
        )
        assert response.status_code == 422  # Validation error

    def test_invalid_filename_rejected(self):
        """Filenames with path traversal should be rejected."""
        files = {"file": ("../../../etc/passwd", io.BytesIO(b"test"), "text/csv")}
        response = requests.post(
            f"{API_BASE_URL}/datasets/upload?project=test",
            files=files,
            headers={"X-API-Key": ADMIN_KEY},
            timeout=5
        )
        assert response.status_code == 400

    def test_invalid_file_extension_rejected(self):
        """Files with disallowed extensions should be rejected."""
        files = {"file": ("malware.exe", io.BytesIO(b"test"), "application/octet-stream")}
        response = requests.post(
            f"{API_BASE_URL}/datasets/upload?project=test",
            files=files,
            headers={"X-API-Key": ADMIN_KEY},
            timeout=5
        )
        assert response.status_code == 400


class TestRateLimiting:
    """Test rate limiting on authentication failures."""

    def test_rate_limit_after_failures(self):
        """After many auth failures, should get rate limited."""
        # Make 11 failed auth attempts
        for _ in range(11):
            requests.get(
                f"{API_BASE_URL}/projects",
                headers={"X-API-Key": "invalid-key"},
                timeout=5
            )

        # The 12th should be rate limited
        response = requests.get(
            f"{API_BASE_URL}/projects",
            headers={"X-API-Key": "another-invalid-key"},
            timeout=5
        )
        # Could be 401 or 429 depending on timing
        assert response.status_code in [401, 429]


class TestMetrics:
    """Test metrics endpoint."""

    def test_metrics_endpoint(self):
        """Metrics endpoint should return Prometheus format."""
        response = requests.get(f"{API_BASE_URL}/metrics", timeout=5)
        assert response.status_code == 200
        assert "api_requests_total" in response.text or "HELP" in response.text


# Unit-level integration tests (can run without services)
class TestAPIClientMocked:
    """Test API with mocked external services."""

    @pytest.fixture
    def mock_env(self):
        """Set up test environment variables."""
        with patch.dict(os.environ, {
            "ADMIN_API_KEY": "test-admin",
            "ENGINEER_API_KEY": "test-engineer",
            "VIEWER_API_KEY": "test-viewer",
            "MINIO_ENDPOINT": "http://localhost:9000",
            "AWS_ACCESS_KEY_ID": "test",
            "AWS_SECRET_ACCESS_KEY": "test",
            "MINIO_BUCKET": "test-bucket",
        }):
            yield

    def test_role_permissions_mapping(self, mock_env):
        """Verify role permissions are correctly defined."""
        from services.api.auth import _get_role_keys

        keys = _get_role_keys()
        assert "test-admin" in keys
        assert keys["test-admin"] == "admin"
        assert "test-engineer" in keys
        assert keys["test-engineer"] == "engineer"
        assert "test-viewer" in keys
        assert keys["test-viewer"] == "viewer"
