"""Tests for API endpoints."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "api"))

# Set environment before imports
os.environ["ADMIN_API_KEY"] = "test-admin-key"
os.environ["ENGINEER_API_KEY"] = "test-engineer-key"
os.environ["VIEWER_API_KEY"] = "test-viewer-key"

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def admin_headers():
    return {"X-API-Key": "test-admin-key"}


@pytest.fixture
def engineer_headers():
    return {"X-API-Key": "test-engineer-key"}


@pytest.fixture
def viewer_headers():
    return {"X-API-Key": "test-viewer-key"}


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestMetricsEndpoint:
    def test_metrics_returns_prometheus_format(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200


class TestAuthRequired:
    def test_projects_requires_auth(self, client):
        response = client.post("/projects", json={"name": "test"})
        assert response.status_code == 401

    def test_datasets_upload_requires_auth(self, client):
        response = client.post("/datasets/upload?project=test")
        assert response.status_code == 401

    def test_datasets_validate_requires_auth(self, client):
        response = client.post(
            "/datasets/validate",
            json={"s3_uri": "s3://bucket/file.csv"},
        )
        assert response.status_code == 401

    def test_train_requires_auth(self, client):
        response = client.post(
            "/experiments/train",
            json={
                "project": "test",
                "dataset_s3_uri": "s3://bucket/file.csv",
                "target": "label",
                "model_name": "model",
            },
        )
        assert response.status_code == 401

    def test_invalid_key_returns_401(self, client):
        response = client.post(
            "/projects",
            json={"name": "test"},
            headers={"X-API-Key": "invalid-key"},
        )
        assert response.status_code == 401


class TestProjectsEndpoint:
    @patch("app.put_fileobj")
    def test_create_project_with_viewer(self, mock_put, client, viewer_headers):
        mock_put.return_value = "s3://bucket/projects/test/.keep"

        response = client.post(
            "/projects",
            json={"name": "test-project"},
            headers=viewer_headers,
        )

        assert response.status_code == 200
        assert response.json()["ok"] is True

    @patch("app.put_fileobj")
    def test_create_project_validates_name(self, mock_put, client, admin_headers):
        response = client.post(
            "/projects",
            json={"name": "invalid<name>"},
            headers=admin_headers,
        )

        assert response.status_code == 422  # Validation error


class TestRolePermissions:
    @patch("app.put_fileobj")
    def test_viewer_can_create_project(self, mock_put, client, viewer_headers):
        mock_put.return_value = "s3://bucket/test"
        response = client.post(
            "/projects",
            json={"name": "test"},
            headers=viewer_headers,
        )
        assert response.status_code == 200

    def test_viewer_cannot_train(self, client, viewer_headers):
        response = client.post(
            "/experiments/train",
            json={
                "project": "test",
                "dataset_s3_uri": "s3://bucket/file.csv",
                "target": "label",
                "model_name": "model",
            },
            headers=viewer_headers,
        )
        assert response.status_code == 403  # Forbidden


class TestInputValidation:
    def test_validate_rejects_path_traversal_in_uri(self, client, admin_headers):
        response = client.post(
            "/datasets/validate",
            json={"s3_uri": "s3://bucket/../etc/passwd"},
            headers=admin_headers,
        )
        assert response.status_code == 422

    def test_train_rejects_invalid_test_size(self, client, admin_headers):
        response = client.post(
            "/experiments/train",
            json={
                "project": "test",
                "dataset_s3_uri": "s3://bucket/data.csv",
                "target": "label",
                "model_name": "model",
                "test_size": 2.0,  # Invalid
            },
            headers=admin_headers,
        )
        assert response.status_code == 422

    def test_deploy_rejects_invalid_stage(self, client, admin_headers):
        response = client.post(
            "/deploy",
            json={"model_name": "model", "stage": "InvalidStage"},
            headers=admin_headers,
        )
        assert response.status_code == 422
