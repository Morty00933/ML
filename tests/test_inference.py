"""Tests for inference service."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "inference"))

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from server import app, holder


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_model_holder():
    """Reset model holder before each test."""
    holder.model = None
    holder.cur_uri = None
    holder.prev_uri = None
    holder.prev_model = None


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data

    def test_health_shows_model_not_loaded(self, client):
        response = client.get("/health")
        assert response.json()["model_loaded"] is False


class TestLoadModelEndpoint:
    @patch("server.holder.load")
    def test_load_model_valid_uri(self, mock_load, client):
        response = client.post(
            "/load_model",
            json={"model_uri": "models:/test-model/1"},
        )
        assert response.status_code == 200
        assert response.json()["ok"] is True
        mock_load.assert_called_once_with("models:/test-model/1")

    def test_load_model_rejects_path_traversal(self, client):
        response = client.post(
            "/load_model",
            json={"model_uri": "models:/../etc/passwd"},
        )
        assert response.status_code == 422

    def test_load_model_rejects_invalid_uri(self, client):
        response = client.post(
            "/load_model",
            json={"model_uri": "http://malicious.com/model"},
        )
        assert response.status_code == 422

    def test_load_model_requires_uri(self, client):
        response = client.post("/load_model", json={})
        assert response.status_code == 422


class TestPredictEndpoint:
    def test_predict_without_model_returns_error(self, client):
        response = client.post(
            "/predict",
            json={"rows": [{"feature": 1}]},
        )
        assert response.status_code == 400
        assert "not loaded" in response.json()["detail"]

    @patch("server.holder")
    def test_predict_with_model(self, mock_holder, client):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5, 0.3]
        mock_holder.model = mock_model

        response = client.post(
            "/predict",
            json={"rows": [{"x": 1}, {"x": 2}]},
        )

        assert response.status_code == 200
        assert "preds" in response.json()

    def test_predict_rejects_empty_rows(self, client):
        response = client.post("/predict", json={"rows": []})
        assert response.status_code == 422

    def test_predict_rejects_too_many_rows(self, client):
        # MAX_PREDICTION_ROWS = 1000
        response = client.post(
            "/predict",
            json={"rows": [{"x": i} for i in range(1001)]},
        )
        assert response.status_code == 422


class TestRollbackEndpoint:
    def test_rollback_without_previous_model(self, client):
        response = client.post("/rollback")
        assert response.status_code == 400
        assert "no previous" in response.json()["detail"].lower()

    @patch("server.holder")
    def test_rollback_success(self, mock_holder, client):
        mock_holder.rollback.return_value = True
        mock_holder.cur_uri = "models:/old-model/1"

        response = client.post("/rollback")

        assert response.status_code == 200
        assert response.json()["ok"] is True


class TestInputValidation:
    def test_load_model_accepts_runs_uri(self, client):
        with patch("server.holder.load"):
            response = client.post(
                "/load_model",
                json={"model_uri": "runs:/abc123/model"},
            )
            assert response.status_code == 200

    def test_load_model_accepts_s3_uri(self, client):
        with patch("server.holder.load"):
            response = client.post(
                "/load_model",
                json={"model_uri": "s3://bucket/path/model"},
            )
            assert response.status_code == 200
