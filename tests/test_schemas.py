"""Tests for schema validation."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "api"))

import pytest
from pydantic import ValidationError

from schemas import (
    CreateProject,
    ValidateRequest,
    ValidationRule,
    TrainRequest,
    DeployRequest,
    PromoteRequest,
)


class TestCreateProject:
    def test_valid_name(self):
        project = CreateProject(name="my-project_123")
        assert project.name == "my-project_123"

    def test_rejects_path_traversal(self):
        with pytest.raises(ValidationError) as exc:
            CreateProject(name="../etc/passwd")
        assert "alphanumeric" in str(exc.value).lower() or "invalid" in str(exc.value).lower()

    def test_rejects_special_characters(self):
        with pytest.raises(ValidationError):
            CreateProject(name="project<script>")

    def test_rejects_empty_name(self):
        with pytest.raises(ValidationError):
            CreateProject(name="")

    def test_rejects_too_long_name(self):
        with pytest.raises(ValidationError):
            CreateProject(name="a" * 101)


class TestValidateRequest:
    def test_valid_s3_uri(self):
        req = ValidateRequest(s3_uri="s3://bucket/path/to/file.csv")
        assert req.s3_uri == "s3://bucket/path/to/file.csv"

    def test_rejects_non_s3_uri(self):
        with pytest.raises(ValidationError):
            ValidateRequest(s3_uri="http://bucket/path")

    def test_rejects_path_traversal(self):
        with pytest.raises(ValidationError):
            ValidateRequest(s3_uri="s3://bucket/../etc/passwd")

    def test_with_rules(self):
        req = ValidateRequest(
            s3_uri="s3://bucket/data.csv",
            rules=[
                ValidationRule(column="age", min=0, max=150),
                ValidationRule(column="name", allow_nulls=False),
            ],
        )
        assert len(req.rules) == 2


class TestValidationRule:
    def test_valid_rule(self):
        rule = ValidationRule(column="age", min=0, max=100)
        assert rule.column == "age"
        assert rule.min == 0
        assert rule.max == 100

    def test_rejects_invalid_column_name(self):
        with pytest.raises(ValidationError):
            ValidationRule(column="col<script>")

    def test_rejects_min_greater_than_max(self):
        with pytest.raises(ValidationError):
            ValidationRule(column="age", min=100, max=0)


class TestTrainRequest:
    def test_valid_request(self):
        req = TrainRequest(
            project="my-project",
            dataset_s3_uri="s3://bucket/data.csv",
            target="label",
            model_name="my-model",
        )
        assert req.project == "my-project"
        assert req.test_size == 0.2  # default
        assert req.max_trials == 10  # default

    def test_rejects_invalid_project_name(self):
        with pytest.raises(ValidationError):
            TrainRequest(
                project="../bad",
                dataset_s3_uri="s3://bucket/data.csv",
                target="label",
                model_name="model",
            )

    def test_rejects_invalid_test_size(self):
        with pytest.raises(ValidationError):
            TrainRequest(
                project="project",
                dataset_s3_uri="s3://bucket/data.csv",
                target="label",
                model_name="model",
                test_size=1.5,  # > 0.9
            )

    def test_rejects_invalid_max_trials(self):
        with pytest.raises(ValidationError):
            TrainRequest(
                project="project",
                dataset_s3_uri="s3://bucket/data.csv",
                target="label",
                model_name="model",
                max_trials=0,  # < 1
            )


class TestDeployRequest:
    def test_valid_request(self):
        req = DeployRequest(model_name="my-model", stage="Production")
        assert req.model_name == "my-model"
        assert req.stage == "Production"

    def test_default_stage(self):
        req = DeployRequest(model_name="my-model")
        assert req.stage == "Staging"

    def test_rejects_invalid_stage(self):
        with pytest.raises(ValidationError):
            DeployRequest(model_name="model", stage="InvalidStage")

    def test_rejects_invalid_model_name(self):
        with pytest.raises(ValidationError):
            DeployRequest(model_name="model<>bad")


class TestPromoteRequest:
    def test_valid_request(self):
        req = PromoteRequest(model_name="my-model", version=1, to_stage="Production")
        assert req.version == 1
        assert req.to_stage == "Production"

    def test_rejects_invalid_version(self):
        with pytest.raises(ValidationError):
            PromoteRequest(model_name="model", version=0)  # < 1

    def test_rejects_invalid_stage(self):
        with pytest.raises(ValidationError):
            PromoteRequest(model_name="model", version=1, to_stage="BadStage")
