"""Schema definitions with input validation.

Security features:
- Path traversal prevention
- Input length limits
- S3 URI validation
- Stage whitelist validation
"""
import re
from typing import Optional, List

from pydantic import BaseModel, Field, field_validator, model_validator


# Security: Allowed characters for names (alphanumeric, dash, underscore)
NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')
# Security: Valid S3 URI pattern
S3_URI_PATTERN = re.compile(r'^s3://[a-zA-Z0-9._-]+/[a-zA-Z0-9._/-]+$')
# Security: Allowed MLflow stages
ALLOWED_STAGES = {"None", "Staging", "Production", "Archived"}


class CreateProject(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Project name")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not NAME_PATTERN.match(v):
            raise ValueError("Project name must contain only alphanumeric characters, dashes, and underscores")
        if ".." in v or "/" in v or "\\" in v:
            raise ValueError("Invalid characters in project name")
        return v


class UploadDatasetResponse(BaseModel):
    s3_uri: str


class ValidationRule(BaseModel):
    column: str = Field(..., min_length=1, max_length=100)
    min: Optional[float] = Field(default=None, ge=-1e15, le=1e15)
    max: Optional[float] = Field(default=None, ge=-1e15, le=1e15)
    allow_nulls: bool = True

    @field_validator("column")
    @classmethod
    def validate_column(cls, v: str) -> str:
        if not NAME_PATTERN.match(v):
            raise ValueError("Column name must contain only alphanumeric characters, dashes, and underscores")
        return v

    @model_validator(mode="after")
    def validate_min_max(self):
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("min cannot be greater than max")
        return self


class ValidateRequest(BaseModel):
    s3_uri: str = Field(..., min_length=5, max_length=500)
    rules: Optional[List[ValidationRule]] = Field(default=None, max_length=100)

    @field_validator("s3_uri")
    @classmethod
    def validate_s3_uri(cls, v: str) -> str:
        if not v.startswith("s3://"):
            raise ValueError("URI must start with s3://")
        if ".." in v:
            raise ValueError("Path traversal not allowed")
        if not S3_URI_PATTERN.match(v):
            raise ValueError("Invalid S3 URI format")
        return v


class TrainRequest(BaseModel):
    project: str = Field(..., min_length=1, max_length=100)
    dataset_s3_uri: str = Field(..., min_length=5, max_length=500)
    target: str = Field(..., min_length=1, max_length=100)
    model_name: str = Field(..., min_length=1, max_length=100, description="Model name in MLflow")
    test_size: float = Field(default=0.2, ge=0.01, le=0.9)
    max_trials: int = Field(default=10, ge=1, le=100)

    @field_validator("project", "model_name")
    @classmethod
    def validate_names(cls, v: str) -> str:
        if not NAME_PATTERN.match(v):
            raise ValueError("Name must contain only alphanumeric characters, dashes, and underscores")
        if ".." in v:
            raise ValueError("Invalid characters in name")
        return v

    @field_validator("target")
    @classmethod
    def validate_target(cls, v: str) -> str:
        if not NAME_PATTERN.match(v):
            raise ValueError("Target column must contain only alphanumeric characters, dashes, and underscores")
        return v

    @field_validator("dataset_s3_uri")
    @classmethod
    def validate_dataset_uri(cls, v: str) -> str:
        if not v.startswith("s3://"):
            raise ValueError("URI must start with s3://")
        if ".." in v:
            raise ValueError("Path traversal not allowed")
        if not S3_URI_PATTERN.match(v):
            raise ValueError("Invalid S3 URI format")
        return v


class DeployRequest(BaseModel):
    model_name: str = Field(..., min_length=1, max_length=100)
    stage: str = Field(default="Staging", min_length=1, max_length=20)

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        if not NAME_PATTERN.match(v):
            raise ValueError("Model name must contain only alphanumeric characters, dashes, and underscores")
        return v

    @field_validator("stage")
    @classmethod
    def validate_stage(cls, v: str) -> str:
        if v not in ALLOWED_STAGES:
            raise ValueError(f"Stage must be one of: {', '.join(ALLOWED_STAGES)}")
        return v


class PromoteRequest(BaseModel):
    model_name: str = Field(..., min_length=1, max_length=100)
    version: int = Field(..., ge=1, le=10000)
    to_stage: str = Field(default="Production", min_length=1, max_length=20)

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        if not NAME_PATTERN.match(v):
            raise ValueError("Model name must contain only alphanumeric characters, dashes, and underscores")
        return v

    @field_validator("to_stage")
    @classmethod
    def validate_stage(cls, v: str) -> str:
        if v not in ALLOWED_STAGES:
            raise ValueError(f"Stage must be one of: {', '.join(ALLOWED_STAGES)}")
        return v
