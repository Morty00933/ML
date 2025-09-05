from typing import Optional, List

from pydantic import BaseModel, Field


class CreateProject(BaseModel):
    name: str


class UploadDatasetResponse(BaseModel):
    s3_uri: str


class ValidationRule(BaseModel):
    column: str
    min: Optional[float] = None
    max: Optional[float] = None
    allow_nulls: bool = True


class ValidateRequest(BaseModel):
    s3_uri: str
    rules: Optional[List[ValidationRule]] = None


class TrainRequest(BaseModel):
    project: str
    dataset_s3_uri: str
    target: str
    model_name: str = Field(..., description="Имя модели в MLflow")
    test_size: float = 0.2
    max_trials: int = 10


class DeployRequest(BaseModel):
    model_name: str
    stage: str = "Staging"


class PromoteRequest(BaseModel):
    model_name: str
    version: int
    to_stage: str = "Production"
