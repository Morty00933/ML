# services/api/app.py
"""Mini-ML Platform API with security best practices.

Security features:
- Restricted CORS origins
- Input validation on all endpoints
- Proper error handling and logging
- Path traversal prevention
- Rate limiting
"""
import io
import logging
import os
import re

import pandas as pd
import requests as http_requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from auth import Role, require_roles
from mlflow_client import latest_model_uri
from schemas import (
    CreateProject,
    UploadDatasetResponse,
    ValidateRequest,
    TrainRequest,
    DeployRequest,
    PromoteRequest,
)
from storage import put_fileobj
from validation import simple_validate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Rate limiting configuration
RATE_LIMIT_DEFAULT = os.getenv("RATE_LIMIT_DEFAULT", "100/minute")
RATE_LIMIT_TRAINING = os.getenv("RATE_LIMIT_TRAINING", "5/minute")
RATE_LIMIT_UPLOAD = os.getenv("RATE_LIMIT_UPLOAD", "20/minute")

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Mini-ML Platform API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Security: Configure CORS with explicit origins
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:8080"
).split(",")
ALLOWED_ORIGINS = [origin.strip() for origin in ALLOWED_ORIGINS if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["X-API-Key", "Content-Type", "Authorization"],
)

# Security: Filename validation pattern
SAFE_FILENAME_PATTERN = re.compile(r'^[a-zA-Z0-9._-]+$')

REQS = Counter("api_requests_total", "API requests", ["path"])
LAT = Histogram("api_latency_seconds", "API latency", ["path"])


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
@limiter.exempt
def health(request: Request):
    """Health check endpoint."""
    return {"status": "ok"}


# ============================================================================
# READ-ONLY ENDPOINTS (Viewer, Engineer, Admin)
# ============================================================================

@app.get("/projects", dependencies=[require_roles(Role.VIEWER, Role.ENGINEER, Role.ADMIN)])
@limiter.limit(RATE_LIMIT_DEFAULT)
def list_projects(request: Request):
    """List all projects. Available to all authenticated users."""
    import boto3
    from botocore.client import Config

    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=os.getenv("MINIO_ENDPOINT"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            config=Config(signature_version="s3v4"),
        )
        bucket = os.getenv("MINIO_BUCKET", "mlflow")

        # List all prefixes under projects/
        response = s3.list_objects_v2(Bucket=bucket, Prefix="projects/", Delimiter="/")
        projects = []
        for prefix in response.get("CommonPrefixes", []):
            # Extract project name from prefix like "projects/my-project/"
            project_name = prefix["Prefix"].replace("projects/", "").rstrip("/")
            if project_name:
                projects.append(project_name)

        return {"projects": projects}
    except Exception as e:
        logger.error(f"Failed to list projects: {e}")
        raise HTTPException(status_code=500, detail="Failed to list projects")


@app.get("/models", dependencies=[require_roles(Role.VIEWER, Role.ENGINEER, Role.ADMIN)])
@limiter.limit(RATE_LIMIT_DEFAULT)
def list_models(request: Request):
    """List all registered models. Available to all authenticated users."""
    from mlflow.tracking import MlflowClient

    try:
        client = MlflowClient()
        models = client.search_registered_models()
        return {
            "models": [
                {
                    "name": m.name,
                    "latest_versions": [
                        {"version": v.version, "stage": v.current_stage}
                        for v in (m.latest_versions or [])
                    ]
                }
                for m in models
            ]
        }
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail="Failed to list models")


@app.get("/experiments", dependencies=[require_roles(Role.VIEWER, Role.ENGINEER, Role.ADMIN)])
@limiter.limit(RATE_LIMIT_DEFAULT)
def list_experiments(request: Request):
    """List all experiments. Available to all authenticated users."""
    from mlflow.tracking import MlflowClient

    try:
        client = MlflowClient()
        experiments = client.search_experiments()
        return {
            "experiments": [
                {
                    "id": exp.experiment_id,
                    "name": exp.name,
                    "lifecycle_stage": exp.lifecycle_stage,
                }
                for exp in experiments
            ]
        }
    except Exception as e:
        logger.error(f"Failed to list experiments: {e}")
        raise HTTPException(status_code=500, detail="Failed to list experiments")


@app.post("/projects", dependencies=[require_roles(Role.ENGINEER, Role.ADMIN)])
@limiter.limit(RATE_LIMIT_DEFAULT)
def create_project(request: Request, payload: CreateProject):
    """Create a new project (S3 prefix). Requires Engineer or Admin role."""
    logger.info(f"Creating project: {payload.name}")
    try:
        key = f"projects/{payload.name}/.keep"
        put_fileobj(key, io.BytesIO(b""))
        return {"ok": True, "project": payload.name}
    except Exception as e:
        logger.error(f"Failed to create project {payload.name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to create project")


def _validate_filename(filename: str) -> str:
    """Validate and sanitize uploaded filename.

    Security: Prevents path traversal and restricts allowed characters.
    """
    if not filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    # Remove any path components
    filename = os.path.basename(filename)

    # Check for path traversal attempts
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Check filename pattern
    if not SAFE_FILENAME_PATTERN.match(filename):
        raise HTTPException(
            status_code=400,
            detail="Filename must contain only alphanumeric characters, dots, dashes, and underscores"
        )

    # Check extension
    if not filename.lower().endswith((".csv", ".parquet", ".json")):
        raise HTTPException(
            status_code=400,
            detail="Only CSV, Parquet, and JSON files are allowed"
        )

    return filename


def _validate_project_name(project: str) -> str:
    """Validate project name parameter.

    Security: Prevents path traversal.
    """
    if not project:
        raise HTTPException(status_code=400, detail="Project name is required")

    if not re.match(r'^[a-zA-Z0-9_-]+$', project):
        raise HTTPException(
            status_code=400,
            detail="Project name must contain only alphanumeric characters, dashes, and underscores"
        )

    if ".." in project:
        raise HTTPException(status_code=400, detail="Invalid project name")

    return project


@app.post(
    "/datasets/upload",
    response_model=UploadDatasetResponse,
    dependencies=[require_roles(Role.ENGINEER, Role.ADMIN)],
)
@limiter.limit(RATE_LIMIT_UPLOAD)
async def upload_dataset(request: Request, project: str, file: UploadFile = File(...)):
    """Upload a dataset file to S3."""
    with LAT.labels("/datasets/upload").time():
        REQS.labels("/datasets/upload").inc()

        # Security: Validate inputs
        project = _validate_project_name(project)
        filename = _validate_filename(file.filename)

        logger.info(f"Uploading dataset: project={project}, filename={filename}")

        try:
            key = f"projects/{project}/datasets/{filename}"
            uri = put_fileobj(key, file.file)
            return {"s3_uri": uri}
        except Exception as e:
            logger.error(f"Failed to upload dataset: {e}")
            raise HTTPException(status_code=500, detail="Failed to upload dataset")


@app.post("/datasets/validate", dependencies=[require_roles(Role.ENGINEER, Role.ADMIN)])
@limiter.limit(RATE_LIMIT_DEFAULT)
def validate(request: Request, req: ValidateRequest):
    """Validate a dataset against rules."""
    import boto3
    from botocore.client import Config
    from botocore.exceptions import ClientError

    logger.info(f"Validating dataset: {req.s3_uri}")

    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=os.getenv("MINIO_ENDPOINT"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            config=Config(signature_version="s3v4"),
        )

        # Parse S3 URI (already validated by schema)
        _, path = req.s3_uri.split("s3://", 1)
        bucket, key = path.split("/", 1)

        obj = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(io.BytesIO(obj["Body"].read()))

        rules = req.rules or []
        report = simple_validate(df, rules)
        return report

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        if error_code == "NoSuchKey":
            raise HTTPException(status_code=404, detail="Dataset not found")
        logger.error(f"S3 error during validation: {e}")
        raise HTTPException(status_code=500, detail="Failed to access dataset")
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Dataset is empty")
    except Exception as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to validate dataset")


@app.post("/experiments/train", dependencies=[require_roles(Role.ENGINEER, Role.ADMIN)])
@limiter.limit(RATE_LIMIT_TRAINING)
def train(request: Request, req: TrainRequest):
    """Start model training."""
    logger.info(f"Starting training: project={req.project}, model={req.model_name}")

    training_service = os.getenv("TRAINING_SERVICE_URL", "http://training:8001")
    try:
        r = http_requests.post(f"{training_service}/train", json=req.model_dump(), timeout=3600)
        r.raise_for_status()
        return r.json()
    except http_requests.Timeout:
        logger.error("Training service timeout")
        raise HTTPException(status_code=504, detail="Training service timeout")
    except http_requests.RequestException as e:
        logger.error(f"Training service error: {e}")
        raise HTTPException(status_code=502, detail="Training service unavailable")


@app.post("/models/promote", dependencies=[require_roles(Role.ENGINEER, Role.ADMIN)])
@limiter.limit(RATE_LIMIT_DEFAULT)
def promote(request: Request, req: PromoteRequest):
    """Promote a model version to a new stage."""
    from mlflow.tracking import MlflowClient
    from mlflow.exceptions import MlflowException

    logger.info(f"Promoting model: {req.model_name} v{req.version} to {req.to_stage}")

    try:
        client = MlflowClient()
        client.transition_model_version_stage(
            name=req.model_name,
            version=req.version,
            stage=req.to_stage,
            archive_existing_versions=False,
        )
        return {"ok": True, "model": req.model_name, "version": req.version, "to_stage": req.to_stage}
    except MlflowException as e:
        logger.error(f"MLflow error during promotion: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to promote model: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during promotion: {e}")
        raise HTTPException(status_code=500, detail="Failed to promote model")


@app.post("/deploy", dependencies=[require_roles(Role.ENGINEER, Role.ADMIN)])
@limiter.limit(RATE_LIMIT_TRAINING)
def deploy(request: Request, req: DeployRequest):
    """Deploy a model to the inference service."""
    logger.info(f"Deploying model: {req.model_name} stage={req.stage}")

    try:
        uri = latest_model_uri(req.model_name, stage=req.stage)
    except Exception as e:
        logger.error(f"Failed to get model URI: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model info")

    if not uri:
        raise HTTPException(status_code=404, detail="No model found in requested stage")

    inference_url = os.getenv("INFERENCE_SERVICE_URL", "http://inference:8080")
    try:
        r = http_requests.post(f"{inference_url}/load_model", json={"model_uri": uri}, timeout=60)
        r.raise_for_status()
    except http_requests.Timeout:
        logger.error("Inference service timeout")
        raise HTTPException(status_code=504, detail="Inference service timeout")
    except http_requests.RequestException as e:
        logger.error(f"Inference service error: {e}")
        raise HTTPException(status_code=502, detail="Inference service unavailable")

    return {"ok": True, "loaded": uri}
