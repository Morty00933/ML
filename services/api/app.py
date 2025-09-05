# services/api/app.py
import io
import os

import pandas as pd
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

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

app = FastAPI(title="Mini-ML Platform API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
)

REQS = Counter("api_requests_total", "API requests", ["path"])
LAT = Histogram("api_latency_seconds", "API latency", ["path"])


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/projects", dependencies=[require_roles(Role.VIEWER, Role.ENGINEER, Role.ADMIN)])
def create_project(payload: CreateProject):
    # MVP: проект = префикс в S3 (в проде: таблицы в БД)
    key = f"projects/{payload.name}/.keep"
    put_fileobj(key, io.BytesIO(b""))
    return {"ok": True, "project": payload.name}


@app.post(
    "/datasets/upload",
    response_model=UploadDatasetResponse,
    dependencies=[require_roles(Role.ENGINEER, Role.ADMIN)],
)
async def upload_dataset(project: str, file: UploadFile = File(...)):
    with LAT.labels("/datasets/upload").time():
        REQS.labels("/datasets/upload").inc()
        key = f"projects/{project}/datasets/{file.filename}"
        uri = put_fileobj(key, file.file)
        return {"s3_uri": uri}


@app.post("/datasets/validate", dependencies=[require_roles(Role.ENGINEER, Role.ADMIN)])
def validate(req: ValidateRequest):
    import boto3
    from botocore.client import Config

    s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("MINIO_ENDPOINT"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        config=Config(signature_version="s3v4"),
    )
    # parse s3 uri
    _, path = req.s3_uri.split("s3://", 1)
    bucket, key = path.split("/", 1)
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()))

    rules = req.rules or []
    report = simple_validate(df, rules)
    return report


@app.post("/experiments/train", dependencies=[require_roles(Role.ENGINEER, Role.ADMIN)])
def train(req: TrainRequest):
    training_service = os.getenv("TRAINING_SERVICE_URL", "http://training:8001")
    try:
        r = requests.post(f"{training_service}/train", json=req.model_dump(), timeout=3600)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        return JSONResponse(status_code=502, content={"ok": False, "error": f"training service error: {e}"})


@app.post("/models/promote", dependencies=[require_roles(Role.ENGINEER, Role.ADMIN)])
def promote(req: PromoteRequest):
    # Перевод конкретной версии в нужную стадию
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    client.transition_model_version_stage(
        name=req.model_name,
        version=req.version,
        stage=req.to_stage,
        archive_existing_versions=False,
    )
    return {"ok": True, "model": req.model_name, "version": req.version, "to_stage": req.to_stage}


@app.post("/deploy", dependencies=[require_roles(Role.ENGINEER, Role.ADMIN)])
def deploy(req: DeployRequest):
    # Дёргаем inference-сервис, чтобы он подтянул модель со стадии
    uri = latest_model_uri(req.model_name, stage=req.stage)
    if not uri:
        return JSONResponse(status_code=404, content={"ok": False, "error": "no model in requested stage"})
    try:
        r = requests.post("http://inference:8080/load_model", json={"model_uri": uri}, timeout=60)
        r.raise_for_status()
    except requests.RequestException as e:
        return JSONResponse(status_code=502, content={"ok": False, "error": f"inference error: {e}"})
    return {"ok": True, "loaded": uri}
