"""Inference service for model predictions.

Security features:
- Input validation
- Proper error handling
- Health check endpoint

Features:
- Single and batch predictions
- Async batch processing with job status
- Model versioning with rollback
- Prediction probabilities
- Structured JSON logging
"""
import os
import re
import uuid
from typing import List, Dict
from datetime import datetime, timezone
from enum import Enum

import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, field_validator

from metrics import REQS, ERRS, LAT, metrics_endpoint
from model_loader import holder

# Structured logging setup
import sys
sys.path.insert(0, "/app/common")
try:
    from logging_config import setup_logging, get_logger
    USE_JSON_LOGGING = os.getenv("LOG_FORMAT", "json").lower() == "json"
    logger = setup_logging("inference", level=os.getenv("LOG_LEVEL", "INFO"), json_format=USE_JSON_LOGGING)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

app = FastAPI(title="Inference Service")

# Security: Maximum rows per prediction request
MAX_PREDICTION_ROWS = 1000
MAX_BATCH_ROWS = 10000

# Batch job storage (in production, use Redis or database)
batch_jobs: Dict[str, dict] = {}


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PredictReq(BaseModel):
    rows: List[dict] = Field(..., min_length=1, max_length=MAX_PREDICTION_ROWS)


class BatchPredictReq(BaseModel):
    rows: List[dict] = Field(..., min_length=1, max_length=MAX_BATCH_ROWS)
    include_probabilities: bool = False


class LoadModelReq(BaseModel):
    model_uri: str = Field(..., min_length=1, max_length=500)

    @field_validator("model_uri")
    @classmethod
    def validate_uri(cls, v: str) -> str:
        # Security: Validate model URI format
        if not re.match(r'^(models:/|runs:/|s3://)[a-zA-Z0-9._/-]+$', v):
            raise ValueError("Invalid model URI format")
        if ".." in v:
            raise ValueError("Path traversal not allowed")
        return v


def _convert_predictions(preds) -> List:
    """Convert numpy predictions to Python types."""
    out = []
    for x in preds:
        try:
            out.append(x.item())
        except (AttributeError, TypeError):
            out.append(x)
    return out


def _run_batch_prediction(job_id: str, rows: List[dict], include_probabilities: bool):
    """Background task for batch predictions."""
    batch_jobs[job_id]["status"] = JobStatus.RUNNING
    batch_jobs[job_id]["started_at"] = datetime.now(timezone.utc).isoformat()

    try:
        df = pd.DataFrame(rows)
        preds = holder.model.predict(df)
        result = {"predictions": _convert_predictions(preds)}

        # Add probabilities if requested and model supports it
        if include_probabilities:
            try:
                if hasattr(holder.model, "predict_proba"):
                    proba = holder.model.predict_proba(df)
                    result["probabilities"] = proba.tolist()
                elif hasattr(holder.model, "_model_impl"):
                    # MLflow pyfunc wrapper
                    inner = holder.model._model_impl
                    if hasattr(inner, "predict_proba"):
                        proba = inner.predict_proba(df)
                        result["probabilities"] = proba.tolist()
            except Exception as e:
                logger.warning(f"Could not get probabilities: {e}")
                result["probabilities"] = None

        batch_jobs[job_id]["status"] = JobStatus.COMPLETED
        batch_jobs[job_id]["result"] = result
        batch_jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        batch_jobs[job_id]["rows_processed"] = len(rows)

    except Exception as e:
        logger.error(f"Batch prediction failed for job {job_id}: {e}")
        batch_jobs[job_id]["status"] = JobStatus.FAILED
        batch_jobs[job_id]["error"] = str(e)
        batch_jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": holder.model is not None}


@app.get("/metrics")
def metrics():
    return metrics_endpoint()


@app.post("/load_model")
def load_model(payload: LoadModelReq):
    """Load a model from MLflow."""
    logger.info(f"Loading model: {payload.model_uri}")
    try:
        holder.load(payload.model_uri)
        return {"ok": True, "loaded": payload.model_uri}
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail="Failed to load model")


@app.post("/rollback")
def rollback():
    """Rollback to previous model version."""
    logger.info("Rolling back model")
    ok = holder.rollback()
    if not ok:
        raise HTTPException(status_code=400, detail="No previous model to rollback to")
    return {"ok": True, "loaded": holder.cur_uri}


@app.post("/predict")
def predict(req: PredictReq):
    """Make predictions using the loaded model (synchronous)."""
    if holder.model is None:
        raise HTTPException(status_code=400, detail="Model is not loaded. Call /load_model first.")

    REQS.inc()
    with LAT.time():
        try:
            df = pd.DataFrame(req.rows)
            preds = holder.model.predict(df)
            return {"preds": _convert_predictions(preds)}
        except ValueError as e:
            ERRS.inc()
            logger.warning(f"Prediction validation error: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid input data: {str(e)}")
        except Exception as e:
            ERRS.inc()
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail="Prediction failed")


@app.post("/predict/batch")
def batch_predict(req: BatchPredictReq, background_tasks: BackgroundTasks):
    """Submit a batch prediction job (asynchronous).

    Returns a job_id that can be used to check status and retrieve results.
    """
    if holder.model is None:
        raise HTTPException(status_code=400, detail="Model is not loaded. Call /load_model first.")

    job_id = str(uuid.uuid4())
    batch_jobs[job_id] = {
        "job_id": job_id,
        "status": JobStatus.PENDING,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_rows": len(req.rows),
        "result": None,
        "error": None,
    }

    background_tasks.add_task(
        _run_batch_prediction,
        job_id,
        req.rows,
        req.include_probabilities
    )

    logger.info(f"Batch prediction job {job_id} submitted with {len(req.rows)} rows")
    return {"job_id": job_id, "status": JobStatus.PENDING}


@app.get("/predict/batch/{job_id}")
def get_batch_status(job_id: str):
    """Get the status and results of a batch prediction job."""
    if job_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = batch_jobs[job_id]
    response = {
        "job_id": job_id,
        "status": job["status"],
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at"),
        "total_rows": job.get("total_rows"),
        "rows_processed": job.get("rows_processed"),
    }

    if job["status"] == JobStatus.COMPLETED:
        response["result"] = job["result"]
    elif job["status"] == JobStatus.FAILED:
        response["error"] = job["error"]

    return response


@app.delete("/predict/batch/{job_id}")
def delete_batch_job(job_id: str):
    """Delete a batch prediction job and its results."""
    if job_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    del batch_jobs[job_id]
    logger.info(f"Batch job {job_id} deleted")
    return {"ok": True, "deleted": job_id}


@app.get("/predict/batch")
def list_batch_jobs():
    """List all batch prediction jobs."""
    return {
        "jobs": [
            {
                "job_id": job_id,
                "status": job["status"],
                "created_at": job.get("created_at"),
                "total_rows": job.get("total_rows"),
            }
            for job_id, job in batch_jobs.items()
        ]
    }


@app.get("/model/info")
def model_info():
    """Get information about the currently loaded model."""
    if holder.model is None:
        return {"loaded": False}

    info = {
        "loaded": True,
        "current_uri": holder.cur_uri,
        "previous_uri": holder.prev_uri,
        "can_rollback": holder.prev_uri is not None,
    }

    # Try to get model metadata
    try:
        if hasattr(holder.model, "metadata"):
            info["metadata"] = holder.model.metadata.to_dict()
    except Exception:
        pass

    return info
