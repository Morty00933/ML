# services/training/flow.py
"""Training service API with cross-validation support and metrics."""
import os
import time
import logging
import uvicorn
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from training import train_once  # абсолютный импорт

# Structured logging setup
import sys
sys.path.insert(0, "/app/common")
try:
    from logging_config import setup_logging
    USE_JSON_LOGGING = os.getenv("LOG_FORMAT", "json").lower() == "json"
    logger = setup_logging("training", level=os.getenv("LOG_LEVEL", "INFO"), json_format=USE_JSON_LOGGING)
except ImportError:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

app = FastAPI(title="Training Service")

# Prometheus metrics
TRAINING_JOBS_TOTAL = Counter(
    'training_jobs_total',
    'Total number of training jobs started',
    ['project', 'model_name']
)

TRAINING_JOBS_SUCCESS = Counter(
    'training_jobs_success_total',
    'Total number of successful training jobs',
    ['project', 'model_name']
)

TRAINING_JOBS_FAILED = Counter(
    'training_jobs_failed_total',
    'Total number of failed training jobs',
    ['project', 'model_name']
)

TRAINING_DURATION = Histogram(
    'training_duration_seconds',
    'Training job duration in seconds',
    ['project', 'model_name'],
    buckets=[60, 120, 300, 600, 1200, 1800, 3600]  # 1min to 1hr
)

TRAINING_IN_PROGRESS = Gauge(
    'training_jobs_in_progress',
    'Number of training jobs currently in progress'
)

LAST_TRAINING_METRICS = Gauge(
    'training_last_metric',
    'Last training job metrics',
    ['project', 'model_name', 'metric']
)


class TrainReq(BaseModel):
    project: str
    dataset_s3_uri: str
    target: str
    model_name: str
    test_size: float = 0.2
    max_trials: int = 10
    cv_folds: int = 5
    use_cross_validation: bool = True


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    return PlainTextResponse(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.post("/train")
def train(req: TrainReq):
    """Train a model with metrics tracking."""
    project = req.project
    model_name = req.model_name

    logger.info("Training job started", extra={
        "project": project,
        "model_name": model_name,
        "cv_folds": req.cv_folds,
        "use_cross_validation": req.use_cross_validation
    })

    TRAINING_JOBS_TOTAL.labels(project=project, model_name=model_name).inc()
    TRAINING_IN_PROGRESS.inc()

    start_time = time.time()

    try:
        result = train_once(**req.model_dump())

        # Record success
        TRAINING_JOBS_SUCCESS.labels(project=project, model_name=model_name).inc()

        # Record metrics from result
        if 'f1' in result:
            LAST_TRAINING_METRICS.labels(
                project=project, model_name=model_name, metric='f1'
            ).set(result['f1'])
        if 'accuracy' in result:
            LAST_TRAINING_METRICS.labels(
                project=project, model_name=model_name, metric='accuracy'
            ).set(result['accuracy'])
        if 'cv_mean_f1' in result:
            LAST_TRAINING_METRICS.labels(
                project=project, model_name=model_name, metric='cv_mean_f1'
            ).set(result['cv_mean_f1'])

        duration = time.time() - start_time
        logger.info("Training job completed", extra={
            "project": project,
            "model_name": model_name,
            "duration_seconds": round(duration, 2),
            "f1": result.get('f1'),
            "accuracy": result.get('accuracy'),
            "cv_mean_f1": result.get('cv_mean_f1')
        })

        return result

    except Exception as e:
        TRAINING_JOBS_FAILED.labels(project=project, model_name=model_name).inc()
        logger.error("Training job failed", extra={
            "project": project,
            "model_name": model_name,
            "error": str(e)
        })
        raise

    finally:
        TRAINING_IN_PROGRESS.dec()
        duration = time.time() - start_time
        TRAINING_DURATION.labels(project=project, model_name=model_name).observe(duration)


if __name__ == "__main__":
    uvicorn.run("flow:app", host="0.0.0.0", port=8001)
