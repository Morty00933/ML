# services/training/training.py
import io
import os

import boto3
import mlflow
import pandas as pd
from botocore.client import Config
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split, ParameterGrid

from featurize import build_pipeline, PARAM_GRID  # <-- абсолютный импорт, БЕЗ точки

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

s3 = boto3.client(
    "s3",
    endpoint_url=os.getenv("MINIO_ENDPOINT", "http://minio:9000"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    config=Config(signature_version="s3v4"),
)


def read_s3_csv(uri: str) -> pd.DataFrame:
    _, path = uri.split("s3://", 1)
    bucket, key = path.split("/", 1)
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


def train_once(project: str, dataset_s3_uri: str, target: str, model_name: str, test_size: float, max_trials: int):
    df = read_s3_csv(dataset_s3_uri)
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    best = {"score": -1, "params": None, "model_key": None, "run_id": None}

    for model_key, grid in PARAM_GRID.items():
        params_list = list(ParameterGrid(grid))
        # равномерно распределим лимит попыток между моделями
        per_model_trials = max(1, max_trials // max(1, len(PARAM_GRID)))
        for params in params_list[:per_model_trials]:
            with mlflow.start_run(run_name=f"{project}-{model_key}") as run:
                pipe = build_pipeline(df, target, model_key)
                pipe.set_params(**params)
                pipe.fit(X_train, y_train)
                preds = pipe.predict(X_valid)
                acc = accuracy_score(y_valid, preds)
                f1 = f1_score(y_valid, preds, average="macro")

                mlflow.sklearn.log_model(pipe, "model")
                mlflow.log_params(params)
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("f1_macro", f1)
                mlflow.set_tag("project", project)
                mlflow.set_tag("model_key", model_key)

                if f1 > best["score"]:
                    best = {"score": f1, "params": params, "model_key": model_key, "run_id": run.info.run_id}

    # Регистрируем лучшую
    client = mlflow.tracking.MlflowClient()
    result = mlflow.register_model(f"runs:/{best['run_id']}/model", model_name)
    client.transition_model_version_stage(model_name, result.version, stage="Staging", archive_existing_versions=False)

    return {
        "best_f1": best["score"],
        "best_model": best["model_key"],
        "best_params": best["params"],
        "run_id": best["run_id"],
        "model_name": model_name,
        "version": result.version,
    }
