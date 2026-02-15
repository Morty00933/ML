# services/training/training.py
"""Training module with cross-validation and proper data handling.

Features:
- Cross-validation to prevent data leakage
- Stratified splits for imbalanced data
- Proper train/validation/test splits
"""
import io
import os

import boto3
import mlflow
import numpy as np
import pandas as pd
from botocore.client import Config
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import (
    train_test_split,
    ParameterGrid,
    StratifiedKFold,
    cross_val_score,
)

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


def train_once(
    project: str,
    dataset_s3_uri: str,
    target: str,
    model_name: str,
    test_size: float,
    max_trials: int,
    cv_folds: int = 5,
    use_cross_validation: bool = True,
):
    """Train models with proper cross-validation to prevent data leakage.

    Args:
        project: Project name for MLflow tracking
        dataset_s3_uri: S3 URI of the dataset
        target: Target column name
        model_name: Name for registered model
        test_size: Fraction of data for final test set
        max_trials: Maximum hyperparameter combinations to try
        cv_folds: Number of cross-validation folds (default: 5)
        use_cross_validation: Whether to use CV (default: True)

    Returns:
        dict with best model info and metrics
    """
    df = read_s3_csv(dataset_s3_uri)
    X = df.drop(columns=[target])
    y = df[target]

    # Split into train+val and holdout test set (NEVER touch test during training)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Cross-validation setup
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    best = {"score": -1, "params": None, "model_key": None, "run_id": None, "cv_scores": None}

    for model_key, grid in PARAM_GRID.items():
        params_list = list(ParameterGrid(grid))
        # равномерно распределим лимит попыток между моделями
        per_model_trials = max(1, max_trials // max(1, len(PARAM_GRID)))

        for params in params_list[:per_model_trials]:
            with mlflow.start_run(run_name=f"{project}-{model_key}") as run:
                pipe = build_pipeline(df, target, model_key)
                pipe.set_params(**params)

                if use_cross_validation:
                    # Use cross-validation on train+val set
                    cv_scores = cross_val_score(
                        pipe, X_trainval, y_trainval,
                        cv=cv, scoring='f1_macro', n_jobs=-1
                    )
                    mean_f1 = np.mean(cv_scores)
                    std_f1 = np.std(cv_scores)

                    # Log CV metrics
                    mlflow.log_metric("cv_f1_mean", mean_f1)
                    mlflow.log_metric("cv_f1_std", std_f1)
                    for i, score in enumerate(cv_scores):
                        mlflow.log_metric(f"cv_fold_{i}_f1", score)

                    # Refit on full train+val for final model
                    pipe.fit(X_trainval, y_trainval)
                    score_for_selection = mean_f1
                else:
                    # Simple train/val split (legacy behavior)
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval
                    )
                    pipe.fit(X_train, y_train)
                    preds_val = pipe.predict(X_val)
                    score_for_selection = f1_score(y_val, preds_val, average="macro")
                    cv_scores = None

                # Calculate metrics on train set (for logging only)
                preds_train = pipe.predict(X_trainval)
                train_acc = accuracy_score(y_trainval, preds_train)
                train_f1 = f1_score(y_trainval, preds_train, average="macro")

                # Log model and metrics
                mlflow.sklearn.log_model(pipe, "model")
                mlflow.log_params(params)
                mlflow.log_param("cv_folds", cv_folds)
                mlflow.log_param("use_cross_validation", use_cross_validation)
                mlflow.log_metric("train_accuracy", train_acc)
                mlflow.log_metric("train_f1_macro", train_f1)
                mlflow.set_tag("project", project)
                mlflow.set_tag("model_key", model_key)

                if score_for_selection > best["score"]:
                    best = {
                        "score": score_for_selection,
                        "params": params,
                        "model_key": model_key,
                        "run_id": run.info.run_id,
                        "cv_scores": cv_scores.tolist() if cv_scores is not None else None,
                    }

    # Final evaluation on holdout test set (ONLY for the best model)
    with mlflow.start_run(run_id=best["run_id"]):
        # Load best model and evaluate on test set
        best_pipe = build_pipeline(df, target, best["model_key"])
        best_pipe.set_params(**best["params"])
        best_pipe.fit(X_trainval, y_trainval)

        preds_test = best_pipe.predict(X_test)
        test_acc = accuracy_score(y_test, preds_test)
        test_f1 = f1_score(y_test, preds_test, average="macro")

        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_f1_macro", test_f1)

    # Register best model
    client = mlflow.tracking.MlflowClient()
    result = mlflow.register_model(f"runs:/{best['run_id']}/model", model_name)
    client.transition_model_version_stage(
        model_name, result.version, stage="Staging", archive_existing_versions=False
    )

    return {
        "best_cv_f1": best["score"],
        "test_f1": test_f1,
        "test_accuracy": test_acc,
        "best_model": best["model_key"],
        "best_params": best["params"],
        "cv_scores": best["cv_scores"],
        "run_id": best["run_id"],
        "model_name": model_name,
        "version": result.version,
    }
