import os

import mlflow

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))


def register_and_transition(run_id: str, model_name: str, to_stage: str = "Staging"):
    result = mlflow.register_model(f"runs:/{run_id}/model", model_name)
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name, version=result.version, stage=to_stage, archive_existing_versions=False
    )
    return result


def latest_model_uri(model_name: str, stage: str = "Production"):
    client = mlflow.tracking.MlflowClient()
    mv = client.get_latest_versions(model_name, stages=[stage])
    if not mv:
        return None
    return f"models:/{model_name}/{stage}"
