import os

import requests

PREFECT_API_URL = os.getenv("PREFECT_API_URL", "http://prefect:4200/api")

# В MVP дергаем тренинг через простой HTTP на контейнер training (можно заменить на prefect deployments)
TRAINING_SERVICE_URL = os.getenv("TRAINING_SERVICE_URL", "http://training:8001")


def trigger_training(payload: dict):
    # training контейнер поднимает простой HTTP для запуска флоу (см. services/training/flow.py)
    r = requests.post(f"{TRAINING_SERVICE_URL}/train", json=payload, timeout=3600)
    r.raise_for_status()
    return r.json()
