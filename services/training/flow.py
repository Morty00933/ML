# services/training/flow.py
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from training import train_once  # абсолютный импорт

app = FastAPI(title="Training Service")


class TrainReq(BaseModel):
    project: str
    dataset_s3_uri: str
    target: str
    model_name: str
    test_size: float = 0.2
    max_trials: int = 10


@app.post("/train")
def train(req: TrainReq):
    return train_once(**req.dict())


if __name__ == "__main__":
    uvicorn.run("flow:app", host="0.0.0.0", port=8001)
