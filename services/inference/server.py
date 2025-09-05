import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from metrics import REQS, ERRS, LAT, metrics_endpoint
from model_loader import holder

app = FastAPI(title="Inference Service")


class PredictReq(BaseModel):
    rows: list


@app.get("/metrics")
def metrics():
    return metrics_endpoint()


@app.post("/load_model")
def load_model(payload: dict):
    uri = payload.get("model_uri")
    if not uri:
        raise HTTPException(status_code=400, detail="model_uri is required")
    holder.load(uri)
    return {"ok": True, "loaded": uri}


@app.post("/rollback")
def rollback():
    ok = holder.rollback()
    if not ok:
        raise HTTPException(status_code=400, detail="no previous model to rollback to")
    return {"ok": True, "loaded": holder.cur_uri}


@app.post("/predict")
def predict(req: PredictReq):
    if holder.model is None:
        raise HTTPException(status_code=400, detail="model is not loaded. Call /load_model first.")
    REQS.inc()
    with LAT.time():
        try:
            df = pd.DataFrame(req.rows)
            preds = holder.model.predict(df)
            out = []
            for x in preds:
                try:
                    out.append(x.item())
                except Exception:
                    out.append(x)
            return {"preds": out}
        except Exception as e:
            ERRS.inc()
            raise HTTPException(status_code=400, detail=str(e))
