from fastapi.responses import Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

REQS = Counter("inference_requests_total", "Inference requests")
ERRS = Counter("inference_errors_total", "Inference errors")
LAT = Histogram("inference_latency_seconds", "Inference latency")


def metrics_endpoint():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
