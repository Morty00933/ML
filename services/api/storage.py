import io
import os

import boto3
from botocore.client import Config

BUCKET = os.getenv("MINIO_BUCKET", "mlflow")
ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

_session = boto3.session.Session()
s3 = _session.client(
    service_name="s3",
    endpoint_url=ENDPOINT,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    config=Config(signature_version="s3v4"),
)


def put_bytes(key: str, data: bytes, content_type: str = "application/octet-stream"):
    s3.put_object(Bucket=BUCKET, Key=key, Body=io.BytesIO(data), ContentType=content_type)
    return f"s3://{BUCKET}/{key}"


def put_fileobj(key: str, fileobj):
    s3.upload_fileobj(fileobj, BUCKET, key)
    return f"s3://{BUCKET}/{key}"
