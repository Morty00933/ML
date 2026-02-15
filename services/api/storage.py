"""S3/MinIO storage utilities with retry logic.

Features:
- Automatic retries with exponential backoff
- Configurable retry parameters
- Proper error handling and logging
"""
import io
import logging
import os
import time
from functools import wraps
from typing import Callable, Any

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError, EndpointConnectionError

logger = logging.getLogger(__name__)

BUCKET = os.getenv("MINIO_BUCKET", "mlflow")
ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Retry configuration
MAX_RETRIES = int(os.getenv("S3_MAX_RETRIES", "3"))
RETRY_DELAY = float(os.getenv("S3_RETRY_DELAY", "1.0"))  # seconds
RETRY_BACKOFF = float(os.getenv("S3_RETRY_BACKOFF", "2.0"))  # multiplier

_session = boto3.session.Session()
s3 = _session.client(
    service_name="s3",
    endpoint_url=ENDPOINT,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    config=Config(
        signature_version="s3v4",
        retries={"max_attempts": MAX_RETRIES, "mode": "adaptive"},
    ),
)


def with_retry(
    max_retries: int = MAX_RETRIES,
    delay: float = RETRY_DELAY,
    backoff: float = RETRY_BACKOFF,
    exceptions: tuple = (ClientError, EndpointConnectionError, ConnectionError),
) -> Callable:
    """Decorator for retrying S3 operations with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry

    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"S3 operation '{func.__name__}' failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"S3 operation '{func.__name__}' failed after {max_retries + 1} attempts: {e}"
                        )

            raise last_exception

        return wrapper
    return decorator


@with_retry()
def put_bytes(key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
    """Upload bytes to S3 with retry logic.

    Args:
        key: S3 object key
        data: Bytes to upload
        content_type: MIME type of the content

    Returns:
        S3 URI of uploaded object
    """
    s3.put_object(Bucket=BUCKET, Key=key, Body=io.BytesIO(data), ContentType=content_type)
    logger.info(f"Uploaded {len(data)} bytes to s3://{BUCKET}/{key}")
    return f"s3://{BUCKET}/{key}"


@with_retry()
def put_fileobj(key: str, fileobj) -> str:
    """Upload file object to S3 with retry logic.

    Args:
        key: S3 object key
        fileobj: File-like object to upload

    Returns:
        S3 URI of uploaded object
    """
    s3.upload_fileobj(fileobj, BUCKET, key)
    logger.info(f"Uploaded file to s3://{BUCKET}/{key}")
    return f"s3://{BUCKET}/{key}"


@with_retry()
def get_object(key: str) -> bytes:
    """Download object from S3 with retry logic.

    Args:
        key: S3 object key

    Returns:
        Object content as bytes
    """
    response = s3.get_object(Bucket=BUCKET, Key=key)
    data = response["Body"].read()
    logger.info(f"Downloaded {len(data)} bytes from s3://{BUCKET}/{key}")
    return data


@with_retry()
def list_objects(prefix: str = "", delimiter: str = "/") -> dict:
    """List objects in S3 with retry logic.

    Args:
        prefix: Filter by prefix
        delimiter: Delimiter for grouping

    Returns:
        S3 list_objects_v2 response
    """
    return s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, Delimiter=delimiter)


@with_retry()
def delete_object(key: str) -> None:
    """Delete object from S3 with retry logic.

    Args:
        key: S3 object key
    """
    s3.delete_object(Bucket=BUCKET, Key=key)
    logger.info(f"Deleted s3://{BUCKET}/{key}")


def check_connection() -> bool:
    """Check if S3/MinIO connection is working.

    Returns:
        True if connection is successful, False otherwise
    """
    try:
        s3.head_bucket(Bucket=BUCKET)
        return True
    except Exception as e:
        logger.error(f"S3 connection check failed: {e}")
        return False
