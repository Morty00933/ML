"""Model loading and management with proper error handling.

Features:
- Thread-safe model loading
- Rollback support
- Retry logic for transient failures
- Proper error categorization
"""
import logging
import os
import threading
import time
from typing import Optional

import mlflow
from mlflow.exceptions import MlflowException

logger = logging.getLogger(__name__)

# Configuration
MAX_LOAD_RETRIES = int(os.getenv("MODEL_LOAD_RETRIES", "3"))
RETRY_DELAY = float(os.getenv("MODEL_LOAD_RETRY_DELAY", "2.0"))


class ModelLoadError(Exception):
    """Custom exception for model loading errors."""

    def __init__(self, message: str, cause: Optional[Exception] = None, retryable: bool = False):
        super().__init__(message)
        self.cause = cause
        self.retryable = retryable


class ModelHolder:
    """Thread-safe model holder with rollback support."""

    def __init__(self):
        self.model = None
        self.prev_uri: Optional[str] = None
        self.cur_uri: Optional[str] = None
        self.load_timestamp: Optional[float] = None
        self._lock = threading.Lock()

    def load(self, model_uri: str, retries: int = MAX_LOAD_RETRIES) -> None:
        """Load a model with retry logic and proper error handling.

        Args:
            model_uri: MLflow model URI (runs:/, models:/, s3://)
            retries: Number of retry attempts for transient failures

        Raises:
            ModelLoadError: If model cannot be loaded after all retries
        """
        last_error = None

        for attempt in range(retries + 1):
            try:
                with self._lock:
                    logger.info(f"Loading model from {model_uri} (attempt {attempt + 1}/{retries + 1})")

                    # Validate URI format
                    if not self._validate_uri(model_uri):
                        raise ModelLoadError(f"Invalid model URI format: {model_uri}", retryable=False)

                    # Load the model
                    new_model = mlflow.pyfunc.load_model(model_uri)

                    # Validate the model has predict method
                    if not hasattr(new_model, "predict"):
                        raise ModelLoadError("Loaded model does not have predict method", retryable=False)

                    # Update state only after successful load
                    self.prev_uri = self.cur_uri
                    self.model = new_model
                    self.cur_uri = model_uri
                    self.load_timestamp = time.time()

                    logger.info(f"Successfully loaded model from {model_uri}")
                    return

            except MlflowException as e:
                last_error = e
                error_msg = str(e).lower()

                # Categorize MLflow errors
                if "not found" in error_msg or "does not exist" in error_msg:
                    raise ModelLoadError(
                        f"Model not found: {model_uri}",
                        cause=e,
                        retryable=False
                    )
                elif "permission" in error_msg or "access denied" in error_msg:
                    raise ModelLoadError(
                        f"Permission denied accessing model: {model_uri}",
                        cause=e,
                        retryable=False
                    )
                elif "connection" in error_msg or "timeout" in error_msg:
                    # Transient error, can retry
                    logger.warning(f"Transient error loading model (attempt {attempt + 1}): {e}")
                    if attempt < retries:
                        time.sleep(RETRY_DELAY * (attempt + 1))
                        continue
                else:
                    # Unknown MLflow error
                    raise ModelLoadError(
                        f"MLflow error loading model: {str(e)}",
                        cause=e,
                        retryable=False
                    )

            except OSError as e:
                # File system or network errors - potentially retryable
                last_error = e
                logger.warning(f"OS error loading model (attempt {attempt + 1}): {e}")
                if attempt < retries:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue

            except Exception as e:
                # Unexpected error
                raise ModelLoadError(
                    f"Unexpected error loading model: {str(e)}",
                    cause=e,
                    retryable=False
                )

        # All retries exhausted
        raise ModelLoadError(
            f"Failed to load model after {retries + 1} attempts",
            cause=last_error,
            retryable=True
        )

    def rollback(self) -> bool:
        """Rollback to the previous model version.

        Returns:
            True if rollback succeeded, False if no previous model
        """
        if not self.prev_uri:
            logger.warning("Rollback requested but no previous model available")
            return False

        try:
            old_uri = self.prev_uri
            logger.info(f"Rolling back to previous model: {old_uri}")
            self.load(old_uri)
            return True
        except ModelLoadError as e:
            logger.error(f"Rollback failed: {e}")
            return False

    def unload(self) -> None:
        """Unload the current model."""
        with self._lock:
            self.prev_uri = self.cur_uri
            self.model = None
            self.cur_uri = None
            self.load_timestamp = None
            logger.info("Model unloaded")

    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self.model is not None

    def get_info(self) -> dict:
        """Get information about the current model state."""
        return {
            "loaded": self.is_loaded(),
            "current_uri": self.cur_uri,
            "previous_uri": self.prev_uri,
            "load_timestamp": self.load_timestamp,
            "can_rollback": self.prev_uri is not None,
        }

    @staticmethod
    def _validate_uri(uri: str) -> bool:
        """Validate model URI format."""
        if not uri:
            return False

        valid_prefixes = ("runs:/", "models:/", "s3://", "file://")
        if not uri.startswith(valid_prefixes):
            return False

        # Check for path traversal
        if ".." in uri:
            return False

        return True


# Global singleton
holder = ModelHolder()
