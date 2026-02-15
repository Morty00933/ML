"""Feature engineering and model pipeline building.

This module provides utilities for building sklearn pipelines
with proper column handling and model configuration.
"""
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Model configurations (templates - will be cloned for each use)
_MODEL_TEMPLATES = {
    "logreg": LogisticRegression(max_iter=1000),
    "rf": RandomForestClassifier(n_estimators=200, random_state=42)
}

PARAM_GRID = {
    "logreg": {"clf__C": [0.1, 1.0, 10.0]},
    "rf": {"clf__n_estimators": [100, 200, 400], "clf__max_depth": [None, 5, 10]},
}

# For backward compatibility
MODELS = _MODEL_TEMPLATES


def build_pipeline(df: pd.DataFrame, target: str, model_key: str) -> Pipeline:
    """Build a sklearn pipeline for the given data and model type.

    Args:
        df: DataFrame containing features and target
        target: Name of the target column
        model_key: Key for model type ('logreg' or 'rf')

    Returns:
        sklearn Pipeline ready for fitting

    Raises:
        KeyError: If model_key is not recognized
    """
    if model_key not in _MODEL_TEMPLATES:
        raise KeyError(f"Unknown model key: {model_key}. Available: {list(_MODEL_TEMPLATES.keys())}")

    # Determine numeric columns (all columns except target)
    numeric_cols = [c for c in df.columns if c != target]

    # Build preprocessor
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols)
    ])

    # Clone model template to avoid state pollution between runs
    model = clone(_MODEL_TEMPLATES[model_key])

    # Build pipeline
    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", model)
    ])

    return pipe
