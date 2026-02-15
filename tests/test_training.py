# tests/test_training.py
"""Tests for training module functionality.

Tests featurize.py, training.py, and drift.py
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import io
import os
import sys

# Add services/training to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'services', 'training'))


class TestFeaturize:
    """Tests for featurize.py module."""

    @pytest.fixture
    def sample_df(self):
        """Create sample dataframe for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })

    def test_build_pipeline_logreg(self, sample_df):
        """Test building logistic regression pipeline."""
        from featurize import build_pipeline

        pipe = build_pipeline(sample_df, 'target', 'logreg')

        assert pipe is not None
        assert len(pipe.steps) == 2
        assert pipe.steps[0][0] == 'prep'
        assert pipe.steps[1][0] == 'clf'

    def test_build_pipeline_rf(self, sample_df):
        """Test building random forest pipeline."""
        from featurize import build_pipeline

        pipe = build_pipeline(sample_df, 'target', 'rf')

        assert pipe is not None
        assert len(pipe.steps) == 2

    def test_pipeline_fit_predict(self, sample_df):
        """Test pipeline can fit and predict."""
        from featurize import build_pipeline

        X = sample_df.drop(columns=['target'])
        y = sample_df['target']

        pipe = build_pipeline(sample_df, 'target', 'logreg')
        pipe.fit(X, y)
        preds = pipe.predict(X)

        assert len(preds) == len(y)
        assert set(preds).issubset({0, 1})

    def test_pipeline_hyperparameters(self, sample_df):
        """Test pipeline hyperparameter setting."""
        from featurize import build_pipeline, PARAM_GRID

        pipe = build_pipeline(sample_df, 'target', 'logreg')

        # Test setting parameters from grid
        params = {"clf__C": 0.1}
        pipe.set_params(**params)

        assert pipe.get_params()['clf__C'] == 0.1

    def test_param_grid_has_expected_models(self):
        """Verify PARAM_GRID contains expected model configs."""
        from featurize import PARAM_GRID, MODELS

        assert 'logreg' in PARAM_GRID
        assert 'rf' in PARAM_GRID
        assert 'logreg' in MODELS
        assert 'rf' in MODELS

    def test_invalid_model_key_raises(self, sample_df):
        """Invalid model key should raise KeyError."""
        from featurize import build_pipeline

        with pytest.raises(KeyError):
            build_pipeline(sample_df, 'target', 'invalid_model')


class TestTrainingModule:
    """Tests for training.py module."""

    @pytest.fixture
    def mock_s3(self):
        """Mock S3 client."""
        with patch('training.s3') as mock:
            csv_data = b"feature1,feature2,target\n1.0,2.0,0\n3.0,4.0,1\n5.0,6.0,0\n7.0,8.0,1"
            mock.get_object.return_value = {
                'Body': MagicMock(read=lambda: csv_data)
            }
            yield mock

    @pytest.fixture
    def mock_mlflow(self):
        """Mock MLflow client."""
        with patch('training.mlflow') as mock:
            mock_run = MagicMock()
            mock_run.info.run_id = "test-run-123"
            mock.start_run.return_value.__enter__ = lambda x: mock_run
            mock.start_run.return_value.__exit__ = lambda x, *args: None

            mock_result = MagicMock()
            mock_result.version = "1"
            mock.register_model.return_value = mock_result

            yield mock

    def test_read_s3_csv(self, mock_s3):
        """Test reading CSV from S3."""
        from training import read_s3_csv

        df = read_s3_csv("s3://bucket/path/file.csv")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        assert 'target' in df.columns

    def test_read_s3_csv_invalid_uri(self, mock_s3):
        """Invalid S3 URI should raise."""
        from training import read_s3_csv

        with pytest.raises(ValueError):
            read_s3_csv("invalid-uri")


class TestDriftDetection:
    """Tests for drift.py module."""

    @pytest.fixture
    def reference_df(self):
        """Create reference dataset."""
        np.random.seed(42)
        return pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(5, 2, 1000),
        })

    @pytest.fixture
    def drifted_df(self):
        """Create dataset with distribution drift."""
        np.random.seed(42)
        return pd.DataFrame({
            'feature1': np.random.normal(2, 1, 1000),  # Mean shifted
            'feature2': np.random.normal(5, 5, 1000),  # Variance increased
        })

    @pytest.fixture
    def no_drift_df(self):
        """Create dataset without drift."""
        np.random.seed(123)
        return pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(5, 2, 1000),
        })

    def test_drift_module_exists(self):
        """Verify drift module can be imported."""
        from drift import detect_drift

        assert callable(detect_drift)

    def test_detect_drift_with_drift(self, reference_df, drifted_df):
        """Should detect drift in shifted distribution."""
        from drift import detect_drift

        result = detect_drift(reference_df, drifted_df)

        # At least feature1 should show drift
        assert 'feature1' in result
        assert result['feature1']['drift_detected'] is True

    def test_detect_drift_no_drift(self, reference_df, no_drift_df):
        """Should not detect drift in similar distribution."""
        from drift import detect_drift

        result = detect_drift(reference_df, no_drift_df)

        # Neither should show significant drift
        for col, stats in result.items():
            # P-value should be relatively high (no significant drift)
            assert stats['p_value'] > 0.01 or stats['drift_detected'] is False


class TestCrossValidation:
    """Tests for cross-validation functionality."""

    @pytest.fixture
    def classification_data(self):
        """Create classification dataset."""
        np.random.seed(42)
        n_samples = 200
        X = pd.DataFrame({
            'f1': np.random.randn(n_samples),
            'f2': np.random.randn(n_samples),
            'f3': np.random.randn(n_samples),
        })
        y = (X['f1'] + X['f2'] > 0).astype(int)
        return X, y

    def test_cross_validation_scores(self, classification_data):
        """Test that cross-validation produces reasonable scores."""
        from sklearn.model_selection import cross_val_score
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        X, y = classification_data

        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression())
        ])

        scores = cross_val_score(pipe, X, y, cv=5, scoring='f1')

        assert len(scores) == 5
        assert all(0 <= s <= 1 for s in scores)
        assert np.mean(scores) > 0.5  # Should be better than random

    def test_stratified_split_maintains_ratio(self, classification_data):
        """Stratified split should maintain class ratio."""
        from sklearn.model_selection import train_test_split

        X, y = classification_data
        original_ratio = y.mean()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        train_ratio = y_train.mean()
        test_ratio = y_test.mean()

        # Ratios should be similar
        assert abs(train_ratio - original_ratio) < 0.05
        assert abs(test_ratio - original_ratio) < 0.05


class TestModelEvaluation:
    """Tests for model evaluation metrics."""

    def test_accuracy_calculation(self):
        """Test accuracy metric calculation."""
        from sklearn.metrics import accuracy_score

        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]

        acc = accuracy_score(y_true, y_pred)
        assert acc == 0.8

    def test_f1_macro_calculation(self):
        """Test F1 macro metric calculation."""
        from sklearn.metrics import f1_score

        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]

        f1 = f1_score(y_true, y_pred, average='macro')
        assert 0 < f1 < 1

    def test_metrics_with_imbalanced_classes(self):
        """Test metrics handle imbalanced classes."""
        from sklearn.metrics import f1_score, accuracy_score

        # 90% class 0, 10% class 1
        y_true = [0] * 90 + [1] * 10
        y_pred = [0] * 100  # Naive predictor

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')

        # Accuracy looks good but F1 is bad
        assert acc == 0.9
        assert f1 < 0.5  # F1 penalizes missing minority class
