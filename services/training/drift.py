"""Data drift detection utilities.

This module provides tools for detecting distribution shifts between
reference (training) and production data.

Supported methods:
- Population Stability Index (PSI)
- Kolmogorov-Smirnov test
- Chi-square test for categorical features
- Jensen-Shannon divergence
"""
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# PSI thresholds
PSI_THRESHOLD_LOW = 0.1  # No significant change
PSI_THRESHOLD_MEDIUM = 0.2  # Moderate change
# Above MEDIUM = Significant change

# KS test default significance level
KS_ALPHA = 0.05


def psi(
    reference: pd.Series,
    current: pd.Series,
    buckets: int = 10,
    epsilon: float = 1e-6
) -> float:
    """Calculate Population Stability Index (PSI).

    PSI measures how much a distribution has shifted.
    - PSI < 0.1: No significant change
    - 0.1 <= PSI < 0.2: Moderate change
    - PSI >= 0.2: Significant change

    Args:
        reference: Reference (training) data
        current: Current (production) data
        buckets: Number of buckets for discretization
        epsilon: Small value to avoid division by zero

    Returns:
        PSI value
    """
    try:
        # Create buckets from reference data
        ref_bins = pd.qcut(reference, q=buckets, duplicates="drop")
        bin_edges = ref_bins.cat.categories

        # Apply same bins to current data
        cur_bins = pd.cut(current, bins=bin_edges)

        # Calculate proportions
        ref_counts = ref_bins.value_counts(normalize=True).sort_index()
        cur_counts = cur_bins.value_counts(normalize=True).reindex(ref_counts.index).fillna(epsilon)

        # Replace zeros with epsilon
        ref_counts = ref_counts.replace(0, epsilon)
        cur_counts = cur_counts.replace(0, epsilon)

        # Calculate PSI
        psi_values = (cur_counts - ref_counts) * np.log(cur_counts / ref_counts)
        return float(psi_values.sum())

    except Exception as e:
        logger.warning(f"PSI calculation failed: {e}")
        return float("nan")


def ks_test(
    reference: pd.Series,
    current: pd.Series
) -> Tuple[float, float, bool]:
    """Perform Kolmogorov-Smirnov test for distribution comparison.

    Args:
        reference: Reference (training) data
        current: Current (production) data

    Returns:
        Tuple of (KS statistic, p-value, drift_detected)
    """
    try:
        statistic, p_value = stats.ks_2samp(reference.dropna(), current.dropna())
        drift_detected = p_value < KS_ALPHA
        return float(statistic), float(p_value), drift_detected
    except Exception as e:
        logger.warning(f"KS test failed: {e}")
        return float("nan"), float("nan"), False


def chi_square_test(
    reference: pd.Series,
    current: pd.Series
) -> Tuple[float, float, bool]:
    """Perform Chi-square test for categorical features.

    Args:
        reference: Reference (training) data
        current: Current (production) data

    Returns:
        Tuple of (Chi-square statistic, p-value, drift_detected)
    """
    try:
        # Get all unique categories
        all_categories = set(reference.unique()) | set(current.unique())

        # Count occurrences
        ref_counts = reference.value_counts()
        cur_counts = current.value_counts()

        # Align categories
        ref_freq = np.array([ref_counts.get(cat, 0) for cat in all_categories])
        cur_freq = np.array([cur_counts.get(cat, 0) for cat in all_categories])

        # Normalize to same total
        ref_freq = ref_freq / ref_freq.sum() * cur_freq.sum()

        # Perform chi-square test
        statistic, p_value = stats.chisquare(cur_freq, f_exp=ref_freq)
        drift_detected = p_value < KS_ALPHA

        return float(statistic), float(p_value), drift_detected

    except Exception as e:
        logger.warning(f"Chi-square test failed: {e}")
        return float("nan"), float("nan"), False


def jensen_shannon_divergence(
    reference: pd.Series,
    current: pd.Series,
    buckets: int = 10
) -> float:
    """Calculate Jensen-Shannon divergence between distributions.

    JSD is a symmetric version of KL divergence, bounded between 0 and 1.
    - JSD = 0: Identical distributions
    - JSD = 1: Completely different distributions

    Args:
        reference: Reference (training) data
        current: Current (production) data
        buckets: Number of buckets for discretization

    Returns:
        Jensen-Shannon divergence value
    """
    try:
        # Create histogram bins from combined data
        combined = pd.concat([reference, current])
        bins = pd.qcut(combined, q=buckets, duplicates="drop").cat.categories

        # Discretize both datasets
        ref_hist = pd.cut(reference, bins=bins).value_counts(normalize=True).sort_index()
        cur_hist = pd.cut(current, bins=bins).value_counts(normalize=True).sort_index()

        # Align and fill missing
        all_bins = ref_hist.index.union(cur_hist.index)
        p = ref_hist.reindex(all_bins).fillna(1e-10).values
        q = cur_hist.reindex(all_bins).fillna(1e-10).values

        # Normalize
        p = p / p.sum()
        q = q / q.sum()

        # Calculate JSD
        m = 0.5 * (p + q)
        jsd = 0.5 * (stats.entropy(p, m) + stats.entropy(q, m))

        return float(jsd)

    except Exception as e:
        logger.warning(f"JSD calculation failed: {e}")
        return float("nan")


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    methods: List[str] = ["psi", "ks"]
) -> Dict[str, Dict]:
    """Detect drift for all features in the dataset.

    Args:
        reference: Reference (training) DataFrame
        current: Current (production) DataFrame
        features: List of features to check (default: all numeric)
        categorical_features: List of categorical features
        methods: List of methods to use ("psi", "ks", "chi2", "jsd")

    Returns:
        Dictionary with drift statistics for each feature
    """
    if features is None:
        features = reference.select_dtypes(include=[np.number]).columns.tolist()

    if categorical_features is None:
        categorical_features = []

    results = {}

    for feature in features:
        if feature not in reference.columns or feature not in current.columns:
            logger.warning(f"Feature {feature} not found in both datasets")
            continue

        ref_series = reference[feature].dropna()
        cur_series = current[feature].dropna()

        if len(ref_series) == 0 or len(cur_series) == 0:
            logger.warning(f"Feature {feature} has no valid data")
            continue

        feature_result = {
            "reference_mean": float(ref_series.mean()) if feature not in categorical_features else None,
            "reference_std": float(ref_series.std()) if feature not in categorical_features else None,
            "current_mean": float(cur_series.mean()) if feature not in categorical_features else None,
            "current_std": float(cur_series.std()) if feature not in categorical_features else None,
            "reference_count": len(ref_series),
            "current_count": len(cur_series),
            "drift_detected": False,
        }

        # Apply appropriate tests based on feature type
        if feature in categorical_features:
            if "chi2" in methods:
                chi2_stat, chi2_p, chi2_drift = chi_square_test(ref_series, cur_series)
                feature_result["chi2_statistic"] = chi2_stat
                feature_result["chi2_p_value"] = chi2_p
                feature_result["chi2_drift"] = chi2_drift
                if chi2_drift:
                    feature_result["drift_detected"] = True
        else:
            # Numeric features
            if "psi" in methods:
                psi_value = psi(ref_series, cur_series)
                feature_result["psi"] = psi_value
                feature_result["psi_drift"] = psi_value >= PSI_THRESHOLD_MEDIUM
                if psi_value >= PSI_THRESHOLD_MEDIUM:
                    feature_result["drift_detected"] = True

            if "ks" in methods:
                ks_stat, ks_p, ks_drift = ks_test(ref_series, cur_series)
                feature_result["ks_statistic"] = ks_stat
                feature_result["p_value"] = ks_p
                feature_result["ks_drift"] = ks_drift
                if ks_drift:
                    feature_result["drift_detected"] = True

            if "jsd" in methods:
                jsd_value = jensen_shannon_divergence(ref_series, cur_series)
                feature_result["jsd"] = jsd_value

        results[feature] = feature_result

    return results


def generate_drift_report(
    drift_results: Dict[str, Dict],
    threshold: float = PSI_THRESHOLD_MEDIUM
) -> Dict:
    """Generate a summary report from drift detection results.

    Args:
        drift_results: Results from detect_drift()
        threshold: PSI threshold for drift detection

    Returns:
        Summary report dictionary
    """
    total_features = len(drift_results)
    drifted_features = [f for f, r in drift_results.items() if r.get("drift_detected", False)]

    report = {
        "total_features": total_features,
        "drifted_features_count": len(drifted_features),
        "drifted_features": drifted_features,
        "drift_percentage": len(drifted_features) / total_features * 100 if total_features > 0 else 0,
        "overall_drift_detected": len(drifted_features) > 0,
        "severity": "none",
    }

    # Determine severity
    drift_pct = report["drift_percentage"]
    if drift_pct == 0:
        report["severity"] = "none"
    elif drift_pct < 10:
        report["severity"] = "low"
    elif drift_pct < 25:
        report["severity"] = "medium"
    else:
        report["severity"] = "high"

    return report
