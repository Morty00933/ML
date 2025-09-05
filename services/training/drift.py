import numpy as np
import pandas as pd


def psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    """Population Stability Index (простая реализация)."""
    e = pd.qcut(expected, q=buckets, duplicates='drop')
    a = pd.cut(actual, bins=e.cat.categories.categories)
    e_counts = e.value_counts(normalize=True)
    a_counts = a.value_counts(normalize=True)
    # выравниваем индексы
    a_counts = a_counts.reindex(e_counts.index).fillna(1e-6)
    e_counts = e_counts.fillna(1e-6)
    return float(((a_counts - e_counts) * np.log(a_counts / e_counts)).sum())
