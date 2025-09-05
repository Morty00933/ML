# services/api/validation.py
from typing import List

import pandas as pd

from schemas import ValidationRule  # <-- абсолютный импорт, БЕЗ точки


def simple_validate(df: pd.DataFrame, rules: List[ValidationRule]):
    report = {"rows": len(df), "columns": list(df.columns), "issues": []}
    for r in rules:
        if r.column not in df.columns:
            report["issues"].append({"column": r.column, "error": "missing"})
            continue
        col = df[r.column]
        if not r.allow_nulls and col.isna().any():
            report["issues"].append({"column": r.column, "error": "nulls"})
        if r.min is not None and (col < r.min).any():
            report["issues"].append({"column": r.column, "error": f"less_than_min({r.min})"})
        if r.max is not None and (col > r.max).any():
            report["issues"].append({"column": r.column, "error": f"greater_than_max({r.max})"})
    report["ok"] = len(report["issues"]) == 0
    return report
