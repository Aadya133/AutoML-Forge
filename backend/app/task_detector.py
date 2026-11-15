"""Classification/regression task inference."""
from __future__ import annotations
import pandas as pd

def detect_task(target: pd.Series, *, classification_max_unique: int = 15,
                classification_max_ratio: float = 0.05) -> str:
    """Infer task using dtype and cardinality; low-cardinality integer labels are classes."""
    if not isinstance(target, pd.Series):
        target = pd.Series(target)
    values = target.dropna()
    if values.empty:
        raise ValueError("Target column has no non-missing values.")
    if not pd.api.types.is_numeric_dtype(values):
        return "classification"
    unique = int(values.nunique())
    ratio = unique / max(len(values), 1)
    if pd.api.types.is_integer_dtype(values) and (unique <= classification_max_unique or ratio <= classification_max_ratio):
        return "classification"
    if unique <= 2:
        return "classification"
    return "regression"
