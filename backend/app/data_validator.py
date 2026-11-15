"""Dataset profiling and validation."""
from __future__ import annotations
import pandas as pd

def profile_dataset(frame: pd.DataFrame) -> dict:
    """Return serializable dimensions, schema, missingness, duplicates and summary stats."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    return {
        "rows": int(len(frame)), "columns": int(len(frame.columns)),
        "column_names": [str(c) for c in frame.columns],
        "dtypes": {str(c): str(t) for c, t in frame.dtypes.items()},
        "missing": {str(c): int(n) for c, n in frame.isna().sum().items()},
        "duplicates": int(frame.duplicated().sum()),
        "empty_columns": [str(c) for c in frame.columns if frame[c].isna().all()],
        "statistics": frame.describe(include="all").fillna("").to_dict(),
    }

def validate_target(frame: pd.DataFrame, target: str) -> None:
    """Raise a clear ValueError when the selected target cannot be used."""
    if target not in frame.columns:
        raise ValueError(f"Target column '{target}' does not exist.")
    if frame[target].isna().all():
        raise ValueError(f"Target column '{target}' contains only missing values.")
    if frame[target].nunique(dropna=True) < 2:
        raise ValueError(f"Target column '{target}' must contain at least two distinct values.")
