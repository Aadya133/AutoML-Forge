"""Leakage-safe train/test splitting and reusable preprocessing pipelines."""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .data_validator import validate_target
from .task_detector import detect_task

@dataclass
class PreparedData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer
    task_type: str
    feature_names: list[str]

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create a transformer; it must be fitted only on training data."""
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    if not transformers:
        raise ValueError("No usable feature columns remain.")
    return ColumnTransformer(transformers, remainder="drop")

def prepare_data(frame: pd.DataFrame, target: str, *, task_type: str = "auto",
                 excluded_columns: list[str] | None = None, test_size: float = .2,
                 random_state: int = 42) -> PreparedData:
    """Clean duplicate/missing-target rows and split before any learned transforms."""
    validate_target(frame, target)
    if not 0.05 < test_size < 0.5:
        raise ValueError("test_size must be greater than 0.05 and less than 0.5.")
    clean = frame.drop_duplicates().dropna(subset=[target]).copy()
    excluded = set(excluded_columns or []) - {target}
    X = clean.drop(columns=[target, *[c for c in excluded if c in clean.columns]])
    y = clean[target]
    if X.empty:
        raise ValueError("No feature columns remain after exclusions.")
    task = detect_task(y) if task_type == "auto" else task_type
    if task not in {"classification", "regression"}:
        raise ValueError("task_type must be auto, classification or regression.")
    if task == "regression" and not pd.api.types.is_numeric_dtype(y):
        raise ValueError("Regression requires a numeric target.")
    stratify = None
    if task == "classification":
        counts = y.value_counts()
        if counts.min() >= 2 and int(len(y) * test_size) >= len(counts):
            stratify = y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify)
    return PreparedData(X_train, X_test, y_train, y_test,
                        build_preprocessor(X_train), task, list(X.columns))
