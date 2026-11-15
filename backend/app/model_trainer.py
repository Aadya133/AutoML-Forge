"""Train the required baseline model families independently."""
from __future__ import annotations
import time
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from .preprocessing import build_preprocessor
from sklearn.pipeline import Pipeline

def candidate_estimators(task_type: str, random_state: int = 42) -> dict:
    if task_type == "classification":
        return {"Logistic Regression": LogisticRegression(max_iter=2000, random_state=random_state),
                "Decision Tree Classifier": DecisionTreeClassifier(random_state=random_state),
                "Random Forest Classifier": RandomForestClassifier(n_estimators=150, random_state=random_state, n_jobs=-1),
                "K-Nearest Neighbors Classifier": KNeighborsClassifier(n_neighbors=3)}
    if task_type == "regression":
        return {"Linear Regression": LinearRegression(),
                "Decision Tree Regressor": DecisionTreeRegressor(random_state=random_state),
                "Random Forest Regressor": RandomForestRegressor(n_estimators=150, random_state=random_state, n_jobs=-1),
                "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=random_state)}
    raise ValueError("task_type must be classification or regression")

def train_models(X_train, y_train, task_type: str, *, random_state: int = 42) -> list[dict]:
    """Fit each full preprocessing+estimator pipeline; one failure does not stop others."""
    results = []
    for name, estimator in candidate_estimators(task_type, random_state).items():
        started = time.perf_counter()
        try:
            pipeline = Pipeline([("preprocessor", build_preprocessor(X_train)), ("model", clone(estimator))])
            pipeline.fit(X_train, y_train)
            results.append({"name": name, "pipeline": pipeline,
                            "duration_seconds": time.perf_counter() - started, "error": None})
        except Exception as exc:
            results.append({"name": name, "pipeline": None,
                            "duration_seconds": time.perf_counter() - started, "error": str(exc)})
    return results
