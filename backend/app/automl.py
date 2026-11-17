import time
import uuid
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold,
    cross_val_score, train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectPercentile, f_classif, f_regression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
try:
    import optuna
except ImportError:
    optuna = None


from .task_detector import detect_task as _detect_task
from .preprocessing import build_preprocessor
from .metrics import calculate_metrics
from .model_trainer import candidate_estimators
from .model_selector import choose_best_result
from .model_evaluator import evaluate_models

RANDOM_STATE = 42

CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1),
    "Extra Trees": ExtraTreesClassifier(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),
    "Support Vector Machine": SVC(probability=True, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}
REGRESSORS = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso": Lasso(max_iter=5000),
    "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
    "Random Forest": RandomForestRegressor(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1),
    "Extra Trees": ExtraTreesRegressor(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1),
    "HistGradientBoosting": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
    "Support Vector Regressor": SVR(),
    "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
}


def detect_task(y: pd.Series) -> str:
    """Delegate task inference to the shared Member 1 task detector."""
    return _detect_task(y)


def make_pipeline(X: pd.DataFrame, estimator) -> Pipeline:
    """Compose the shared leakage-safe preprocessor with a candidate estimator."""
    return Pipeline([("preprocessor", build_preprocessor(X)), ("model", estimator)])


def make_enhanced_pipeline(X, estimator, task, feature_selection=False):
    steps = [("preprocessor", build_preprocessor(X))]
    if feature_selection:
        steps.append(("feature_selection", SelectPercentile(
            score_func=f_classif if task == "classification" else f_regression,
            percentile=80,
        )))
    steps.append(("model", estimator))
    return Pipeline(steps)


def _cv_strategy(task, y_train, requested_folds, random_state=RANDOM_STATE):
    if task == "classification":
        counts = y_train.value_counts()
        smallest_class = int(counts.min())
        if smallest_class >= 2:
            return StratifiedKFold(
                n_splits=min(requested_folds, smallest_class),
                shuffle=True,
                random_state=random_state,
            )
    return KFold(
        n_splits=min(requested_folds, len(y_train)),
        shuffle=True,
        random_state=random_state,
    )


def train_all(
    df, target, excluded, task_type, test_size, cv_folds,
    selection_metric, tune, algorithms, model_dir, random_state=RANDOM_STATE,
    feature_selection=False, balance_classes=False,
):
    if target not in df.columns:
        raise ValueError("Target column does not exist.")
    if not 0.05 < float(test_size) < 0.5:
        raise ValueError("test_size must be greater than 0.05 and less than 0.5.")
    if int(cv_folds) < 2:
        raise ValueError("At least two cross-validation folds are required.")

    clean = df.drop_duplicates().dropna(subset=[target]).copy()
    if clean.empty:
        raise ValueError("No usable rows remain after removing missing target values.")
    y = clean[target]
    excluded_columns = set(excluded or [])
    excluded_columns.discard(target)
    X = clean.drop(columns=[target, *[c for c in excluded_columns if c in clean.columns]])
    if X.empty:
        raise ValueError("No feature columns remain after exclusions.")
    if len(clean) < 20:
        raise ValueError("At least 20 usable rows are required for training.")

    task = detect_task(y) if task_type == "auto" else task_type
    if task not in ("classification", "regression"):
        raise ValueError("Task must be classification or regression.")
    if task == "classification" and y.nunique() < 2:
        raise ValueError("Classification requires at least two target classes.")
    if task == "regression" and not pd.api.types.is_numeric_dtype(y):
        raise ValueError("Regression requires a numeric target column.")

    stratify = None
    if task == "classification":
        counts = y.value_counts()
        # Stratification needs at least two examples per class and enough test rows.
        if counts.min() >= 2 and int(np.ceil(len(y) * test_size)) >= len(counts):
            stratify = y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    if len(X_train) < 2:
        raise ValueError("Not enough training rows for cross-validation.")
    if task == "classification" and y_train.nunique() < 2:
        raise ValueError("Training split contains only one target class; use more data.")
    cv = _cv_strategy(task, y_train, cv_folds, random_state)

    if task == "classification":
        models = CLASSIFIERS.copy()
        baselines = candidate_estimators(task)
        models.update({
            "Logistic Regression": baselines["Logistic Regression"],
            "Decision Tree": baselines["Decision Tree Classifier"],
            "Random Forest": baselines["Random Forest Classifier"],
            "K-Nearest Neighbors": baselines["K-Nearest Neighbors Classifier"],
        })
        default_metric = "f1_weighted"
        scoring = {
            "accuracy": "accuracy",
            "precision_weighted": "precision_weighted",
            "recall_weighted": "recall_weighted",
            "f1_weighted": "f1_weighted",
            "roc_auc_ovr_weighted": "roc_auc_ovr_weighted",
        }
    else:
        models = REGRESSORS.copy()
        baselines = candidate_estimators(task)
        models.update({
            "Linear Regression": baselines["Linear Regression"],
            "Decision Tree": baselines["Decision Tree Regressor"],
            "Random Forest": baselines["Random Forest Regressor"],
            "Gradient Boosting": baselines["Gradient Boosting Regressor"],
        })
        default_metric = "neg_root_mean_squared_error"
        scoring = {
            "neg_root_mean_squared_error": "neg_root_mean_squared_error",
            "r2": "r2",
            "neg_mean_absolute_error": "neg_mean_absolute_error",
        }

    metric = selection_metric or default_metric
    if metric not in scoring:
        raise ValueError(f"Unsupported selection metric '{metric}' for {task}.")
    selected = list(algorithms) if algorithms else list(models)
    unknown = set(selected) - set(models)
    if unknown:
        raise ValueError(f"Unsupported algorithms: {', '.join(sorted(unknown))}")
    if not selected:
        raise ValueError("Select at least one algorithm.")

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    results = []
    for name in selected:
        started = time.perf_counter()
        try:
            estimator = clone(models[name])
            if "random_state" in estimator.get_params(deep=False):
                estimator.set_params(random_state=random_state)
            if balance_classes and task == "classification" and "class_weight" in estimator.get_params(deep=False):
                estimator.set_params(class_weight="balanced")
            pipe = make_enhanced_pipeline(X_train, estimator, task, feature_selection)
            if tune and name in ("Random Forest", "Extra Trees"):
                if optuna is not None:
                    def objective(trial):
                        params = {
                            "model__n_estimators": trial.suggest_int("n_estimators", 100, 250, step=50),
                            "model__max_depth": trial.suggest_int("max_depth", 4, 20),
                            "model__min_samples_split": trial.suggest_int("min_samples_split", 2, 8),
                        }
                        candidate = clone(pipe).set_params(**params)
                        return float(np.mean(cross_val_score(
                            candidate, X_train, y_train, cv=cv, scoring=metric, error_score="raise"
                        )))
                    study = optuna.create_study(direction="maximize",
                                                sampler=optuna.samplers.TPESampler(seed=random_state))
                    study.optimize(objective, n_trials=6, show_progress_bar=False)
                    params_out = {f"model__{key}": value for key, value in study.best_params.items()}
                    fitted = clone(pipe).set_params(**params_out).fit(X_train, y_train)
                    cv_score = float(study.best_value)
                else:
                    search_space = {
                        "model__n_estimators": [100, 150, 200, 300],
                        "model__max_depth": [None, 5, 10, 20],
                        "model__min_samples_split": [2, 4, 8],
                    }
                    search = RandomizedSearchCV(
                        pipe, search_space, n_iter=8, cv=cv, scoring=metric,
                        n_jobs=-1, random_state=random_state, error_score="raise",
                    )
                    search.fit(X_train, y_train)
                    fitted = search.best_estimator_
                    params_out = search.best_params_
                    cv_score = float(search.best_score_)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    scores = cross_val_score(
                        pipe, X_train, y_train, cv=cv, scoring=metric,
                        n_jobs=None, error_score="raise",
                    )
                cv_score = float(np.mean(scores))
                fitted = pipe.fit(X_train, y_train)
                params_out = fitted.named_steps["model"].get_params()

            duration = time.perf_counter() - started
            comparison = evaluate_models(
                [{"name": name, "pipeline": fitted, "duration_seconds": duration, "error": None}],
                X_test, y_test, task,
            )
            evaluated = comparison.iloc[0].to_dict()
            metric_names = ("accuracy", "precision", "recall", "f1") if task == "classification" else ("r2", "mae", "mse", "rmse")
            metrics = {key: float(evaluated[key]) for key in metric_names if key in evaluated and pd.notna(evaluated[key])}
            predictions = fitted.predict(X_test)
            if task == "classification":
                metrics.update({
                    "confusion_matrix": confusion_matrix(y_test, predictions, labels=fitted.classes_).tolist(),
                    "labels": [str(value) for value in fitted.classes_],
                })
                if hasattr(fitted, "predict_proba") and len(fitted.classes_) == 2:
                    try:
                        positive_index = list(fitted.classes_).index(fitted.classes_[1])
                        probabilities = fitted.predict_proba(X_test)[:, positive_index]
                        metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities))
                        from sklearn.metrics import roc_curve
                        fpr, tpr, thresholds = roc_curve(y_test, probabilities)
                        metrics["roc_curve"] = {
                            "fpr": fpr.tolist(), "tpr": tpr.tolist(),
                            "thresholds": thresholds.tolist(),
                        }
                    except (ValueError, IndexError):
                        pass
            else:
                metrics["residuals"] = {
                    "actual": y_test.tolist()[:200],
                    "predicted": predictions.tolist()[:200],
                    "residual": (y_test.to_numpy() - predictions).tolist()[:200],
                }

            model_id = uuid.uuid4().hex[:12]
            artifact_path = str(Path(model_dir) / f"{model_id}.joblib")
            joblib.dump(fitted, artifact_path)
            results.append({
                "id": model_id, "name": name, "algorithm": name,
                "metrics": metrics, "cv_score": cv_score,
                "duration_seconds": duration, "artifact_path": artifact_path,
                "feature_names": list(X.columns), "target_name": target,
                "params": str(params_out),
            })
        except Exception as exc:
            results.append({"name": name, "algorithm": name, "error": str(exc)})

    eligible = [item for item in results if "error" not in item]
    if not eligible:
        errors = "; ".join(f"{item['algorithm']}: {item['error']}" for item in results)
        raise RuntimeError(f"Every model failed. {errors}")
    # sklearn scorers are oriented so higher is always better, including negated losses.
    best = choose_best_result(eligible, "cv_score", higher_is_better=True)
    return {
        "task_type": task,
        "selection_metric": metric,
        "best_model_id": best["id"],
        "models": results,
        "rows_used": len(clean),
        "features": list(X.columns),
        "target": target,
    }
