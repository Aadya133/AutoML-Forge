import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from app.automl import build_voting_ensemble, make_pipeline
from app.main import app, explain_model, permutation_explanation

client = TestClient(app)


def classification_frame(rows=80):
    rng = np.random.default_rng(0)
    age = rng.integers(18, 70, rows)
    income = rng.integers(20000, 120000, rows)
    city = rng.choice(["Agra", "Delhi", "Pune"], rows)
    target = np.where(income + age * 500 > 80000, "yes", "no")
    return pd.DataFrame({"age": age, "income": income, "city": city, "purchased": target})


def regression_frame(rows=80):
    rng = np.random.default_rng(1)
    size = rng.integers(400, 3000, rows)
    rooms = rng.integers(1, 6, rows)
    price = size * 100 + rooms * 5000 + rng.normal(0, 1000, rows)
    return pd.DataFrame({"size": size, "rooms": rooms, "price": price})


@pytest.fixture(scope="module")
def token():
    response = client.post("/api/auth/login", json={"email": "demo@automl.local", "password": "demo123"})
    return response.json()["access_token"]


def headers(token):
    return {"Authorization": f"Bearer {token}"}


def upload(frame, token, name="data.csv"):
    buffer = io.BytesIO(frame.to_csv(index=False).encode())
    response = client.post("/api/datasets", files={"file": (name, buffer, "text/csv")}, headers=headers(token))
    assert response.status_code == 200, response.text
    return response.json()["id"]


def train(dataset_id, target, token, algorithms):
    # TestClient runs background tasks before returning, so training is finished here.
    response = client.post("/api/experiments", headers=headers(token), json={
        "dataset_id": dataset_id, "target": target, "cv_folds": 3, "algorithms": algorithms})
    assert response.status_code == 200, response.text
    experiment = client.get(f"/api/experiments/{response.json()['id']}", headers=headers(token)).json()
    assert experiment["status"] == "completed", experiment
    assert experiment["progress"] == 100
    return experiment


def test_tree_model_explanation_uses_importances():
    frame = classification_frame()
    X, y = frame.drop(columns="purchased"), frame["purchased"]
    pipe = make_pipeline(X, RandomForestClassifier(n_estimators=20, random_state=0)).fit(X, y)
    result = explain_model(pipe, X, list(X.columns))
    assert result["method"] == "tree importance"
    names = [item["feature"] for item in result["feature_importance"]]
    assert "income" in names
    assert not any("__" in name for name in names)
    values = [item["importance"] for item in result["feature_importance"]]
    assert values == sorted(values, reverse=True)


def test_linear_model_explanation_uses_coefficients():
    frame = classification_frame()
    X, y = frame.drop(columns="purchased"), frame["purchased"]
    pipe = make_pipeline(X, LogisticRegression(max_iter=500)).fit(X, y)
    result = explain_model(pipe, X, list(X.columns))
    assert result["method"] == "model coefficients"
    assert all(item["importance"] >= 0 for item in result["feature_importance"])


def test_ensemble_falls_back_to_permutation_importance():
    frame = classification_frame()
    X, y = frame.drop(columns="purchased"), frame["purchased"]
    members = [make_pipeline(X, LogisticRegression(max_iter=500)),
               make_pipeline(X, RandomForestClassifier(n_estimators=20, random_state=0))]
    voter = build_voting_ensemble(members, "classification").fit(X, y)
    assert explain_model(voter, X, list(X.columns))["method"] == "unavailable"
    result = permutation_explanation(voter, frame, "purchased", "classification", list(X.columns))
    assert result["method"] == "permutation importance"
    assert {item["feature"] for item in result["feature_importance"]} == {"age", "income", "city"}


def test_explain_requires_auth_and_existing_model(token):
    assert client.get("/api/models/1/explain").status_code == 401
    assert client.get("/api/models/999999/explain", headers=headers(token)).status_code == 404


def test_classification_explain_report_and_history(token):
    dataset_id = upload(classification_frame(), token, "customers.csv")
    detail = client.get(f"/api/datasets/{dataset_id}", headers=headers(token)).json()
    assert detail["quality_score"] == 100
    assert detail["quality_label"] == "good"
    assert detail["version"] == 1

    experiment = train(dataset_id, "purchased", token, ["Logistic Regression", "Random Forest"])
    models = client.get(f"/api/experiments/{experiment['id']}/models", headers=headers(token)).json()
    assert {m["algorithm"] for m in models} == {"Logistic Regression", "Random Forest", "Voting Ensemble"}

    for model in models:
        info = client.get(f"/api/models/{model['id']}/explain", headers=headers(token))
        assert info.status_code == 200
        body = info.json()
        assert body["feature_importance"], model["algorithm"]
        assert body["plots"]["labels"] == ["no", "yes"]
        assert len(body["plots"]["confusion_matrix"]) == 2
        assert body["plots"]["roc_curve"]["fpr"][0] == 0
        assert "accuracy" in body["metrics"]

    ensemble = next(m for m in models if m["algorithm"] == "Voting Ensemble")
    report = client.get(f"/api/models/{ensemble['id']}/report", headers=headers(token))
    assert report.status_code == 200
    assert "attachment" in report.headers["content-disposition"]
    assert "Evaluation report: Voting Ensemble" in report.text
    assert "Confusion matrix" in report.text

    row = {"age": 40, "income": 90000, "city": "Delhi"}
    prediction = client.post(f"/api/models/{ensemble['id']}/predict", json={"rows": [row]}, headers=headers(token))
    assert prediction.status_code == 200
    assert prediction.json()["predictions"][0] in ("yes", "no")
    history = client.get(f"/api/predictions?model_id={ensemble['id']}", headers=headers(token)).json()
    assert history[0]["inputs"] == [row]

    filtered = client.get("/api/models?algorithm=Voting%20Ensemble", headers=headers(token)).json()
    assert filtered and all(m["algorithm"] == "Voting Ensemble" for m in filtered)

    analytics = client.get("/api/analytics", headers=headers(token)).json()
    assert any(d["id"] == dataset_id and d["quality_score"] == 100 for d in analytics["datasets"])
    assert analytics["experiments_by_status"]["completed"] >= 1
    assert analytics["predictions"] >= 1


def test_regression_explain_has_residuals(token):
    dataset_id = upload(regression_frame(), token, "houses.csv")
    experiment = train(dataset_id, "price", token, ["Linear Regression", "Decision Tree"])
    models = client.get(f"/api/experiments/{experiment['id']}/models", headers=headers(token)).json()
    assert "Voting Ensemble" in {m["algorithm"] for m in models}
    for model in models:
        body = client.get(f"/api/models/{model['id']}/explain", headers=headers(token)).json()
        residuals = body["plots"]["residual_plot"]
        assert len(residuals["actual"]) == len(residuals["predicted"]) == len(residuals["residual"])
        assert body["plots"]["roc_curve"] is None
        assert {item["feature"] for item in body["feature_importance"]} <= {"size", "rooms"}


def test_training_survives_unreachable_mlflow(token, monkeypatch):
    import app.main as main
    if main.mlflow is None:
        pytest.skip("mlflow not installed")

    def boom(*args, **kwargs):
        raise RuntimeError("tracking server down")

    monkeypatch.setattr(main.mlflow, "set_experiment", boom)
    dataset_id = upload(classification_frame(), token, "mlflow_down.csv")
    experiment = train(dataset_id, "purchased", token, ["Logistic Regression", "Decision Tree"])
    assert experiment["best_model_id"] is not None
