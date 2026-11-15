from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_api_root_is_available():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"


def test_health_checks_database():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["database"] == "connected"


def test_public_health_alias():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["service"] == "AutoML Forge API"


def test_predict_rejects_empty_rows():
    response = client.post("/predict", json={"rows": []})
    assert response.status_code == 404 or response.status_code == 400
