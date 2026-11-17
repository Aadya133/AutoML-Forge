from datetime import datetime, timedelta, timezone
import uuid

import jwt
from fastapi.testclient import TestClient

from app.auth import hash_password, verify_password
from app.core import settings
from app.db import SessionLocal
from app.entities import User
from app.main import app

client = TestClient(app)


def unique_email():
    return f"user-{uuid.uuid4().hex[:8]}@example.com"


def register(email=None, password="secret123", name="Test User"):
    return client.post("/api/auth/register", json={"email": email or unique_email(),
                                                   "password": password, "name": name})


def auth(token):
    return {"Authorization": f"Bearer {token}"}


def test_password_hash_roundtrip():
    encoded = hash_password("s3cret")
    assert encoded.startswith("pbkdf2_sha256$")
    assert "s3cret" not in encoded
    assert verify_password("s3cret", encoded)
    assert not verify_password("wrong", encoded)


def test_same_password_gets_different_salt():
    assert hash_password("same") != hash_password("same")


def test_verify_rejects_malformed_hash():
    assert not verify_password("anything", "not-a-valid-hash")
    assert not verify_password("anything", "md5$1$abc$def")


def test_demo_login_returns_token():
    response = client.post("/api/auth/login", json={"email": "demo@automl.local", "password": "demo123"})
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["user"]["role"] == "admin"
    me = client.get("/api/auth/me", headers=auth(body["access_token"]))
    assert me.status_code == 200
    assert me.json()["email"] == "demo@automl.local"


def test_login_email_is_case_insensitive():
    response = client.post("/api/auth/login", json={"email": "DEMO@automl.local", "password": "demo123"})
    assert response.status_code == 200


def test_login_rejects_wrong_password():
    response = client.post("/api/auth/login", json={"email": "demo@automl.local", "password": "nope"})
    assert response.status_code == 401


def test_login_rejects_unknown_user():
    response = client.post("/api/auth/login", json={"email": unique_email(), "password": "demo123"})
    assert response.status_code == 401


def test_register_then_login():
    email = unique_email()
    created = register(email)
    assert created.status_code == 200
    assert created.json()["user"]["role"] == "user"
    login = client.post("/api/auth/login", json={"email": email, "password": "secret123"})
    assert login.status_code == 200


def test_register_rejects_duplicate_email():
    email = unique_email()
    assert register(email).status_code == 200
    assert register(email).status_code == 409


def test_register_validates_name():
    assert register(name="x").status_code == 422


def test_protected_routes_require_token():
    for path in ("/api/datasets", "/api/experiments", "/api/models", "/api/dashboard",
                 "/api/analytics", "/api/predictions", "/api/auth/me"):
        assert client.get(path).status_code == 401, path


def test_invalid_token_is_rejected():
    assert client.get("/api/auth/me", headers=auth("garbage.token.value")).status_code == 401


def test_token_signed_with_other_secret_is_rejected():
    token = jwt.encode({"sub": "1", "exp": datetime.now(timezone.utc) + timedelta(minutes=5)},
                       "some-other-secret-that-is-long-enough", algorithm="HS256")
    assert client.get("/api/auth/me", headers=auth(token)).status_code == 401


def test_expired_token_is_rejected():
    token = jwt.encode({"sub": "1", "exp": datetime.now(timezone.utc) - timedelta(minutes=1)},
                       settings.jwt_secret, algorithm="HS256")
    assert client.get("/api/auth/me", headers=auth(token)).status_code == 401


def test_inactive_user_is_rejected():
    email = unique_email()
    token = register(email).json()["access_token"]
    db = SessionLocal()
    try:
        user = db.query(User).filter_by(email=email).first()
        user.is_active = False
        db.commit()
    finally:
        db.close()
    assert client.get("/api/auth/me", headers=auth(token)).status_code == 401


def test_admin_route_requires_admin_role():
    user_token = register().json()["access_token"]
    assert client.get("/api/admin/users", headers=auth(user_token)).status_code == 403
    admin_token = client.post("/api/auth/login", json={"email": "demo@automl.local",
                                                       "password": "demo123"}).json()["access_token"]
    response = client.get("/api/admin/users", headers=auth(admin_token))
    assert response.status_code == 200
    assert any(u["email"] == "demo@automl.local" for u in response.json())
