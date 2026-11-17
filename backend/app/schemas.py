from pydantic import BaseModel, Field
from typing import Any

class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(LoginRequest):
    name: str = Field(min_length=2, max_length=255)

class ExperimentRequest(BaseModel):
    dataset_id: int
    target: str
    excluded_columns: list[str] = Field(default_factory=list)
    task_type: str = "auto"
    test_size: float = Field(default=0.2, gt=0.05, lt=0.5)
    cv_folds: int = Field(default=5, ge=2, le=10)
    random_state: int = Field(default=42, ge=0, le=2147483647)
    selection_metric: str | None = None
    tune: bool = False
    algorithms: list[str] | None = None
    feature_selection: bool = False
    balance_classes: bool = False
    ensemble: bool = True

class PredictionRequest(BaseModel):
    rows: list[dict[str, Any]]

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict
