import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str = "sqlite:///./automl.db"
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "AutoML Forge"
    upload_dir: str = "./storage/uploads"
    model_dir: str = "./storage/models"
    max_upload_mb: int = 25
    cors_origins: list[str] = ["http://localhost:5173"]
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
Path(settings.model_dir).mkdir(parents=True, exist_ok=True)
