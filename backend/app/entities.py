from sqlalchemy import Boolean, Column, Integer, String, Text, DateTime, Float, ForeignKey
from sqlalchemy.sql import func
from .db import Base


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    password_hash = Column(Text, nullable=False)
    role = Column(String(30), nullable=False, default="user")
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    filename = Column(String(255), nullable=False)
    path = Column(Text, nullable=False)
    rows = Column(Integer, default=0)
    columns = Column(Integer, default=0)
    owner = Column(String(255), default="demo@automl.local")
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    version = Column(Integer, nullable=False)
    checksum = Column(String(64), nullable=False)
    path = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Experiment(Base):
    __tablename__ = "experiments"
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    target = Column(String(255), nullable=False)
    task_type = Column(String(30), nullable=False)
    status = Column(String(30), default="queued")
    config = Column(Text, default="{}")
    best_model_id = Column(Integer, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class ModelRecord(Base):
    __tablename__ = "models"
    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False)
    name = Column(String(255), nullable=False)
    algorithm = Column(String(100), nullable=False)
    task_type = Column(String(30), nullable=False)
    metrics = Column(Text, default="{}")
    cv_score = Column(Float, nullable=True)
    duration_seconds = Column(Float, default=0)
    artifact_path = Column(Text, nullable=False)
    feature_names = Column(Text, default="[]")
    target_name = Column(String(255), nullable=False)
    mlflow_run_id = Column(String(100), nullable=True)
    registered_version = Column(String(30), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    version = Column(Integer, nullable=False)
    artifact_path = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
