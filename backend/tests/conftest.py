"""Point the API at throwaway storage so tests never touch a developer database."""
import os
import tempfile
from pathlib import Path

_tmp = Path(tempfile.mkdtemp(prefix="automl-tests-"))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp / 'test.db'}")
os.environ.setdefault("UPLOAD_DIR", str(_tmp / "uploads"))
os.environ.setdefault("MODEL_DIR", str(_tmp / "models"))
os.environ.setdefault("MLFLOW_TRACKING_URI", (_tmp / "mlruns").as_uri())
os.environ.setdefault("JWT_SECRET", "test-secret-for-pytest-only-0123456789")
