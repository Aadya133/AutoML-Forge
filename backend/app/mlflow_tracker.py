"""Best-effort MLflow experiment tracking that never blocks local training."""
from __future__ import annotations
import logging
logger = logging.getLogger(__name__)

class MLflowTracker:
    def __init__(self, tracking_uri: str | None = None, experiment_name: str = "AutoML Forge"):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

    def log_run(self, run_name: str, params: dict, metrics: dict, artifact_path: str | None = None) -> dict:
        """Log a run; return status and error details instead of raising tracking failures."""
        try:
            import mlflow
            import mlflow.sklearn
            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            with mlflow.start_run(run_name=run_name) as run:
                for key, value in params.items():
                    if value is not None and isinstance(value, (str, int, float, bool)):
                        mlflow.log_param(str(key), value)
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(str(key), float(value))
                if artifact_path:
                    mlflow.log_artifact(artifact_path)
                return {"logged": True, "run_id": run.info.run_id, "error": None}
        except Exception as exc:
            logger.exception("MLflow tracking unavailable; continuing without tracking.")
            return {"logged": False, "run_id": None, "error": str(exc)}
