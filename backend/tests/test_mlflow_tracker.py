import sys
from app.mlflow_tracker import MLflowTracker

def test_tracking_failure_does_not_raise_when_mlflow_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlflow", None)
    result = MLflowTracker().log_run("test", {"p": 1}, {"score": 0.5})
    assert result["logged"] is False
    assert result["error"]
