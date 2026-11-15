"""Reproducible DVC training stage for the bundled sample dataset."""
from __future__ import annotations
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
import yaml
from app.automl import train_all

params = yaml.safe_load((ROOT / "params.yaml").read_text())["training"]
frame_path = ROOT / "data" / "sample_classification.csv"
import pandas as pd
frame = pd.read_csv(frame_path)
result = train_all(frame, "purchased", [], "auto", params["test_size"], params["cv_folds"],
                   params["selection_metric_classification"], False, None,
                   str(ROOT / "models" / "dvc_runs"), params["random_state"])
best = next(x for x in result["models"] if x.get("id") == result["best_model_id"])
import shutil
Path(ROOT / "models").mkdir(exist_ok=True)
shutil.copy2(best["artifact_path"], ROOT / "models" / "dvc_best_model.joblib")
shutil.rmtree(ROOT / "models" / "dvc_runs", ignore_errors=True)
Path(ROOT / "reports").mkdir(exist_ok=True)
serializable = {k:v for k,v in result.items() if k != "models"}
serializable["models"] = [{k:v for k,v in item.items() if k not in {"artifact_path", "params"}} for item in result["models"]]
(ROOT / "reports" / "automl_results.json").write_text(json.dumps(serializable, indent=2))
print(f"Trained {len(result['models'])} models; selected {best['name']}.")
