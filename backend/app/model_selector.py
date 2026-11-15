"""Metric-direction-aware best model selection and artifact persistence."""
from __future__ import annotations
from pathlib import Path
import joblib
import pandas as pd

_HIGHER = {"accuracy", "precision", "recall", "f1", "r2", "cv_score"}
_LOWER = {"mae", "mse", "rmse"}

def choose_best_result(results: list[dict], metric: str, *, higher_is_better: bool = True,
                       require_pipeline: bool = False) -> dict:
    """Choose a successful result using the requested metric direction."""
    candidates = [item for item in results
                  if not item.get("error")
                  and (not require_pipeline or item.get("pipeline") is not None)
                  and metric in item and pd.notna(item[metric])]
    if not candidates:
        raise ValueError("No successfully evaluated models are eligible for selection.")
    chooser = max if higher_is_better else min
    return chooser(candidates, key=lambda item: float(item[metric]))

def select_best_model(results: list[dict], *, metric: str, output_path: str | Path) -> dict:
    """Select from successful fitted pipelines and persist the entire pipeline."""
    if metric not in _HIGHER | _LOWER:
        raise ValueError(f"Unsupported selection metric: {metric}")
    best = choose_best_result(results, metric, higher_is_better=metric in _HIGHER, require_pipeline=True)
    path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["pipeline"], path)
    return {"model": best["name"], "metric": metric, "score": float(best[metric]),
            "artifact_path": str(path), "duration_seconds": float(best["duration_seconds"])}
