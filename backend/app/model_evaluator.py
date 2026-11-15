"""Evaluate fitted model pipelines and optionally save a comparison chart."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from .metrics import calculate_metrics

def evaluate_models(results: list[dict], X_test, y_test, task_type: str) -> pd.DataFrame:
    rows = []
    for item in results:
        row = {"model": item["name"], "duration_seconds": item["duration_seconds"], "error": item.get("error")}
        if item.get("error") is None and item.get("pipeline") is not None:
            try:
                row.update(calculate_metrics(y_test, item["pipeline"].predict(X_test), task_type))
            except Exception as exc:
                row["error"] = str(exc)
        rows.append(row)
    return pd.DataFrame(rows)

def save_comparison_plot(comparison: pd.DataFrame, output_path: str | Path, metric: str) -> str:
    """Save a metric comparison bar chart; matplotlib is imported only when requested."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if metric not in comparison:
        raise ValueError(f"Metric '{metric}' is not in comparison results.")
    valid = comparison.dropna(subset=[metric])
    if valid.empty:
        raise ValueError("No successful model metrics are available to plot.")
    path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(valid["model"], valid[metric]); ax.set_ylabel(metric); ax.tick_params(axis="x", rotation=25)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return str(path)
