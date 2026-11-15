import pandas as pd
from app.model_trainer import train_models
from app.model_evaluator import evaluate_models, save_comparison_plot

def test_evaluator_returns_metrics_and_duration(tmp_path):
    X = pd.DataFrame({"x":range(40)})
    y = pd.Series([0,1]*20)
    result = train_models(X,y,"classification")
    table = evaluate_models(result,X,y,"classification")
    assert len(table) == 4 and "f1" in table.columns
    chart = save_comparison_plot(table, tmp_path / "comparison.png", "f1")
    assert (tmp_path / "comparison.png").is_file()
