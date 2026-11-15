import numpy as np
from app.metrics import calculate_metrics

def test_classification_metrics_support_multiple_classes():
    values = calculate_metrics([0,1,2,2],[0,2,2,2],"classification")
    assert set(values) == {"accuracy","precision","recall","f1"}
    assert all(np.isfinite(v) for v in values.values())

def test_regression_metrics():
    values = calculate_metrics([1,2,3],[1,2,4],"regression")
    assert values["mae"] > 0 and values["rmse"] > 0
