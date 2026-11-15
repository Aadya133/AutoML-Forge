import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from app.automl import detect_task, make_pipeline, train_all


def test_detect_classification():
    assert detect_task(pd.Series([0, 1, 0, 1, 1])) == "classification"


def test_detect_regression():
    assert detect_task(pd.Series(range(100))) == "regression"


def test_pipeline_handles_missing_and_categories():
    X = pd.DataFrame({"age": [20, 30, None, 40], "city": ["A", "B", "A", None]})
    pipe = make_pipeline(X, LogisticRegression())
    pipe.fit(X, [0, 1, 0, 1])
    assert len(pipe.predict(X)) == 4


def test_pipeline_handles_all_missing_feature():
    X = pd.DataFrame({"age": [20, 30, 40, 50], "empty": [None] * 4})
    pipe = make_pipeline(X, LogisticRegression())
    pipe.fit(X, [0, 1, 0, 1])
    assert len(pipe.predict(X)) == 4


def test_train_all_returns_model_and_metrics(tmp_path):
    frame = pd.DataFrame({
        "age": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
        "city": ["A", "B", "A"] * 10,
        "target": [0, 1] * 15,
    })
    result = train_all(
        frame, "target", [], "auto", 0.2, 3, "f1_weighted",
        False, ["Logistic Regression"], str(tmp_path),
    )
    assert result["task_type"] == "classification"
    assert len(result["models"]) == 1
    assert "f1" in result["models"][0]["metrics"]
    assert (tmp_path / f"{result['best_model_id']}.joblib").exists()


def test_rejects_missing_target(tmp_path):
    with pytest.raises(ValueError, match="Target column"):
        train_all(pd.DataFrame({"x": [1] * 20}), "missing", [], "auto",
                  0.2, 3, None, False, None, str(tmp_path))
