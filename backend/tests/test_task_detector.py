import pandas as pd
from app.task_detector import detect_task

def test_integer_class_labels_are_classification():
    assert detect_task(pd.Series([0,1,0,1,1])) == "classification"

def test_continuous_numeric_target_is_regression():
    assert detect_task(pd.Series(range(100))) == "regression"

def test_text_target_is_classification():
    assert detect_task(pd.Series(["low","high","medium"])) == "classification"
