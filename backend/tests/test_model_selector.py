import pandas as pd
from app.model_selector import select_best_model

def test_selector_uses_lower_is_better_and_saves_pipeline(tmp_path):
    from sklearn.dummy import DummyRegressor
    from sklearn.pipeline import Pipeline
    p1=Pipeline([("model",DummyRegressor())])
    p2=Pipeline([("model",DummyRegressor())])
    rows=[{"name":"a","pipeline":p1,"duration_seconds":1,"rmse":3},
          {"name":"b","pipeline":p2,"duration_seconds":2,"rmse":1}]
    selected=select_best_model(rows,metric="rmse",output_path=tmp_path/"best.joblib")
    assert selected["model"]=="b" and (tmp_path/"best.joblib").exists()
