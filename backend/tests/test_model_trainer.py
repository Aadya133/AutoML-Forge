import pandas as pd
from app.model_trainer import train_models

def test_required_classifiers_train_independently():
    X = pd.DataFrame({"x":range(40),"kind":["a","b"]*20})
    y = pd.Series([0,1]*20)
    results = train_models(X,y,"classification")
    assert len(results) == 4
    assert all(r["pipeline"] is not None for r in results)

def test_required_regressors_train_independently():
    X = pd.DataFrame({"x":range(40),"kind":["a","b"]*20})
    y = pd.Series([float(i)*1.5 for i in range(40)])
    results = train_models(X,y,"regression")
    assert len(results) == 4
    assert all(r["pipeline"] is not None for r in results)
