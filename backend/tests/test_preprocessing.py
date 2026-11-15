import pandas as pd
from app.preprocessing import prepare_data

def test_split_and_preprocessor_handle_missing_and_unseen_category():
    df = pd.DataFrame({"x":[1,2,None,4,5,6,7,8,9,10,11,12], "city":["a","b"]*6,
                       "target":[0,1]*6})
    prepared = prepare_data(df, "target", test_size=.25)
    prepared.preprocessor.fit(prepared.X_train)
    transformed = prepared.preprocessor.transform(pd.DataFrame({"x":[None],"city":["unseen"]}))
    assert transformed.shape[0] == 1
    assert prepared.task_type == "classification"
