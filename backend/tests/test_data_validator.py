import pandas as pd
import pytest
from app.data_validator import profile_dataset, validate_target

def test_profile_counts_and_missing():
    df = pd.DataFrame({"x":[1,1,None], "target":["a","a","b"]})
    result = profile_dataset(df)
    assert result["rows"] == 3 and result["duplicates"] == 1
    assert result["missing"]["x"] == 1

def test_validate_target_errors():
    with pytest.raises(ValueError): validate_target(pd.DataFrame({"x":[1]}), "missing")
    with pytest.raises(ValueError): validate_target(pd.DataFrame({"x":[None,None]}), "x")
