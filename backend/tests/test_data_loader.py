import pytest
from app.data_loader import load_csv, DatasetLoadError

def test_load_valid_csv():
    frame = load_csv(b"x,y\n1,a\n2,b\n")
    assert frame.shape == (2, 2)

@pytest.mark.parametrize("raw", [b"", b"x,y\n", b"only\n1\n", b"x,y\n\"unterminated,2\n"])
def test_reject_invalid_csv(raw):
    with pytest.raises(DatasetLoadError):
        load_csv(raw)

def test_reject_non_utf8():
    with pytest.raises(DatasetLoadError, match="UTF-8"):
        load_csv(b"x,y\n1,\xff\n")
