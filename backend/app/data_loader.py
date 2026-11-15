"""Safe CSV loading helpers used by the API, CLI and tests."""
from __future__ import annotations
from pathlib import Path
from typing import BinaryIO
import io
import pandas as pd

class DatasetLoadError(ValueError):
    """Raised when an uploaded CSV cannot be loaded safely."""

def load_csv(source: str | Path | bytes | BinaryIO, *, max_bytes: int | None = None) -> pd.DataFrame:
    """Load a non-empty CSV, validating size, encoding, shape and column names."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists() or not path.is_file():
            raise DatasetLoadError(f"CSV file does not exist: {path}")
        if path.stat().st_size == 0:
            raise DatasetLoadError("The uploaded CSV file is empty.")
        if max_bytes is not None and path.stat().st_size > max_bytes:
            raise DatasetLoadError(f"CSV exceeds the {max_bytes}-byte upload limit.")
        handle = path
    else:
        raw = source if isinstance(source, bytes) else source.read()
        if not raw:
            raise DatasetLoadError("The uploaded CSV file is empty.")
        if max_bytes is not None and len(raw) > max_bytes:
            raise DatasetLoadError(f"CSV exceeds the {max_bytes}-byte upload limit.")
        handle = io.BytesIO(raw)
    try:
        frame = pd.read_csv(handle, encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DatasetLoadError("CSV encoding is not UTF-8. Save the file as UTF-8 and retry.") from exc
    except pd.errors.EmptyDataError as exc:
        raise DatasetLoadError("CSV contains no tabular data or column headers.") from exc
    except pd.errors.ParserError as exc:
        raise DatasetLoadError(f"Invalid CSV structure: {exc}") from exc
    except (OSError, ValueError) as exc:
        raise DatasetLoadError(f"Unable to read CSV: {exc}") from exc
    if frame.empty:
        raise DatasetLoadError("CSV contains headers but no data rows.")
    if len(frame.columns) < 2:
        raise DatasetLoadError("CSV must contain at least one feature column and one target column.")
    if frame.columns.has_duplicates:
        raise DatasetLoadError("CSV contains duplicate column names. Rename them and retry.")
    if any(not str(c).strip() for c in frame.columns):
        raise DatasetLoadError("CSV contains an empty column name.")
    return frame
