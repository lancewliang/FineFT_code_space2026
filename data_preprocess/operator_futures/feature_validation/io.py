from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

import pandas as pd


COMMODITY_FEATURE_ROOT = "PREPROCESS_DATASET/commodity-futures"


def commodity_root(root_path: Path) -> Path:
    return root_path / COMMODITY_FEATURE_ROOT


def date_range_exclusive(start_date: str, end_date: str) -> Iterator[str]:
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    while current < end:
        yield current.strftime("%Y-%m-%d")
        current += timedelta(days=1)


def read_feather_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_feather(path)


def write_feather_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_feather(path)


def read_state_features(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    import numpy as np

    values = np.load(path, allow_pickle=True)
    return [str(value) for value in values.tolist()]


def read_csv_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path, index_col=0)
