"""Causal, partition-local price statistics for exit-memory observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import polars as pl


CAUSAL_PRICE_SCALE_COLUMN = "causal_mark_price_mean_abs_change_18"
CAUSAL_PRICE_SCALE_VALID_COLUMN = "causal_mark_price_scale_valid"
CAUSAL_PRICE_SCALE_ROW_COLUMN = "source_row_index"
CAUSAL_PRICE_SCALE_WINDOW = 18
CAUSAL_PRICE_SCALE_PRICE_SOURCE = "mark_price"
CAUSAL_PRICE_SCALE_PRICE_UNIT = "yuan_per_tonne"


@dataclass(frozen=True, slots=True)
class CausalPriceStatistics:
    scale: np.ndarray
    valid: np.ndarray
    window: int = CAUSAL_PRICE_SCALE_WINDOW
    price_source: str = CAUSAL_PRICE_SCALE_PRICE_SOURCE
    price_unit: str = CAUSAL_PRICE_SCALE_PRICE_UNIT

    @property
    def valid_start(self) -> int | None:
        indices = np.flatnonzero(self.valid)
        return int(indices[0]) if indices.size else None

    @property
    def valid_row_count(self) -> int:
        return int(np.count_nonzero(self.valid))

    def metadata(self, *, contract: str, stage: str, row_coordinates: np.ndarray) -> dict[str, Any]:
        coordinates = np.asarray(row_coordinates)
        if coordinates.ndim != 1 or coordinates.size != self.scale.size:
            raise ValueError("row_coordinates must align with causal price statistics")
        return {
            "schema": "causal_price_scale.v1",
            "window": self.window,
            "observations": self.window + 1,
            "price_source": self.price_source,
            "price_unit": self.price_unit,
            "contract": contract,
            "stage": stage,
            "row_count": int(self.scale.size),
            "valid_start": self.valid_start,
            "valid_row_count": self.valid_row_count,
            "row_coordinate_first": int(coordinates[0]) if coordinates.size else None,
            "row_coordinate_last": int(coordinates[-1]) if coordinates.size else None,
            "scale_column": CAUSAL_PRICE_SCALE_COLUMN,
            "valid_column": CAUSAL_PRICE_SCALE_VALID_COLUMN,
            "row_coordinate_column": CAUSAL_PRICE_SCALE_ROW_COLUMN,
        }


def _validate_prices_and_timestamps(prices: Any, timestamps: Any) -> tuple[np.ndarray, np.ndarray]:
    price_array = np.asarray(prices, dtype=np.float64).reshape(-1)
    timestamp_array = np.asarray(timestamps).reshape(-1)
    if price_array.size != timestamp_array.size:
        raise ValueError("mark_price and timestamp must have equal row counts")
    if not np.isfinite(price_array).all() or (price_array <= 0).any():
        raise ValueError("mark_price must be finite and strictly positive")
    if timestamp_array.size:
        timestamp_index = pd.Index(timestamp_array)
        if timestamp_index.has_duplicates or not timestamp_index.is_monotonic_increasing:
            raise ValueError("timestamps must be strictly increasing without duplicates")
    return price_array, timestamp_array


def prepare_causal_price_statistics(
    prices: Any,
    timestamps: Any,
    *,
    window: int = CAUSAL_PRICE_SCALE_WINDOW,
) -> CausalPriceStatistics:
    """Compute current-and-past-only absolute price changes for one partition.

    The first ``window`` physical rows remain present but are explicitly
    invalid and have a zero placeholder.  A constant complete window is
    valid with a zero statistic.
    """
    if window != CAUSAL_PRICE_SCALE_WINDOW:
        raise ValueError(f"window must be {CAUSAL_PRICE_SCALE_WINDOW}")
    price_array, _ = _validate_prices_and_timestamps(prices, timestamps)
    scales = np.zeros(price_array.size, dtype=np.float64)
    valid = np.zeros(price_array.size, dtype=bool)
    if price_array.size > window:
        changes = np.abs(np.diff(price_array))
        for row in range(window, price_array.size):
            scales[row] = float(changes[row - window : row].mean())
            valid[row] = True
    return CausalPriceStatistics(scale=scales, valid=valid)


def validate_causal_price_statistics(
    scale: Any,
    valid: Any,
    *,
    row_count: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the persisted auxiliary arrays without treating invalid rows as usable."""
    scale_array = np.asarray(scale, dtype=np.float64).reshape(-1)
    valid_array = np.asarray(valid, dtype=bool).reshape(-1)
    if scale_array.size != valid_array.size:
        raise ValueError("causal scale and valid arrays must have equal length")
    if row_count is not None and scale_array.size != row_count:
        raise ValueError("causal auxiliary arrays do not match row count")
    if not np.isfinite(scale_array).all() or (scale_array < 0).any():
        raise ValueError("causal price scales must be finite and non-negative")
    warmup = min(CAUSAL_PRICE_SCALE_WINDOW, scale_array.size)
    if valid_array[:warmup].any() or np.any(scale_array[:warmup] != 0):
        raise ValueError("causal warmup rows must be exactly zero and invalid")
    return scale_array, valid_array


def attach_causal_price_statistics(
    frame: pl.DataFrame,
    *,
    price_column: str = "mark_price",
    timestamp_column: str = "timestamp",
    row_coordinate_column: str | None = None,
) -> tuple[pl.DataFrame, CausalPriceStatistics]:
    """Return a frame with independent auxiliary columns and aligned coordinates."""
    missing = [column for column in (price_column, timestamp_column) if column not in frame.columns]
    if missing:
        raise ValueError(f"missing causal price input columns: {missing}")
    statistics = prepare_causal_price_statistics(
        frame.get_column(price_column).to_numpy(),
        frame.get_column(timestamp_column).to_numpy(),
    )
    if row_coordinate_column is not None:
        if row_coordinate_column not in frame.columns:
            raise ValueError(f"missing row coordinate column: {row_coordinate_column}")
        coordinates = frame.get_column(row_coordinate_column).to_numpy()
        if len(np.unique(coordinates)) != len(coordinates):
            raise ValueError("row coordinates must be unique")
    else:
        coordinates = np.arange(frame.height, dtype=np.int64)
    validate_causal_price_statistics(statistics.scale, statistics.valid, row_count=frame.height)
    result = frame.with_columns(
        pl.Series(CAUSAL_PRICE_SCALE_COLUMN, statistics.scale),
        pl.Series(CAUSAL_PRICE_SCALE_VALID_COLUMN, statistics.valid),
        pl.Series(CAUSAL_PRICE_SCALE_ROW_COLUMN, np.arange(frame.height, dtype=np.int64)),
    )
    return result, statistics
