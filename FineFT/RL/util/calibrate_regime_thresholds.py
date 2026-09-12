from __future__ import annotations

from typing import Any
import numpy as np

from datahandler.regime_calibration_engine import (
    calibrate_regime_thresholds as engine_calibrate_regime_thresholds,
)


def calibrate_regime_thresholds(
    slopes: np.ndarray,
    vols: np.ndarray,
) -> dict[str, Any]:
    """基于池化样本计算三分位数分档阈值，委托至 datahandler.regime_calibration_engine。"""
    return engine_calibrate_regime_thresholds(slopes, vols, dynamic_number=3)


def map_regime_grid_id(
    slope: float | np.ndarray,
    vol: float | np.ndarray,
    slope_thresholds: tuple[float, float] | list[float],
    vol_thresholds: tuple[float, float] | list[float],
) -> int | np.ndarray:
    """将单步或向量化 (slope, vol) 映射为 3x3 的 grid_id (0..8)。"""
    s_thresh = np.asarray(slope_thresholds, dtype=float)
    v_thresh = np.asarray(vol_thresholds, dtype=float)

    is_scalar = np.isscalar(slope) and np.isscalar(vol)
    vol_arr = np.asarray(vol, dtype=float)
    slope_arr = np.asarray(slope, dtype=float)

    vol_bin = np.searchsorted(v_thresh, vol_arr, side="right")
    slope_bin = np.searchsorted(s_thresh, slope_arr, side="right")
    grid_id = vol_bin * 3 + slope_bin

    if is_scalar:
        return int(grid_id.item())
    return grid_id


def ensure_regime_grid_id_column(df: Any) -> Any:
    """确保 DataFrame 包含 regime_grid_id 列；若缺失直接 Fail-fast 抛错。"""
    if "regime_grid_id" in df.columns:
        return df

    raise ValueError(
        "DataFrame lacks pre-computed 'regime_grid_id' column. "
        "Please run commodity_contract_dataset generation to materialize macro-segment regime labels."
    )
