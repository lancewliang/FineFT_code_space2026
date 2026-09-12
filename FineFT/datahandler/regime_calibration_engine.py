from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from . import label_util as util
except ImportError:
    import label_util as util


@dataclass
class ContractSegments:
    """Represents market dynamic turning-point segments and extracted 2D scores for a contract."""

    contract: str
    turning_points: list[int]
    slopes: list[float]
    vols: list[float]
    row_positions: np.ndarray


def _point(value: Any) -> int:
    arr = np.asarray(value).reshape(-1)
    return int(arr[0])


def _finite_slopes(values: Any, contract: str) -> list[float]:
    slopes = [float(np.asarray(value).reshape(-1)[0]) for value in values]
    if not slopes or not np.isfinite(slopes).all():
        raise ValueError(f"{contract} produced non-finite segment slopes")
    return slopes


def _segment_log_return_volatility(prices: np.ndarray) -> float:
    if (prices <= 0.0).any():
        raise ValueError("prices contain non-positive values required by volatility calculation")
    if len(prices) < 2:
        return 0.0
    return float(np.std(np.diff(np.log(prices)), ddof=0) * 100.0)


def _label_for_score(score: float, thresholds: list[float]) -> int:
    for index, threshold in enumerate(thresholds):
        if score <= threshold:
            return index
    return len(thresholds)


def _limit_state_masks(
    prepared: pd.DataFrame, key_indicator: str
) -> tuple[np.ndarray, np.ndarray]:
    limit_up = np.zeros(len(prepared), dtype=bool)
    limit_down = np.zeros(len(prepared), dtype=bool)

    if "limit_up_single_sided_ratio" in prepared.columns:
        limit_up |= prepared["limit_up_single_sided_ratio"].to_numpy() > 0
    if "limit_down_single_sided_ratio" in prepared.columns:
        limit_down |= prepared["limit_down_single_sided_ratio"].to_numpy() > 0
    if "is_limit_up" in prepared.columns:
        limit_up |= prepared["is_limit_up"].fillna(False).to_numpy(dtype=bool)
    if "is_limit_down" in prepared.columns:
        limit_down |= prepared["is_limit_down"].fillna(False).to_numpy(dtype=bool)

    prices = prepared[key_indicator].to_numpy(dtype=float)
    if "UpperLimitPrice" in prepared.columns:
        upper_limits = pd.to_numeric(
            prepared["UpperLimitPrice"], errors="coerce"
        ).to_numpy(dtype=float)
        limit_up |= (
            np.isfinite(upper_limits)
            & (upper_limits > 0)
            & (prices >= upper_limits)
        )
    if "LowerLimitPrice" in prepared.columns:
        lower_limits = pd.to_numeric(
            prepared["LowerLimitPrice"], errors="coerce"
        ).to_numpy(dtype=float)
        limit_down |= (
            np.isfinite(lower_limits)
            & (lower_limits > 0)
            & (prices <= lower_limits)
        )
    return limit_up, limit_down


def extract_contract_segments(
    prepared: pd.DataFrame,
    contract: str,
    stage_root: Path,
    *,
    key_indicator: str = "mark_price",
    timestamp: str = "timestamp",
    tic: str = "symbol",
    filter_strength: int = 1,
    min_length_limit: int = 288,
    merging_threshold: float = 0.0003,
    merging_metric: str = "DTW_distance",
    merging_dynamic_constraint: int = 1,
    max_length_expectation: int = 864,
    filter_padlen: int = 15,
) -> ContractSegments:
    """Extract turning points, normalized slopes, and return volatilities for a contract."""
    if key_indicator not in prepared.columns:
        raise ValueError(f"{contract} missing required key_indicator column: {key_indicator}")

    prices = pd.to_numeric(prepared[key_indicator], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0.0).any():
        raise ValueError(f"{contract} contains non-finite or non-positive {key_indicator} values")

    if len(prepared) <= filter_padlen:
        raise ValueError(
            f"{contract} has insufficient rows for dynamic slicing: {len(prepared)} <= {filter_padlen}"
        )

    clean_data = prepared.copy()
    if timestamp == "index":
        clean_data[timestamp] = clean_data.index

    # Build worker-specific normalized data: prices normalized by first price per tic
    worker_data = clean_data.copy()
    first_tic = str(prepared[tic].iloc[0]) if tic in prepared.columns else contract
    if tic not in worker_data.columns:
        worker_data[tic] = contract

    for _, positions in worker_data.groupby(tic, sort=False).groups.items():
        first_price = worker_data.loc[positions[0], key_indicator]
        worker_data.loc[positions, key_indicator] = (
            worker_data.loc[positions, key_indicator] / first_price
        )

    worker_file = stage_root / f"worker_{contract}.feather"
    worker_file.parent.mkdir(parents=True, exist_ok=True)
    worker_data.to_feather(worker_file)

    worker = util.Worker(
        str(worker_file),
        "slice_and_merge",
        filter_strength=filter_strength,
        key_indicator=key_indicator,
        timestamp=timestamp,
        tic=tic,
        labeling_method="slope",
        min_length_limit=min_length_limit,
        merging_threshold=merging_threshold,
        merging_metric=merging_metric,
        merging_dynamic_constraint=merging_dynamic_constraint,
    )
    worker.fit(3, max_length_expectation, min_length_limit)

    worker_tic = worker.tics[0]
    points = [_point(val) for val in worker.turning_points_dict[worker_tic]]
    slopes = _finite_slopes(worker.norm_coef_list_dict[worker_tic], contract)

    if len(points) != len(slopes) + 1:
        raise ValueError(f"{contract} has inconsistent segment boundaries: {len(points)} vs {len(slopes)}")

    row_positions = np.flatnonzero(worker_data[tic].to_numpy() == worker_tic)
    if points[0] != 0 or points[-1] != len(row_positions):
        raise ValueError(f"{contract} segment boundaries do not cover all rows")

    vols = [
        _segment_log_return_volatility(
            clean_data.iloc[row_positions[start:end]][key_indicator].to_numpy(dtype=float)
        )
        for start, end in zip(points[:-1], points[1:])
    ]

    return ContractSegments(
        contract=contract,
        turning_points=points,
        slopes=slopes,
        vols=vols,
        row_positions=row_positions,
    )


def calibrate_regime_thresholds(
    pooled_slopes: list[float] | np.ndarray,
    pooled_vols: list[float] | np.ndarray,
    dynamic_number: int = 3,
) -> dict[str, Any]:
    """Calibrate tercile thresholds across pooled segment scores and assert invariants."""
    s_arr = np.asarray(pooled_slopes, dtype=float)
    v_arr = np.asarray(pooled_vols, dtype=float)

    if len(s_arr) == 0 or len(v_arr) == 0:
        raise ValueError("pooled slopes and vols cannot be empty")
    if len(s_arr) != len(v_arr):
        raise ValueError(f"pooled slopes ({len(s_arr)}) and vols ({len(v_arr)}) length mismatch")

    quantiles = [i / dynamic_number for i in range(1, dynamic_number)]
    q_slope = np.quantile(s_arr, quantiles)
    q_vol = np.quantile(v_arr, quantiles)

    s0, s1 = float(q_slope[0]), float(q_slope[1])
    v0, v1 = float(q_vol[0]), float(q_vol[1])

    if s0 >= 0.0:
        raise ValueError(
            f"Invalid slope calibration: lower slope threshold T_slope[0]={s0:.6f} "
            "must be strictly negative (< 0.0) to ensure slope_bin=0 isolates negative trends"
        )
    if s1 <= 0.0:
        raise ValueError(
            f"Invalid slope calibration: upper slope threshold T_slope[1]={s1:.6f} "
            "must be strictly positive (> 0.0) to ensure slope_bin=2 isolates positive trends"
        )
    if v0 <= 0.0 or v1 <= v0:
        raise ValueError(f"Invalid volatility thresholds: v0={v0:.6f}, v1={v1:.6f}")

    return {
        "slope_thresholds": [s0, s1],
        "vol_thresholds": [v0, v1],
        "dynamic_number": dynamic_number,
        "sample_count": int(len(s_arr)),
        "invariant_checked": True,
    }


def apply_regime_labels_to_dataframe(
    prepared: pd.DataFrame,
    segments: ContractSegments,
    slope_thresholds: list[float],
    vol_thresholds: list[float],
    key_indicator: str = "mark_price",
) -> pd.DataFrame:
    """Annotate DataFrame rows with slope_label, volatility_label, and regime_grid_id."""
    df = prepared.copy()
    n_rows = len(df)
    row_slopes = np.full(n_rows, -1, dtype=np.int64)
    row_vols = np.full(n_rows, -1, dtype=np.int64)
    row_grids = np.full(n_rows, -1, dtype=np.int64)

    for start, end, slope, vol in zip(
        segments.turning_points[:-1],
        segments.turning_points[1:],
        segments.slopes,
        segments.vols,
    ):
        s_label = _label_for_score(slope, slope_thresholds)
        v_label = _label_for_score(vol, vol_thresholds)
        grid_id = v_label * 3 + s_label

        idx_range = segments.row_positions[start:end]
        row_slopes[idx_range] = s_label
        row_vols[idx_range] = v_label
        row_grids[idx_range] = grid_id

    limit_up, limit_down = _limit_state_masks(df, key_indicator)
    # Project conventions: limit up -> max slope & high vol; limit down -> min slope & high vol
    row_slopes[limit_up] = 2
    row_vols[limit_up] = 2
    row_grids[limit_up] = 8

    row_slopes[limit_down] = 0
    row_vols[limit_down] = 2
    row_grids[limit_down] = 6

    if (row_grids < 0).any():
        raise ValueError(f"{segments.contract} has unaccounted rows during regime labeling")

    df["slope_label"] = row_slopes
    df["volatility_label"] = row_vols
    df["regime_grid_id"] = row_grids
    return df
