from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np


def compute_causal_rolling_slope_and_volatility(
    prices: np.ndarray,
    window: int = 48,
) -> tuple[np.ndarray, np.ndarray]:
    """计算单条价格序列的纯历史因果滚动 OLS 斜率与对数收益率波动率。

    斜率单位：% 对数价格变化 / bar。
    波动率单位：% 对数收益率总体标准差（ddof=0）。
    """
    if len(prices) < window:
        raise ValueError(
            f"price length ({len(prices)}) is smaller than rolling window ({window})"
        )
    if np.any(prices <= 0.0) or not np.all(np.isfinite(prices)):
        raise ValueError("prices must be strictly positive and finite")

    log_p = np.log(prices.astype(float))
    centered_steps = np.arange(window, dtype=float) - (window - 1) / 2.0
    denom = float(np.square(centered_steps).sum())

    windows = np.lib.stride_tricks.sliding_window_view(log_p, window)
    slopes = (windows @ centered_steps) / denom * 100.0

    log_rets = np.diff(log_p)
    ret_windows = np.lib.stride_tricks.sliding_window_view(log_rets, window - 1)
    vols = ret_windows.std(axis=1, ddof=0) * 100.0

    return slopes, vols


def calibrate_regime_thresholds(
    slopes: np.ndarray,
    vols: np.ndarray,
) -> dict[str, Any]:
    """基于池化样本计算三分位数分档阈值，并严格检验斜率下界必须为负。"""
    if len(slopes) == 0 or len(vols) == 0:
        raise ValueError("slopes and vols cannot be empty")
    if len(slopes) != len(vols):
        raise ValueError(f"slopes ({len(slopes)}) and vols ({len(vols)}) length mismatch")

    q_slope = np.quantile(slopes, [1.0 / 3.0, 2.0 / 3.0])
    q_vol = np.quantile(vols, [1.0 / 3.0, 2.0 / 3.0])

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
        "method": "pooled_terciles",
        "sample_count": int(len(slopes)),
        "invariant_checked": True,
    }


def calibrate_from_slice_files(
    slice_paths: list[Path | str],
    window: int = 48,
    price_col: str = "close",
) -> dict[str, Any]:
    """遍历训练切片计算并返回全局分位数 Manifest。"""
    import polars as pl

    all_slopes = []
    all_vols = []

    for path in slice_paths:
        df = pl.read_ipc(str(path))
        prices = df[price_col].to_numpy().astype(float)
        if len(prices) < window:
            continue
        slopes, vols = compute_causal_rolling_slope_and_volatility(prices, window=window)
        all_slopes.append(slopes)
        all_vols.append(vols)

    if not all_slopes:
        raise ValueError(f"no valid slices found with length >= {window}")

    pooled_slopes = np.concatenate(all_slopes)
    pooled_vols = np.concatenate(all_vols)

    manifest = calibrate_regime_thresholds(pooled_slopes, pooled_vols)
    manifest["window"] = window
    manifest["price_col"] = price_col
    manifest["num_slices"] = len(slice_paths)
    return manifest


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


def ensure_regime_grid_id_column(
    df: Any,
    manifest: dict[str, Any] | None = None,
    price_col: str = "close",
) -> Any:
    """确保 DataFrame 包含 regime_grid_id 列；若缺失则依据 manifest 因果计算注入。"""
    if "regime_grid_id" in df.columns:
        return df

    if manifest is None:
        raise ValueError("manifest is required when df lacks 'regime_grid_id' column")

    window = int(manifest["window"])
    s_thresh = manifest["slope_thresholds"]
    v_thresh = manifest["vol_thresholds"]

    prices = df[price_col].to_numpy().astype(float)
    n_rows = len(prices)
    grid_ids = np.full(n_rows, -1, dtype=np.int64)

    if n_rows >= window:
        slopes, vols = compute_causal_rolling_slope_and_volatility(prices, window=window)
        mapped_ids = map_regime_grid_id(slopes, vols, s_thresh, v_thresh)
        grid_ids[window - 1 :] = mapped_ids

    df["regime_grid_id"] = grid_ids
    return df


def apply_regime_grid_ids_to_slice_files(
    slice_paths: list[Path | str],
    manifest: dict[str, Any],
    price_col: str = "close",
) -> int:
    """遍历切片文件并将 regime_grid_id 列原地物化写回文件。"""
    import pandas as pd

    count = 0
    for path in slice_paths:
        df = pd.read_feather(str(path))
        df = ensure_regime_grid_id_column(df, manifest=manifest, price_col=price_col)
        df.to_feather(str(path))
        count += 1
    return count


def cli_main() -> None:
    import argparse
    import glob
    import json
    import os

    parser = argparse.ArgumentParser(
        description="Calibrate regime thresholds and materialize regime_grid_id on slices."
    )
    parser.add_argument(
        "--slice_dir",
        type=str,
        required=True,
        help="Directory containing train slice df_*.feather files",
    )
    parser.add_argument(
        "--output_manifest",
        type=str,
        default=None,
        help="Output path for regime_thresholds.json",
    )
    parser.add_argument(
        "--price_col",
        type=str,
        default="close",
        help="Price column name",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=48,
        help="Rolling window size",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Materialize regime_grid_id into slice feather files on disk",
    )
    args = parser.parse_args()

    slice_paths = sorted(glob.glob(os.path.join(args.slice_dir, "df_*.feather")))
    if not slice_paths:
        raise FileNotFoundError(f"No df_*.feather slices found in {args.slice_dir}")

    manifest = calibrate_from_slice_files(
        slice_paths, window=args.window, price_col=args.price_col
    )
    manifest_path = args.output_manifest or os.path.join(
        os.path.dirname(args.slice_dir.rstrip("/")), "regime_thresholds.json"
    )
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved regime thresholds to {manifest_path}")

    if args.apply:
        applied_count = apply_regime_grid_ids_to_slice_files(
            slice_paths, manifest=manifest, price_col=args.price_col
        )
        print(
            f"Materialized regime_grid_id into {applied_count} slice files in {args.slice_dir}"
        )


if __name__ == "__main__":
    cli_main()
