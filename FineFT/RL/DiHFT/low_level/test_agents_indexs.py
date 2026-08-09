"""Isolated Agent evaluation and pattern-analysis entry point.

This module intentionally does not import the legacy ``test_agent_index``
entry point or consume its output files.  It produces the complete isolated
step-detail, window, classifier, summary, and manifest artifact set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch


_FINEFT_ROOT = Path(__file__).resolve().parents[3]
if str(_FINEFT_ROOT) not in sys.path:
    sys.path.insert(0, str(_FINEFT_ROOT))

from env.env_class.futures_util import map_action_to_position_leverage
from env.env_initiate.base_initiate import initiate_base_env
from model.low_level import ensemble_Qnet


EPOCH_PATTERN = re.compile(r"^epoch_(\d+)$")
LABEL_PATTERN = re.compile(r"^label_\d+$")

STEP_DETAIL_COLUMNS = [
    "epoch",
    "label",
    "contract",
    "df_path",
    "initial_action",
    "bin_index",
    "timestep",
    "timestamp",
    "close",
    "volume",
    "mark_price",
    "action",
    "target_position",
    "target_leverage",
    "position_before",
    "leverage_before",
    "position_after",
    "leverage_after",
    "action_change_step",
    "trade_count_step",
    "cumulative_action_change_count",
    "cumulative_trade_count",
    "step_reward",
    "realized_pnl_step",
    "cumulative_realized_pnl",
    "commission_fee_step",
    "cumulative_commission_fee",
    "slippage_step",
    "cumulative_slippage",
    "wallet_balance",
    "unrealized_pnl",
    "margin_balance",
    "notional_asset_value",
    "cash_balance",
    "total_value",
]

COVERAGE_COLUMNS = [
    "record_type",
    "epoch",
    "label",
    "bin_index",
    "contract",
    "df_path",
    "initial_action",
    "expected_count",
    "observed_count",
    "coverage_ratio",
    "status",
    "window_count",
    "dropped_tail_steps",
    "dropped_tail_gross_pnl",
    "dropped_tail_net_pnl",
    "message",
]

WINDOW_COLUMNS = [
    "label",
    "epoch",
    "bin_index",
    "contract",
    "df_path",
    "initial_action",
    "window_index",
    "start_timestep",
    "end_timestep",
    "start_timestamp",
    "end_timestamp",
    "step_count",
    "window_id",
    "kline_patterns",
    "strategy_patterns",
    "realized_pnl_sum",
    "unrealized_pnl_before_start",
    "unrealized_pnl_end",
    "commission_fee_sum",
    "slippage_sum",
    "gross_pnl",
    "net_pnl",
]

EXPANDED_COLUMNS = [
    "label",
    "epoch",
    "bin_index",
    "contract",
    "df_path",
    "initial_action",
    "window_index",
    "start_timestep",
    "end_timestep",
    "window_id",
    "kline_pattern",
    "strategy_pattern",
    "gross_pnl",
    "net_pnl",
]

DEFAULT_THRESHOLDS = {
    "breakout_ratio": 0.003,
    "extension_ratio": 0.005,
    "breakout_hold_ratio": 0.8,
    "retrace_ratio": 0.003,
    "retrace_band_ratio": 0.005,
    "acceleration_ratio": 1.5,
    "cumulative_return_ratio": 0.005,
    "z_extreme": 2.0,
    "min_leg_return": 0.005,
    "min_leg_steps": 3,
    "return_std_threshold": 0.003,
    "outer_band_ratio": 0.005,
    "touch_band_ratio": 0.0025,
    "min_band_transitions": 4,
    "volume_drop_ratio": 0.2,
    "min_price_move": 0.005,
    "near_full_ratio": 0.8,
    "min_hold_steps": 10,
}

REQUIRED_MARKET_COLUMNS = ("contract", "volume", "mark_price")


def build_position_levels(max_holding_number: float, position_choices: int) -> list[float]:
    """Return the ordered signed position grid used by the environment."""

    if position_choices < 3 or position_choices % 2 == 0:
        raise ValueError("position_choices must be an odd integer >= 3")
    if not np.isfinite(max_holding_number) or max_holding_number <= 0:
        raise ValueError("max_holding_number must be finite and positive")
    side_count = (position_choices - 1) // 2
    step = float(max_holding_number) / side_count
    levels = [-step * i for i in range(side_count, 0, -1)]
    levels.append(0.0)
    levels.extend(step * i for i in range(1, side_count + 1))
    return levels


def discover_model_epochs(model_root: Path) -> list[tuple[int, Path]]:
    """Discover direct epoch directories that contain a trained model."""

    model_root = Path(model_root)
    if not model_root.is_dir():
        raise FileNotFoundError(f"model_root does not exist: {model_root}")
    epochs: list[tuple[int, Path]] = []
    for child in model_root.iterdir():
        match = EPOCH_PATTERN.fullmatch(child.name)
        if match and child.is_dir() and (child / "trained_model.pkl").is_file():
            epochs.append((int(match.group(1)), child))
    return sorted(epochs, key=lambda item: item[0])


def discover_missing_model_epochs(model_root: Path) -> list[int]:
    """Return direct epoch directory numbers without a trained model."""

    model_root = Path(model_root)
    missing: list[int] = []
    for child in model_root.iterdir():
        match = EPOCH_PATTERN.fullmatch(child.name)
        if match and child.is_dir() and not (child / "trained_model.pkl").is_file():
            missing.append(int(match.group(1)))
    return sorted(missing)


def discover_validation_files(valid_root: Path) -> list[dict[str, str]]:
    """Discover contract/Label validation files in deterministic order."""

    valid_root = Path(valid_root)
    if not valid_root.is_dir():
        raise FileNotFoundError(f"valid data root does not exist: {valid_root}")
    entries: list[dict[str, str]] = []
    for contract_dir in sorted(valid_root.iterdir(), key=lambda path: path.name):
        if not contract_dir.is_dir() or contract_dir.name == "processed":
            continue
        for label_dir in sorted(contract_dir.iterdir(), key=lambda path: path.name):
            if not label_dir.is_dir() or not LABEL_PATTERN.fullmatch(label_dir.name):
                continue
            for data_file in sorted(label_dir.glob("df_*.feather"), key=lambda path: path.name):
                entries.append(
                    {
                        "contract": contract_dir.name,
                        "label": label_dir.name,
                        "df_path": data_file.relative_to(valid_root).as_posix(),
                        "abs_path": str(data_file),
                    }
                )
    if not entries:
        raise FileNotFoundError(
            f"no validation files found under {valid_root}; expected contract/label_*/df_*.feather"
        )
    return entries


def validate_required_market_columns(data_frame: pd.DataFrame) -> None:
    """Validate the raw market fields required by the new evaluator."""

    missing = [column for column in REQUIRED_MARKET_COLUMNS if column not in data_frame.columns]
    if missing:
        raise ValueError(f"validation data missing required columns: {missing}")
    for column in ("volume", "mark_price"):
        values = pd.to_numeric(data_frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"validation data column {column} contains non-finite values")
    if data_frame["contract"].isna().any() or (data_frame["contract"].astype(str).str.len() == 0).any():
        raise ValueError("validation data column contract contains empty values")


def _number(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"non-finite execution value: {value!r}")
    return numeric


def build_step_detail_row(
    *,
    epoch: int,
    label: str,
    contract: str,
    df_path: str,
    initial_action: int,
    bin_index: int,
    timestep: int,
    market_row: Mapping[str, Any],
    action: int,
    target_position: float,
    target_leverage: float,
    position_before: float,
    leverage_before: float,
    position_after: float,
    leverage_after: float,
    step_reward: float,
    info: Mapping[str, Any],
    wallet_balance: float,
    unrealized_pnl: float,
    action_change_step: int,
    trade_count_step: int,
    cumulative_action_change_count: int,
    cumulative_trade_count: int,
) -> dict[str, Any]:
    """Build one stable English Detail row from public evaluator values."""

    if timestep < 0:
        raise ValueError("timestep must be non-negative")
    if not contract:
        raise ValueError("contract must not be empty")
    if "volume" not in market_row or "mark_price" not in market_row:
        raise ValueError("market row must contain raw volume and mark_price")
    mark_price = _number(market_row["mark_price"])
    volume = _number(market_row["volume"])
    wallet = _number(wallet_balance)
    unrealized = _number(unrealized_pnl)
    row = {
        "epoch": int(epoch),
        "label": str(label),
        "contract": str(contract),
        "df_path": str(df_path),
        "initial_action": int(initial_action),
        "bin_index": int(bin_index),
        "timestep": int(timestep),
        "timestamp": market_row.get("timestamp"),
        "close": market_row.get("close"),
        "volume": volume,
        "mark_price": mark_price,
        "action": int(action),
        "target_position": _number(target_position),
        "target_leverage": _number(target_leverage),
        "position_before": _number(position_before),
        "leverage_before": _number(leverage_before),
        "position_after": _number(position_after),
        "leverage_after": _number(leverage_after),
        "action_change_step": int(action_change_step),
        "trade_count_step": int(trade_count_step),
        "cumulative_action_change_count": int(cumulative_action_change_count),
        "cumulative_trade_count": int(cumulative_trade_count),
        "step_reward": _number(step_reward),
        "realized_pnl_step": _number(info.get("realized_pnl_step")),
        "cumulative_realized_pnl": _number(info.get("cumulative_realized_pnl")),
        "commission_fee_step": _number(info.get("commission_fee_step")),
        "cumulative_commission_fee": _number(info.get("cumulative_commission_fee")),
        "slippage_step": _number(info.get("slippage_step")),
        "cumulative_slippage": _number(info.get("cumulative_slippage")),
        "wallet_balance": wallet,
        "unrealized_pnl": unrealized,
        "margin_balance": wallet + unrealized,
        "notional_asset_value": mark_price * _number(position_after),
        "cash_balance": wallet,
        "total_value": wallet + unrealized,
    }
    return {column: row[column] for column in STEP_DETAIL_COLUMNS}


def write_step_detail_csv(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    """Write one epoch's Detail rows with a stable English schema."""

    frame = pd.DataFrame(rows, columns=STEP_DETAIL_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8")


def write_coverage_report(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    """Write deterministic epoch and trajectory coverage records."""

    frame = pd.DataFrame(rows, columns=COVERAGE_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8")


def _canonical_window_identity(window: Mapping[str, Any]) -> str:
    identity = {
        key: window[key]
        for key in (
            "label",
            "epoch",
            "bin_index",
            "contract",
            "df_path",
            "initial_action",
            "window_index",
            "start_timestep",
            "end_timestep",
        )
    }
    payload = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _trajectory_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["epoch"]),
        str(row["label"]),
        int(row["bin_index"]),
        str(row["contract"]),
        str(row["df_path"]),
        int(row["initial_action"]),
    )


def _validated_trajectory(
    rows: Sequence[Mapping[str, Any]], *, require_zero_start: bool = True
) -> list[dict[str, Any]]:
    ordered = sorted((dict(row) for row in rows), key=lambda row: int(row["timestep"]))
    if not ordered:
        raise ValueError("trajectory must contain at least one step")
    start = 0 if require_zero_start else int(ordered[0]["timestep"])
    expected = list(range(start, start + len(ordered)))
    actual = [int(row["timestep"]) for row in ordered]
    if actual != expected:
        raise ValueError(f"trajectory timestep must be 0-based and continuous: {actual}")
    return ordered


def _window_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    window_index: int,
    kline_pattern: str,
    strategy_patterns: Sequence[str],
) -> dict[str, Any]:
    ordered = _validated_trajectory(rows, require_zero_start=False)
    first = ordered[0]
    last = ordered[-1]
    start_timestep = int(first["timestep"])
    end_timestep = int(last["timestep"])
    realized_sum = float(sum(_number(row.get("realized_pnl_step")) for row in ordered))
    commission_sum = float(sum(_number(row.get("commission_fee_step")) for row in ordered))
    slippage_sum = float(sum(_number(row.get("slippage_step")) for row in ordered))
    unrealized_before = 0.0 if start_timestep == 0 else _number(first.get("unrealized_pnl_before_start"))
    if start_timestep != 0 and "unrealized_pnl_before_start" not in first:
        raise ValueError("non-initial window requires unrealized_pnl_before_start")
    unrealized_end = _number(last.get("unrealized_pnl"))
    window = {
        "label": str(first["label"]),
        "epoch": int(first["epoch"]),
        "bin_index": int(first["bin_index"]),
        "contract": str(first["contract"]),
        "df_path": str(first["df_path"]),
        "initial_action": int(first["initial_action"]),
        "window_index": int(window_index),
        "start_timestep": start_timestep,
        "end_timestep": end_timestep,
        "start_timestamp": first.get("timestamp"),
        "end_timestamp": last.get("timestamp"),
        "step_count": len(ordered),
        "kline_patterns": json.dumps([kline_pattern], ensure_ascii=False),
        "strategy_patterns": json.dumps(list(strategy_patterns), ensure_ascii=False),
        "realized_pnl_sum": realized_sum,
        "unrealized_pnl_before_start": unrealized_before,
        "unrealized_pnl_end": unrealized_end,
        "commission_fee_sum": commission_sum,
        "slippage_sum": slippage_sum,
        "gross_pnl": realized_sum + unrealized_end - unrealized_before,
        "net_pnl": realized_sum + unrealized_end - unrealized_before - commission_sum,
    }
    window["window_id"] = _canonical_window_identity(window)
    return {column: window[column] for column in WINDOW_COLUMNS}


def build_window_rows(
    detail_rows: Sequence[Mapping[str, Any]],
    *,
    position_levels: Sequence[float] = (-1.0, 0.0, 1.0),
    thresholds: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Build deterministic Window rows from step facts.

    Classifier rules are added in later slices; the initial slice preserves
    every window with an explicit diagnostic sentinel.
    """

    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in detail_rows:
        grouped.setdefault(_trajectory_key(row), []).append(row)
    windows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        trajectory = _validated_trajectory(grouped[key])
        label = str(trajectory[0]["label"])
        if label in {"label_0", "label_6"}:
            chunks = [trajectory]
        else:
            chunks = [trajectory[start : start + 20] for start in range(0, len(trajectory), 20)]
            chunks = [chunk for chunk in chunks if len(chunk) == 20]
        for window_index, chunk in enumerate(chunks):
            chunk = [dict(row) for row in chunk]
            if window_index > 0:
                start = window_index * 20
                chunk[0]["unrealized_pnl_before_start"] = trajectory[start - 1].get(
                    "unrealized_pnl", 0.0
                )
            kline_pattern = classify_kline_pattern(
                chunk,
                label=label,
                thresholds=thresholds,
            )
            strategy_patterns = classify_strategy_patterns(
                chunk,
                label=label,
                kline_pattern=kline_pattern,
                position_levels=position_levels,
                thresholds=thresholds,
            )
            windows.append(
                _window_from_rows(
                    chunk,
                    window_index=window_index,
                    kline_pattern=kline_pattern,
                    strategy_patterns=strategy_patterns,
                )
            )
    return windows


def expand_window_rows(window_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Expand JSON pattern arrays without changing Window rows."""

    expanded: list[dict[str, Any]] = []
    for window in window_rows:
        kline_patterns = json.loads(str(window["kline_patterns"]))
        strategy_patterns = json.loads(str(window["strategy_patterns"]))
        if not isinstance(kline_patterns, list) or not isinstance(strategy_patterns, list):
            raise ValueError("pattern fields must contain JSON arrays")
        for kline_pattern in kline_patterns:
            for strategy_pattern in strategy_patterns:
                row = {
                    "label": window["label"],
                    "epoch": window["epoch"],
                    "bin_index": window["bin_index"],
                    "contract": window["contract"],
                    "df_path": window["df_path"],
                    "initial_action": window["initial_action"],
                    "window_index": window["window_index"],
                    "start_timestep": window["start_timestep"],
                    "end_timestep": window["end_timestep"],
                    "window_id": window["window_id"],
                    "kline_pattern": kline_pattern,
                    "strategy_pattern": strategy_pattern,
                    "gross_pnl": window["gross_pnl"],
                    "net_pnl": window["net_pnl"],
                }
                expanded.append({column: row[column] for column in EXPANDED_COLUMNS})
    return expanded


def write_window_table(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    pd.DataFrame(rows, columns=WINDOW_COLUMNS).to_csv(output_path, index=False, encoding="utf-8")


def write_expanded_table(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    pd.DataFrame(rows, columns=EXPANDED_COLUMNS).to_csv(output_path, index=False, encoding="utf-8")


DIAGNOSTIC_COLUMNS = [
    "scope",
    "label",
    "epoch",
    "bin_index",
    "pattern_axis",
    "kline_pattern",
    "strategy_pattern",
    "is_unclassified",
    "window_count",
    "window_ratio",
    "total_net_pnl",
    "pnl_p25",
    "pnl_p50",
    "pnl_p75",
    "pnl_median_range",
    "warning_code",
    "warning_message",
]

SCENARIO_KLINE_COLUMNS = [
    "label", "epoch", "bin_index", "contract", "df_path", "initial_action",
    "kline_pattern", "total_net_pnl", "window_count", "pnl_p25", "pnl_p50", "pnl_p75",
]
SCENARIO_STRATEGY_COLUMNS = [
    "label", "epoch", "bin_index", "contract", "df_path", "initial_action",
    "strategy_pattern", "total_net_pnl", "window_count", "pnl_p25", "pnl_p50", "pnl_p75",
]
SCENARIO_CROSS_COLUMNS = [
    "label", "epoch", "bin_index", "contract", "df_path", "initial_action",
    "kline_pattern", "strategy_pattern", "total_net_pnl", "window_count", "pnl_p25", "pnl_p50", "pnl_p75",
]
TRIPLE_KLINE_COLUMNS = [
    "label", "epoch", "bin_index", "kline_pattern", "mean_initial_action_total_net_pnl",
    "mean_initial_action_window_count", "mean_initial_action_pnl_p25", "mean_initial_action_pnl_p50",
    "mean_initial_action_pnl_p75", "observed_initial_action_count", "expected_initial_action_count",
    "initial_action_coverage_ratio",
]
TRIPLE_STRATEGY_COLUMNS = [
    "label", "epoch", "bin_index", "strategy_pattern", "mean_initial_action_total_net_pnl",
    "mean_initial_action_window_count", "mean_initial_action_pnl_p25", "mean_initial_action_pnl_p50",
    "mean_initial_action_pnl_p75", "observed_initial_action_count", "expected_initial_action_count",
    "initial_action_coverage_ratio",
]
TRIPLE_CROSS_COLUMNS = [
    "label", "epoch", "bin_index", "kline_pattern", "strategy_pattern",
    "mean_initial_action_total_net_pnl", "mean_initial_action_window_count",
    "mean_initial_action_pnl_p25", "mean_initial_action_pnl_p50", "mean_initial_action_pnl_p75",
    "observed_initial_action_count", "expected_initial_action_count", "initial_action_coverage_ratio",
]

UNCLASSIFIED_PATTERNS = {"未分类", "策略未分类"}


def _percentiles(values: Sequence[float]) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    result = np.percentile(np.asarray(values, dtype=float), [25, 50, 75])
    return float(result[0]), float(result[1]), float(result[2])


def _scenario_rows(
    expanded_rows: Sequence[Mapping[str, Any]],
    *,
    axis: str,
) -> list[dict[str, Any]]:
    if axis == "kline":
        pattern_key = ("kline_pattern",)
        pattern_columns = ("kline_pattern",)
        output_columns = SCENARIO_KLINE_COLUMNS
    elif axis == "strategy":
        pattern_key = ("strategy_pattern",)
        pattern_columns = ("strategy_pattern",)
        output_columns = SCENARIO_STRATEGY_COLUMNS
    elif axis == "cross":
        pattern_key = ("kline_pattern", "strategy_pattern")
        pattern_columns = pattern_key
        output_columns = SCENARIO_CROSS_COLUMNS
    else:
        raise ValueError(f"unknown summary axis: {axis}")
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in expanded_rows:
        if any(str(row[column]) in UNCLASSIFIED_PATTERNS for column in pattern_columns):
            continue
        key = (
            str(row["label"]), int(row["epoch"]), int(row["bin_index"]),
            str(row["contract"]), str(row["df_path"]), int(row["initial_action"]),
            *(str(row[column]) for column in pattern_key),
        )
        grouped.setdefault(key, []).append(row)
    rows: list[dict[str, Any]] = []
    for key in sorted(grouped, key=str):
        values = grouped[key]
        p25, p50, p75 = _percentiles([float(row["net_pnl"]) for row in values])
        row: dict[str, Any] = {
            "label": key[0], "epoch": key[1], "bin_index": key[2], "contract": key[3],
            "df_path": key[4], "initial_action": key[5],
            "total_net_pnl": float(sum(float(item["net_pnl"]) for item in values)),
            "window_count": len(values), "pnl_p25": p25, "pnl_p50": p50, "pnl_p75": p75,
        }
        offset = 6
        for index, column in enumerate(pattern_key):
            row[column] = key[offset + index]
        rows.append({column: row[column] for column in output_columns})
    return rows


def _triple_rows(
    scenario_rows: Sequence[Mapping[str, Any]],
    all_expanded_rows: Sequence[Mapping[str, Any]],
    *,
    axis: str,
) -> list[dict[str, Any]]:
    if axis == "kline":
        pattern_columns = ("kline_pattern",)
        output_columns = TRIPLE_KLINE_COLUMNS
    elif axis == "strategy":
        pattern_columns = ("strategy_pattern",)
        output_columns = TRIPLE_STRATEGY_COLUMNS
    elif axis == "cross":
        pattern_columns = ("kline_pattern", "strategy_pattern")
        output_columns = TRIPLE_CROSS_COLUMNS
    else:
        raise ValueError(f"unknown summary axis: {axis}")
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    expected: dict[tuple[Any, ...], set[tuple[Any, ...]]] = {}
    universe: dict[tuple[Any, ...], set[tuple[Any, ...]]] = {}
    for row in all_expanded_rows:
        base = (str(row["label"]), int(row["epoch"]), int(row["bin_index"]))
        scenario = (str(row["contract"]), str(row["df_path"]), int(row["initial_action"]))
        universe.setdefault(base, set()).add(scenario)
        pattern = tuple(str(row[column]) for column in pattern_columns)
        if not any(str(row[column]) in UNCLASSIFIED_PATTERNS for column in pattern_columns):
            expected.setdefault(base + pattern, set()).add(scenario)
    for row in scenario_rows:
        base = (str(row["label"]), int(row["epoch"]), int(row["bin_index"]))
        pattern = tuple(str(row[column]) for column in pattern_columns)
        grouped.setdefault(base + pattern, []).append(row)
    rows: list[dict[str, Any]] = []
    for key in sorted(grouped, key=str):
        values = grouped[key]
        p25 = float(np.mean([float(row["pnl_p25"]) for row in values]))
        p50 = float(np.mean([float(row["pnl_p50"]) for row in values]))
        p75 = float(np.mean([float(row["pnl_p75"]) for row in values]))
        observed = len({(row["contract"], row["df_path"], int(row["initial_action"])) for row in values})
        expected_count = len(universe.get(key[:3], set()))
        row: dict[str, Any] = {
            "label": key[0], "epoch": key[1], "bin_index": key[2],
            "mean_initial_action_total_net_pnl": float(np.mean([float(item["total_net_pnl"]) for item in values])),
            "mean_initial_action_window_count": float(np.mean([float(item["window_count"]) for item in values])),
            "mean_initial_action_pnl_p25": p25,
            "mean_initial_action_pnl_p50": p50,
            "mean_initial_action_pnl_p75": p75,
            "observed_initial_action_count": observed,
            "expected_initial_action_count": expected_count,
            "initial_action_coverage_ratio": observed / expected_count if expected_count else 0.0,
        }
        for index, column in enumerate(pattern_columns):
            row[column] = key[3 + index]
        rows.append({column: row[column] for column in output_columns})
    return rows


def build_diagnostics(window_rows: Sequence[Mapping[str, Any]], expanded_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    scope_specs: list[tuple[str, Callable[[Mapping[str, Any]], bool], str, Any]] = [("overall", lambda row: True, "", "")]
    scope_specs.extend(("label", lambda row, value=value: str(row["label"]) == value, value, "") for value in sorted({str(row["label"]) for row in window_rows}))
    scope_specs.extend(("epoch", lambda row, value=value: int(row["epoch"]) == value, "", value) for value in sorted({int(row["epoch"]) for row in window_rows}))
    scope_specs.append(("triple", lambda row: True, "", ""))
    for axis in ("kline", "strategy", "cross"):
        if axis == "kline":
            source = expanded_rows
            pattern_columns = ("kline_pattern",)
        elif axis == "strategy":
            source = expanded_rows
            pattern_columns = ("strategy_pattern",)
        else:
            source = expanded_rows
            pattern_columns = ("kline_pattern", "strategy_pattern")
        for scope, predicate, scope_label, scope_epoch in scope_specs:
            source = [row for row in expanded_rows if predicate(row)]
            total_windows = sum(1 for row in window_rows if predicate(row))
            grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = {}
            for row in source:
                key = tuple(str(row[column]) for column in pattern_columns)
                grouped.setdefault(key, []).append(row)
            for key, values in sorted(grouped.items(), key=str):
                is_unclassified = any(item in UNCLASSIFIED_PATTERNS for item in key)
                pnl_values = [float(item["net_pnl"]) for item in values]
                p25, p50, p75 = _percentiles(pnl_values)
                ratio = len(values) / total_windows if total_windows else 0.0
                warnings = []
                if is_unclassified:
                    warnings.append("unclassified")
                    if ratio >= 0.30:
                        warnings.append("unclassified_rate_high")
                diagnostics.append(
                    {
                        "scope": scope,
                        "label": scope_label,
                        "epoch": scope_epoch,
                        "bin_index": "",
                        "pattern_axis": axis,
                        "kline_pattern": key[0] if axis in {"kline", "cross"} else "",
                        "strategy_pattern": key[-1] if axis in {"strategy", "cross"} else "",
                        "is_unclassified": is_unclassified,
                        "window_count": len(values),
                        "window_ratio": ratio,
                        "total_net_pnl": float(sum(pnl_values)),
                        "pnl_p25": p25,
                        "pnl_p50": p50,
                        "pnl_p75": p75,
                        "pnl_median_range": float(p75 - p25),
                        "warning_code": ";".join(warnings),
                        "warning_message": "classified pattern diagnostics",
                    }
                )
    return diagnostics


def write_csv_rows(rows: Sequence[Mapping[str, Any]], columns: Sequence[str], output_path: Path) -> None:
    pd.DataFrame(rows, columns=list(columns)).to_csv(output_path, index=False, encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(path: Path, logical_root: Path) -> dict[str, Any]:
    path = path.resolve()
    logical_root = logical_root.resolve()
    return {
        "logical_path": path.relative_to(logical_root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_analysis_manifest(
    config: "EvaluationConfig",
    *,
    validation_files: Sequence[Mapping[str, str]],
    epochs: Sequence[tuple[int, Path]],
    missing_epochs: Sequence[int],
    output_paths: Sequence[Path],
    window_count: int,
) -> dict[str, Any]:
    """Build the reproducibility manifest for one isolated run."""

    model_inputs = [epoch_path / "trained_model.pkl" for _, epoch_path in epochs]
    data_inputs = [Path(entry["abs_path"]) for entry in validation_files]
    input_records = [
        _fingerprint(path, config.model_root) for path in model_inputs
    ] + [
        _fingerprint(path, config.valid_root) for path in data_inputs
    ] + [
        _fingerprint(config.state_features_path, config.state_features_path.parent),
        _fingerprint(config.maintenance_margin_path, config.maintenance_margin_path.parent),
    ]
    output_records = [
        _fingerprint(path, config.output_dir) for path in sorted(output_paths)
    ]
    output_records.extend(
        _fingerprint(path, config.output_dir)
        for path in sorted(config.output_dir.rglob("*.csv"))
        if path not in output_paths
    )
    return {
        "schema_version": "1.0",
        "dataset_name": config.dataset_name,
        "experiment_name": config.experiment_name,
        "model_root": str(config.model_root),
        "data_root": str(config.valid_root),
        "evaluation_config": {
            "max_holding_number": config.max_holding_number,
            "position_choices": config.position_choices,
            "leverage_choices": list(config.leverage_choices),
            "hidden_nodes": config.hidden_nodes,
            "ensemble_number": config.ensemble_number,
            "time_info_dim": config.time_info_dim,
            "order_book_depth": config.order_book_depth,
            "long_estimated_rate": config.long_estimated_rate,
            "short_estimated_rate": config.short_estimated_rate,
            "transaction_cost": config.transaction_cost,
            "initial_wallet_balance": config.initial_wallet_balance,
            "initial_unrealized_pnl": config.initial_unrealized_pnl,
            "initial_leverage": config.initial_leverage,
            "allow_reverse_position": config.allow_reverse_position,
        },
        "action_space": {
            "position_levels": config.position_levels,
            "action_count": config.action_count,
            "initial_actions": list(config.initial_actions),
        },
        "window_config": {"ordinary_window_steps": 20, "ordinary_stride": 20, "limit_labels": ["label_0", "label_6"]},
        "classifier_thresholds": DEFAULT_THRESHOLDS,
        "epochs_discovered": [epoch for epoch, _ in epochs],
        "epochs_analyzed": [epoch for epoch, _ in epochs],
        "epochs_missing_model": list(missing_epochs),
        "candidate_universe": {
            "epoch_count": len(epochs),
            "validation_file_count": len(validation_files),
            "bin_count": config.ensemble_number,
            "initial_action_count": config.action_count,
            "window_count": window_count,
        },
        "input_files": input_records,
        "output_files": output_records,
        "warnings": [
            "missing_model_epoch" for _ in missing_epochs
        ],
    }


def write_analysis_manifest(manifest: Mapping[str, Any], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _prices_and_volume(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    prices = np.asarray([_number(row.get("mark_price")) for row in rows], dtype=float)
    volume = np.asarray([_number(row.get("volume")) for row in rows], dtype=float)
    if len(prices) == 0 or np.any(prices <= 0) or np.any(volume < 0):
        raise ValueError("pattern window requires positive prices and non-negative volume")
    return prices, volume


def _z_scores(values: np.ndarray) -> np.ndarray | None:
    std = float(np.std(values, ddof=0))
    if std == 0.0:
        return None
    return (values - float(np.mean(values))) / std


def _linear_slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.polyfit(np.arange(len(values), dtype=float), np.log(values), 1)[0])


def _breakout_event(prices: np.ndarray, thresholds: Mapping[str, float]) -> tuple[int, int] | None:
    if len(prices) < 10:
        return None
    baseline_high = float(np.max(prices[:5]))
    baseline_low = float(np.min(prices[:5]))
    for index in range(5, min(10, len(prices))):
        if prices[index] >= baseline_high * (1 + thresholds["breakout_ratio"]):
            return index, 1
        if prices[index] <= baseline_low * (1 - thresholds["breakout_ratio"]):
            return index, -1
    return None


def _is_kline_divergence(
    prices: np.ndarray, volume: np.ndarray, thresholds: Mapping[str, float]
) -> bool:
    if len(prices) < 6:
        return False
    base_price = float(np.mean(prices[:5]))
    base_volume = float(np.median(volume[:5]))
    if base_volume <= 0:
        return False
    direction = 1 if prices[-1] > base_price else -1
    if direction > 0:
        breakout = float(np.max(prices[5:])) >= float(np.max(prices[:5])) * (1 + thresholds["breakout_ratio"])
        move = (float(np.max(prices[5:])) - base_price) / base_price
    else:
        breakout = float(np.min(prices[5:])) <= float(np.min(prices[:5])) * (1 - thresholds["breakout_ratio"])
        move = (base_price - float(np.min(prices[5:]))) / base_price
    volume_near_breakout = float(np.median(volume[max(5, len(volume) - 3) :]))
    return breakout and move >= thresholds["min_price_move"] and volume_near_breakout <= base_volume * (1 - thresholds["volume_drop_ratio"])


def classify_kline_pattern(
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
    thresholds: Mapping[str, float] | None = None,
) -> str:
    """Return one deterministic Kline Pattern for a recognition window."""

    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    if label in {"label_0", "label_6"}:
        return "KX1"
    prices, volume = _prices_and_volume(rows)
    z_scores = _z_scores(prices)
    if z_scores is not None and len(prices) >= thresholds["min_leg_steps"] * 2 + 1:
        extreme = int(np.argmax(np.abs(z_scores)))
        if thresholds["min_leg_steps"] <= extreme <= len(prices) - thresholds["min_leg_steps"] - 1:
            left = (prices[extreme] - prices[0]) / prices[0]
            right = (prices[-1] - prices[extreme]) / prices[extreme]
            if abs(left) >= thresholds["min_leg_return"] and abs(right) >= thresholds["min_leg_return"] and left * right < 0:
                return "KM1"
    breakout = _breakout_event(prices, thresholds)
    if breakout is not None:
        trigger, direction = breakout
        baseline = float(np.max(prices[:5]) if direction > 0 else np.min(prices[:5]))
        post = prices[trigger + 1 :]
        if len(post):
            extreme = float(np.max(post) if direction > 0 else np.min(post))
            retrace_target = baseline * (1 - thresholds["retrace_band_ratio"] if direction > 0 else 1 + thresholds["retrace_band_ratio"])
            retraced = bool(np.min(post) <= retrace_target if direction > 0 else np.max(post) >= retrace_target)
            final_extension = (prices[-1] - baseline) / baseline if direction > 0 else (baseline - prices[-1]) / baseline
            if abs(extreme - baseline) / baseline >= thresholds["retrace_ratio"] and retraced and final_extension >= thresholds["extension_ratio"]:
                return "KT2"
    if _is_kline_divergence(prices, volume, thresholds):
        return "KM3"
    if breakout is not None:
        trigger, direction = breakout
        baseline = float(np.max(prices[:5]) if direction > 0 else np.min(prices[:5]))
        extension = (prices[-1] - baseline) / baseline if direction > 0 else (baseline - prices[-1]) / baseline
        post = prices[trigger:]
        held = np.mean(post >= baseline if direction > 0 else post <= baseline)
        if extension >= thresholds["extension_ratio"] and held >= thresholds["breakout_hold_ratio"]:
            return "KT1"
    if len(prices) >= 20:
        first_slope = _linear_slope(prices[:10])
        second_slope = _linear_slope(prices[10:20])
        cumulative = abs(prices[-1] / prices[0] - 1)
        if first_slope * second_slope > 0 and abs(second_slope) >= abs(first_slope) * thresholds["acceleration_ratio"] and cumulative >= thresholds["cumulative_return_ratio"]:
            return "KT3"
    log_returns = np.diff(np.log(prices))
    mean_price = float(np.mean(prices))
    states: list[str] = []
    for price in prices:
        if price <= mean_price * (1 - thresholds["touch_band_ratio"]):
            state = "low"
        elif price >= mean_price * (1 + thresholds["touch_band_ratio"]):
            state = "high"
        else:
            continue
        if not states or states[-1] != state:
            states.append(state)
    if len(log_returns) and float(np.std(log_returns, ddof=0)) <= thresholds["return_std_threshold"] and all(
        mean_price * (1 - thresholds["outer_band_ratio"]) <= price <= mean_price * (1 + thresholds["outer_band_ratio"])
        for price in prices
    ) and sum(a != b for a, b in zip(states, states[1:])) >= thresholds["min_band_transitions"]:
        return "KM2"
    return "未分类"


def _same_direction_increase(before: float, after: float, levels: Sequence[float]) -> bool:
    if before == 0 or after == 0 or np.sign(before) != np.sign(after):
        return False
    return abs(after) > abs(before) and any(np.isclose(after, level) for level in levels)


def classify_strategy_patterns(
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
    kline_pattern: str,
    position_levels: Sequence[float],
    thresholds: Mapping[str, float] | None = None,
) -> list[str]:
    """Return compatible Strategy Second-order Patterns for one window."""

    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    ordered = _validated_trajectory(rows, require_zero_start=False)
    prices, volume = _prices_and_volume(ordered)
    before = np.asarray([_number(row.get("position_before")) for row in ordered])
    after = np.asarray([_number(row.get("position_after")) for row in ordered])
    unrealized = np.asarray([_number(row.get("unrealized_pnl")) for row in ordered])
    patterns: list[str] = []
    max_position = max(abs(float(level)) for level in position_levels)
    breakout = _breakout_event(prices, thresholds)
    if len(ordered) >= thresholds["min_hold_steps"] and breakout is not None:
        trigger, direction = breakout
        for index in range(trigger, min(trigger + 2, len(ordered))):
            if before[index] == 0 and after[index] * direction > 0 and abs(after[index]) / max_position >= thresholds["near_full_ratio"]:
                held = after[index:] * direction > 0
                if len(held) >= thresholds["min_hold_steps"] and held[: thresholds["min_hold_steps"]].all():
                    patterns.append("ST1")
                    break
    profitable_add = any(
        _same_direction_increase(float(b), float(a), position_levels)
        and (index == 0 or float(unrealized[index - 1]) > 0)
        for index, (b, a) in enumerate(zip(before, after))
    )
    if profitable_add:
        patterns.append("ST3")
    if kline_pattern == "KT2" and any(
        _same_direction_increase(float(b), float(a), position_levels)
        for b, a in zip(before, after)
    ):
        patterns.append("ST2")
    z_scores = _z_scores(prices)
    if label == "label_0" or label == "label_6":
        extreme_direction = -1 if label == "label_0" else 1
        reverse = any(
            a * extreme_direction > 0 and abs(a) > abs(b) and abs(a) / max_position >= thresholds["near_full_ratio"]
            for b, a in zip(before, after)
        )
    else:
        reverse = bool(z_scores is not None and any(
            (z <= -thresholds["z_extreme"] and a > 0 and a > b)
            or (z >= thresholds["z_extreme"] and a < 0 and abs(a) > abs(b))
            for z, b, a in zip(z_scores, before, after)
        ))
    if reverse:
        patterns.append("SM1")
    if z_scores is not None and len(ordered) >= 2:
        levels = list(position_levels)
        adjacent_steps = {
            abs(levels[index + 1] - levels[index])
            for index in range(len(levels) - 1)
        }
        high_side = any(
            z >= 0.5 and a < b and any(np.isclose(abs(a - b), step) for step in adjacent_steps)
            for z, b, a in zip(z_scores, before, after)
        )
        low_side = any(
            z <= -0.5 and a > b and any(np.isclose(abs(a - b), step) for step in adjacent_steps)
            for z, b, a in zip(z_scores, before, after)
        )
        if high_side and low_side:
            patterns.append("SM2")
    if kline_pattern == "KM3" and "ST1" not in patterns and len(ordered) >= 20:
        direction = 1 if prices[-1] > prices[0] else -1
        divergence_segment = slice(5, len(ordered))
        aligned_action = any(
            (direction > 0 and a > b) or (direction < 0 and a < b)
            for b, a in zip(before[divergence_segment], after[divergence_segment])
        )
        non_divergence_action = any(b != a for b, a in zip(before[:5], after[:5]))
        if not aligned_action and non_divergence_action:
            patterns.append("SM3")
    return patterns or ["策略未分类"]


@dataclass(frozen=True)
class EvaluationConfig:
    model_root: Path
    valid_root: Path
    output_dir: Path
    state_features_path: Path
    maintenance_margin_path: Path
    max_holding_number: float = 8.0
    position_choices: int = 9
    leverage_choices: tuple[int, ...] = (1,)
    hidden_nodes: int = 128
    ensemble_number: int = 7
    time_info_dim: int = 2
    order_book_depth: int = 25
    long_estimated_rate: float = 0.0005
    short_estimated_rate: float = 0.0
    transaction_cost: float = 0.0002
    initial_wallet_balance: float = 100000.0
    initial_unrealized_pnl: float = 0.0
    initial_leverage: float = 5.0
    allow_reverse_position: bool = False
    dataset_name: str = ""
    experiment_name: str = ""

    @property
    def position_levels(self) -> list[float]:
        return build_position_levels(self.max_holding_number, self.position_choices)

    @property
    def action_count(self) -> int:
        return (self.position_choices - 1) * len(self.leverage_choices) + 1

    @property
    def initial_actions(self) -> range:
        return range(self.action_count)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class IsolatedAgentEvaluator:
    """Run new evaluations without depending on the legacy test entry point."""

    def __init__(
        self,
        config: EvaluationConfig,
        *,
        model_factory: Callable[..., Any] = ensemble_Qnet,
        environment_factory: Callable[..., Any] = initiate_base_env,
        model_loader: Callable[[Path], Any] | None = None,
    ) -> None:
        self.config = config
        self.model_factory = model_factory
        self.environment_factory = environment_factory
        self.model_loader = model_loader
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.state_features = np.load(config.state_features_path, allow_pickle=True).tolist()
        self.maintenance_margin = np.load(
            config.maintenance_margin_path, allow_pickle=True
        ).item()
        if not self.state_features:
            raise ValueError("state feature list must not be empty")

    def _load_model(self, epoch_path: Path) -> Any:
        if self.model_loader is not None:
            return self.model_loader(epoch_path)
        network = self.model_factory(
            N_STATES=len(self.state_features),
            N_ACTIONS=self.config.action_count,
            hidden_nodes=self.config.hidden_nodes,
            TIME_INFO_DIM=self.config.time_info_dim,
            ensemble_number=self.config.ensemble_number,
        ).to(self.device)
        network.load_state_dict(torch.load(epoch_path / "trained_model.pkl", map_location=self.device))
        network.eval()
        return network

    def _select_action(self, network: Any, state: Any, info: Mapping[str, Any], bin_index: int) -> int:
        state_tensor = torch.as_tensor(np.asarray(state).reshape(-1), dtype=torch.float32, device=self.device).unsqueeze(0)
        previous_action = torch.tensor([[info["previous_action"]]], dtype=torch.float32, device=self.device)
        available = torch.as_tensor([info["avaliable_action"]], device=self.device)
        time_input = torch.tensor(
            [[info["funding_count_down_hour"], info["funding_count_down_minute"]]],
            dtype=torch.float32,
            device=self.device,
        )
        trading_info = torch.as_tensor([info["trading_info"]], dtype=torch.float32, device=self.device)
        if bin_index < 0 or bin_index >= len(network.qnet_list):
            raise ValueError(f"bin_index out of range: {bin_index}")
        with torch.inference_mode():
            values = network.qnet_list[bin_index](
                state=state_tensor,
                time=time_input,
                previous_action=previous_action,
                avaliable_action=available,
                trading_info=trading_info,
            )
        return int(torch.argmax(values, dim=1).item())

    def _build_environment(self, data_frame: pd.DataFrame, initial_action: int) -> Any:
        initial_position, initial_leverage = map_action_to_position_leverage(
            initial_action, list(self.config.leverage_choices), self.config.position_levels
        )
        initial_margin = abs(
            float(initial_position) * float(data_frame["mark_price"].iloc[0]) / float(initial_leverage)
        )
        return self.environment_factory(
            df=data_frame,
            feature_list=self.state_features,
            max_holding_number=self.config.max_holding_number,
            order_book_depth=self.config.order_book_depth,
            position_choices=self.config.position_choices,
            leverage_choice=list(self.config.leverage_choices),
            long_estimated_rate=self.config.long_estimated_rate,
            short_estimated_rate=self.config.short_estimated_rate,
            commission_rate=self.config.transaction_cost,
            maintenance_margin_ratio_dict=self.maintenance_margin,
            early_stop=0,
            initial_state=(
                self.config.initial_wallet_balance,
                initial_margin,
                self.config.initial_unrealized_pnl,
                initial_position,
                initial_leverage,
            ),
            allow_reverse_position=self.config.allow_reverse_position,
        )

    def evaluate_trajectory(
        self,
        *,
        network: Any,
        epoch: int,
        entry: Mapping[str, str],
        initial_action: int,
        bin_index: int,
    ) -> list[dict[str, Any]]:
        data_frame = pd.read_feather(entry["abs_path"])
        validate_required_market_columns(data_frame)
        if len(data_frame) == 0:
            raise ValueError(f"empty validation file: {entry['abs_path']}")
        environment = self._build_environment(data_frame, initial_action)
        state, info = environment.reset()
        rows: list[dict[str, Any]] = []
        previous_action = initial_action
        action_changes = 0
        trade_count = 0
        done = False
        timestep = 0
        while not done:
            position_before = float(getattr(environment, "position"))
            leverage_before = float(getattr(environment, "leverage"))
            action = self._select_action(network, state, info, bin_index)
            target_position, target_leverage = map_action_to_position_leverage(
                action, list(self.config.leverage_choices), self.config.position_levels
            )
            action_change = int(action != previous_action)
            action_changes += action_change
            next_state, reward, done, info = environment.step(action)
            if timestep >= len(data_frame):
                raise RuntimeError("environment exceeded validation data length")
            position_after = float(getattr(environment, "position"))
            leverage_after = float(getattr(environment, "leverage"))
            trade = int(position_after != position_before or leverage_after != leverage_before)
            trade_count += trade
            row = build_step_detail_row(
                epoch=epoch,
                label=entry["label"],
                contract=entry["contract"],
                df_path=entry["df_path"],
                initial_action=initial_action,
                bin_index=bin_index,
                timestep=timestep,
                market_row=data_frame.iloc[timestep].to_dict(),
                action=action,
                target_position=target_position,
                target_leverage=target_leverage,
                position_before=position_before,
                leverage_before=leverage_before,
                position_after=position_after,
                leverage_after=leverage_after,
                step_reward=reward,
                info=info,
                wallet_balance=getattr(environment, "wallet_balance"),
                unrealized_pnl=getattr(environment, "unrealized_pnl"),
                action_change_step=action_change,
                trade_count_step=trade,
                cumulative_action_change_count=action_changes,
                cumulative_trade_count=trade_count,
            )
            rows.append(row)
            state = next_state
            previous_action = action
            timestep += 1
        return rows

    def run(self, *, seed: int = 0) -> list[Path]:
        _seed(seed)
        self.config.output_dir.mkdir(parents=True, exist_ok=False)
        entries = discover_validation_files(self.config.valid_root)
        epochs = discover_model_epochs(self.config.model_root)
        missing_epochs = discover_missing_model_epochs(self.config.model_root)
        if not epochs:
            raise FileNotFoundError("no direct epoch_<N>/trained_model.pkl found")
        output_paths: list[Path] = []
        all_window_rows: list[dict[str, Any]] = []
        coverage_rows: list[dict[str, Any]] = [
            {
                "record_type": "epoch",
                "epoch": epoch,
                "status": "missing_model",
                "expected_count": 0,
                "observed_count": 0,
                "coverage_ratio": 0.0,
                "message": "epoch directory has no trained_model.pkl",
            }
            for epoch in missing_epochs
        ]
        for epoch, epoch_path in epochs:
            network = self._load_model(epoch_path)
            epoch_rows: list[dict[str, Any]] = []
            expected_lengths = {
                entry["abs_path"]: len(pd.read_feather(entry["abs_path"]))
                for entry in entries
            }
            for entry in entries:
                for initial_action in self.config.initial_actions:
                    bin_count = len(network.qnet_list)
                    if bin_count != self.config.ensemble_number:
                        raise ValueError(
                            f"epoch {epoch} ensemble count {bin_count} does not match "
                            f"configured ensemble_number {self.config.ensemble_number}"
                        )
                    for bin_index in range(bin_count):
                        trajectory_rows = self.evaluate_trajectory(
                            network=network,
                            epoch=epoch,
                            entry=entry,
                            initial_action=initial_action,
                            bin_index=bin_index,
                        )
                        epoch_rows.extend(trajectory_rows)
                        expected_count = expected_lengths[entry["abs_path"]]
                        observed_count = len(trajectory_rows)
                        status = "complete" if observed_count == expected_count else "failed"
                        message = "" if status == "complete" else (
                            f"observed_count={observed_count} expected_count={expected_count}; "
                            "trajectory identity is incomplete"
                        )
                        is_limit_label = entry["label"] in {"label_0", "label_6"}
                        window_count = int(observed_count > 0) if is_limit_label else observed_count // 20
                        dropped_tail_steps = 0 if is_limit_label else observed_count % 20
                        tail_rows = trajectory_rows[-dropped_tail_steps:] if dropped_tail_steps else []
                        coverage_rows.append(
                            {
                                "record_type": "trajectory",
                                "epoch": epoch,
                                "label": entry["label"],
                                "bin_index": bin_index,
                                "contract": entry["contract"],
                                "df_path": entry["df_path"],
                                "initial_action": initial_action,
                                "expected_count": expected_count,
                                "observed_count": observed_count,
                                "coverage_ratio": observed_count / expected_count,
                                "status": status,
                                "window_count": window_count,
                                "dropped_tail_steps": dropped_tail_steps,
                                "dropped_tail_gross_pnl": float(sum(_number(row.get("realized_pnl_step")) for row in tail_rows)),
                                "dropped_tail_net_pnl": float(sum(_number(row.get("realized_pnl_step")) - _number(row.get("commission_fee_step")) for row in tail_rows)),
                                "message": message,
                            }
                        )
            output_path = self.config.output_dir / "step_detail" / f"agent_pattern_step_detail_epoch_{epoch}.csv"
            write_step_detail_csv(epoch_rows, output_path)
            output_paths.append(output_path)
            all_window_rows.extend(
                build_window_rows(
                    epoch_rows,
                    position_levels=self.config.position_levels,
                )
            )
        write_coverage_report(
            coverage_rows,
            self.config.output_dir / "agent_pattern_coverage_report.csv",
        )
        write_window_table(
            all_window_rows,
            self.config.output_dir / "agent_pattern_window_table.csv",
        )
        expanded_rows = expand_window_rows(all_window_rows)
        write_expanded_table(
            expanded_rows,
            self.config.output_dir / "agent_pattern_expanded_table.csv",
        )
        write_csv_rows(
            build_diagnostics(all_window_rows, expanded_rows),
            DIAGNOSTIC_COLUMNS,
            self.config.output_dir / "agent_pattern_classifier_diagnostics.csv",
        )
        for axis in ("kline", "strategy", "cross"):
            scenario_rows = _scenario_rows(expanded_rows, axis=axis)
            triple_rows = _triple_rows(scenario_rows, expanded_rows, axis=axis)
            write_csv_rows(
                scenario_rows,
                {
                    "kline": SCENARIO_KLINE_COLUMNS,
                    "strategy": SCENARIO_STRATEGY_COLUMNS,
                    "cross": SCENARIO_CROSS_COLUMNS,
                }[axis],
                self.config.output_dir / f"agent_pattern_{axis}_scenario_summary.csv",
            )
            write_csv_rows(
                triple_rows,
                {
                    "kline": TRIPLE_KLINE_COLUMNS,
                    "strategy": TRIPLE_STRATEGY_COLUMNS,
                    "cross": TRIPLE_CROSS_COLUMNS,
                }[axis],
                self.config.output_dir / f"agent_pattern_{axis}_triple_summary.csv",
            )
        manifest = build_analysis_manifest(
            self.config,
            validation_files=entries,
            epochs=epochs,
            missing_epochs=missing_epochs,
            output_paths=output_paths,
            window_count=len(all_window_rows),
        )
        write_analysis_manifest(
            manifest,
            self.config.output_dir / "analysis_manifest.json",
        )
        return output_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_root", type=Path, required=True)
    parser.add_argument("--valid_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--state_features_path", type=Path, required=True)
    parser.add_argument("--maintenance_margin_path", type=Path, required=True)
    parser.add_argument("--max_holding_number", type=float, default=8.0)
    parser.add_argument("--position_choices", type=int, default=9)
    parser.add_argument("--leverage_choices", type=int, action="append", default=[1])
    parser.add_argument("--hidden_nodes", type=int, default=128)
    parser.add_argument("--ensemble_number", type=int, default=7)
    parser.add_argument("--time_info_dim", type=int, default=2)
    parser.add_argument("--order_book_depth", type=int, default=25)
    parser.add_argument("--long_estimated_rate", type=float, default=0.0005)
    parser.add_argument("--short_estimated_rate", type=float, default=0.0)
    parser.add_argument("--transaction_cost", type=float, default=0.0002)
    parser.add_argument("--initial_wallet_balance", type=float, default=100000.0)
    parser.add_argument("--initial_unrealized_pnl", type=float, default=0.0)
    parser.add_argument("--initial_leverage", type=float, default=5.0)
    parser.add_argument("--allow_reverse_position", action="store_true")
    parser.add_argument("--dataset_name", default="")
    parser.add_argument("--experiment_name", default="")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def config_from_args(args: argparse.Namespace) -> EvaluationConfig:
    return EvaluationConfig(
        model_root=args.model_root,
        valid_root=args.valid_root,
        output_dir=args.output_dir,
        state_features_path=args.state_features_path,
        maintenance_margin_path=args.maintenance_margin_path,
        max_holding_number=args.max_holding_number,
        position_choices=args.position_choices,
        leverage_choices=tuple(args.leverage_choices),
        hidden_nodes=args.hidden_nodes,
        ensemble_number=args.ensemble_number,
        time_info_dim=args.time_info_dim,
        order_book_depth=args.order_book_depth,
        long_estimated_rate=args.long_estimated_rate,
        short_estimated_rate=args.short_estimated_rate,
        transaction_cost=args.transaction_cost,
        initial_wallet_balance=args.initial_wallet_balance,
        initial_unrealized_pnl=args.initial_unrealized_pnl,
        initial_leverage=args.initial_leverage,
        allow_reverse_position=args.allow_reverse_position,
        dataset_name=args.dataset_name,
        experiment_name=args.experiment_name,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    IsolatedAgentEvaluator(config).run(seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
