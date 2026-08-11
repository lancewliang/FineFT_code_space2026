"""Build per-epoch trade lifecycle details and typed analysis results.

The input is the per-step ``trading_action_detail_epoch_{epoch}.csv`` emitted by
``test_agent_index.py``.  Each output lifecycle row represents one continuous
directional position episode.  ``analysis_result_with_type.npy`` and
``analysis_result_with_type.csv`` are rebuilt from that lifecycle CSV.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import json
import numpy as np
import pandas as pd
import polars as pl


HEADER_MAPPING = {
    "标签": "label",
    "label": "label",
    "数据文件": "df_path",
    "df_path": "df_path",
    "初始动作": "initial_action",
    "initial_action": "initial_action",
    "分箱索引": "bin_index",
    "bin_index": "bin_index",
    "时间步": "timestep",
    "timestep": "timestep",
    "时间戳": "timestamp",
    "timestamp": "timestamp",
    "动作": "action",
    "action": "action",
    "执行前仓位": "position_before",
    "position_before": "position_before",
    "执行后仓位": "position_after",
    "position_after": "position_after",
    "单步奖励": "step_reward",
    "step_reward": "step_reward",
}

CSV_HEADER_LABELS = {
    "label": "标签",
    "initial_action": "初始动作",
    "bin_index": "分箱索引",
    "trend_type": "趋势类型",
    "contract": "合约",
    "df_path": "数据文件",
    "reward_sum": "奖励总和",
    "df_length": "数据长度",
    "turnover": "换手率",
    "mean_position": "平均仓位",
    "mean_abs_position": "平均绝对仓位",
    "long_step_ratio": "多头步数占比",
    "short_step_ratio": "空头步数占比",
    "flat_step_ratio": "空仓步数占比",
    "long_reward_sum": "多头奖励总和",
    "short_reward_sum": "空头奖励总和",
    "flat_reward_sum": "空仓奖励总和",
    "net_position_exposure": "净仓位敞口",
    "limit_up_step_ratio": "涨停步数占比",
    "limit_down_step_ratio": "跌停步数占比",
    "limit_up_long_reward_sum": "涨停多头奖励总和",
    "limit_down_short_reward_sum": "跌停空头奖励总和",
    "limit_up_reverse_short_ratio": "涨停反向空头占比",
    "limit_down_reverse_long_ratio": "跌停反向多头占比",
}

REQUIRED_DETAIL_COLUMNS = {
    "label",
    "df_path",
    "initial_action",
    "bin_index",
    "timestep",
    "timestamp",
    "position_before",
    "position_after",
    "step_reward",
}

LIFECYCLE_COLUMNS = [
    "label",
    "df_path",
    "initial_action",
    "bin_index",
    "start_timestep",
    "end_timestep",
    "start_timestamp",
    "end_timestamp",
    "holding_duration",
    "trade_direction",
    "segment_type",
    "trend_type",
    "turnover",
    "reward_sum",
    "mean_position",
    "mean_abs_position",
    "long_step",
    "short_step",
    "flat_step",
    "long_reward_sum",
    "short_reward_sum",
    "flat_reward_sum",
]

ANALYSIS_LIST_COLUMNS = [
    "contract",
    "df_path",
    "reward_sum",
    "df_length",
    "turnover",
    "mean_position",
    "mean_abs_position",
    "long_step_ratio",
    "short_step_ratio",
    "flat_step_ratio",
    "long_reward_sum",
    "short_reward_sum",
    "flat_reward_sum",
    "net_position_exposure",
    "limit_up_step_ratio",
    "limit_down_step_ratio",
    "limit_up_long_reward_sum",
    "limit_down_short_reward_sum",
    "limit_up_reverse_short_ratio",
    "limit_down_reverse_long_ratio",
]

TRAJECTORY_COLUMNS = ["label", "df_path", "initial_action", "bin_index"]
LABEL_PATTERN = re.compile(r"^label_(\d+)$")


def _standardize_detail_frame(frame: pl.DataFrame) -> pl.DataFrame:
    rename = {
        column: HEADER_MAPPING[column.strip()]
        for column in frame.columns
        if column.strip() in HEADER_MAPPING
        and HEADER_MAPPING[column.strip()] != column
    }
    standardized = frame.rename(rename) if rename else frame
    missing = sorted(REQUIRED_DETAIL_COLUMNS - set(standardized.columns))
    if missing:
        raise ValueError(f"detail CSV is missing required columns: {missing}")
    return standardized


def _finite_number(value: Any, *, field: str) -> float:
    if value is None:
        return 0.0
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return number


def _label_index(label: str) -> int | None:
    match = LABEL_PATTERN.fullmatch(label.strip())
    return int(match.group(1)) if match else None


def _segment_type(label: str) -> str:
    index = _label_index(label)
    if index in {0, 1, 2}:
        return "下跌分片"
    if index == 3:
        return "震荡分片"
    if index in {4, 5, 6}:
        return "上涨分片"
    return "未知分片"


def _trend_type(segment_type: str, position: float) -> str:
    if segment_type == "下跌分片" and position < 0:
        return "趋势跟随"
    if segment_type == "下跌分片" and position > 0:
        return "趋势回归"
    if segment_type == "上涨分片" and position > 0:
        return "趋势跟随"
    if segment_type == "上涨分片" and position < 0:
        return "趋势回归"
    return "未分类"


def _new_lifecycle(row: Mapping[str, Any], position: float) -> dict[str, Any]:
    label = str(row["label"])
    segment_type = _segment_type(label)
    timestep = int(row["timestep"])
    timestamp = str(row.get("timestamp") or "")
    return {
        "label": label,
        "df_path": str(row["df_path"]),
        "initial_action": int(row["initial_action"]),
        "bin_index": int(row["bin_index"]),
        "start_timestep": timestep,
        "end_timestep": timestep,
        "start_timestamp": timestamp,
        "end_timestamp": timestamp,
        "holding_duration": 1,
        "trade_direction": "Long" if position > 0 else "Short",
        "segment_type": segment_type,
        "trend_type": _trend_type(segment_type, position),
        "turnover": 0.0,
        "reward_sum": 0.0,
        "positions": [],
        "long_step": 0,
        "short_step": 0,
        "flat_step": 0,
        "long_reward_sum": 0.0,
        "short_reward_sum": 0.0,
        "flat_reward_sum": 0.0,
    }


def _add_exposure(lifecycle: dict[str, Any], position: float) -> None:
    lifecycle["positions"].append(position)
    if position > 0:
        lifecycle["long_step"] += 1
    elif position < 0:
        lifecycle["short_step"] += 1
    else:
        lifecycle["flat_step"] += 1


def _add_reward(lifecycle: dict[str, Any], reward: float, position: float) -> None:
    lifecycle["reward_sum"] += reward
    if position > 0:
        lifecycle["long_reward_sum"] += reward
    elif position < 0:
        lifecycle["short_reward_sum"] += reward
    else:
        lifecycle["flat_reward_sum"] += reward


def _finish_lifecycle(
    lifecycle: dict[str, Any], row: Mapping[str, Any]
) -> dict[str, Any]:
    lifecycle["end_timestep"] = int(row["timestep"])
    lifecycle["end_timestamp"] = str(row.get("timestamp") or "")
    lifecycle["holding_duration"] = (
        lifecycle["end_timestep"] - lifecycle["start_timestep"] + 1
    )
    positions = lifecycle.pop("positions")
    lifecycle["mean_position"] = float(np.mean(positions)) if positions else 0.0
    lifecycle["mean_abs_position"] = (
        float(np.mean(np.abs(positions))) if positions else 0.0
    )
    return {column: lifecycle[column] for column in LIFECYCLE_COLUMNS}


def extract_trade_lifecycles(
    detail_frame: pl.DataFrame,
    *,
    max_holding_number: float,
) -> list[dict[str, Any]]:
    """Convert per-step facts into continuous directional position episodes."""

    if not np.isfinite(max_holding_number) or max_holding_number <= 0:
        raise ValueError("max_holding_number must be finite and positive")
    frame = _standardize_detail_frame(detail_frame)
    if frame.is_empty():
        return []

    lifecycle_rows: list[dict[str, Any]] = []
    groups = frame.group_by(TRAJECTORY_COLUMNS, maintain_order=True)
    turnover_scale = 2.0 * float(max_holding_number)

    for _, group in groups:
        rows = group.sort("timestep").to_dicts()
        active: dict[str, Any] | None = None
        previous_after: float | None = None

        for row in rows:
            before = _finite_number(row["position_before"], field="position_before")
            after = _finite_number(row["position_after"], field="position_after")
            reward = _finite_number(row["step_reward"], field="step_reward")
            if previous_after is not None and not np.isclose(before, previous_after):
                raise ValueError(
                    "trajectory positions are discontinuous for "
                    f"{row['df_path']} at timestep {row['timestep']}: "
                    f"previous position_after={previous_after}, "
                    f"position_before={before}"
                )
            previous_after = after
            reversal = before * after < 0
            turnover = abs(after - before) / turnover_scale

            if active is None:
                opening_position = before if before != 0 else after
                if opening_position == 0:
                    continue
                active = _new_lifecycle(row, opening_position)

            if reversal:
                _add_exposure(active, before)
                _add_reward(active, reward, before)
                active["turnover"] += turnover
                lifecycle_rows.append(_finish_lifecycle(active, row))

                active = _new_lifecycle(row, after)
                _add_exposure(active, after)
                continue

            _add_exposure(active, after)
            _add_reward(active, reward, after)
            active["turnover"] += turnover

            if after == 0:
                lifecycle_rows.append(_finish_lifecycle(active, row))
                active = None

        if active is not None:
            lifecycle_rows.append(_finish_lifecycle(active, rows[-1]))

    return lifecycle_rows


def _contract_from_df_path(df_path: str) -> str:
    parts = Path(df_path).parts
    return parts[0] if parts else ""


def _weighted_mean(rows: list[dict[str, Any]], field: str) -> float:
    total_steps = sum(int(row["holding_duration"]) for row in rows)
    if total_steps == 0:
        return 0.0
    return float(
        sum(float(row[field]) * int(row["holding_duration"]) for row in rows)
        / total_steps
    )


def _scenario_metrics(
    rows: list[dict[str, Any]], *, max_holding_number: float
) -> dict[str, Any]:
    first = rows[0]
    df_length = sum(int(row["holding_duration"]) for row in rows)
    long_step = sum(int(row["long_step"]) for row in rows)
    short_step = sum(int(row["short_step"]) for row in rows)
    flat_step = sum(int(row["flat_step"]) for row in rows)
    mean_position = _weighted_mean(rows, "mean_position")
    label_index = _label_index(str(first["label"]))
    long_reward_sum = float(sum(float(row["long_reward_sum"]) for row in rows))
    short_reward_sum = float(sum(float(row["short_reward_sum"]) for row in rows))
    denominator = float(df_length) if df_length else 1.0

    return {
        "contract": _contract_from_df_path(str(first["df_path"])),
        "df_path": str(first["df_path"]),
        "reward_sum": float(sum(float(row["reward_sum"]) for row in rows)),
        "df_length": df_length,
        "turnover": float(sum(float(row["turnover"]) for row in rows)),
        "mean_position": mean_position,
        "mean_abs_position": _weighted_mean(rows, "mean_abs_position"),
        "long_step_ratio": long_step / denominator,
        "short_step_ratio": short_step / denominator,
        "flat_step_ratio": flat_step / denominator,
        "long_reward_sum": long_reward_sum,
        "short_reward_sum": short_reward_sum,
        "flat_reward_sum": float(
            sum(float(row["flat_reward_sum"]) for row in rows)
        ),
        "net_position_exposure": mean_position / max_holding_number,
        "limit_up_step_ratio": 1.0 if label_index == 6 else 0.0,
        "limit_down_step_ratio": 1.0 if label_index == 0 else 0.0,
        "limit_up_long_reward_sum": long_reward_sum if label_index == 6 else 0.0,
        "limit_down_short_reward_sum": (
            short_reward_sum if label_index == 0 else 0.0
        ),
        "limit_up_reverse_short_ratio": (
            short_step / denominator if label_index == 6 else 0.0
        ),
        "limit_down_reverse_long_ratio": (
            long_step / denominator if label_index == 0 else 0.0
        ),
    }


def build_analysis_result(
    lifecycle_frame: pl.DataFrame,
    *,
    max_holding_number: float,
) -> list[dict[str, Any]]:
    """Build analysis_result-compatible records grouped by lifecycle type."""

    if lifecycle_frame.is_empty():
        return []
    missing = sorted(set(LIFECYCLE_COLUMNS) - set(lifecycle_frame.columns))
    if missing:
        raise ValueError(f"lifecycle CSV is missing required columns: {missing}")

    rows = lifecycle_frame.to_dicts()
    grouped: dict[tuple[str, int, int, str], dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        key = (
            str(row["label"]),
            int(row["initial_action"]),
            int(row["bin_index"]),
            str(row["trend_type"]),
        )
        grouped.setdefault(key, {}).setdefault(str(row["df_path"]), []).append(row)

    overall_result: list[dict[str, Any]] = []
    for (label, initial_action, bin_index, trend_type), path_groups in sorted(
        grouped.items()
    ):
        result: dict[str, Any] = {
            "label": label,
            "initial_action": initial_action,
            "bin_index": bin_index,
            "trend_type": trend_type,
        }
        for column in ANALYSIS_LIST_COLUMNS:
            result[column] = []
        for df_path in sorted(path_groups):
            metrics = _scenario_metrics(
                path_groups[df_path], max_holding_number=max_holding_number
            )
            for column in ANALYSIS_LIST_COLUMNS:
                result[column].append(metrics[column])
        overall_result.append(result)
    return overall_result


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_array(value: Any) -> str:
    return json.dumps(list(value), default=_json_default)


def write_analysis_csv(
    overall_result: Sequence[Mapping[str, Any]], csv_path: Path
) -> None:
    analysis_df = pd.DataFrame(overall_result)
    for column in ANALYSIS_LIST_COLUMNS:
        if column in analysis_df.columns:
            analysis_df[column] = analysis_df[column].apply(_json_array)
    analysis_df.rename(columns=CSV_HEADER_LABELS).to_csv(csv_path, index=False)


def epoch_directory(
    result_path: Path,
    dataset_name: str,
    experiment_name: str,
    epoch: int,
) -> Path:
    return (
        result_path
        / dataset_name
        / experiment_name
        / "weights_advantage_pretrain"
        / f"epoch_{epoch}"
    )


def generate_epoch_artifacts(
    *,
    result_path: Path,
    dataset_name: str,
    experiment_name: str,
    epoch: int,
    max_holding_number: float,
) -> tuple[Path, Path, Path]:
    output_dir = epoch_directory(
        result_path, dataset_name, experiment_name, epoch
    )
    source_path = output_dir / f"trading_action_detail_epoch_{epoch}.csv"
    if not source_path.is_file():
        raise FileNotFoundError(f"trading detail CSV does not exist: {source_path}")

    detail_frame = pl.read_csv(source_path)
    lifecycle_rows = extract_trade_lifecycles(
        detail_frame, max_holding_number=max_holding_number
    )
    lifecycle_path = output_dir / f"agent_trade_lifecycle_detail_{epoch}.csv"
    pl.DataFrame(lifecycle_rows, schema=LIFECYCLE_COLUMNS, orient="row").write_csv(
        lifecycle_path
    )

    lifecycle_frame = pl.read_csv(lifecycle_path)
    analysis_result = build_analysis_result(
        lifecycle_frame, max_holding_number=max_holding_number
    )
    analysis_path = output_dir / "analysis_result_with_type.npy"
    np.save(analysis_path, np.asarray(analysis_result, dtype=object))

    analysis_csv_path = output_dir / "analysis_result_with_type.csv"
    write_analysis_csv(analysis_result, analysis_csv_path)

    return lifecycle_path, analysis_path, analysis_csv_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-epoch agent trade lifecycle CSVs and "
            "analysis_result_with_type.npy files"
        )
    )
    parser.add_argument(
        "--result_path", type=Path, default=Path("result/DiHFT/low_level")
    )
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--epoch_start", type=int, required=True)
    parser.add_argument("--epoch_end", type=int)
    parser.add_argument("--max_holding_number", type=float, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    epoch_end = args.epoch_start if args.epoch_end is None else args.epoch_end
    if epoch_end < args.epoch_start:
        raise ValueError("epoch_end must be greater than or equal to epoch_start")
    for epoch in range(args.epoch_start, epoch_end + 1):
        lifecycle_path, analysis_path, analysis_csv_path = generate_epoch_artifacts(
            result_path=args.result_path,
            dataset_name=args.dataset_name,
            experiment_name=args.experiment_name,
            epoch=epoch,
            max_holding_number=args.max_holding_number,
        )
        print(f"Generated lifecycle CSV: {lifecycle_path}", flush=True)
        print(f"Generated typed analysis NPY: {analysis_path}", flush=True)
        print(f"Generated typed analysis CSV: {analysis_csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
