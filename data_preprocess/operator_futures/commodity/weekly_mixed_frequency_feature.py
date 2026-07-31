from pathlib import Path
import argparse

import polars as pl

from .daily_base_feature import read_base_feature
from .daily_mixed_frequency_feature import (
    aggregate_period_rows,
    daily_rolling_state_features,
    period_features,
    previous_window_rows,
)
from .weekly_base_feature import WEEKLY_BASE_FEATURE_COLUMNS, calendar_week_text


WEEK_ROLLING_WINDOWS: tuple[int, ...] = (1, 2, 4, 6)

_WEEKLY_STATE_FEATURE_SUFFIXES: tuple[str, ...] = (
    "return",
    "range_pct",
    "body_pct",
    "close_position",
    "body_to_range",
    "upper_shadow_to_range",
    "lower_shadow_to_range",
    "vwap_deviation_pct",
    "twap_deviation_pct",
    "trade_up_ratio",
    "trade_down_ratio",
    "trade_imbalance",
    "open_interest_change",
    "turnover_rate",
)

_WEEKLY_ROLLING_STATE_FEATURE_SUFFIXES: tuple[str, ...] = (
    "trade_up_ratio",
    "trade_down_ratio",
    "trade_imbalance",
    "open_interest_change",
    "turnover_rate",
)


def _week_prefix(window: int) -> str:
    return "prev_week" if window == 1 else f"prev_{window}_week"


PREV_WEEK_FEATURE_COLUMNS: list[str] = [
    f"prev_week_{suffix}" for suffix in _WEEKLY_STATE_FEATURE_SUFFIXES
]
PREV_WEEK_FEATURE_COLUMNS.extend(
    f"{_week_prefix(window)}_{suffix}"
    for window in WEEK_ROLLING_WINDOWS
    if window != 1
    for suffix in _WEEKLY_ROLLING_STATE_FEATURE_SUFFIXES
)

_WEEKLY_PRICE_ROLLING_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    f"{_week_prefix(window)}_{suffix}"
    for window in WEEK_ROLLING_WINDOWS
    if window != 1
    for suffix in (
        "return",
        "range_pct",
        "body_pct",
        "close_position",
        "body_to_range",
        "upper_shadow_to_range",
        "lower_shadow_to_range",
        "vwap_deviation_pct",
        "twap_deviation_pct",
    )
)

_WEEKLY_ABSOLUTE_LEVEL_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    f"{_week_prefix(window)}_{suffix}"
    for window in WEEK_ROLLING_WINDOWS
    for suffix in ("volume", "tradeval")
)

_ABSOLUTE_LEVEL_FEATURE_COLUMNS: tuple[str, ...] = (
    *_WEEKLY_ABSOLUTE_LEVEL_FEATURE_COLUMNS,
    *_WEEKLY_PRICE_ROLLING_FEATURE_COLUMNS,
)


def _rows_from_weekly_base(weekly_base: pl.DataFrame) -> dict[str, dict[str, object]]:
    missing = [
        column for column in WEEKLY_BASE_FEATURE_COLUMNS if column not in weekly_base.columns
    ]
    if missing:
        raise ValueError(f"Missing weekly Mixed-frequency Base Data columns: {missing}")
    return {
        str(row["calendar_week"]): row
        for row in weekly_base.iter_rows(named=True)
    }


def generate_weekly_mixed_frequency_features_from_base(
    *, target_bars: pl.DataFrame, weekly_base: pl.DataFrame
) -> pl.DataFrame:
    if "timestamp" not in target_bars.columns or "trading_day" not in target_bars.columns:
        raise ValueError("target_bars must contain timestamp and trading_day columns")

    ordered = target_bars.sort(["trading_day", "timestamp"])
    weekly_rows = _rows_from_weekly_base(weekly_base)
    weekly_keys = sorted(weekly_rows)

    rows: list[dict[str, float | object]] = []
    for row in ordered.select("timestamp", "trading_day").iter_rows(named=True):
        week_key = calendar_week_text(row["trading_day"])
        feature_row: dict[str, float | object] = {"timestamp": row["timestamp"]}
        for window in WEEK_ROLLING_WINDOWS:
            window_rows = previous_window_rows(
                rows=weekly_rows,
                keys=weekly_keys,
                current=week_key,
                window=window,
            )
            prefix = _week_prefix(window)
            if window == 1:
                weekly_features = period_features(
                    prefix, aggregate_period_rows(window_rows)
                )
                weekly_features.pop(f"{prefix}_upper_shadow_pct")
                weekly_features.pop(f"{prefix}_lower_shadow_pct")
            else:
                weekly_features = daily_rolling_state_features(prefix, window_rows)
            feature_row.update(weekly_features)
        rows.append(feature_row)

    result = pl.DataFrame(rows).select(["timestamp", *PREV_WEEK_FEATURE_COLUMNS])
    validate_weekly_mixed_frequency_output(result)
    return result


def validate_weekly_mixed_frequency_output(df: pl.DataFrame) -> None:
    illegal_columns = [
        column for column in _ABSOLUTE_LEVEL_FEATURE_COLUMNS if column in df.columns
    ]
    if illegal_columns:
        raise ValueError(
            "Weekly Mixed-frequency State Feature violates No Absolute Level Rule: "
            f"{illegal_columns}"
        )
    for column in PREV_WEEK_FEATURE_COLUMNS:
        if column not in df.columns:
            raise ValueError(f"Missing weekly Mixed-frequency State Feature column: {column}")
        series = df.get_column(column).cast(pl.Float64, strict=False)
        invalid = series.is_null() | series.is_nan() | series.is_infinite()
        if invalid.any():
            row = df.filter(invalid).row(0, named=True)
            raise ValueError(
                f"Invalid weekly Mixed-frequency State Feature output {column!r}: "
                f"value={row.get(column)!r} timestamp={row.get('timestamp')!r}"
            )


def _resolve_base_path(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"MIXED_FREQUENCY_BASE WEEKLY path does not exist: {path}")
    return path


def write_weekly_mixed_frequency_feature_for_day(
    *,
    root_path: str | Path,
    symbol: str,
    contract: str,
    target_freq: str,
    date: str,
    start_date: str,
    end_date: str,
    data_path: str = "PREPROCESS_DATASET/commodity-futures",
    base_path: str = "PREPROCESS_DATASET/commodity-futures/MIXED_FREQUENCY_BASE",
    save_path: str = "PREPROCESS_DATASET/commodity-futures/MIXED_FREQUENCY_FEATURE",
) -> Path:
    root = Path(root_path)
    range_name = f"{start_date}-{end_date}"
    weekly_base_path = (
        root
        / base_path
        / symbol
        / contract
        / target_freq
        / "WEEKLY"
        / f"{range_name}.feather"
    )
    weekly_base = pl.read_ipc(_resolve_base_path(weekly_base_path))
    current = read_base_feature(
        root / data_path / "BASE_FEATURE" / symbol / contract / target_freq / f"{date}.feather",
        date,
    )
    output = generate_weekly_mixed_frequency_features_from_base(
        target_bars=current.select("timestamp", "trading_day"),
        weekly_base=weekly_base,
    )
    if output.get_column("timestamp").to_list() != current.get_column("timestamp").to_list():
        raise ValueError(
            f"WEEKLY_MIXED_FREQUENCY_FEATURE timestamp set does not match BASE_FEATURE for date {date}"
        )
    out_dir = root / save_path / symbol / contract / target_freq / "WEEKLY"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date}.feather"
    output.write_ipc(out_path)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=Path, default=Path("."))
    parser.add_argument("--symbol", "--symbols", dest="symbol", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--target_freq", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--start_date", required=True)
    parser.add_argument("--end_date", required=True)
    parser.add_argument(
        "--data_path",
        default="PREPROCESS_DATASET/commodity-futures",
    )
    parser.add_argument(
        "--base_path",
        default="PREPROCESS_DATASET/commodity-futures/MIXED_FREQUENCY_BASE",
    )
    parser.add_argument(
        "--save_path",
        default="PREPROCESS_DATASET/commodity-futures/MIXED_FREQUENCY_FEATURE",
    )
    return parser


def main(args=None) -> Path:
    parsed = build_parser().parse_args(args)
    return write_weekly_mixed_frequency_feature_for_day(
        root_path=parsed.root_path,
        symbol=parsed.symbol,
        contract=parsed.contract,
        target_freq=parsed.target_freq,
        date=parsed.date,
        start_date=parsed.start_date,
        end_date=parsed.end_date,
        data_path=parsed.data_path,
        base_path=parsed.base_path,
        save_path=parsed.save_path,
    )
