from pathlib import Path
import argparse

import polars as pl

from .daily_mixed_frequency_feature import (
    PREV_DAY_FEATURE_COLUMNS,
    validate_daily_mixed_frequency_output,
)
from .weekly_mixed_frequency_feature import (
    PREV_WEEK_FEATURE_COLUMNS,
    validate_weekly_mixed_frequency_output,
)


MIXED_FREQUENCY_FEATURE_COLUMNS: list[str] = (
    PREV_DAY_FEATURE_COLUMNS + PREV_WEEK_FEATURE_COLUMNS
)


def _resolve_feature_path(path: Path, *, required_name: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{required_name} path does not exist: {path}")
    return path


def combine_daily_weekly_mixed_frequency_features(
    *,
    daily_feature: pl.DataFrame,
    weekly_feature: pl.DataFrame,
) -> pl.DataFrame:
    validate_daily_mixed_frequency_output(daily_feature)
    validate_weekly_mixed_frequency_output(weekly_feature)
    missing_daily = [
        column for column in PREV_DAY_FEATURE_COLUMNS if column not in daily_feature.columns
    ]
    if missing_daily:
        raise ValueError(f"Missing daily Mixed-frequency State Feature columns: {missing_daily}")
    missing_weekly = [
        column for column in PREV_WEEK_FEATURE_COLUMNS if column not in weekly_feature.columns
    ]
    if missing_weekly:
        raise ValueError(f"Missing weekly Mixed-frequency State Feature columns: {missing_weekly}")
    if daily_feature.get_column("timestamp").to_list() != weekly_feature.get_column("timestamp").to_list():
        raise ValueError("daily and weekly Mixed-frequency State Feature timestamps do not match")
    result = daily_feature.select(["timestamp", *PREV_DAY_FEATURE_COLUMNS]).join(
        weekly_feature.select(["timestamp", *PREV_WEEK_FEATURE_COLUMNS]),
        on="timestamp",
        how="left",
    )
    _validate_finite_output(result)
    return result.select(["timestamp", *MIXED_FREQUENCY_FEATURE_COLUMNS])


def _validate_finite_output(df: pl.DataFrame) -> None:
    for column in MIXED_FREQUENCY_FEATURE_COLUMNS:
        if column not in df.columns:
            raise ValueError(f"Missing Mixed-frequency State Feature column: {column}")
        series = df.get_column(column).cast(pl.Float64, strict=False)
        invalid = series.is_null() | series.is_nan() | series.is_infinite()
        if invalid.any():
            row = df.filter(invalid).row(0, named=True)
            raise ValueError(
                f"Invalid Mixed-frequency State Feature output {column!r}: "
                f"value={row.get(column)!r} timestamp={row.get('timestamp')!r}"
            )


def write_mixed_frequency_feature_for_day(
    *,
    root_path: str | Path,
    symbol: str,
    contract: str,
    target_freq: str,
    date: str,
    feature_path: str = "PREPROCESS_DATASET/commodity-futures/MIXED_FREQUENCY_FEATURE",
    save_path: str = "PREPROCESS_DATASET/commodity-futures/MIXED_FREQUENCY_FEATURE",
) -> Path:
    root = Path(root_path)
    feature_root = root / feature_path / symbol / contract / target_freq
    daily_path = feature_root / "DAILY" / f"{date}.feather"
    weekly_path = feature_root / "WEEKLY" / f"{date}.feather"
    daily_feature = pl.read_ipc(
        _resolve_feature_path(daily_path, required_name="DAILY_MIXED_FREQUENCY_FEATURE")
    )
    weekly_feature = pl.read_ipc(
        _resolve_feature_path(weekly_path, required_name="WEEKLY_MIXED_FREQUENCY_FEATURE")
    )
    output = combine_daily_weekly_mixed_frequency_features(
        daily_feature=daily_feature,
        weekly_feature=weekly_feature,
    )
    out_dir = root / save_path / symbol / contract / target_freq
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
    parser.add_argument(
        "--feature_path",
        default="PREPROCESS_DATASET/commodity-futures/MIXED_FREQUENCY_FEATURE",
    )
    parser.add_argument(
        "--save_path",
        default="PREPROCESS_DATASET/commodity-futures/MIXED_FREQUENCY_FEATURE",
    )
    return parser


def main(args=None) -> Path:
    parsed = build_parser().parse_args(args)
    return write_mixed_frequency_feature_for_day(
        root_path=parsed.root_path,
        symbol=parsed.symbol,
        contract=parsed.contract,
        target_freq=parsed.target_freq,
        date=parsed.date,
        feature_path=parsed.feature_path,
        save_path=parsed.save_path,
    )
