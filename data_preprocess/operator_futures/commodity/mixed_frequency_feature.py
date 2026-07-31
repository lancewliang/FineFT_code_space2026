from pathlib import Path
import argparse
import warnings

import polars as pl

from .daily_base_feature import read_base_feature, trading_day_text
from .daily_mixed_frequency_feature import (
    PREV_DAY_FEATURE_COLUMNS,
    validate_daily_mixed_frequency_output,
)
from .weekly_base_feature import calendar_week_text
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
    target_bars: pl.DataFrame,
    daily_feature: pl.DataFrame,
    weekly_feature: pl.DataFrame,
) -> pl.DataFrame:
    if (
        "timestamp" not in target_bars.columns
        or "trading_day" not in target_bars.columns
    ):
        raise ValueError("target_bars must contain timestamp and trading_day columns")
    validate_daily_mixed_frequency_output(daily_feature)
    validate_weekly_mixed_frequency_output(weekly_feature)
    if daily_feature.get_column("trading_day").n_unique() != daily_feature.height:
        raise ValueError(
            "daily Mixed-frequency State Feature trading_day values must be unique"
        )
    if weekly_feature.get_column("calendar_week").n_unique() != weekly_feature.height:
        raise ValueError(
            "weekly Mixed-frequency State Feature calendar_week values must be unique"
        )

    missing_daily = [
        column for column in PREV_DAY_FEATURE_COLUMNS if column not in daily_feature.columns
    ]
    if missing_daily:
        raise ValueError(
            f"Missing daily Mixed-frequency State Feature columns: {missing_daily}"
        )
    missing_weekly = [
        column for column in PREV_WEEK_FEATURE_COLUMNS if column not in weekly_feature.columns
    ]
    if missing_weekly:
        raise ValueError(
            f"Missing weekly Mixed-frequency State Feature columns: {missing_weekly}"
        )

    target = target_bars.select(["timestamp", "trading_day"]).sort(
        ["trading_day", "timestamp"]
    ).with_columns(
        pl.col("trading_day")
        .map_elements(trading_day_text, return_dtype=pl.Utf8)
        .alias("trading_day"),
        pl.col("trading_day")
        .map_elements(calendar_week_text, return_dtype=pl.Utf8)
        .alias("calendar_week"),
    )
    daily_aligned = target.join(
        daily_feature.select(["trading_day", *PREV_DAY_FEATURE_COLUMNS]),
        on="trading_day",
        how="left",
    ).select(["timestamp", *PREV_DAY_FEATURE_COLUMNS])
    weekly_aligned = target.join(
        weekly_feature.select(["calendar_week", *PREV_WEEK_FEATURE_COLUMNS]),
        on="calendar_week",
        how="left",
    ).select(["timestamp", *PREV_WEEK_FEATURE_COLUMNS])
    result = daily_aligned.join(
        weekly_aligned,
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
    start_date: str,
    end_date: str,
    data_path: str = "PREPROCESS_DATASET/commodity-futures",
    feature_path: str = "PREPROCESS_DATASET/commodity-futures/MIXED_FREQUENCY_FEATURE",
    save_path: str = "PREPROCESS_DATASET/commodity-futures/MIXED_FREQUENCY_FEATURE",
) -> Path | None:
    root = Path(root_path)
    base_feature_path = (
        root
        / data_path
        / "BASE_FEATURE"
        / symbol
        / contract
        / target_freq
        / f"{date}.feather"
    )
    if not base_feature_path.exists():
        warnings.warn(
            "Skipping commodity mixed-frequency feature date with missing "
            f"BASE_FEATURE: symbol={symbol} contract={contract} "
            f"target_freq={target_freq} date={date} path={base_feature_path}",
            stacklevel=2,
        )
        return None

    target_bars = read_base_feature(base_feature_path, date).select(
        ["timestamp", "trading_day"]
    )
    feature_root = root / feature_path / symbol / contract / target_freq
    range_name = f"{start_date}-{end_date}"
    daily_path = feature_root / "DAILY" / f"{range_name}.feather"
    weekly_path = feature_root / "WEEKLY" / f"{range_name}.feather"
    daily_feature = pl.read_ipc(
        _resolve_feature_path(daily_path, required_name="DAILY_MIXED_FREQUENCY_FEATURE")
    )
    weekly_feature = pl.read_ipc(
        _resolve_feature_path(
            weekly_path,
            required_name="WEEKLY_MIXED_FREQUENCY_FEATURE",
        )
    )
    output = combine_daily_weekly_mixed_frequency_features(
        target_bars=target_bars,
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
    parser.add_argument("--start_date", required=True)
    parser.add_argument("--end_date", required=True)
    parser.add_argument(
        "--data_path",
        default="PREPROCESS_DATASET/commodity-futures",
    )
    parser.add_argument(
        "--feature_path",
        default="PREPROCESS_DATASET/commodity-futures/MIXED_FREQUENCY_FEATURE",
    )
    parser.add_argument(
        "--save_path",
        default="PREPROCESS_DATASET/commodity-futures/MIXED_FREQUENCY_FEATURE",
    )
    return parser


def main(args=None) -> Path | None:
    parsed = build_parser().parse_args(args)
    return write_mixed_frequency_feature_for_day(
        root_path=parsed.root_path,
        symbol=parsed.symbol,
        contract=parsed.contract,
        target_freq=parsed.target_freq,
        date=parsed.date,
        start_date=parsed.start_date,
        end_date=parsed.end_date,
        data_path=parsed.data_path,
        feature_path=parsed.feature_path,
        save_path=parsed.save_path,
    )


if __name__ == "__main__":
    main()
