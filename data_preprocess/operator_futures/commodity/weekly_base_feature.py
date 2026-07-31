from datetime import date
from pathlib import Path
import argparse

import polars as pl

from .daily_base_feature import (
    PeriodStats,
    parse_trading_day,
    read_base_features_for_range,
    stats_for_frame,
    validate_bar_input,
)


WEEKLY_BASE_FEATURE_COLUMNS: list[str] = [
    "calendar_week",
    "week_start",
    "week_end",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "tradeval",
    "open_interest_first",
    "open_interest_last",
    "vwap",
    "twap",
    "ntrade_estimated",
    "ntrade_up_estimated",
    "ntrade_down_estimated",
    "ntrade_flat_estimated",
]


def calendar_week_text(trading_day: object) -> str:
    iso = parse_trading_day(trading_day).isocalendar()
    return f"{int(iso.year):04d}-W{int(iso.week):02d}"


def calendar_week_start(trading_day: object) -> date:
    day = parse_trading_day(trading_day)
    iso = day.isocalendar()
    return day.fromisocalendar(iso.year, iso.week, 1)


def _stats_to_weekly_row(
    stats: PeriodStats, *, week_start: date, week_end: date
) -> dict[str, object]:
    return {
        "calendar_week": str(stats.period_key),
        "week_start": week_start.strftime("%Y-%m-%d"),
        "week_end": week_end.strftime("%Y-%m-%d"),
        "open": stats.open,
        "high": stats.high,
        "low": stats.low,
        "close": stats.close,
        "volume": stats.volume,
        "tradeval": stats.tradeval,
        "open_interest_first": stats.open_interest_first,
        "open_interest_last": stats.open_interest_last,
        "vwap": stats.vwap,
        "twap": stats.twap,
        "ntrade_estimated": stats.ntrade_estimated,
        "ntrade_up_estimated": stats.ntrade_up_estimated,
        "ntrade_down_estimated": stats.ntrade_down_estimated,
        "ntrade_flat_estimated": stats.ntrade_flat_estimated,
    }


def validate_weekly_base_output(weekly_base: pl.DataFrame) -> None:
    for column in WEEKLY_BASE_FEATURE_COLUMNS:
        if column not in weekly_base.columns:
            raise ValueError(f"Missing weekly Mixed-frequency Base Data column: {column}")
    for column in WEEKLY_BASE_FEATURE_COLUMNS:
        if column in {"calendar_week", "week_start", "week_end"}:
            continue
        series = weekly_base.get_column(column).cast(pl.Float64, strict=False)
        invalid = series.is_null() | series.is_nan() | series.is_infinite()
        if invalid.any():
            row = weekly_base.filter(invalid).row(0, named=True)
            raise ValueError(
                f"Invalid weekly Mixed-frequency Base Data output {column!r}: "
                f"value={row.get(column)!r}"
            )


def generate_weekly_base_features(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate target-frequency bars into one row per natural calendar week."""
    validate_bar_input(df)
    weekly_input = df.sort(["trading_day", "timestamp"]).with_columns(
        pl.col("trading_day")
        .map_elements(calendar_week_text, return_dtype=pl.Utf8)
        .alias("calendar_week")
    )
    rows: list[dict[str, object]] = []
    for calendar_week, frame in weekly_input.group_by(
        "calendar_week", maintain_order=True
    ):
        week_text = str(calendar_week[0])
        trading_days = [
            parse_trading_day(value) for value in frame["trading_day"].to_list()
        ]
        week_start = calendar_week_start(trading_days[0])
        stats = stats_for_frame(frame.drop("calendar_week"), week_text)
        rows.append(
            _stats_to_weekly_row(
                stats,
                week_start=week_start,
                week_end=max(trading_days),
            )
        )
    weekly_base = pl.DataFrame(rows).sort("calendar_week")
    validate_weekly_base_output(weekly_base)
    return weekly_base


def write_weekly_base_feature_for_range(
    *,
    root_path: str | Path,
    symbol: str,
    contract: str,
    target_freq: str,
    start_date: str,
    end_date: str,
    data_path: str = "PREPROCESS_DATASET/commodity-futures",
    save_path: str = "PREPROCESS_DATASET/commodity-futures/MIXED_FREQUENCY_BASE",
) -> Path:
    frame = read_base_features_for_range(
        root_path=root_path,
        symbol=symbol,
        contract=contract,
        target_freq=target_freq,
        start_date=start_date,
        end_date=end_date,
        data_path=data_path,
    )
    weekly_base = generate_weekly_base_features(frame)
    out_dir = Path(root_path) / save_path / symbol / contract / target_freq / "WEEKLY"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{start_date}-{end_date}.feather"
    weekly_base.write_ipc(out_path)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=Path, default=Path("."))
    parser.add_argument("--symbol", "--symbols", dest="symbol", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--target_freq", required=True)
    parser.add_argument("--start_date", required=True)
    parser.add_argument("--end_date", required=True)
    parser.add_argument(
        "--data_path",
        default="PREPROCESS_DATASET/commodity-futures",
    )
    parser.add_argument(
        "--save_path",
        default="PREPROCESS_DATASET/commodity-futures/MIXED_FREQUENCY_BASE",
    )
    return parser


def main(args=None) -> Path:
    parsed = build_parser().parse_args(args)
    return write_weekly_base_feature_for_range(
        root_path=parsed.root_path,
        symbol=parsed.symbol,
        contract=parsed.contract,
        target_freq=parsed.target_freq,
        start_date=parsed.start_date,
        end_date=parsed.end_date,
        data_path=parsed.data_path,
        save_path=parsed.save_path,
    )


if __name__ == "__main__":
    main()
