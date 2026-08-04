from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
import argparse
import math

import polars as pl


REQUIRED_BAR_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "tradeval",
    "open_interest",
    "vwap",
    "twap",
    "ntrade_estimated",
    "ntrade_up_estimated",
    "ntrade_down_estimated",
    "ntrade_flat_estimated",
)

LIMIT_RATIO_BAR_COLUMNS: tuple[str, ...] = (
    "limit_up_single_sided_ratio",
    "limit_down_single_sided_ratio",
)

DAILY_BASE_FEATURE_COLUMNS: list[str] = [
    "trading_day",
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
    *LIMIT_RATIO_BAR_COLUMNS,
]


@dataclass(frozen=True)
class PeriodStats:
    period_key: object
    open: float
    high: float
    low: float
    close: float
    volume: float
    tradeval: float
    open_interest_first: float
    open_interest_last: float
    vwap: float
    twap: float
    ntrade_estimated: float
    ntrade_up_estimated: float
    ntrade_down_estimated: float
    ntrade_flat_estimated: float
    limit_up_single_sided_ratio: float = 0.0
    limit_down_single_sided_ratio: float = 0.0


def parse_trading_day(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value)
    if "-" in text:
        return datetime.strptime(text, "%Y-%m-%d").date()
    return datetime.strptime(text, "%Y%m%d").date()


def trading_day_text(value: object) -> str:
    return parse_trading_day(value).strftime("%Y-%m-%d")


def validate_bar_input(df: pl.DataFrame) -> None:
    missing = [column for column in REQUIRED_BAR_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for Mixed-frequency Base Data: {missing}")
    if "trading_day" not in df.columns:
        raise ValueError("Missing required column for Mixed-frequency Base Data: trading_day")

    numeric_columns = (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "tradeval",
        "open_interest",
        "vwap",
        "twap",
        "ntrade_estimated",
        "ntrade_up_estimated",
        "ntrade_down_estimated",
        "ntrade_flat_estimated",
    )
    for column in numeric_columns:
        series = df.get_column(column).cast(pl.Float64, strict=False)
        invalid = series.is_null() | series.is_nan() | series.is_infinite()
        if column in {"open", "high", "low", "close", "vwap", "twap"}:
            invalid = invalid | (series <= 0.0)
        if column.startswith("ntrade_"):
            invalid = invalid | (series < 0.0)
        if invalid.any():
            row = df.filter(invalid).row(0, named=True)
            raise ValueError(
                f"Invalid Mixed-frequency Base Data input {column!r}: "
                f"value={row.get(column)!r} timestamp={row.get('timestamp')!r}"
            )


def _weighted_average(values: pl.Series, weights: pl.Series, fallback: float) -> float:
    weighted_sum = float((values.cast(pl.Float64) * weights.cast(pl.Float64)).sum())
    total_weight = float(weights.cast(pl.Float64).sum())
    if total_weight <= 0.0:
        return fallback
    value = weighted_sum / total_weight
    return float(value) if math.isfinite(value) else fallback


def stats_for_frame(frame: pl.DataFrame, period_key: object) -> PeriodStats:
    sorted_frame = frame.sort("timestamp")
    close = float(sorted_frame.get_column("close")[-1])
    return PeriodStats(
        period_key=period_key,
        open=float(sorted_frame.get_column("open")[0]),
        high=float(sorted_frame.get_column("high").max()),
        low=float(sorted_frame.get_column("low").min()),
        close=close,
        volume=float(sorted_frame.get_column("volume").sum()),
        tradeval=float(sorted_frame.get_column("tradeval").sum()),
        open_interest_first=float(sorted_frame.get_column("open_interest")[0]),
        open_interest_last=float(sorted_frame.get_column("open_interest")[-1]),
        vwap=_weighted_average(
            sorted_frame.get_column("vwap"),
            sorted_frame.get_column("volume"),
            close,
        ),
        twap=float(sorted_frame.get_column("twap").mean()),
        ntrade_estimated=float(sorted_frame.get_column("ntrade_estimated").sum()),
        ntrade_up_estimated=float(sorted_frame.get_column("ntrade_up_estimated").sum()),
        ntrade_down_estimated=float(
            sorted_frame.get_column("ntrade_down_estimated").sum()
        ),
        ntrade_flat_estimated=float(
            sorted_frame.get_column("ntrade_flat_estimated").sum()
        ),
        limit_up_single_sided_ratio=float(sorted_frame.get_column("limit_up_single_sided_ratio")[-1]) if "limit_up_single_sided_ratio" in sorted_frame.columns else 0.0,
        limit_down_single_sided_ratio=float(sorted_frame.get_column("limit_down_single_sided_ratio")[-1]) if "limit_down_single_sided_ratio" in sorted_frame.columns else 0.0,
    )


def period_stats_from_base_row(row: dict[str, object], *, key_column: str) -> PeriodStats:
    return PeriodStats(
        period_key=str(row[key_column]),
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        volume=float(row["volume"]),
        tradeval=float(row["tradeval"]),
        open_interest_first=float(row["open_interest_first"]),
        open_interest_last=float(row["open_interest_last"]),
        vwap=float(row["vwap"]),
        twap=float(row["twap"]),
        ntrade_estimated=float(row["ntrade_estimated"]),
        ntrade_up_estimated=float(row["ntrade_up_estimated"]),
        ntrade_down_estimated=float(row["ntrade_down_estimated"]),
        ntrade_flat_estimated=float(row["ntrade_flat_estimated"]),
        limit_up_single_sided_ratio=float(row.get("limit_up_single_sided_ratio", 0.0)),
        limit_down_single_sided_ratio=float(row.get("limit_down_single_sided_ratio", 0.0)),
    )


def read_base_feature(path: Path, trading_day: str) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"BASE_FEATURE path does not exist: {path}")
    frame = pl.read_ipc(path)
    if "trading_day" not in frame.columns:
        frame = frame.with_columns(
            pl.lit(trading_day_text(trading_day)).alias("trading_day")
        )
    return frame


def base_feature_paths(
    *,
    data_root: Path,
    symbol: str,
    contract: str,
    target_freq: str,
    through_date: str,
) -> list[Path]:
    base_dir = data_root / "BASE_FEATURE" / symbol / contract / target_freq
    if not base_dir.exists():
        raise FileNotFoundError(f"BASE_FEATURE directory does not exist: {base_dir}")
    through = parse_trading_day(through_date)
    paths = []
    for path in sorted(base_dir.glob("*.feather")):
        try:
            day = parse_trading_day(path.stem)
        except ValueError:
            continue
        if day <= through:
            paths.append(path)
    if not paths:
        raise FileNotFoundError(
            f"No BASE_FEATURE files found through {through_date} in {base_dir}"
        )
    return paths


def read_base_features_for_range(
    *,
    root_path: str | Path,
    symbol: str,
    contract: str,
    target_freq: str,
    start_date: str,
    end_date: str,
    data_path: str = "PREPROCESS_DATASET/commodity-futures",
) -> pl.DataFrame:
    root = Path(root_path)
    data_root = root / data_path
    paths = base_feature_paths(
        data_root=data_root,
        symbol=symbol,
        contract=contract,
        target_freq=target_freq,
        through_date=end_date,
    )
    start = parse_trading_day(start_date)
    end = parse_trading_day(end_date)
    frames = [
        read_base_feature(path, path.stem)
        for path in paths
        if start <= parse_trading_day(path.stem) < end
    ]
    if not frames:
        raise FileNotFoundError(
            f"No BASE_FEATURE files found in [{start_date}, {end_date}) "
            f"for {symbol}/{contract}/{target_freq}"
        )
    return pl.concat(frames, how="vertical")


def _stats_to_daily_row(stats: PeriodStats) -> dict[str, object]:
    return {
        "trading_day": str(stats.period_key),
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
        "limit_up_single_sided_ratio": stats.limit_up_single_sided_ratio,
        "limit_down_single_sided_ratio": stats.limit_down_single_sided_ratio,
    }


def validate_daily_base_output(daily_base: pl.DataFrame) -> None:
    for column in DAILY_BASE_FEATURE_COLUMNS:
        if column not in daily_base.columns:
            raise ValueError(f"Missing daily Mixed-frequency Base Data column: {column}")
    for column in DAILY_BASE_FEATURE_COLUMNS:
        if column == "trading_day":
            continue
        series = daily_base.get_column(column).cast(pl.Float64, strict=False)
        invalid = series.is_null() | series.is_nan() | series.is_infinite()
        if invalid.any():
            row = daily_base.filter(invalid).row(0, named=True)
            raise ValueError(
                f"Invalid daily Mixed-frequency Base Data output {column!r}: "
                f"value={row.get(column)!r}"
            )


def generate_daily_base_features(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate target-frequency bars into one row per TradingDay."""
    validate_bar_input(df)
    ordered = df.sort(["trading_day", "timestamp"]).with_columns(
        pl.col("trading_day")
        .map_elements(trading_day_text, return_dtype=pl.Utf8)
        .alias("trading_day")
    )
    rows: list[dict[str, object]] = []
    for trading_day, frame in ordered.group_by("trading_day", maintain_order=True):
        day_text = trading_day_text(trading_day[0])
        rows.append(_stats_to_daily_row(stats_for_frame(frame, day_text)))
    daily_base = pl.DataFrame(rows).sort("trading_day")
    validate_daily_base_output(daily_base)
    return daily_base


def write_daily_base_feature_for_range(
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
    daily_base = generate_daily_base_features(frame)
    out_dir = Path(root_path) / save_path / symbol / contract / target_freq / "DAILY"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{start_date}-{end_date}.feather"
    daily_base.write_ipc(out_path)
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
    return write_daily_base_feature_for_range(
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
