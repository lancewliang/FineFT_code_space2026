from pathlib import Path
import argparse
import math

import polars as pl

from .daily_base_feature import (
    DAILY_BASE_FEATURE_COLUMNS,
    parse_trading_day,
    read_base_feature,
)


PREV_DAY_FEATURE_COLUMNS: list[str] = [
    "prev_day_return",
    "prev_day_range_pct",
    "prev_day_body_pct",
    "prev_day_upper_shadow_pct",
    "prev_day_lower_shadow_pct",
    "prev_day_volume",
    "prev_day_tradeval",
    "prev_day_open_interest_change",
    "prev_day_turnover_rate",
]

_FEATURE_SUFFIXES: tuple[str, ...] = (
    "return",
    "range_pct",
    "body_pct",
    "upper_shadow_pct",
    "lower_shadow_pct",
    "volume",
    "tradeval",
    "open_interest_change",
    "turnover_rate",
)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    value = numerator / denominator
    return float(value) if math.isfinite(value) else 0.0


def period_features(prefix: str, row: dict[str, object] | None) -> dict[str, float]:
    if row is None:
        return {f"{prefix}_{suffix}": 0.0 for suffix in _FEATURE_SUFFIXES}

    open_price = float(row["open"])
    high = float(row["high"])
    low = float(row["low"])
    close = float(row["close"])
    volume = float(row["volume"])
    tradeval = float(row["tradeval"])
    oi_first = float(row["open_interest_first"])
    oi_last = float(row["open_interest_last"])
    body_top = max(open_price, close)
    body_bottom = min(open_price, close)
    values = {
        f"{prefix}_return": _safe_ratio(close - open_price, open_price),
        f"{prefix}_range_pct": _safe_ratio(high - low, close),
        f"{prefix}_body_pct": _safe_ratio(close - open_price, open_price),
        f"{prefix}_upper_shadow_pct": _safe_ratio(high - body_top, open_price),
        f"{prefix}_lower_shadow_pct": _safe_ratio(body_bottom - low, open_price),
        f"{prefix}_volume": volume,
        f"{prefix}_tradeval": tradeval,
        f"{prefix}_open_interest_change": _safe_ratio(oi_last - oi_first, oi_first),
        f"{prefix}_turnover_rate": _safe_ratio(volume, oi_last),
    }
    return {
        key: float(value) if math.isfinite(float(value)) else 0.0
        for key, value in values.items()
    }


def previous_key(keys: list[str], current: str) -> str | None:
    previous = [key for key in keys if key < current]
    return previous[-1] if previous else None


def _rows_from_daily_base(daily_base: pl.DataFrame) -> dict[str, dict[str, object]]:
    missing = [
        column for column in DAILY_BASE_FEATURE_COLUMNS if column not in daily_base.columns
    ]
    if missing:
        raise ValueError(f"Missing daily Mixed-frequency Base Data columns: {missing}")
    return {
        str(row["trading_day"]): row
        for row in daily_base.iter_rows(named=True)
    }


def generate_daily_mixed_frequency_features_from_base(
    *, target_bars: pl.DataFrame, daily_base: pl.DataFrame
) -> pl.DataFrame:
    if "timestamp" not in target_bars.columns or "trading_day" not in target_bars.columns:
        raise ValueError("target_bars must contain timestamp and trading_day columns")

    ordered = target_bars.sort(["trading_day", "timestamp"]).with_columns(
        pl.col("trading_day")
        .map_elements(
            lambda value: parse_trading_day(value).strftime("%Y-%m-%d"),
            return_dtype=pl.Utf8,
        )
        .alias("trading_day")
    )
    daily_rows = _rows_from_daily_base(daily_base)
    daily_keys = sorted(daily_rows)

    rows: list[dict[str, float | object]] = []
    for row in ordered.select("timestamp", "trading_day").iter_rows(named=True):
        trading_day = str(row["trading_day"])
        prev_day_key = previous_key(daily_keys, trading_day)
        feature_row: dict[str, float | object] = {"timestamp": row["timestamp"]}
        feature_row.update(period_features("prev_day", daily_rows.get(prev_day_key)))
        rows.append(feature_row)

    result = pl.DataFrame(rows).select(["timestamp", *PREV_DAY_FEATURE_COLUMNS])
    validate_daily_mixed_frequency_output(result)
    return result


def validate_daily_mixed_frequency_output(df: pl.DataFrame) -> None:
    for column in PREV_DAY_FEATURE_COLUMNS:
        if column not in df.columns:
            raise ValueError(f"Missing daily Mixed-frequency State Feature column: {column}")
        series = df.get_column(column).cast(pl.Float64, strict=False)
        invalid = series.is_null() | series.is_nan() | series.is_infinite()
        if invalid.any():
            row = df.filter(invalid).row(0, named=True)
            raise ValueError(
                f"Invalid daily Mixed-frequency State Feature output {column!r}: "
                f"value={row.get(column)!r} timestamp={row.get('timestamp')!r}"
            )


def _resolve_base_path(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"MIXED_FREQUENCY_BASE DAILY path does not exist: {path}")
    return path


def write_daily_mixed_frequency_feature_for_day(
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
    daily_base_path = (
        root
        / base_path
        / symbol
        / contract
        / target_freq
        / "DAILY"
        / f"{range_name}.feather"
    )
    daily_base = pl.read_ipc(_resolve_base_path(daily_base_path))
    current = read_base_feature(
        root / data_path / "BASE_FEATURE" / symbol / contract / target_freq / f"{date}.feather",
        date,
    )
    output = generate_daily_mixed_frequency_features_from_base(
        target_bars=current.select("timestamp", "trading_day"),
        daily_base=daily_base,
    )
    if output.get_column("timestamp").to_list() != current.get_column("timestamp").to_list():
        raise ValueError(
            f"DAILY_MIXED_FREQUENCY_FEATURE timestamp set does not match BASE_FEATURE for date {date}"
        )
    out_dir = root / save_path / symbol / contract / target_freq / "DAILY"
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
    return write_daily_mixed_frequency_feature_for_day(
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
