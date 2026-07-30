import logging
import numbers
from datetime import datetime, timedelta
from typing import List

import polars as pl

from .config import TradingSession, get_commodity_config
from .main_contract import with_normalized_timestamp


logger = logging.getLogger(__name__)

BID_PRICE_COLUMNS = [f"BidPrice{level}" for level in range(1, 6)]
ASK_PRICE_COLUMNS = [f"AskPrice{level}" for level in range(1, 6)]
BID_VOLUME_COLUMNS = [f"BidVolume{level}" for level in range(1, 6)]
ASK_VOLUME_COLUMNS = [f"AskVolume{level}" for level in range(1, 6)]
QUOTE_DEPTH_IMBALANCE_LEVELS = (1, 3, 5)
DEPTH_PRICE_COLUMNS = BID_PRICE_COLUMNS + ASK_PRICE_COLUMNS
SECOND_LEVEL_PRICE_COLUMNS = (
    DEPTH_PRICE_COLUMNS
    + [
        "LastPrice",
        "OpenPrice",
        "LowPrice",
        "HighPrice",
        "LowerLimitPrice",
        "UpperLimitPrice",
    ]
)
OHLC_PRICE_COLUMNS = ("OpenPrice", "HighPrice", "LowPrice")
SOURCE_LINE_COLUMN = "_source_line_number"
SECOND_LEVEL_ROW_COLUMN = "_second_level_row_number"


def _polars_freq(target_freq: str) -> str:
    return target_freq.replace("min", "m")


def _target_freq_delta(target_freq: str) -> timedelta:
    normalized = target_freq.strip().lower()
    for suffix, multiplier in (
        ("min", 60),
        ("m", 60),
        ("s", 1),
        ("h", 3600),
    ):
        if normalized.endswith(suffix):
            value = int(normalized[: -len(suffix)])
            return timedelta(seconds=value * multiplier)
    raise ValueError(f"Unsupported target frequency: {target_freq}")


def _session_for_timestamp(
    timestamp: datetime, sessions: tuple[TradingSession, ...]
) -> TradingSession | None:
    timestamp_time = timestamp.time()
    for session in sessions:
        if session.start <= timestamp_time <= session.end:
            return session
    return None


def _same_trading_session(
    previous: datetime,
    current: datetime,
    sessions: tuple[TradingSession, ...],
) -> bool:
    previous_session = _session_for_timestamp(previous, sessions)
    current_session = _session_for_timestamp(current, sessions)
    return (
        previous_session is not None
        and previous_session == current_session
        and previous.date() == current.date()
    )


def _resample(frame: pl.DataFrame, target_freq: str, aggs: List[pl.Expr]) -> pl.DataFrame:
    return (
        frame.sort("timestamp")
        .group_by_dynamic(
            "timestamp",
            every=_polars_freq(target_freq),
            closed="right",
            label="right",
        )
        .agg(*aggs)
        .sort("timestamp")
    )


def validate_best_quotes(df: pl.DataFrame, contract: str) -> None:
    bid_price1 = pl.col("BidPrice1").cast(pl.Float64, strict=False)
    ask_price1 = pl.col("AskPrice1").cast(pl.Float64, strict=False)
    last_price = pl.col("LastPrice").cast(pl.Float64, strict=False)
    low_price = pl.col("LowPrice").cast(pl.Float64, strict=False)
    high_price = pl.col("HighPrice").cast(pl.Float64, strict=False)
    lower_limit_price = pl.col("LowerLimitPrice").cast(pl.Float64, strict=False)
    upper_limit_price = pl.col("UpperLimitPrice").cast(pl.Float64, strict=False)
    normalized = df.with_columns(
        bid_price1.alias("BidPrice1"),
        ask_price1.alias("AskPrice1"),
    )
    invalid = pl.any_horizontal(
        bid_price1.is_null(),
        ask_price1.is_null(),
        bid_price1 <= 0,
        ask_price1 <= 0,
        bid_price1 >= ask_price1,
    )
    limit_down_single_sided = (
        last_price.is_not_null()
        & lower_limit_price.is_not_null()
        & ((last_price == lower_limit_price) | (low_price == lower_limit_price))
        & pl.all_horizontal(
            [pl.col(column).is_null() for column in BID_PRICE_COLUMNS]
        )
        & pl.all_horizontal(
            [pl.col(column).fill_null(0) == 0 for column in BID_VOLUME_COLUMNS]
        )
        & ask_price1.is_not_null()
        & (ask_price1 > 0)
    ).fill_null(False)
    limit_up_single_sided = (
        last_price.is_not_null()
        & upper_limit_price.is_not_null()
        & (
            (last_price == upper_limit_price)
            | (high_price == upper_limit_price)
            | (last_price == low_price)
        )
        & pl.all_horizontal(
            [pl.col(column).is_null() for column in ASK_PRICE_COLUMNS]
        )
        & pl.all_horizontal(
            [pl.col(column).fill_null(0) == 0 for column in ASK_VOLUME_COLUMNS]
        )
        & bid_price1.is_not_null()
        & (bid_price1 > 0)
    ).fill_null(False)
    invalid_rows = normalized.filter(
        invalid & ~(limit_down_single_sided | limit_up_single_sided)
    )
    if invalid_rows.height:
        first = invalid_rows.row(0, named=True)
        bid_price = first.get("BidPrice1")
        ask_price = first.get("AskPrice1")
        reasons = []
        if bid_price is None:
            reasons.append("BidPrice1 is null")
        if ask_price is None:
            reasons.append("AskPrice1 is null")
        if bid_price is not None and bid_price <= 0:
            reasons.append("BidPrice1 <= 0")
        if ask_price is not None and ask_price <= 0:
            reasons.append("AskPrice1 <= 0")
        if (
            bid_price is not None
            and ask_price is not None
            and bid_price >= ask_price
        ):
            reasons.append("BidPrice1 >= AskPrice1")
        raise ValueError(
            "Invalid best quote for "
            f"{contract}: fields=['BidPrice1', 'AskPrice1'], "
            f"TradingDay={first.get('TradingDay')}, "
            f"UpdateTime={first.get('UpdateTime')}, "
            f"BidPrice1={bid_price}, AskPrice1={ask_price}, "
            f"reason={'; '.join(reasons)}, "
            f"row={first}"
        )


def drop_empty_depth_price_rows(df: pl.DataFrame) -> pl.DataFrame:
    empty_depth_prices = pl.all_horizontal(
        [pl.col(column).is_null() for column in DEPTH_PRICE_COLUMNS]
    )
    return df.filter(~empty_depth_prices)


def _log_second_level_gap_fill_rows(
    df: pl.DataFrame,
    mask: pl.Expr,
    *,
    source_file: str | None,
    rule: str,
    columns: list[str],
    fill_value_column: str,
) -> None:
    if not logger.isEnabledFor(logging.INFO):
        return

    logged = df.with_row_index(SECOND_LEVEL_ROW_COLUMN, offset=1).filter(mask)
    if logged.is_empty():
        return

    select_columns = [SECOND_LEVEL_ROW_COLUMN]
    if SOURCE_LINE_COLUMN in logged.columns:
        select_columns.append(SOURCE_LINE_COLUMN)
    if "timestamp" in logged.columns:
        select_columns.append("timestamp")
    select_columns.extend(columns)
    select_columns.append(fill_value_column)

    for row in logged.select(select_columns).iter_rows(named=True):
        old_values = {column: row.get(column) for column in columns}
        logger.info(
            "Second-level gap filled: source_file=%s source_line=%s second_level_row=%s timestamp=%s rule=%s columns=%s old_values=%s new_value=%s",
            source_file,
            row.get(SOURCE_LINE_COLUMN),
            row.get(SECOND_LEVEL_ROW_COLUMN),
            row.get("timestamp"),
            rule,
            ",".join(columns),
            old_values,
            row.get(fill_value_column),
        )


def _fill_second_level_price_gaps(
    df: pl.DataFrame, source_file: str | None = None
) -> pl.DataFrame:
    if all(
        column in df.columns
        for column in (*OHLC_PRICE_COLUMNS, "Volume", "Turnover", "LastPrice")
    ):
        empty_ohlc_no_trade = (
            pl.all_horizontal(
                [pl.col(column).is_null() for column in OHLC_PRICE_COLUMNS]
            )
            & (pl.col("Volume").fill_null(0) == 0)
            & (pl.col("Turnover").fill_null(0) == 0)
        )
        _log_second_level_gap_fill_rows(
            df,
            empty_ohlc_no_trade,
            source_file=source_file,
            rule="empty_ohlc_no_trade",
            columns=list(OHLC_PRICE_COLUMNS),
            fill_value_column="LastPrice",
        )
        df = df.with_columns(
            [
                pl.when(empty_ohlc_no_trade)
                .then(pl.col("LastPrice"))
                .otherwise(pl.col(column))
                .alias(column)
                for column in OHLC_PRICE_COLUMNS
            ]
        )

    if all(
        column in df.columns
        for column in (
            "LastPrice",
            "LowPrice",
            "UpperLimitPrice",
            *ASK_PRICE_COLUMNS,
            *ASK_VOLUME_COLUMNS,
        )
    ):
        limit_up_empty_asks = (
            (
                (pl.col("LastPrice") == pl.col("UpperLimitPrice"))
                | (pl.col("HighPrice") == pl.col("UpperLimitPrice"))
                | (pl.col("LastPrice") == pl.col("LowPrice"))
            )
            & pl.all_horizontal(
                [pl.col(column).is_null() for column in ASK_PRICE_COLUMNS]
            )
            & pl.all_horizontal(
                [pl.col(column).fill_null(0) == 0 for column in ASK_VOLUME_COLUMNS]
            )
        ).fill_null(False)
        _log_second_level_gap_fill_rows(
            df,
            limit_up_empty_asks,
            source_file=source_file,
            rule="limit_up_empty_asks",
            columns=ASK_PRICE_COLUMNS,
            fill_value_column="UpperLimitPrice",
        )
        df = df.with_columns(
            [
                pl.when(limit_up_empty_asks)
                .then(pl.col("UpperLimitPrice"))
                .otherwise(pl.col(column))
                .alias(column)
                for column in ASK_PRICE_COLUMNS
            ]
        )

    if all(
        column in df.columns
        for column in (
            "LastPrice",
            "LowerLimitPrice",
            *BID_PRICE_COLUMNS,
            *BID_VOLUME_COLUMNS,
        )
    ):
        limit_down_empty_bids = (
            (
                (pl.col("LastPrice") == pl.col("LowerLimitPrice"))
                | (pl.col("LowPrice") == pl.col("LowerLimitPrice"))
            )
            & pl.all_horizontal(
                [pl.col(column).is_null() for column in BID_PRICE_COLUMNS]
            )
            & pl.all_horizontal(
                [pl.col(column).fill_null(0) == 0 for column in BID_VOLUME_COLUMNS]
            )
        ).fill_null(False)
        _log_second_level_gap_fill_rows(
            df,
            limit_down_empty_bids,
            source_file=source_file,
            rule="limit_down_empty_bids",
            columns=BID_PRICE_COLUMNS,
            fill_value_column="LowerLimitPrice",
        )
        df = df.with_columns(
            [
                pl.when(limit_down_empty_bids)
                .then(pl.col("LowerLimitPrice"))
                .otherwise(pl.col(column))
                .alias(column)
                for column in BID_PRICE_COLUMNS
            ]
        )

    for level in range(2, 6):
        price_column = f"AskPrice{level}"
        volume_column = f"AskVolume{level}"
        previous_price_column = f"AskPrice{level - 1}"
        previous_volume_column = f"AskVolume{level - 1}"
        if all(
            column in df.columns
            for column in (
                price_column,
                volume_column,
                previous_price_column,
                previous_volume_column,
            )
        ):
            empty_ask_level = (
                (pl.col(volume_column).fill_null(0) == 0)
                & pl.col(price_column).is_null()
            )
            _log_second_level_gap_fill_rows(
                df,
                empty_ask_level,
                source_file=source_file,
                rule=f"empty_ask_level_{level}",
                columns=[volume_column],
                fill_value_column=previous_volume_column,
            )
            df = df.with_columns(
                [
                    pl.when(empty_ask_level)
                    .then(pl.col(previous_price_column))
                    .otherwise(pl.col(price_column))
                    .alias(price_column),
                    pl.when(empty_ask_level)
                    .then(pl.col(previous_volume_column))
                    .otherwise(pl.col(volume_column))
                    .alias(volume_column),
                ]
            )
    for level in range(2, 6):
        price_column = f"BidPrice{level}"
        volume_column = f"BidVolume{level}"
        previous_price_column = f"BidPrice{level - 1}"
        previous_volume_column = f"BidVolume{level - 1}"
        if all(
            column in df.columns
            for column in (
                price_column,
                volume_column,
                previous_price_column,
                previous_volume_column,
            )
        ):
            empty_bid_level = (
                (pl.col(volume_column).fill_null(0) == 0)
                & pl.col(price_column).is_null()
            )
            _log_second_level_gap_fill_rows(
                df,
                empty_bid_level,
                source_file=source_file,
                rule=f"empty_bid_level_{level}",
                columns=[volume_column],
                fill_value_column=previous_volume_column,
            )
            df = df.with_columns(
                [
                    pl.when(empty_bid_level)
                    .then(pl.col(previous_price_column))
                    .otherwise(pl.col(price_column))
                    .alias(price_column),
                    pl.when(empty_bid_level)
                    .then(pl.col(previous_volume_column))
                    .otherwise(pl.col(volume_column))
                    .alias(volume_column),
                ]
            )
    if SOURCE_LINE_COLUMN in df.columns:
        return df.drop(SOURCE_LINE_COLUMN)
    return df


def create_second_level_snapshots(
    df: pl.DataFrame, source_file: str | None = None
) -> pl.DataFrame:
    contract = (
        str(df.item(0, "InstrumentID"))
        if "InstrumentID" in df.columns and df.height
        else "unknown"
    )
    normalized = with_normalized_timestamp(df)
    if SOURCE_LINE_COLUMN not in normalized.columns:
        normalized = normalized.with_row_index(SOURCE_LINE_COLUMN, offset=2)
    copied = drop_empty_depth_price_rows(
        normalized.with_columns(
            [
                pl.col(column).cast(pl.Float64, strict=False).alias(column)
                for column in SECOND_LEVEL_PRICE_COLUMNS
                if column in df.columns
            ]
        )
    )
    validate_best_quotes(copied, contract)
    second = (
        copied.sort("timestamp")
        .with_columns(pl.col("timestamp").dt.truncate("1s").alias("timestamp"))
        .group_by("timestamp", maintain_order=True)
        .agg(pl.exclude("timestamp").last())
        .sort("timestamp")
    )
    return _fill_second_level_price_gaps(second, source_file=source_file)


def _with_reference_price(df: pl.DataFrame) -> pl.DataFrame:
    mid = (pl.col("BidPrice1") + pl.col("AskPrice1")) / 2
    valid = pl.col("LastPrice").is_not_null() & (pl.col("LastPrice") > 0)
    if "UpperLimitPrice" in df.columns:
        valid = valid & (pl.col("LastPrice") <= pl.col("UpperLimitPrice"))
    if "LowerLimitPrice" in df.columns:
        valid = valid & (pl.col("LastPrice") >= pl.col("LowerLimitPrice"))
    return df.with_columns(
        pl.when(valid)
        .then(pl.col("LastPrice"))
        .otherwise(mid)
        .alias("_reference_price")
    )


def downscale_derivative_reference(
    second_df: pl.DataFrame, target_freq: str, symbol: str
) -> pl.DataFrame:
    frame = _with_reference_price(second_df).select(
        "timestamp",
        pl.lit(symbol).alias("symbol"),
        pl.col("timestamp").alias("funding_timestamp"),
        pl.lit(0.0).alias("funding_rate"),
        pl.col("_reference_price").alias("index_price"),
        pl.col("_reference_price").alias("mark_price"),
    )
    return _resample(
        frame,
        target_freq,
        [
            pl.col("symbol").first(),
            pl.col("funding_timestamp").first(),
            pl.col("funding_rate").first(),
            pl.col("index_price").first(),
            pl.col("mark_price").first(),
        ],
    ).drop_nulls("mark_price")


def downscale_orderbook(
    second_df: pl.DataFrame, target_freq: str, depth: int = 5
) -> pl.DataFrame:
    expressions = [pl.col("timestamp")]
    output_columns: List[str] = []
    for level in range(1, depth + 1):
        for output, source in (
            (f"ask{level}_price", f"AskPrice{level}"),
            (f"ask{level}_size", f"AskVolume{level}"),
            (f"bid{level}_price", f"BidPrice{level}"),
            (f"bid{level}_size", f"BidVolume{level}"),
        ):
            expressions.append(pl.col(source).alias(output))
            output_columns.append(output)
    for column in ("LowerLimitPrice", "UpperLimitPrice"):
        if column in second_df.columns:
            expressions.append(pl.col(column))
            output_columns.append(column)
    renamed = second_df.select(expressions)
    result = _resample(
        renamed,
        target_freq,
        [pl.col(column).last().alias(column) for column in output_columns],
    )
    return result.filter(
        ~pl.all_horizontal([pl.col(column).is_null() for column in output_columns])
    )


def _second_trade_frame(second_df: pl.DataFrame, contract_unit: float) -> pl.DataFrame:
    frame = second_df.sort("timestamp").with_columns(
        pl.col("Volume")
        .cast(pl.Float64, strict=False)
        .diff()
        .alias("second_volume"),
        pl.col("Turnover")
        .cast(pl.Float64, strict=False)
        .diff()
        .alias("second_tradeval"),
    )
    invalid_rows = frame.filter(
        (pl.col("second_volume") > 0)
        & (pl.col("second_tradeval").is_null() | (pl.col("second_tradeval") <= 0))
    )
    if invalid_rows.height:
        row = invalid_rows.row(0, named=True)
        raise ValueError(
            "Invalid turnover delta with positive volume: "
            f"timestamp={row.get('timestamp')}, contract={row.get('InstrumentID')}, "
            f"second_volume={row['second_volume']}, "
            f"second_tradeval={row['second_tradeval']}"
        )

    frame = frame.with_columns(
        pl.when(pl.col("second_volume") > 0)
        .then(pl.col("second_tradeval") / pl.col("second_volume") / contract_unit)
        .otherwise(None)
        .alias("second_avg_price")
    ).with_row_index("_row_nr")
    directions = (
        frame.filter(pl.col("second_avg_price").is_not_null())
        .select(
            "_row_nr",
            pl.col("second_avg_price").diff().alias("_price_diff"),
        )
        .with_columns(
            pl.when(pl.col("_price_diff") > 0)
            .then(pl.lit("buy_estimated"))
            .when(pl.col("_price_diff") < 0)
            .then(pl.lit("sell_estimated"))
            .when(pl.col("_price_diff") == 0)
            .then(pl.lit("flat"))
            .otherwise(pl.lit("none"))
            .alias("direction_estimated")
        )
        .select("_row_nr", "direction_estimated")
    )
    return (
        frame.join(directions, on="_row_nr", how="left")
        .with_columns(pl.col("direction_estimated").fill_null("none"))
        .drop("_row_nr")
    )


def downscale_base_features(
    second_df: pl.DataFrame, target_freq: str, symbol: str = "fu"
) -> pl.DataFrame:
    if "OpenInterest" not in second_df.columns:
        raise ValueError("Missing required column for BASE_FEATURE downscaling: OpenInterest")
    oi_series = second_df.get_column("OpenInterest")
    if oi_series.null_count() > 0:
        raise ValueError("OpenInterest contains null values")
    try:
        oi_float = oi_series.cast(pl.Float64, strict=False)
    except Exception as err:
        raise ValueError(f"OpenInterest is non-numeric: {err}")
    if oi_float.null_count() > 0 or oi_float.is_nan().any() or oi_float.is_infinite().any():
        raise ValueError("OpenInterest contains null, non-numeric, or non-finite values")

    contract_unit = get_commodity_config(symbol).contract_unit
    frame = _with_reference_price(
        _second_trade_frame(second_df, contract_unit)
    ).with_columns(
        pl.when(pl.col("second_avg_price").is_not_null())
        .then(pl.col("second_avg_price"))
        .otherwise(pl.col("_reference_price"))
        .alias("price"),
        pl.when(pl.col("second_volume") > 0)
        .then(pl.col("second_volume"))
        .otherwise(0.0)
        .fill_null(0.0)
        .alias("volume"),
        pl.when(pl.col("second_volume") > 0)
        .then(pl.col("second_tradeval"))
        .otherwise(0.0)
        .fill_null(0.0)
        .alias("tradeval"),
    )
    grouped = _resample(
        frame,
        target_freq,
        [
            pl.col("price").first().alias("open"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
            pl.col("tradeval").sum().alias("tradeval"),
            pl.col("OpenInterest").cast(pl.Float64, strict=False).last().alias("open_interest"),
            pl.col("price").mean().alias("awap"),
            (pl.col("second_volume") > 0)
            .fill_null(False)
            .sum()
            .alias("ntrade_estimated"),
            (pl.col("direction_estimated") == "buy_estimated")
            .sum()
            .alias("ntrade_up_estimated"),
            (pl.col("direction_estimated") == "sell_estimated")
            .sum()
            .alias("ntrade_down_estimated"),
            (pl.col("direction_estimated") == "flat")
            .sum()
            .alias("ntrade_flat_estimated"),
        ],
    ).drop_nulls("open")
    return grouped.with_columns(
        pl.when(pl.col("volume") > 0)
        .then(pl.col("tradeval") / pl.col("volume") / contract_unit)
        .otherwise(pl.col("close"))
        .alias("vwap"),
        pl.col("awap").alias("twap"),
    ).select(
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "tradeval",
        "open_interest",
        "vwap",
        "awap",
        "twap",
        "ntrade_estimated",
        "ntrade_up_estimated",
        "ntrade_down_estimated",
        "ntrade_flat_estimated",
    )


def _change_count_expr(column: str, direction: str | None = None) -> pl.Expr:
    diff = pl.col(column).diff()
    if direction == "up":
        return (diff > 0).fill_null(False)
    if direction == "down":
        return (diff < 0).fill_null(False)
    return diff.ne(0).fill_null(True)


def _ofi_required_columns(depth: int) -> list[str]:
    columns = ["timestamp"]
    for level in range(1, depth + 1):
        columns.extend(
            [
                f"BidPrice{level}",
                f"AskPrice{level}",
                f"BidVolume{level}",
                f"AskVolume{level}",
            ]
        )
    return columns


def _non_finite_expr(column: str) -> pl.Expr:
    value = pl.col(column).cast(pl.Float64, strict=False)
    return (
        value.is_nan()
        | (value == float("inf"))
        | (value == -float("inf"))
    ).fill_null(False)


def _validate_ofi_input(
    second_df: pl.DataFrame, window_rows: int, depth: int
) -> list[str]:
    window_rows = _validate_positive_integer_window_rows(window_rows)
    if second_df.height == 0:
        raise ValueError("OFI input has no quote snapshots")

    required_columns = _ofi_required_columns(depth)
    missing = [column for column in required_columns if column not in second_df.columns]
    if missing:
        raise ValueError(f"Missing OFI columns: {', '.join(missing)}")

    null_counts = second_df.select(
        [pl.col(column).null_count().alias(column) for column in required_columns]
    ).row(0, named=True)
    null_columns = [column for column, count in null_counts.items() if count > 0]
    if null_columns:
        raise ValueError(
            f"OFI columns contain null values: {', '.join(null_columns)}"
        )

    numeric_columns = [column for column in required_columns if column != "timestamp"]
    non_finite_counts = second_df.select(
        [_non_finite_expr(column).sum().alias(column) for column in numeric_columns]
    ).row(0, named=True)
    non_finite_columns = [
        column for column, count in non_finite_counts.items() if count > 0
    ]
    if non_finite_columns:
        raise ValueError(
            f"OFI columns contain non-finite values: {', '.join(non_finite_columns)}"
        )

    return required_columns


def _ofi_bid_expr(level: int) -> pl.Expr:
    price = pl.col(f"BidPrice{level}").cast(pl.Float64, strict=False)
    size = pl.col(f"BidVolume{level}").cast(pl.Float64, strict=False)
    previous_price = price.shift(1)
    previous_size = size.shift(1)
    return (
        pl.when(previous_price.is_null())
        .then(pl.lit(0.0))
        .when(price > previous_price)
        .then(size)
        .when(price == previous_price)
        .then(size - previous_size)
        .otherwise(-previous_size)
        .alias(f"ofi_bid{level}")
    )


def _ofi_ask_expr(level: int) -> pl.Expr:
    price = pl.col(f"AskPrice{level}").cast(pl.Float64, strict=False)
    size = pl.col(f"AskVolume{level}").cast(pl.Float64, strict=False)
    previous_price = price.shift(1)
    previous_size = size.shift(1)
    return (
        pl.when(previous_price.is_null())
        .then(pl.lit(0.0))
        .when(price < previous_price)
        .then(-size)
        .when(price == previous_price)
        .then(-(size - previous_size))
        .otherwise(previous_size)
        .alias(f"ofi_ask{level}")
    )


def _safe_divide(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    invalid_denominator = (
        denominator.is_null()
        | (denominator == 0)
        | denominator.is_nan()
        | (denominator == float("inf"))
        | (denominator == -float("inf"))
    ).fill_null(True)
    return pl.when(invalid_denominator).then(0.0).otherwise(numerator / denominator)


def _validate_positive_integer_window_rows(window_rows: int) -> int:
    if isinstance(window_rows, bool) or not isinstance(window_rows, numbers.Integral):
        raise ValueError("window_rows must be a positive integer")
    if window_rows <= 0:
        raise ValueError("window_rows must be positive")
    return int(window_rows)


def _quote_microstructure_required_columns() -> list[str]:
    return [
        "timestamp",
        "BidPrice1",
        "AskPrice1",
        "BidVolume1",
        "AskVolume1",
        "LastPrice",
        "LowPrice",
        "HighPrice",
        "LowerLimitPrice",
        "UpperLimitPrice",
    ]


def _validate_quote_microstructure_input(
    second_df: pl.DataFrame, window_rows: int
) -> list[str]:
    window_rows = _validate_positive_integer_window_rows(window_rows)
    if second_df.height == 0:
        raise ValueError("Microstructure input has no quote snapshots")

    required_columns = _quote_microstructure_required_columns()
    missing = [column for column in required_columns if column not in second_df.columns]
    if missing:
        raise ValueError(
            f"Missing microstructure columns: {', '.join(missing)}"
        )

    null_counts = second_df.select(
        [pl.col(column).null_count().alias(column) for column in required_columns]
    ).row(0, named=True)
    null_columns = [column for column, count in null_counts.items() if count > 0]
    if null_columns:
        raise ValueError(
            f"Microstructure columns contain null values: {', '.join(null_columns)}"
        )

    numeric_columns = [column for column in required_columns if column != "timestamp"]
    non_numeric_counts = second_df.select(
        [
            (
                pl.col(column).is_not_null()
                & pl.col(column).cast(pl.Float64, strict=False).is_null()
            )
            .sum()
            .alias(column)
            for column in numeric_columns
        ]
    ).row(0, named=True)
    non_numeric_columns = [
        column for column, count in non_numeric_counts.items() if count > 0
    ]
    if non_numeric_columns:
        raise ValueError(
            "Microstructure columns contain non-numeric values: "
            f"{', '.join(non_numeric_columns)}"
        )

    non_finite_counts = second_df.select(
        [_non_finite_expr(column).sum().alias(column) for column in numeric_columns]
    ).row(0, named=True)
    non_finite_columns = [
        column for column, count in non_finite_counts.items() if count > 0
    ]
    if non_finite_columns:
        raise ValueError(
            "Microstructure columns contain non-finite values: "
            f"{', '.join(non_finite_columns)}"
        )

    return required_columns


def _quote_queue_event_expr(
    price_column: str, size_column: str, direction: str
) -> pl.Expr:
    same_price = pl.col(price_column).diff() == 0
    size_diff = pl.col(size_column).diff()
    if direction == "refill":
        size_event = size_diff > 0
    elif direction == "deplete":
        size_event = size_diff < 0
    else:
        raise ValueError(f"Unsupported queue event direction: {direction}")

    return (
        same_price
        & size_event
    ).fill_null(False)


def _quote_side_empty_expr(price_column: str, size_column: str) -> pl.Expr:
    return (
        pl.col(price_column).is_null()
        | (pl.col(price_column) <= 0)
        | (pl.col(size_column) <= 0)
    ).fill_null(False)


def _limit_single_sided_expr(side: str) -> pl.Expr:
    if side == "up":
        return (
            _quote_side_empty_expr("ask_price", "ask_size")
            & (
                (pl.col("LastPrice") == pl.col("UpperLimitPrice"))
                | (pl.col("HighPrice") == pl.col("UpperLimitPrice"))
            )
            & (pl.col("bid_price") > 0)
            & (pl.col("bid_size") > 0)
        ).fill_null(False)
    if side == "down":
        return (
            _quote_side_empty_expr("bid_price", "bid_size")
            & (
                (pl.col("LastPrice") == pl.col("LowerLimitPrice"))
                | (pl.col("LowPrice") == pl.col("LowerLimitPrice"))
            )
            & (pl.col("ask_price") > 0)
            & (pl.col("ask_size") > 0)
        ).fill_null(False)
    raise ValueError(f"Unsupported single-sided side: {side}")


def _quote_depth_volume_columns(depth: int = 5) -> list[str]:
    columns = []
    for level in range(1, depth + 1):
        columns.extend([f"BidVolume{level}", f"AskVolume{level}"])
    return columns


def _validate_quote_depth_imbalance_input(second_df: pl.DataFrame) -> None:
    volume_columns = _quote_depth_volume_columns()
    missing = [column for column in volume_columns if column not in second_df.columns]
    if missing:
        raise ValueError(
            f"Missing quote depth volume columns: {', '.join(missing)}"
        )

    non_finite_counts = second_df.select(
        [_non_finite_expr(column).sum().alias(column) for column in volume_columns]
    ).row(0, named=True)
    non_finite_columns = [
        column for column, count in non_finite_counts.items() if count > 0
    ]
    if non_finite_columns:
        raise ValueError(
            "Quote volume columns contain non-finite values: "
            f"{', '.join(non_finite_columns)}"
        )


def _quote_volume_expr(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.Float64, strict=False).fill_null(0.0)


def _depth_imbalance_expr(depth: int) -> pl.Expr:
    bid_volume = pl.sum_horizontal(
        [_quote_volume_expr(f"BidVolume{level}") for level in range(1, depth + 1)]
    )
    ask_volume = pl.sum_horizontal(
        [_quote_volume_expr(f"AskVolume{level}") for level in range(1, depth + 1)]
    )
    return _safe_divide(bid_volume - ask_volume, bid_volume + ask_volume)


def _quote_window_stat_aggs(
    names: list[str], std_names: set[str] | None = None
) -> list[pl.Expr]:
    std_names = std_names or set()
    aggs: list[pl.Expr] = []
    for name in names:
        aggs.extend(
            [
                pl.col(name).first().alias(f"open_{name}"),
                pl.col(name).max().alias(f"high_{name}"),
                pl.col(name).min().alias(f"low_{name}"),
                pl.col(name).last().alias(f"close_{name}"),
                pl.col(name).mean().alias(f"awap_{name}"),
                pl.col(name).mean().alias(f"twap_{name}"),
            ]
        )
        if name in std_names:
            aggs.append(pl.col(name).std().fill_null(0.0).alias(f"std_{name}"))
    return aggs


def _normalize_limit_single_sided_quote_prices(df: pl.DataFrame) -> pl.DataFrame:
    def optional_column(name: str) -> pl.Expr:
        if name in df.columns:
            return pl.col(name)
        return pl.lit(None, dtype=pl.Float64)

    last_price = optional_column("LastPrice")
    low_price = optional_column("LowPrice")
    high_price = optional_column("HighPrice")
    lower_limit = optional_column("LowerLimitPrice")
    upper_limit = optional_column("UpperLimitPrice")
    limit_down = (
        pl.col("BidPrice1").is_null()
        & (pl.col("BidVolume1").fill_null(0) == 0)
        & pl.col("AskPrice1").is_not_null()
        & lower_limit.is_not_null()
        & (
            (last_price == lower_limit)
            | (low_price == lower_limit)
        )
    ).fill_null(False)
    limit_up = (
        pl.col("AskPrice1").is_null()
        & (pl.col("AskVolume1").fill_null(0) == 0)
        & pl.col("BidPrice1").is_not_null()
        & upper_limit.is_not_null()
        & (
            (last_price == upper_limit)
            | (high_price == upper_limit)
        )
    ).fill_null(False)
    return df.with_columns(
        pl.when(limit_down)
        .then(lower_limit)
        .otherwise(pl.col("BidPrice1"))
        .alias("BidPrice1"),
        pl.when(limit_down)
        .then(pl.lit(0))
        .otherwise(pl.col("BidVolume1"))
        .alias("BidVolume1"),
        pl.when(limit_up)
        .then(upper_limit)
        .otherwise(pl.col("AskPrice1"))
        .alias("AskPrice1"),
        pl.when(limit_up)
        .then(pl.lit(0))
        .otherwise(pl.col("AskVolume1"))
        .alias("AskVolume1"),
    )


def downscale_quote_ofi_features(
    second_df: pl.DataFrame, window_rows: int = 12, depth: int = 5
) -> pl.DataFrame:
    required_columns = _validate_ofi_input(second_df, window_rows, depth)
    quote = second_df.sort("timestamp").select(required_columns)

    bid_columns = [f"ofi_bid{level}" for level in range(1, depth + 1)]
    ask_columns = [f"ofi_ask{level}" for level in range(1, depth + 1)]
    bid_size_columns = [f"BidVolume{level}" for level in range(1, depth + 1)]
    ask_size_columns = [f"AskVolume{level}" for level in range(1, depth + 1)]
    ofi_columns = bid_columns + ask_columns

    quote = quote.with_columns(
        *[_ofi_bid_expr(level) for level in range(1, depth + 1)],
        *[_ofi_ask_expr(level) for level in range(1, depth + 1)],
        pl.sum_horizontal(
            [
                pl.col(column).cast(pl.Float64, strict=False)
                for column in bid_size_columns
            ]
        ).alias("_ofi_bid_volume"),
        pl.sum_horizontal(
            [
                pl.col(column).cast(pl.Float64, strict=False)
                for column in ask_size_columns
            ]
        ).alias("_ofi_ask_volume"),
    ).with_columns(
        pl.sum_horizontal([pl.col(column) for column in bid_columns]).alias("ofi_bid"),
        pl.sum_horizontal([pl.col(column) for column in ask_columns]).alias("ofi_ask"),
    )
    quote = quote.with_columns((pl.col("ofi_bid") + pl.col("ofi_ask")).alias("ofi"))
    quote = quote.with_columns(
        (pl.col("_ofi_bid_volume") + pl.col("_ofi_ask_volume")).alias(
            "_ofi_total_volume"
        )
    )

    grouped = (
        quote.with_row_index("_ofi_row_index")
        .with_columns((pl.col("_ofi_row_index") // window_rows).alias("_ofi_window"))
        .group_by("_ofi_window", maintain_order=True)
        .agg(
            pl.col("timestamp").last().alias("timestamp"),
            pl.len().alias("nquote"),
            *[pl.col(column).sum().alias(column) for column in ofi_columns],
            pl.col("ofi_bid").sum().alias("ofi_bid"),
            pl.col("ofi_ask").sum().alias("ofi_ask"),
            pl.col("ofi").sum().alias("ofi"),
            pl.col("_ofi_bid_volume").sum().alias("_ofi_bid_volume"),
            pl.col("_ofi_ask_volume").sum().alias("_ofi_ask_volume"),
            pl.col("_ofi_total_volume").sum().alias("_ofi_total_volume"),
        )
        .sort("_ofi_window")
    )
    grouped = grouped.with_columns(
        _safe_divide(pl.col("ofi_bid"), pl.col("_ofi_bid_volume")).alias(
            "ofi_bid_norm"
        ),
        _safe_divide(pl.col("ofi_ask"), pl.col("_ofi_ask_volume")).alias(
            "ofi_ask_norm"
        ),
        _safe_divide(pl.col("ofi"), pl.col("_ofi_total_volume")).alias("ofi_norm"),
    )
    return grouped.select(
        "timestamp",
        "nquote",
        *ofi_columns,
        "ofi_bid",
        "ofi_ask",
        "ofi",
        "ofi_bid_norm",
        "ofi_ask_norm",
        "ofi_norm",
    )


def downscale_quote_microstructure_features(
    second_df: pl.DataFrame, window_rows: int = 12
) -> pl.DataFrame:
    required_columns = _validate_quote_microstructure_input(second_df, window_rows)
    quote = second_df.sort("timestamp").select(
        required_columns
    ).with_columns(
        pl.col("BidPrice1").cast(pl.Float64, strict=False).alias("bid_price"),
        pl.col("AskPrice1").cast(pl.Float64, strict=False).alias("ask_price"),
        pl.col("BidVolume1").cast(pl.Float64, strict=False).alias("bid_size"),
        pl.col("AskVolume1").cast(pl.Float64, strict=False).alias("ask_size"),
    )
    quote = quote.with_columns(
        (pl.col("ask_price") - pl.col("bid_price")).alias("spread"),
        ((pl.col("ask_price") + pl.col("bid_price")) / 2).alias("mid"),
    ).with_columns(
        _safe_divide(
            pl.col("ask_price") * pl.col("bid_size")
            + pl.col("bid_price") * pl.col("ask_size"),
            pl.col("bid_size") + pl.col("ask_size"),
        ).alias("microprice"),
        _safe_divide(pl.col("spread"), pl.col("mid")).alias("relative_spread"),
        (pl.col("bid_size") + pl.col("ask_size")).alias("_microprice_total_size"),
    ).with_columns(
        pl.when(
            (pl.col("_microprice_total_size") == 0)
            | (pl.col("spread") == 0)
            | (pl.col("mid") == 0)
        )
        .then(0.0)
        .otherwise(
            _safe_divide(
                pl.col("microprice") - pl.col("mid"),
                pl.col("spread"),
            )
        )
        .alias("microprice_pressure"),
    )
    quote = quote.with_row_index("_microstructure_row_index").with_columns(
        (pl.col("_microstructure_row_index") // window_rows).alias(
            "_microstructure_window"
        ),
        (pl.col("_microstructure_row_index") % window_rows).alias(
            "_microstructure_window_pos"
        ),
        pl.col("spread").diff().alias("_spread_diff"),
    )
    quote = quote.with_columns(
        (
            (pl.col("_microstructure_window_pos") == 0)
            | (pl.col("_spread_diff") == 0)
        )
        .fill_null(True)
        .alias("_spread_flat"),
        (
            (pl.col("_microstructure_window_pos") != 0)
            & (pl.col("_spread_diff") > 0)
        )
        .fill_null(False)
        .alias("_spread_widen"),
        (
            (pl.col("_microstructure_window_pos") != 0)
            & (pl.col("_spread_diff") < 0)
        )
        .fill_null(False)
        .alias("_spread_narrow"),
        _quote_queue_event_expr("bid_price", "bid_size", "refill").alias(
            "_bid_refill"
        ),
        _quote_queue_event_expr("bid_price", "bid_size", "deplete").alias(
            "_bid_deplete"
        ),
        _quote_queue_event_expr("ask_price", "ask_size", "refill").alias(
            "_ask_refill"
        ),
        _quote_queue_event_expr("ask_price", "ask_size", "deplete").alias(
            "_ask_deplete"
        ),
        _quote_side_empty_expr("bid_price", "bid_size").alias("_bid_side_empty"),
        _quote_side_empty_expr("ask_price", "ask_size").alias("_ask_side_empty"),
        _limit_single_sided_expr("up").alias("_limit_up_single_sided"),
        _limit_single_sided_expr("down").alias("_limit_down_single_sided"),
    )

    grouped = (
        quote.group_by("_microstructure_window", maintain_order=True)
        .agg(
            pl.col("timestamp").last().alias("timestamp"),
            pl.len().alias("nquote"),
            pl.col("microprice_pressure").mean().alias(
                "mean_microprice_pressure"
            ),
            pl.col("relative_spread").mean().alias("mean_relative_spread"),
            pl.col("_spread_widen").sum().alias("spread_widen_count"),
            pl.col("_spread_narrow").sum().alias("spread_narrow_count"),
            pl.col("_spread_flat").sum().alias("spread_flat_count"),
            pl.col("_bid_refill").sum().alias("bid_refill_count"),
            pl.col("_bid_deplete").sum().alias("bid_deplete_count"),
            pl.col("_ask_refill").sum().alias("ask_refill_count"),
            pl.col("_ask_deplete").sum().alias("ask_deplete_count"),
            pl.col("_bid_side_empty").sum().alias("_bid_side_empty_count"),
            pl.col("_ask_side_empty").sum().alias("_ask_side_empty_count"),
            pl.col("_limit_up_single_sided").sum().alias(
                "_limit_up_single_sided_count"
            ),
            pl.col("_limit_down_single_sided").sum().alias(
                "_limit_down_single_sided_count"
            ),
        )
        .sort("_microstructure_window")
    )
    grouped = grouped.with_columns(
        (
            pl.col("bid_refill_count")
            + pl.col("bid_deplete_count")
            + pl.col("ask_refill_count")
            + pl.col("ask_deplete_count")
        ).alias("_total_queue_events"),
    ).with_columns(
        _safe_divide(
            pl.col("spread_widen_count"),
            pl.col("nquote").cast(pl.Float64, strict=False),
        ).alias("spread_widen_ratio"),
        _safe_divide(
            pl.col("bid_refill_count")
            + pl.col("ask_deplete_count")
            - pl.col("bid_deplete_count")
            - pl.col("ask_refill_count"),
            pl.col("_total_queue_events").cast(pl.Float64, strict=False),
        ).alias("queue_refill_imbalance"),
        _safe_divide(
            pl.col("_bid_side_empty_count"),
            pl.col("nquote").cast(pl.Float64, strict=False),
        ).alias("bid_side_empty_ratio"),
        _safe_divide(
            pl.col("_ask_side_empty_count"),
            pl.col("nquote").cast(pl.Float64, strict=False),
        ).alias("ask_side_empty_ratio"),
        _safe_divide(
            pl.col("_limit_up_single_sided_count"),
            pl.col("nquote").cast(pl.Float64, strict=False),
        ).alias("limit_up_single_sided_ratio"),
        _safe_divide(
            pl.col("_limit_down_single_sided_count"),
            pl.col("nquote").cast(pl.Float64, strict=False),
        ).alias("limit_down_single_sided_ratio"),
    )
    return grouped.select(
        "timestamp",
        "nquote",
        "mean_microprice_pressure",
        "mean_relative_spread",
        "spread_widen_count",
        "spread_narrow_count",
        "spread_flat_count",
        "spread_widen_ratio",
        "bid_refill_count",
        "bid_deplete_count",
        "ask_refill_count",
        "ask_deplete_count",
        "queue_refill_imbalance",
        "bid_side_empty_ratio",
        "ask_side_empty_ratio",
        "limit_up_single_sided_ratio",
        "limit_down_single_sided_ratio",
    )


def downscale_quote_features(
    second_df: pl.DataFrame, target_freq: str, symbol: str = "fu"
) -> pl.DataFrame:
    if second_df.height == 0:
        raise ValueError("Target window has no quote snapshots")
    _validate_quote_depth_imbalance_input(second_df)

    quote = _normalize_limit_single_sided_quote_prices(second_df).sort("timestamp").select(
        "timestamp",
        pl.col("BidPrice1").alias("bid_price"),
        pl.col("AskPrice1").alias("ask_price"),
        pl.col("BidVolume1").alias("bid_amount"),
        pl.col("AskVolume1").alias("ask_amount"),
        *_quote_depth_volume_columns(),
    )
    quote = quote.with_columns(
        (pl.col("ask_price") - pl.col("bid_price")).alias("spread"),
        ((pl.col("ask_price") + pl.col("bid_price")) / 2).alias("mid"),
        _depth_imbalance_expr(1).alias("imbalance_volume"),
        *[
            _depth_imbalance_expr(depth).alias(f"imbalance_{depth}")
            for depth in QUOTE_DEPTH_IMBALANCE_LEVELS
        ],
        pl.col("bid_price").alias("bid"),
        pl.col("ask_price").alias("ask"),
        pl.col("bid_amount").alias("bidsize"),
        pl.col("ask_amount").alias("asksize"),
    )
    quote = quote.with_columns(
        _change_count_expr("bid_price").alias("_nquote_bid"),
        _change_count_expr("ask_price").alias("_nquote_ask"),
        _change_count_expr("bid_price", "up").alias("_nquote_bid_up"),
        _change_count_expr("bid_price", "down").alias("_nquote_bid_down"),
        _change_count_expr("ask_price", "up").alias("_nquote_ask_up"),
        _change_count_expr("ask_price", "down").alias("_nquote_ask_down"),
    )

    aggs: List[pl.Expr] = [
        pl.col("bid_price").count().alias("nquote"),
        pl.col("_nquote_bid").sum().alias("nquote_bid"),
        pl.col("_nquote_ask").sum().alias("nquote_ask"),
        pl.col("_nquote_bid_up").sum().alias("nquote_bid_up"),
        pl.col("_nquote_bid_down").sum().alias("nquote_bid_down"),
        pl.col("_nquote_ask_up").sum().alias("nquote_ask_up"),
        pl.col("_nquote_ask_down").sum().alias("nquote_ask_down"),
    ]
    aggs.extend(
        _quote_window_stat_aggs(
            [
                "spread",
                "mid",
                "imbalance_volume",
                "imbalance_1",
                "imbalance_3",
                "imbalance_5",
                "bid",
                "ask",
                "bidsize",
                "asksize",
            ],
            {"imbalance_volume", "imbalance_1", "imbalance_3", "imbalance_5"},
        )
    )

    result = _resample(quote, target_freq, aggs)
    timestamps = result["timestamp"].to_list()
    target_delta = _target_freq_delta(target_freq)
    trading_sessions = get_commodity_config(symbol).trading_sessions
    for previous, current in zip(timestamps, timestamps[1:]):
        missing = previous + target_delta
        if (
            current - previous > target_delta
            and _same_trading_session(previous, current, trading_sessions)
        ):
            logger.warning(
                "Target window has no quote snapshots: %s", missing
            )

    empty_windows = result.filter(pl.col("nquote") == 0)
    if empty_windows.height:
        first = empty_windows.item(0, "timestamp")
        if _session_for_timestamp(first, trading_sessions) is not None:
            logger.warning(
                "Dropping %d empty quote window(s); first: %s",
                empty_windows.height,
                first,
            )
    return result.filter(pl.col("nquote") > 0)


def downscale_multi_window_quote_ofi_features(
    second_df: pl.DataFrame,
    window_rows_list: tuple[int, ...] | list[int] = (6, 12, 24, 48),
    depth: int = 5,
) -> pl.DataFrame:
    if not window_rows_list:
        raise ValueError("window_rows_list must not be empty")
    result_df: pl.DataFrame | None = None
    for window in window_rows_list:
        ofi_df = downscale_quote_ofi_features(
            second_df, window_rows=window, depth=depth
        )
        rename_dict = {
            col: f"{col}_{window}" for col in ofi_df.columns if col != "timestamp"
        }
        ofi_df = ofi_df.rename(rename_dict)
        if result_df is None:
            result_df = ofi_df
        else:
            result_df = result_df.join(ofi_df, on="timestamp", how="left")
    return result_df if result_df is not None else pl.DataFrame()


def downscale_multi_window_quote_microstructure_features(
    second_df: pl.DataFrame,
    window_rows_list: tuple[int, ...] | list[int] = (6, 12, 24, 48),
) -> pl.DataFrame:
    if not window_rows_list:
        raise ValueError("window_rows_list must not be empty")
    result_df: pl.DataFrame | None = None
    for window in window_rows_list:
        micro_df = downscale_quote_microstructure_features(
            second_df, window_rows=window
        )
        rename_dict = {
            col: f"{col}_{window}" for col in micro_df.columns if col != "timestamp"
        }
        micro_df = micro_df.rename(rename_dict)
        if result_df is None:
            result_df = micro_df
        else:
            result_df = result_df.join(micro_df, on="timestamp", how="left")
    return result_df if result_df is not None else pl.DataFrame()
