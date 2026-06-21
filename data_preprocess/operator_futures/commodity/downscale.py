from typing import List

import polars as pl

from .main_contract import with_normalized_timestamp


BID_PRICE_COLUMNS = [f"BidPrice{level}" for level in range(1, 6)]
ASK_PRICE_COLUMNS = [f"AskPrice{level}" for level in range(1, 6)]
BID_VOLUME_COLUMNS = [f"BidVolume{level}" for level in range(1, 6)]
ASK_VOLUME_COLUMNS = [f"AskVolume{level}" for level in range(1, 6)]
DEPTH_PRICE_COLUMNS = BID_PRICE_COLUMNS + ASK_PRICE_COLUMNS


def _polars_freq(target_freq: str) -> str:
    return target_freq.replace("min", "m")


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
    invalid = pl.any_horizontal(
        pl.col("BidPrice1").is_null(),
        pl.col("AskPrice1").is_null(),
        pl.col("BidPrice1") <= 0,
        pl.col("AskPrice1") <= 0,
        pl.col("BidPrice1") >= pl.col("AskPrice1"),
    )
    limit_down_single_sided = (
        pl.col("LastPrice").is_not_null()
        & pl.col("LowerLimitPrice").is_not_null()
        & (
            (pl.col("LastPrice") == pl.col("LowerLimitPrice"))
            | (pl.col("LowPrice") == pl.col("LowerLimitPrice"))
        )
        & pl.all_horizontal(
            [pl.col(column).is_null() for column in BID_PRICE_COLUMNS]
        )
        & pl.all_horizontal(
            [pl.col(column).fill_null(0) == 0 for column in BID_VOLUME_COLUMNS]
        )
        & pl.col("AskPrice1").is_not_null()
        & (pl.col("AskPrice1") > 0)
    ).fill_null(False)
    limit_up_single_sided = (
        pl.col("LastPrice").is_not_null()
        & pl.col("UpperLimitPrice").is_not_null()
        & (
            (pl.col("LastPrice") == pl.col("UpperLimitPrice"))
            | (pl.col("HighPrice") == pl.col("UpperLimitPrice"))
        )
        & pl.all_horizontal(
            [pl.col(column).is_null() for column in ASK_PRICE_COLUMNS]
        )
        & pl.all_horizontal(
            [pl.col(column).fill_null(0) == 0 for column in ASK_VOLUME_COLUMNS]
        )
        & pl.col("BidPrice1").is_not_null()
        & (pl.col("BidPrice1") > 0)
    ).fill_null(False)
    invalid_rows = df.filter(
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


def create_second_level_snapshots(df: pl.DataFrame) -> pl.DataFrame:
    contract = (
        str(df.item(0, "InstrumentID"))
        if "InstrumentID" in df.columns and df.height
        else "unknown"
    )
    copied = drop_empty_depth_price_rows(with_normalized_timestamp(df))
    validate_best_quotes(copied, contract)
    return (
        copied.sort("timestamp")
        .with_columns(pl.col("timestamp").dt.truncate("1s").alias("timestamp"))
        .group_by("timestamp", maintain_order=True)
        .agg(pl.exclude("timestamp").last())
        .sort("timestamp")
    )


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
    renamed = second_df.select(expressions)
    result = _resample(
        renamed,
        target_freq,
        [pl.col(column).last().alias(column) for column in output_columns],
    )
    return result.filter(
        ~pl.all_horizontal([pl.col(column).is_null() for column in output_columns])
    )


def _second_trade_frame(second_df: pl.DataFrame) -> pl.DataFrame:
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
        .then(pl.col("second_tradeval") / pl.col("second_volume"))
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


def downscale_base_features(second_df: pl.DataFrame, target_freq: str) -> pl.DataFrame:
    frame = _with_reference_price(_second_trade_frame(second_df)).with_columns(
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
        .then(pl.col("tradeval") / pl.col("volume"))
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


def downscale_quote_features(second_df: pl.DataFrame, target_freq: str) -> pl.DataFrame:
    if second_df.height == 0:
        raise ValueError("Target window has no quote snapshots")

    quote = second_df.sort("timestamp").select(
        "timestamp",
        pl.col("BidPrice1").alias("bid_price"),
        pl.col("AskPrice1").alias("ask_price"),
        pl.col("BidVolume1").alias("bid_amount"),
        pl.col("AskVolume1").alias("ask_amount"),
    )
    quote = quote.with_columns(
        (pl.col("ask_price") - pl.col("bid_price")).alias("spread"),
        ((pl.col("ask_price") + pl.col("bid_price")) / 2).alias("mid"),
        (
            (pl.col("bid_amount") - pl.col("ask_amount"))
            / (pl.col("bid_amount") + pl.col("ask_amount"))
        ).alias("imbalance_volume"),
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
    for name in ["spread", "mid", "imbalance_volume", "bid", "ask", "bidsize", "asksize"]:
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

    result = _resample(quote, target_freq, aggs)
    empty_windows = result.filter(pl.col("nquote") == 0)
    if empty_windows.height:
        first = empty_windows.item(0, "timestamp")
        raise ValueError(f"Target window has no quote snapshots: {first}")
    return result
