from datetime import datetime

import polars as pl
import pytest

from operator_futures.commodity.downscale import (
    create_second_level_snapshots,
    downscale_base_features,
    downscale_derivative_reference,
    downscale_orderbook,
    downscale_quote_features,
    validate_best_quotes,
)


SAMPLE_PATH = "docs/上海商品交易所/fu2302.csv"


def test_sample_file_can_create_depth_five_outputs():
    raw = pl.read_csv(SAMPLE_PATH).head(20)
    second = create_second_level_snapshots(raw)
    orderbook = downscale_orderbook(second, "5min", depth=5)
    derivative = downscale_derivative_reference(second, "5min", "fu")
    base = downscale_base_features(second, "5min")
    quote = downscale_quote_features(second, "5min")

    assert "ask5_price" in orderbook.columns
    assert "ask6_price" not in orderbook.columns
    assert "mark_price" in derivative.columns
    assert "ntrade_estimated" in base.columns
    assert "nquote" in quote.columns


def test_invalid_best_quote_fails_fast():
    raw = pl.read_csv(SAMPLE_PATH).head(2)
    ask_price = raw.item(0, "AskPrice1")
    raw = raw.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.lit(ask_price))
        .otherwise(pl.col("BidPrice1"))
        .alias("BidPrice1")
    )

    with pytest.raises(ValueError, match="BidPrice1"):
        validate_best_quotes(raw, "fu2302")


def test_second_level_uses_last_snapshot_per_second():
    raw = pl.read_csv(SAMPLE_PATH).head(4)
    update_time = raw.item(1, "UpdateTime")
    raw = raw.with_columns(
        pl.when(pl.int_range(pl.len()) == 2)
        .then(pl.lit(update_time))
        .otherwise(pl.col("UpdateTime"))
        .alias("UpdateTime"),
        pl.when(pl.int_range(pl.len()) == 2)
        .then(pl.lit(2600.0))
        .otherwise(pl.col("BidPrice1"))
        .alias("BidPrice1"),
    )

    second = create_second_level_snapshots(raw)

    timestamp = datetime.strptime(
        f"{raw.item(2, 'ActionDay')} {raw.item(2, 'UpdateTime')}",
        "%Y%m%d %H:%M:%S.%f",
    ).replace(microsecond=0)
    assert (
        second.filter(pl.col("timestamp") == timestamp).item(0, "BidPrice1")
        == 2600.0
    )


def test_derivative_reference_falls_back_to_midprice_for_invalid_lastprice():
    raw = pl.read_csv(SAMPLE_PATH).head(3)
    second = create_second_level_snapshots(raw)
    first_timestamp = second.item(0, "timestamp")
    second = second.with_columns(
        pl.when(pl.col("timestamp") == first_timestamp)
        .then(pl.lit(0))
        .otherwise(pl.col("LastPrice"))
        .alias("LastPrice")
    )

    derivative = downscale_derivative_reference(second, "5min", "fu")

    expected_mid = (second.item(0, "BidPrice1") + second.item(0, "AskPrice1")) / 2
    assert derivative.item(0, "mark_price") == expected_mid
    assert derivative.item(0, "funding_rate") == 0


def test_empty_quote_window_fails_fast():
    raw = pl.read_csv(SAMPLE_PATH).head(2)
    second = create_second_level_snapshots(raw)

    with pytest.raises(ValueError, match="no quote snapshots"):
        downscale_quote_features(second.head(0), "5min")
