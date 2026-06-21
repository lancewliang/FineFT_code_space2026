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

    with pytest.raises(ValueError) as exc_info:
        validate_best_quotes(raw, "fu2302")

    message = str(exc_info.value)
    assert f"BidPrice1={ask_price}" in message
    assert f"AskPrice1={ask_price}" in message
    assert "reason=BidPrice1 >= AskPrice1" in message
    assert "row={" in message
    assert "'InstrumentID': 'fu2302'" in message
    assert "'LastPrice':" in message


def test_second_level_drops_rows_with_all_depth_prices_null():
    raw = pl.read_csv(SAMPLE_PATH).head(2)
    dropped_timestamp = datetime.strptime(
        f"{raw.item(0, 'ActionDay')} {raw.item(0, 'UpdateTime')}",
        "%Y%m%d %H:%M:%S.%f",
    ).replace(microsecond=0)
    depth_price_columns = [
        f"{side}Price{level}"
        for side in ("Bid", "Ask")
        for level in range(1, 6)
    ]
    raw = raw.with_columns(
        [
            pl.when(pl.int_range(pl.len()) == 0)
            .then(pl.lit(None))
            .otherwise(pl.col(column))
            .alias(column)
            for column in depth_price_columns
        ]
    )

    second = create_second_level_snapshots(raw)

    assert second.height == 1
    assert not second["timestamp"].eq(dropped_timestamp).any()


def test_limit_down_single_sided_book_is_allowed():
    raw = pl.read_csv(SAMPLE_PATH).head(1)
    lower_limit = raw.item(0, "LowerLimitPrice")
    raw = raw.with_columns(
        [pl.lit(lower_limit).alias("LastPrice")]
        + [
            pl.lit(None).alias(f"BidPrice{level}")
            for level in range(1, 6)
        ]
        + [
            pl.lit(0).alias(f"BidVolume{level}")
            for level in range(1, 6)
        ]
    )

    validate_best_quotes(raw, "fu2302")


def test_touched_limit_down_single_sided_book_is_allowed():
    raw = pl.read_csv(SAMPLE_PATH).head(1)
    lower_limit = raw.item(0, "LowerLimitPrice")
    raw = raw.with_columns(
        [
            pl.lit(lower_limit + 1).alias("LastPrice"),
            pl.lit(lower_limit).alias("LowPrice"),
        ]
        + [
            pl.lit(None).alias(f"BidPrice{level}")
            for level in range(1, 6)
        ]
        + [
            pl.lit(0).alias(f"BidVolume{level}")
            for level in range(1, 6)
        ]
    )

    validate_best_quotes(raw, "fu2302")


def test_limit_up_single_sided_book_is_allowed():
    raw = pl.read_csv(SAMPLE_PATH).head(1)
    upper_limit = raw.item(0, "UpperLimitPrice")
    raw = raw.with_columns(
        [pl.lit(upper_limit).alias("LastPrice")]
        + [
            pl.lit(None).alias(f"AskPrice{level}")
            for level in range(1, 6)
        ]
        + [
            pl.lit(0).alias(f"AskVolume{level}")
            for level in range(1, 6)
        ]
    )

    validate_best_quotes(raw, "fu2302")


def test_touched_limit_up_single_sided_book_is_allowed():
    raw = pl.read_csv(SAMPLE_PATH).head(1)
    upper_limit = raw.item(0, "UpperLimitPrice")
    raw = raw.with_columns(
        [
            pl.lit(upper_limit - 1).alias("LastPrice"),
            pl.lit(upper_limit).alias("HighPrice"),
        ]
        + [
            pl.lit(None).alias(f"AskPrice{level}")
            for level in range(1, 6)
        ]
        + [
            pl.lit(0).alias(f"AskVolume{level}")
            for level in range(1, 6)
        ]
    )

    validate_best_quotes(raw, "fu2302")


def test_non_limit_single_sided_book_still_fails():
    raw = pl.read_csv(SAMPLE_PATH).head(1).with_columns(
        [
            pl.lit(None).alias(f"BidPrice{level}")
            for level in range(1, 6)
        ]
        + [
            pl.lit(0).alias(f"BidVolume{level}")
            for level in range(1, 6)
        ]
    )

    with pytest.raises(ValueError, match="BidPrice1 is null"):
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


def test_base_features_use_contract_unit_for_prices_but_keep_raw_tradeval():
    second = pl.DataFrame(
        {
            "timestamp": [
                datetime(2023, 1, 3, 9, 0, 0),
                datetime(2023, 1, 3, 9, 0, 1),
                datetime(2023, 1, 3, 9, 0, 2),
            ],
            "InstrumentID": ["fu2302", "fu2302", "fu2302"],
            "BidPrice1": [2599.0, 2599.0, 2600.0],
            "AskPrice1": [2601.0, 2601.0, 2602.0],
            "LastPrice": [2600.0, 2600.0, 2601.0],
            "Volume": [0, 1, 2],
            "Turnover": [0.0, 26000.0, 52010.0],
        }
    )

    base = downscale_base_features(second, "5min", "fu").filter(pl.col("volume") > 0)

    assert base.item(0, "open") == 2600.0
    assert base.item(0, "close") == 2601.0
    assert base.item(0, "volume") == 2
    assert base.item(0, "tradeval") == 52010.0
    assert base.item(0, "vwap") == 2600.5


def test_empty_quote_window_fails_fast():
    raw = pl.read_csv(SAMPLE_PATH).head(2)
    second = create_second_level_snapshots(raw)

    with pytest.raises(ValueError, match="no quote snapshots"):
        downscale_quote_features(second.head(0), "5min")


def test_intermediate_empty_quote_window_fails_fast():
    second = pl.DataFrame(
        {
            "timestamp": [
                datetime(2023, 1, 3, 9, 0, 0),
                datetime(2023, 1, 3, 9, 10, 0),
            ],
            "BidPrice1": [2600.0, 2601.0],
            "AskPrice1": [2602.0, 2603.0],
            "BidVolume1": [1.0, 1.0],
            "AskVolume1": [1.0, 1.0],
        }
    )

    with pytest.raises(ValueError, match="2023-01-03 09:05:00"):
        downscale_quote_features(second, "5min")
