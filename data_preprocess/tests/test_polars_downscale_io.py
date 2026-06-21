import sys

import polars as pl

sys.path.append("data_preprocess")

from operator_futures.derivative_ticker.down_scale_single_shot import (
    down_scale_single_dertick,
)
from operator_futures.orderbook_25.down_scale_single_shot import (
    down_scale_single_oe_snapshot,
)


def _orderbook_frame() -> pl.DataFrame:
    raw = pl.DataFrame(
        {
            "timestamp": [1_000_000, 1_000_001, 1_010_000],
            "local_timestamp": [1, 2, 3],
            "exchange": ["binance", "binance", "binance"],
            "asks[0].price": [101.0, 102.0, 103.0],
            "asks[0].amount": [1.0, 2.0, 3.0],
            "bids[0].price": [100.0, 99.0, 98.0],
            "bids[0].amount": [4.0, 5.0, 6.0],
        }
    )
    for level in range(1, 25):
        raw = raw.with_columns(
            pl.lit(101.0 + level).alias(f"asks[{level}].price"),
            pl.lit(1.0 + level).alias(f"asks[{level}].amount"),
            pl.lit(100.0 - level).alias(f"bids[{level}].price"),
            pl.lit(4.0 + level).alias(f"bids[{level}].amount"),
        )
    return raw


def test_orderbook_downscale_preserves_first_and_depth_column_names():
    out = down_scale_single_oe_snapshot(_orderbook_frame(), "10s")

    assert out.columns[:5] == [
        "timestamp",
        "ask1_price",
        "ask1_size",
        "bid1_price",
        "bid1_size",
    ]
    assert out["ask1_price"].to_list()[0] == 101.0
    assert out["bid1_price"].to_list()[0] == 100.0


def test_derivative_downscale_preserves_selected_columns_and_first():
    raw = pl.DataFrame(
        {
            "timestamp": [1_000_000, 1_000_001, 1_010_000],
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "funding_timestamp": [2_000_000, 2_000_001, 2_010_000],
            "funding_rate": [0.01, 0.02, 0.03],
            "index_price": [100.0, 101.0, 102.0],
            "mark_price": [100.5, 101.5, 102.5],
        }
    )

    out = down_scale_single_dertick(raw, "10s")

    assert out.columns == [
        "timestamp",
        "symbol",
        "funding_timestamp",
        "funding_rate",
        "index_price",
        "mark_price",
    ]
    assert out["funding_rate"].to_list()[0] == 0.01
