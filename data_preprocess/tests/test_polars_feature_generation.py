import sys

import polars as pl

sys.path.append("data_preprocess")

from operator_futures.features_related.feature_util import (
    create_ohlc_quotes_feature,
    create_quotes_feature,
    intial_process_trades,
    preprocess_quotes,
    preprocess_trades,
    side_group_trades,
)
from operator_futures.cross_section.base_feature_util import (
    process_k_line_feature,
    process_snapshot_features,
)
from operator_futures.time_operator.time_operator_util import process_ohlc


def test_quote_feature_counts_preserve_column_names():
    quotes = pl.DataFrame(
        {
            "timestamp": [1_000_000, 2_000_000, 3_000_000],
            "exchange": ["binance", "binance", "binance"],
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "bid_price": [100.0, 101.0, 101.0],
            "ask_price": [101.0, 101.0, 102.0],
            "bid_amount": [1.0, 2.0, 2.0],
            "ask_amount": [3.0, 3.0, 4.0],
        }
    )
    quotes = preprocess_quotes(quotes)

    out = create_quotes_feature(quotes, "10s")

    assert out.columns == [
        "timestamp",
        "nquote",
        "nquote_bid",
        "nquote_ask",
        "nquote_bid_up",
        "nquote_bid_down",
        "nquote_ask_up",
        "nquote_ask_down",
        "nquote_bidsize",
        "nquote_asksize",
        "nquote_bidsize_up",
        "nquote_bidsize_down",
        "nquote_asksize_up",
        "nquote_asksize_down",
        "nquote_bid_askflat",
        "nquote_bidup_askflat",
        "nquote_biddown_askflat",
        "nquote_ask_bidflat",
        "nquote_askup_bidflat",
        "nquote_askdown_bidflat",
    ]


def test_trade_feature_side_group_columns_exist():
    trades = pl.DataFrame(
        {
            "timestamp": [1_000_000, 2_000_000, 3_000_000],
            "exchange": ["binance", "binance", "binance"],
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "side": ["buy", "sell", "buy"],
            "price": [100.0, 101.0, 102.0],
            "amount": [1.0, 2.0, 3.0],
        }
    )
    trades = preprocess_trades(trades)

    base, processed = intial_process_trades(trades, "10s")
    side = side_group_trades(processed, "10s")

    assert "timestamp" in base.columns
    assert "buy_volume" in side.columns
    assert "sell_volume" in side.columns


def test_cross_section_features_return_polars_with_timestamp():
    base = pl.DataFrame(
        {
            "timestamp": [1, 2],
            "open": [100.0, 102.0],
            "high": [105.0, 106.0],
            "low": [99.0, 101.0],
            "close": [103.0, 104.0],
            "twap": [102.0, 103.0],
            "awap": [101.0, 102.0],
            "vwap": [102.5, 103.5],
        }
    )
    snapshot = pl.DataFrame(
        {
            "timestamp": [1],
            "ask1_price": [101.0],
            "ask2_price": [102.0],
            "ask1_size": [2.0],
            "ask2_size": [3.0],
            "bid1_price": [100.0],
            "bid2_price": [99.0],
            "bid1_size": [4.0],
            "bid2_size": [5.0],
        }
    )

    kline = process_k_line_feature(base)
    snapshot_features = process_snapshot_features(snapshot, topk=1, depth=2)

    assert isinstance(kline, pl.DataFrame)
    assert kline.columns[:2] == ["timestamp", "klen"]
    assert isinstance(snapshot_features, pl.DataFrame)
    assert snapshot_features.columns[:2] == ["timestamp", "midprice"]


def test_time_operator_ohlc_returns_polars_with_causal_rolling_features():
    frame = pl.DataFrame(
        {
            "timestamp": [1, 2, 3, 4, 5],
            "open": [10.0, 11.0, 12.0, 13.0, 100.0],
            "high": [11.0, 12.0, 13.0, 14.0, 101.0],
            "low": [9.0, 10.0, 11.0, 12.0, 99.0],
            "close": [10.0, 11.0, 12.0, 13.0, 100.0],
        }
    )

    out = process_ohlc(frame, [2])

    assert isinstance(out, pl.DataFrame)
    assert out["timestamp"].to_list() == [4, 5]
    assert out["max_2"].to_list()[0] == 13.0 / 13.0


def test_time_operator_ohlc_preserves_rank_and_index_features():
    frame = pl.DataFrame(
        {
            "timestamp": [1, 2, 3, 4, 5, 6],
            "open": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "high": [11.0, 13.0, 12.0, 16.0, 15.0, 17.0],
            "low": [9.0, 10.0, 8.0, 12.0, 11.0, 14.0],
            "close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        }
    )

    out = process_ohlc(frame, [2])

    assert out.height == 3
    assert out["rank_2"].to_list() != [0.0, 0.0, 0.0]
    assert out["imax_2"].to_list() != [0.0, 0.0, 0.0]
    assert out["imin_2"].to_list() != [0.0, 0.0, 0.0]
    assert out["imxd_2"].to_list() != [0.0, 0.0, 0.0]
