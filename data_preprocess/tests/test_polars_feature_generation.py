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
