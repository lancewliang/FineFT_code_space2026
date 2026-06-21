import polars as pl

from operator_futures.merge_concat.concat import concat_concurrent_future_frames
from operator_futures.merge_concat.merge import build_daily_feature_frames


def test_concat_concurrent_future_frames_preserves_shift_and_inner_join():
    concurrent = pl.DataFrame(
        {
            "timestamp": [1, 1, 2, 3],
            "mark_price": [10.0, 11.0, 12.0, 13.0],
        }
    )
    future = pl.DataFrame(
        {
            "timestamp": [1, 2, 3],
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "exchange": ["binance", "binance", "binance"],
            "feature": [100.0, 200.0, 300.0],
        }
    )

    out = concat_concurrent_future_frames(concurrent, future)

    assert out.columns == ["timestamp", "mark_price", "feature"]
    assert out["timestamp"].to_list() == [2, 3]
    assert out["feature"].to_list() == [100.0, 200.0]


def test_build_daily_feature_frames_drops_derivative_symbol_from_reward():
    snapshot = pl.DataFrame({"timestamp": [1], "ask1_price": [101.0]})
    der = pl.DataFrame({"timestamp": [1], "symbol": ["BTCUSDT"], "mark_price": [100.0]})
    base = pl.DataFrame(
        {
            "timestamp": [1],
            "symbol": ["BTCUSDT"],
            "exchange": ["binance"],
            "volume": [1.0],
        }
    )
    snapshot_feature = pl.DataFrame({"timestamp": [1], "snapshot_feature": [2.0]})
    quotes_feature = pl.DataFrame({"timestamp": [1], "quote_feature": [3.0]})
    kline_feature = pl.DataFrame({"timestamp": [1], "kline_feature": [4.0]})

    reward, future = build_daily_feature_frames(
        snapshot,
        der,
        base,
        snapshot_feature,
        quotes_feature,
        kline_feature,
    )

    assert "symbol" not in reward.columns
    assert future.columns[:3] == ["timestamp", "symbol", "exchange"]
