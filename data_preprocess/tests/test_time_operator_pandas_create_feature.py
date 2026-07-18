from types import SimpleNamespace

import pandas as pd
import pytest

from operator_futures.time_operator import create_feature


def _args(tmp_path):
    return SimpleNamespace(
        root_path=str(tmp_path),
        data_path="PREPROCESS_DATASET/binance-futures/MERGE_CONCAT/CONCAT_FEATURE/",
        save_path="PREPROCESS_DATASET/binance-futures/TIME_FEATURE/",
        symbols="BTCUSDT",
        start_date="2026-01-05",
        end_date="2026-01-06",
        target_freq="10s",
        windows="2",
    )


def _write_input(tmp_path, frame):
    input_path = (
        tmp_path
        / "PREPROCESS_DATASET/binance-futures/MERGE_CONCAT/CONCAT_FEATURE/BTCUSDT/10s"
        / "2026-01-05-2026-01-06.feather"
    )
    input_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_feather(input_path)
    return input_path


def _valid_ohlcv_frame():
    return pd.DataFrame(
        {
            "timestamp": list(range(8)),
            "open": [100.0 + idx for idx in range(8)],
            "high": [101.0 + idx for idx in range(8)],
            "low": [99.0 + idx for idx in range(8)],
            "close": [100.5 + idx for idx in range(8)],
            "volume": [1000.0 + idx for idx in range(8)],
            "sell_wap": [101.0 + idx for idx in range(8)],
            "buy_wap": [99.0 + idx for idx in range(8)],
            "buy_sell_wap_spread": [-2.0 for _ in range(8)],
            "ask1_size_n": [0.0 if idx == 2 else 0.2 for idx in range(8)],
            "bid1_size_n": [1.0 for _ in range(8)],
            "ask_side_empty": [idx == 2 for idx in range(8)],
            "bid_side_empty": [False for _ in range(8)],
            "LowerLimitPrice": [80.0 for _ in range(8)],
            "UpperLimitPrice": [120.0 for _ in range(8)],
        }
    )


def test_create_feature_rejects_illegal_input_values(tmp_path):
    frame = _valid_ohlcv_frame()
    frame.loc[2, "close"] = float("nan")
    _write_input(tmp_path, frame)

    with pytest.raises(ValueError) as exc_info:
        create_feature.main(_args(tmp_path))

    message = str(exc_info.value)
    assert "stage=time_feature_input" in message
    assert "close:null=1" in message


def test_create_feature_accepts_enhanced_single_sided_snapshot_input(tmp_path):
    _write_input(tmp_path, _valid_ohlcv_frame())

    create_feature.main(_args(tmp_path))


def test_create_feature_rejects_illegal_output_values(tmp_path, monkeypatch):
    _write_input(tmp_path, _valid_ohlcv_frame())

    def process_ohlcv_with_infinite_feature(df, windows):
        return pd.DataFrame({"bad_feature": [float("inf")] * len(df)}, index=df.index)

    monkeypatch.setattr(create_feature, "process_ohlcv", process_ohlcv_with_infinite_feature)

    with pytest.raises(ValueError) as exc_info:
        create_feature.main(_args(tmp_path))

    message = str(exc_info.value)
    assert "stage=time_feature_output" in message
    assert "bad_feature_origin:infinite=8" in message
