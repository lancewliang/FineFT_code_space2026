from pathlib import Path
import os
import subprocess
import sys

import polars as pl

from operator_futures.merge_concat.concat import concat_concurrent_future_frames
from operator_futures.merge_concat.merge import build_daily_feature_frames


REPO_ROOT = Path(__file__).resolve().parents[2]


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


def test_merge_clean_cli_writes_csv_next_to_feather(tmp_path):
    start_date = "2026-01-05"
    end_date = "2026-01-06"
    date_range = f"{start_date}-{end_date}"
    concat_file = (
        tmp_path
        / "PREPROCESS_DATASET/binance-futures/MERGE_CONCAT/CONCAT_FEATURE/BTCUSDT/10s"
        / f"{date_range}.feather"
    )
    time_file = (
        tmp_path
        / "PREPROCESS_DATASET/binance-futures/TIME_FEATURE/BTCUSDT/10s"
        / f"{date_range}.feather"
    )
    concat_file.parent.mkdir(parents=True, exist_ok=True)
    time_file.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {"timestamp": [1, 2, 3], "cross_feature": [10.0, 20.0, 30.0]}
    ).write_ipc(concat_file)
    pl.DataFrame(
        {"timestamp": [2, 3, 4], "time_feature": [200.0, 300.0, 400.0]}
    ).write_ipc(time_file)

    subprocess.run(
        [
            sys.executable,
            "data_preprocess/operator_futures/merge_all/merge_clean.py",
            "--root_path",
            str(tmp_path),
            "--symbols",
            "BTCUSDT",
            "--target_freq",
            "10s",
            "--start_date",
            start_date,
            "--end_date",
            end_date,
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "data_preprocess")},
        check=True,
    )

    output_file = (
        tmp_path
        / "PREPROCESS_DATASET/binance-futures/ALL_FEATURE/BTCUSDT/10s"
        / f"{date_range}.feather"
    )
    csv_output = output_file.with_suffix(".csv")
    assert output_file.exists()
    assert csv_output.exists()
    assert pl.read_ipc(output_file).height == 2
    assert pl.read_csv(csv_output).height == 2
