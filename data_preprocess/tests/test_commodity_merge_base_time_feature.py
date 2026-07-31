import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import polars as pl

from operator_futures.merge_concat.merge import build_daily_feature_frames, main, parser
from operator_futures.commodity.mixed_frequency_feature import (
    MIXED_FREQUENCY_FEATURE_COLUMNS,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_build_daily_feature_frames_joins_base_time_feature():
    snapshot = pl.DataFrame({"timestamp": [1, 2], "ask1_price": [101.0, 102.0]})
    der = pl.DataFrame({"timestamp": [1, 2], "mark_price": [100.0, 101.0]})
    base = pl.DataFrame({"timestamp": [1, 2], "volume": [1.0, 2.0]})
    snapshot_feature = pl.DataFrame({"timestamp": [1, 2], "sf": [2.0, 3.0]})
    quotes_feature = pl.DataFrame({"timestamp": [1, 2], "qf": [3.0, 4.0]})
    kline_feature = pl.DataFrame({"timestamp": [1, 2], "kf": [4.0, 5.0]})
    base_time = pl.DataFrame({"timestamp": [1, 2], "trading_minute_progress": [0.0, 0.5]})

    reward, future = build_daily_feature_frames(
        snapshot,
        der,
        base,
        snapshot_feature,
        quotes_feature,
        kline_feature,
        base_time_feature=base_time,
        contract="fu2601",
    )

    assert "trading_minute_progress" in future.columns
    assert future["trading_minute_progress"].to_list() == [0.0, 0.5]


def test_build_daily_feature_frames_joins_cross_month_feature_into_future_only():
    snapshot = pl.DataFrame({"timestamp": [1, 2], "ask1_price": [101.0, 102.0]})
    der = pl.DataFrame({"timestamp": [1, 2], "mark_price": [100.0, 101.0]})
    base = pl.DataFrame({"timestamp": [1, 2], "volume": [1.0, 2.0]})
    snapshot_feature = pl.DataFrame({"timestamp": [1, 2], "sf": [2.0, 3.0]})
    quotes_feature = pl.DataFrame({"timestamp": [1, 2], "qf": [3.0, 4.0]})
    kline_feature = pl.DataFrame({"timestamp": [1, 2], "kf": [4.0, 5.0]})
    cross_month = pl.DataFrame(
        {"timestamp": [1], "cm_contract_role_main": [1.0]}
    )

    reward, future = build_daily_feature_frames(
        snapshot,
        der,
        base,
        snapshot_feature,
        quotes_feature,
        kline_feature,
        cross_month_feature=cross_month,
        contract="fu2601",
    )

    assert "cm_contract_role_main" not in reward.columns
    assert future["cm_contract_role_main"].to_list() == [1.0, 0.0]


def test_build_daily_feature_frames_joins_mixed_frequency_feature_into_future_only():
    snapshot = pl.DataFrame({"timestamp": [1, 2], "ask1_price": [101.0, 102.0]})
    der = pl.DataFrame({"timestamp": [1, 2], "mark_price": [100.0, 101.0]})
    base = pl.DataFrame({"timestamp": [1, 2], "volume": [1.0, 2.0]})
    snapshot_feature = pl.DataFrame({"timestamp": [1, 2], "sf": [2.0, 3.0]})
    quotes_feature = pl.DataFrame({"timestamp": [1, 2], "qf": [3.0, 4.0]})
    kline_feature = pl.DataFrame({"timestamp": [1, 2], "kf": [4.0, 5.0]})
    mixed_frequency = pl.DataFrame(
        {
            "timestamp": [1, 2],
            **{column: [1.0, 2.0] for column in MIXED_FREQUENCY_FEATURE_COLUMNS},
        }
    )

    reward, future = build_daily_feature_frames(
        snapshot,
        der,
        base,
        snapshot_feature,
        quotes_feature,
        kline_feature,
        mixed_frequency_feature=mixed_frequency,
        contract="fu2601",
    )

    assert "prev_day_return" not in reward.columns
    assert "prev_week_return" not in reward.columns
    assert future["prev_day_return"].to_list() == [1.0, 2.0]
    assert future["prev_week_return"].to_list() == [1.0, 2.0]


def test_daily_merge_commodity_proceeds_when_base_time_feature_missing(tmp_path):
    date = "2026-01-05"
    parts = ["fu", "fu2601", "5min"]
    
    for folder in ["DOWNSCALE_ORDERBOOK_25", "DOWNSCALE_DERTIC", "BASE_FEATURE"]:
        p = tmp_path / "PREPROCESS_DATASET/commodity-futures" / folder / parts[0] / parts[1] / parts[2]
        p.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"timestamp": [1, 2]}).write_ipc(p / f"{date}.feather")

    for cs_folder in ["SNAPSHOT_FEATURE", "QUOTES_FEATURE", "KLINE_FEATURE"]:
        p = tmp_path / "PREPROCESS_DATASET/commodity-futures/CROSS_SECTION" / cs_folder / parts[0] / parts[1] / parts[2]
        p.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"timestamp": [1, 2]}).write_ipc(p / f"{date}.feather")

    # BASE_TIME_FEATURE is NOT created

    args = parser.parse_args([
        "--symbols", "fu",
        "--contract", "fu2601",
        "--target_freq", "5min",
        "--date", date,
        "--root_path", str(tmp_path),
        "--data_path", "PREPROCESS_DATASET/commodity-futures/",
        "--save_path", "PREPROCESS_DATASET/commodity-futures/MERGE_CONCAT",
        "--no-require_cross_month_feature",
    ])

    main(args)
    out_file = tmp_path / "PREPROCESS_DATASET/commodity-futures/MERGE_CONCAT/MERGED_FEATURE/fu/fu2601/5min/FUTURE_FEATURE/2026-01-05.feather"
    assert out_file.exists()


def test_daily_merge_commodity_joins_cross_month_feature(tmp_path):
    date = "2026-01-05"
    parts = ["fu", "fu2601", "5min"]

    for folder in ["DOWNSCALE_ORDERBOOK_25", "DOWNSCALE_DERTIC", "BASE_FEATURE"]:
        p = tmp_path / "PREPROCESS_DATASET/commodity-futures" / folder / parts[0] / parts[1] / parts[2]
        p.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"timestamp": [1, 2]}).write_ipc(p / f"{date}.feather")

    for cs_folder in ["SNAPSHOT_FEATURE", "QUOTES_FEATURE", "KLINE_FEATURE"]:
        p = tmp_path / "PREPROCESS_DATASET/commodity-futures/CROSS_SECTION" / cs_folder / parts[0] / parts[1] / parts[2]
        p.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"timestamp": [1, 2]}).write_ipc(p / f"{date}.feather")

    cross_month_dir = tmp_path / "PREPROCESS_DATASET/commodity-futures/CROSS_MONTH_FEATURE" / parts[0] / parts[1] / parts[2]
    cross_month_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"timestamp": [1], "cm_contract_role_main": [1.0]}).write_ipc(
        cross_month_dir / f"{date}.feather"
    )

    args = parser.parse_args([
        "--symbols", "fu",
        "--contract", "fu2601",
        "--target_freq", "5min",
        "--date", date,
        "--root_path", str(tmp_path),
        "--data_path", "PREPROCESS_DATASET/commodity-futures/",
        "--save_path", "PREPROCESS_DATASET/commodity-futures/MERGE_CONCAT",
    ])

    main(args)

    out_file = tmp_path / "PREPROCESS_DATASET/commodity-futures/MERGE_CONCAT/MERGED_FEATURE/fu/fu2601/5min/FUTURE_FEATURE/2026-01-05.feather"
    out = pl.read_ipc(out_file)
    assert out["cm_contract_role_main"].to_list() == [1.0, 0.0]


def test_daily_merge_commodity_joins_mixed_frequency_feature(tmp_path):
    date = "2026-01-05"
    parts = ["fu", "fu2601", "5min"]

    for folder in ["DOWNSCALE_ORDERBOOK_25", "DOWNSCALE_DERTIC", "BASE_FEATURE"]:
        p = tmp_path / "PREPROCESS_DATASET/commodity-futures" / folder / parts[0] / parts[1] / parts[2]
        p.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"timestamp": [1, 2]}).write_ipc(p / f"{date}.feather")

    for cs_folder in ["SNAPSHOT_FEATURE", "QUOTES_FEATURE", "KLINE_FEATURE"]:
        p = tmp_path / "PREPROCESS_DATASET/commodity-futures/CROSS_SECTION" / cs_folder / parts[0] / parts[1] / parts[2]
        p.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"timestamp": [1, 2]}).write_ipc(p / f"{date}.feather")

    cross_month_dir = tmp_path / "PREPROCESS_DATASET/commodity-futures/CROSS_MONTH_FEATURE" / parts[0] / parts[1] / parts[2]
    cross_month_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"timestamp": [1, 2], "cm_contract_role_main": [1.0, 1.0]}).write_ipc(
        cross_month_dir / f"{date}.feather"
    )
    mixed_dir = tmp_path / "PREPROCESS_DATASET/commodity-futures/MIXED_FREQUENCY_FEATURE" / parts[0] / parts[1] / parts[2]
    mixed_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "timestamp": [1, 2],
            **{column: [1.0, 2.0] for column in MIXED_FREQUENCY_FEATURE_COLUMNS},
        }
    ).write_ipc(mixed_dir / f"{date}.feather")

    args = parser.parse_args([
        "--symbols", "fu",
        "--contract", "fu2601",
        "--target_freq", "5min",
        "--date", date,
        "--root_path", str(tmp_path),
        "--data_path", "PREPROCESS_DATASET/commodity-futures/",
        "--save_path", "PREPROCESS_DATASET/commodity-futures/MERGE_CONCAT",
    ])

    main(args)

    future_file = tmp_path / "PREPROCESS_DATASET/commodity-futures/MERGE_CONCAT/MERGED_FEATURE/fu/fu2601/5min/FUTURE_FEATURE/2026-01-05.feather"
    reward_file = tmp_path / "PREPROCESS_DATASET/commodity-futures/MERGE_CONCAT/MERGED_FEATURE/fu/fu2601/5min/CONCURRENT_FEATURE/2026-01-05.feather"
    future = pl.read_ipc(future_file)
    reward = pl.read_ipc(reward_file)
    assert future["prev_day_return"].to_list() == [1.0, 2.0]
    assert "prev_day_return" not in reward.columns


def test_daily_merge_commodity_requires_cross_month_feature_when_enabled(tmp_path):
    date = "2026-01-05"
    parts = ["fu", "fu2601", "5min"]

    for folder in ["DOWNSCALE_ORDERBOOK_25", "DOWNSCALE_DERTIC", "BASE_FEATURE"]:
        p = tmp_path / "PREPROCESS_DATASET/commodity-futures" / folder / parts[0] / parts[1] / parts[2]
        p.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"timestamp": [1, 2]}).write_ipc(p / f"{date}.feather")

    for cs_folder in ["SNAPSHOT_FEATURE", "QUOTES_FEATURE", "KLINE_FEATURE"]:
        p = tmp_path / "PREPROCESS_DATASET/commodity-futures/CROSS_SECTION" / cs_folder / parts[0] / parts[1] / parts[2]
        p.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"timestamp": [1, 2]}).write_ipc(p / f"{date}.feather")

    args = parser.parse_args([
        "--symbols", "fu",
        "--contract", "fu2601",
        "--target_freq", "5min",
        "--date", date,
        "--root_path", str(tmp_path),
        "--data_path", "PREPROCESS_DATASET/commodity-futures/",
        "--save_path", "PREPROCESS_DATASET/commodity-futures/MERGE_CONCAT",
        "--require_cross_month_feature",
    ])

    with pytest.raises(ValueError, match="missing required CROSS_MONTH_FEATURE"):
        main(args)


def test_daily_merge_commodity_fails_fast_when_timestamps_mismatch(tmp_path):
    date = "2026-01-05"
    parts = ["fu", "fu2601", "5min"]
    
    for folder in ["DOWNSCALE_ORDERBOOK_25", "DOWNSCALE_DERTIC", "BASE_FEATURE"]:
        p = tmp_path / "PREPROCESS_DATASET/commodity-futures" / folder / parts[0] / parts[1] / parts[2]
        p.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"timestamp": [1, 2]}).write_ipc(p / f"{date}.feather")

    for cs_folder in ["SNAPSHOT_FEATURE", "QUOTES_FEATURE", "KLINE_FEATURE"]:
        p = tmp_path / "PREPROCESS_DATASET/commodity-futures/CROSS_SECTION" / cs_folder / parts[0] / parts[1] / parts[2]
        p.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"timestamp": [1, 2]}).write_ipc(p / f"{date}.feather")

    # Create BASE_TIME_FEATURE with mismatching timestamps: [1, 999]
    bt_p = tmp_path / "PREPROCESS_DATASET/commodity-futures/BASE_TIME_FEATURE" / parts[0] / parts[1] / parts[2]
    bt_p.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"timestamp": [1, 999], "trading_minute_progress": [0.0, 1.0]}).write_ipc(bt_p / f"{date}.feather")

    args = parser.parse_args([
        "--symbols", "fu",
        "--contract", "fu2601",
        "--target_freq", "5min",
        "--date", date,
        "--root_path", str(tmp_path),
        "--data_path", "PREPROCESS_DATASET/commodity-futures/",
        "--save_path", "PREPROCESS_DATASET/commodity-futures/MERGE_CONCAT",
    ])

    with pytest.raises(ValueError, match="timestamp"):
        main(args)
