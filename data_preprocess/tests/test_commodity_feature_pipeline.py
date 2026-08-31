import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from operator_futures.util import symbol_contract_path_parts
from operator_futures.commodity.schema import get_reward_execution_columns
from operator_futures.cross_section.base_feature_util import process_snapshot_features
from operator_futures.feature_selection.ic_correlation import (
    calculate_target,
    select_reward_state_features,
)
from operator_futures.feature_selection.contract_feature_union import (
    build_union_state_features,
    write_contract_feature_union,
)
from operator_futures.feature_selection.manifests import FeatureUnionResult


def _snapshot():
    row = {"timestamp": pd.Timestamp("2023-01-03 21:05:00")}
    for level in range(1, 6):
        row[f"ask{level}_price"] = 2600 + level
        row[f"ask{level}_size"] = level
        row[f"bid{level}_price"] = 2600 - level
        row[f"bid{level}_size"] = level + 1
    return pd.DataFrame([row])


def _single_sided_snapshot(empty_side: str):
    frame = _snapshot()
    if empty_side == "ask":
        for level in range(1, 6):
            frame[f"ask{level}_price"] = 3050.0
            frame[f"ask{level}_size"] = 0
    elif empty_side == "bid":
        for level in range(1, 6):
            frame[f"bid{level}_price"] = 2600.0
            frame[f"bid{level}_size"] = 0
    else:
        raise ValueError(empty_side)
    return frame


def test_snapshot_features_accept_depth_five_without_level_25():
    features = process_snapshot_features(_snapshot(), topk=3, depth=5)

    assert "midprice" in features.columns
    assert "buy_volume_oe" in features.columns
    assert "ask5_size_n" in features.columns
    assert "ask6_size_n" not in features.columns


def test_snapshot_features_handle_empty_ask_side_without_nan():
    features = process_snapshot_features(_single_sided_snapshot("ask"), topk=3, depth=5)

    row = features.row(0, named=True)
    assert bool(row["ask_side_empty"]) is True
    assert bool(row["bid_side_empty"]) is False
    assert row["sell_wap"] == 3050.0
    assert row["buy_sell_wap_spread"] == row["buy_wap"] - row["sell_wap"]
    for level in range(1, 6):
        assert row[f"ask{level}_size_n"] == 0.0
    assert not features.select(pl.any_horizontal(pl.selectors.float().is_nan())).item()
    assert not features.select(pl.any_horizontal(pl.selectors.float().is_infinite())).item()


def test_snapshot_features_handle_empty_bid_side_without_nan():
    features = process_snapshot_features(_single_sided_snapshot("bid"), topk=3, depth=5)

    row = features.row(0, named=True)
    assert bool(row["ask_side_empty"]) is False
    assert bool(row["bid_side_empty"]) is True
    assert row["buy_wap"] == 2600.0
    assert row["buy_sell_wap_spread"] == row["buy_wap"] - row["sell_wap"]
    for level in range(1, 6):
        assert row[f"bid{level}_size_n"] == 0.0
    assert not features.select(pl.any_horizontal(pl.selectors.float().is_nan())).item()
    assert not features.select(pl.any_horizontal(pl.selectors.float().is_infinite())).item()


def test_snapshot_features_flag_normal_two_sided_book_as_not_empty():
    features = process_snapshot_features(_snapshot(), topk=3, depth=5)
    row = features.row(0, named=True)

    assert bool(row["ask_side_empty"]) is False
    assert bool(row["bid_side_empty"]) is False
    assert row["ask1_size_n"] == 1 / sum(range(1, 6))
    assert row["bid1_size_n"] == 2 / sum(range(2, 7))


def test_snapshot_features_reject_both_sides_empty():
    frame = _snapshot()
    for side in ("ask", "bid"):
        for level in range(1, 6):
            frame[f"{side}{level}_size"] = 0

    with pytest.raises(ValueError, match="both sides have zero total size"):
        process_snapshot_features(frame, topk=3, depth=5)


def test_manifest_replaces_first_106_reward_columns():
    reward_columns = get_reward_execution_columns(depth=5)

    assert len(reward_columns) == 32
    assert "contract" in reward_columns
    assert "close" in reward_columns
    assert "volume" in reward_columns
    assert "tradeval" in reward_columns
    assert "ask5_price" in reward_columns
    assert "LowerLimitPrice" in reward_columns
    assert "UpperLimitPrice" in reward_columns
    assert "ask25_price" not in reward_columns


def test_ic_correlation_uses_commodity_manifest_for_reward_columns():
    reward_columns = get_reward_execution_columns(depth=5)
    df = pd.DataFrame({column: [1.0, 2.0] for column in reward_columns})
    df["contract"] = ["fu2601", "fu2601"]
    df["state_alpha"] = [0.1, 0.2]

    selected_reward, selected_state = select_reward_state_features(
        df, market_type="commodity_futures", orderbook_depth=5
    )

    assert selected_reward == reward_columns
    assert "volume" in selected_reward
    assert "tradeval" in selected_reward
    assert selected_state == ["state_alpha"]


def test_feature_selection_target_remains_price_difference():
    df = pd.DataFrame({"mark_price": [10.0, 12.5, 11.0]})

    target = calculate_target(df, "mark_price", 1)

    assert target.tolist() == [2.5, -1.5]


def test_symbol_contract_path_parts_uses_contract_when_present():
    assert symbol_contract_path_parts("fu", "fu2601") == ("fu", "fu2601")


def test_symbol_contract_path_parts_keeps_legacy_symbol_only_path():
    assert symbol_contract_path_parts("BTCUSDT", None) == ("BTCUSDT",)


def test_contract_path_shape_for_daily_outputs(tmp_path):
    parts = symbol_contract_path_parts("fu", "fu2601")
    path = tmp_path.joinpath("BASE_FEATURE", *parts, "5min", "2026-01-05.feather")
    assert path.as_posix().endswith("BASE_FEATURE/fu/fu2601/5min/2026-01-05.feather")


def test_legacy_path_shape_for_daily_outputs(tmp_path):
    parts = symbol_contract_path_parts("fu", None)
    path = tmp_path.joinpath("BASE_FEATURE", *parts, "5min", "2026-01-05.feather")
    assert path.as_posix().endswith("BASE_FEATURE/fu/5min/2026-01-05.feather")


def test_cross_section_parser_accepts_contract():
    from operator_futures.cross_section.create_feature import parser

    args = parser.parse_args(["--contract", "fu2601"])

    assert args.contract == "fu2601"


def test_build_union_state_features_preserves_first_seen_order():
    result = build_union_state_features(
        [
            ["alpha", "beta", "alpha"],
            ["beta", "gamma"],
            ["delta", "alpha"],
        ]
    )

    assert result == ["alpha", "beta", "gamma", "delta"]


def _write_two_contract_summary(path):
    path.write_text(
        json.dumps(
            {
                "symbol": "fu",
                "commodity_name": "燃料油",
                "start_date": "2026-01-01",
                "end_date": "2026-04-01",
                "contracts": [
                    {
                        "contract": "fu2601",
                        "start_trading_day": "20260101",
                        "end_trading_day": "20260102",
                        "trading_day_count": 1,
                        "last_trading_day": "20260105",
                        "total_trading_day_count": 1,
                        "selected_months": ["2026-01"],
                        "trading_days": [
                            {
                                "trading_day": "20260101",
                                "date": "2026-01-01",
                                "source_file": "fu2601.csv",
                                "daily_volume": 1.0,
                            }
                        ],
                    },
                    {
                        "contract": "fu2605",
                        "start_trading_day": "20260201",
                        "end_trading_day": "20260202",
                        "trading_day_count": 1,
                        "last_trading_day": "20260105",
                        "total_trading_day_count": 1,
                        "selected_months": ["2026-02"],
                        "trading_days": [
                            {
                                "trading_day": "20260201",
                                "date": "2026-02-01",
                                "source_file": "fu2605.csv",
                                "daily_volume": 1.0,
                            }
                        ],
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_write_contract_feature_union_writes_symbol_level_manifest(tmp_path):
    summary_path = tmp_path / "main_contract_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "symbol": "fu",
                "commodity_name": "燃料油",
                "start_date": "2026-01-01",
                "end_date": "2026-04-01",
                "contracts": [
                    {
                        "contract": "fu2601",
                        "start_trading_day": "20260101",
                        "end_trading_day": "20260102",
                        "trading_day_count": 1,
                        "last_trading_day": "20260105",
                        "total_trading_day_count": 1,
                        "selected_months": ["2026-01"],
                        "trading_days": [
                            {
                                "trading_day": "20260101",
                                "date": "2026-01-01",
                                "source_file": "fu2601.csv",
                                "daily_volume": 1.0,
                            }
                        ],
                    },
                    {
                        "contract": "fu2605",
                        "start_trading_day": "20260201",
                        "end_trading_day": "20260202",
                        "trading_day_count": 1,
                        "last_trading_day": "20260105",
                        "total_trading_day_count": 1,
                        "selected_months": ["2026-02"],
                        "trading_days": [
                            {
                                "trading_day": "20260201",
                                "date": "2026-02-01",
                                "source_file": "fu2605.csv",
                                "daily_volume": 1.0,
                            }
                        ],
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    base = tmp_path / "PREPROCESS_DATASET" / "commodity-futures"
    first = base / "SCALE_SAVE" / "fu" / "fu2601" / "5min" / "2026-01-01-2026-04-01"
    second = base / "SCALE_SAVE" / "fu" / "fu2605" / "5min" / "2026-01-01-2026-04-01"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    np.save(first / "state_features.npy", np.array(["alpha", "beta"]))
    np.save(second / "state_features.npy", np.array(["beta", "gamma"]))

    result = write_contract_feature_union(
        root_path=tmp_path,
        summary_path=summary_path,
        symbol="fu",
        target_freq="5min",
        start_date="2026-01-01",
        end_date="2026-04-01",
    )
    assert isinstance(result, FeatureUnionResult)
    output_dir = result.output_dir

    assert output_dir == (
        base / "FEATURE_UNION" / "fu" / "5min" / "2026-01-01-2026-04-01"
    )
    assert np.load(output_dir / "state_features.npy", allow_pickle=True).tolist() == [
        "alpha",
        "beta",
        "gamma",
    ]
    manifest = json.loads(
        (output_dir / "feature_union_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest == result.manifest.to_dict()
    assert manifest["contracts"] == ["fu2601", "fu2605"]
    assert manifest["state_features"] == ["alpha", "beta", "gamma"]
    assert manifest["state_feature_count"] == 3
    assert result.manifest.contracts == ["fu2601", "fu2605"]
    assert result.manifest.state_features == ["alpha", "beta", "gamma"]
    assert result.manifest.state_feature_count == 3


def test_write_contract_feature_union_finalizes_ic_result_from_candidates(tmp_path):
    summary_path = tmp_path / "main_contract_summary.json"
    _write_two_contract_summary(summary_path)
    base = tmp_path / "PREPROCESS_DATASET" / "commodity-futures"
    date_range = "2026-01-01-2026-04-01"

    first_candidate = base / "IC_RESULT" / "fu" / "fu2601" / "5min" / date_range
    second_candidate = base / "IC_RESULT" / "fu" / "fu2605" / "5min" / date_range
    first_candidate.mkdir(parents=True)
    second_candidate.mkdir(parents=True)
    np.save(first_candidate / "state_features_candidate.npy", np.array(["alpha", "beta"]))
    np.save(second_candidate / "state_features_candidate.npy", np.array(["beta", "gamma"]))

    for contract in ["fu2601", "fu2605"]:
        all_feature_dir = base / "ALL_FEATURE" / "fu" / contract / "5min"
        all_feature_dir.mkdir(parents=True)
        pl.DataFrame(
            {
                "timestamp": [1, 2],
                "mark_price": [100.0, 101.0],
                "index_price": [100.0, 101.0],
                "funding_timestamp": [1, 2],
                "funding_rate": [0.0, 0.0],
                "ask1_price": [101.0, 102.0],
                "ask1_size": [10.0, 11.0],
                "bid1_price": [99.0, 100.0],
                "bid1_size": [12.0, 13.0],
                "alpha": [1.0, 2.0],
                "beta": [3.0, 4.0],
                "gamma": [5.0, 6.0],
            }
        ).write_ipc(all_feature_dir / f"{date_range}.feather")

    result = write_contract_feature_union(
        root_path=tmp_path,
        summary_path=summary_path,
        symbol="fu",
        target_freq="5min",
        start_date="2026-01-01",
        end_date="2026-04-01",
        candidate_path="PREPROCESS_DATASET/commodity-futures/IC_RESULT",
        all_feature_path="PREPROCESS_DATASET/commodity-futures/ALL_FEATURE",
        ic_result_path="PREPROCESS_DATASET/commodity-futures/IC_RESULT",
        finalize_filtered_df=True,
        market_type="commodity_futures",
        orderbook_depth=5,
    )
    assert isinstance(result, FeatureUnionResult)
    output_dir = result.output_dir

    assert np.load(output_dir / "state_features.npy", allow_pickle=True).tolist() == [
        "alpha",
        "beta",
        "gamma",
    ]
    for contract in ["fu2601", "fu2605"]:
        contract_dir = base / "IC_RESULT" / "fu" / contract / "5min" / date_range
        assert np.load(contract_dir / "state_features.npy", allow_pickle=True).tolist() == [
            "alpha",
            "beta",
            "gamma",
        ]
        frame = pl.read_ipc(contract_dir / "df.feather")
        expected_reward_columns = [
            column
            for column in get_reward_execution_columns(depth=5)
            if column in frame.columns
        ]
        assert frame.columns == [*expected_reward_columns, "alpha", "beta", "gamma"]

    manifest = json.loads(
        (output_dir / "feature_union_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest == result.manifest.to_dict()
    assert manifest["state_feature_count"] == 3
    assert manifest["per_contract_output_shapes"]["fu2601"]["rows"] == 2
    assert manifest["per_contract_output_shapes"]["fu2605"]["columns"] == 12
    assert result.manifest.per_contract_output_shapes["fu2601"].rows == 2
    assert result.manifest.per_contract_output_shapes["fu2605"].columns == 12
    assert manifest["candidate_source_path"] == "PREPROCESS_DATASET/commodity-futures/IC_RESULT"
    assert manifest["all_feature_path"] == "PREPROCESS_DATASET/commodity-futures/ALL_FEATURE"
    assert manifest["ic_result_path"] == "PREPROCESS_DATASET/commodity-futures/IC_RESULT"
    assert manifest["finalize_filtered_df"] is True
    assert set(manifest["per_contract_output_paths"]) == {"fu2601", "fu2605"}
    for output_path in manifest["per_contract_output_paths"].values():
        assert Path(output_path).exists()


def test_write_contract_feature_union_fails_when_contract_state_features_missing(
    tmp_path,
):
    summary_path = tmp_path / "main_contract_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "symbol": "fu",
                "commodity_name": "燃料油",
                "start_date": "2026-01-01",
                "end_date": "2026-04-01",
                "contracts": [
                    {
                        "contract": "fu2605",
                        "start_trading_day": "20260201",
                        "end_trading_day": "20260202",
                        "trading_day_count": 1,
                        "last_trading_day": "20260105",
                        "total_trading_day_count": 1,
                        "selected_months": ["2026-02"],
                        "trading_days": [
                            {
                                "trading_day": "20260201",
                                "date": "2026-02-01",
                                "source_file": "fu2605.csv",
                                "daily_volume": 1.0,
                            }
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError) as excinfo:
        write_contract_feature_union(
            root_path=tmp_path,
            summary_path=summary_path,
            symbol="fu",
            target_freq="5min",
            start_date="2026-01-01",
            end_date="2026-04-01",
            candidate_path="PREPROCESS_DATASET/commodity-futures/IC_RESULT",
            finalize_filtered_df=True,
        )

    message = str(excinfo.value)
    assert "fu2605" in message
    assert "state_features_candidate.npy" in message


def test_write_contract_feature_union_fails_when_candidate_union_empty(tmp_path):
    summary_path = tmp_path / "main_contract_summary.json"
    _write_two_contract_summary(summary_path)
    date_range = "2026-01-01-2026-04-01"
    for contract in ["fu2601", "fu2605"]:
        candidate_dir = (
            tmp_path
            / "PREPROCESS_DATASET/commodity-futures/IC_RESULT/fu"
            / contract
            / "5min"
            / date_range
        )
        candidate_dir.mkdir(parents=True)
        np.save(candidate_dir / "state_features_candidate.npy", np.array([]))

    with pytest.raises(ValueError) as excinfo:
        write_contract_feature_union(
            root_path=tmp_path,
            summary_path=summary_path,
            symbol="fu",
            target_freq="5min",
            start_date="2026-01-01",
            end_date="2026-04-01",
            candidate_path="PREPROCESS_DATASET/commodity-futures/IC_RESULT",
            finalize_filtered_df=True,
        )

    assert "Feature union is empty" in str(excinfo.value)


def test_write_contract_feature_union_fails_when_union_feature_missing_from_contract(
    tmp_path,
):
    summary_path = tmp_path / "main_contract_summary.json"
    _write_two_contract_summary(summary_path)
    base = tmp_path / "PREPROCESS_DATASET" / "commodity-futures"
    date_range = "2026-01-01-2026-04-01"

    for contract, features in {
        "fu2601": ["alpha", "gamma"],
        "fu2605": ["gamma"],
    }.items():
        candidate_dir = base / "IC_RESULT" / "fu" / contract / "5min" / date_range
        candidate_dir.mkdir(parents=True)
        np.save(candidate_dir / "state_features_candidate.npy", np.array(features))

    all_feature_dir = base / "ALL_FEATURE" / "fu" / "fu2601" / "5min"
    all_feature_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "timestamp": [1],
            "mark_price": [100.0],
            "alpha": [1.0],
            "gamma": [2.0],
        }
    ).write_ipc(all_feature_dir / f"{date_range}.feather")

    missing_dir = base / "ALL_FEATURE" / "fu" / "fu2605" / "5min"
    missing_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "timestamp": [1],
            "mark_price": [100.0],
            "gamma": [2.0],
        }
    ).write_ipc(missing_dir / f"{date_range}.feather")

    with pytest.raises(ValueError) as excinfo:
        write_contract_feature_union(
            root_path=tmp_path,
            summary_path=summary_path,
            symbol="fu",
            target_freq="5min",
            start_date="2026-01-01",
            end_date="2026-04-01",
            candidate_path="PREPROCESS_DATASET/commodity-futures/IC_RESULT",
            finalize_filtered_df=True,
            market_type="commodity_futures",
            orderbook_depth=5,
        )

    message = str(excinfo.value)
    assert "fu2605" in message
    assert "alpha" in message
    first_output_dir = base / "IC_RESULT" / "fu" / "fu2601" / "5min" / date_range
    union_output_dir = base / "FEATURE_UNION" / "fu" / "5min" / date_range
    assert not (first_output_dir / "df.feather").exists()
    assert not (first_output_dir / "state_features.npy").exists()
    assert not (union_output_dir / "state_features.npy").exists()
    assert not (union_output_dir / "feature_union_manifest.json").exists()
