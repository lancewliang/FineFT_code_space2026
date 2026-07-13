import json

import numpy as np
import pandas as pd
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


def _snapshot():
    row = {"timestamp": pd.Timestamp("2023-01-03 21:05:00")}
    for level in range(1, 6):
        row[f"ask{level}_price"] = 2600 + level
        row[f"ask{level}_size"] = level
        row[f"bid{level}_price"] = 2600 - level
        row[f"bid{level}_size"] = level + 1
    return pd.DataFrame([row])


def test_snapshot_features_accept_depth_five_without_level_25():
    features = process_snapshot_features(_snapshot(), topk=3, depth=5)

    assert "midprice" in features.columns
    assert "buy_volume_oe" in features.columns
    assert "ask5_size_n" in features.columns
    assert "ask6_size_n" not in features.columns


def test_manifest_replaces_first_106_reward_columns():
    reward_columns = get_reward_execution_columns(depth=5)

    assert len(reward_columns) == 26
    assert "ask5_price" in reward_columns
    assert "ask25_price" not in reward_columns


def test_ic_correlation_uses_commodity_manifest_for_reward_columns():
    reward_columns = get_reward_execution_columns(depth=5)
    df = pd.DataFrame({column: [1.0, 2.0] for column in reward_columns})
    df["state_alpha"] = [0.1, 0.2]

    selected_reward, selected_state = select_reward_state_features(
        df, market_type="commodity_futures", orderbook_depth=5
    )

    assert selected_reward == reward_columns
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

    output_dir = write_contract_feature_union(
        root_path=tmp_path,
        summary_path=summary_path,
        symbol="fu",
        target_freq="5min",
        start_date="2026-01-01",
        end_date="2026-04-01",
    )

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
    assert manifest["contracts"] == ["fu2601", "fu2605"]
    assert manifest["state_features"] == ["alpha", "beta", "gamma"]
    assert manifest["state_feature_count"] == 3


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
        )

    message = str(excinfo.value)
    assert "fu2605" in message
    assert "state_features.npy" in message
