import pandas as pd

from operator_futures.commodity.schema import get_reward_execution_columns
from operator_futures.cross_section.base_feature_util import process_snapshot_features
from operator_futures.feature_selection.ic_correlation import calculate_target


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


def test_feature_selection_target_remains_price_difference():
    df = pd.DataFrame({"mark_price": [10.0, 12.5, 11.0]})

    target = calculate_target(df, "mark_price", 1)

    assert target.tolist() == [2.5, -1.5]
