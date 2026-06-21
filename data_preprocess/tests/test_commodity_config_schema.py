import pytest
import pandas as pd

from operator_futures.commodity.config import CommodityConfig, get_commodity_config
from operator_futures.commodity.schema import (
    build_orderbook_columns,
    get_reward_execution_columns,
    resample_kwargs,
)


def test_fu_config_contract():
    config = get_commodity_config("fu")

    assert config.symbol == "fu"
    assert config.dataset_name == "fu"
    assert config.orderbook_depth == 5
    assert config.funding_enabled is False
    assert config.buy_fee_rate == 0.0001
    assert config.sell_fee_rate == 0.0003
    assert config.main_contract_months == tuple(range(1, 13))
    assert config.contract_unit == 10
    assert config.use_contract_multiplier is False


def test_commodity_config_rejects_non_positive_contract_unit():
    with pytest.raises(ValueError, match="contract_unit must be positive"):
        CommodityConfig(
            symbol="bad",
            display_name="bad",
            dataset_name="bad",
            orderbook_depth=5,
            funding_enabled=False,
            buy_fee_rate=0.0001,
            sell_fee_rate=0.0003,
            main_contract_months=(1,),
            contract_unit=0,
            use_contract_multiplier=False,
        )


def test_depth_five_orderbook_columns_have_no_synthetic_levels():
    columns = build_orderbook_columns(5)

    assert "ask1_price" in columns
    assert "ask5_size" in columns
    assert "bid5_price" in columns
    assert "ask6_price" not in columns
    assert "bid25_price" not in columns
    assert len(columns) == 20


def test_reward_execution_manifest_for_depth_five():
    columns = get_reward_execution_columns(depth=5)

    assert columns[0] == "timestamp"
    assert "mark_price" in columns
    assert "funding_rate" in columns
    assert "ask5_price" in columns
    assert "bid5_size" in columns
    assert "ask6_price" not in columns
    assert len(columns) == 1 + 20 + 5


def test_resample_kwargs_are_right_closed_and_right_labeled():
    kwargs = resample_kwargs()

    assert kwargs == {"closed": "right", "label": "right"}
    series = pd.Series(
        [1, 2],
        index=pd.to_datetime(["2023-01-03 09:00:00", "2023-01-03 09:05:00"]),
    )
    result = series.resample("5min", **kwargs).sum()

    assert result.loc[pd.Timestamp("2023-01-03 09:00:00")] == 1
    assert result.loc[pd.Timestamp("2023-01-03 09:05:00")] == 2
