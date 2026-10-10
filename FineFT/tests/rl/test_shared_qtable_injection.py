from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from env.env_class.demo_env import Demo_Env
from RL.DiHFT.low_level.shared_data_manager import (
    EnvTensorPack,
    SharedMarketDataPack,
    create_demo_env_from_pack,
)


def _create_sample_df(num_rows: int = 20, order_book_depth: int = 5) -> pd.DataFrame:
    data = {
        "timestamp": pd.date_range("2026-01-01", periods=num_rows, freq="10min"),
        "funding_timestamp": pd.date_range("2026-01-01", periods=num_rows, freq="8h"),
        "mark_price": np.linspace(100.0, 110.0, num_rows, dtype=np.float32),
        "funding_rate": np.zeros(num_rows, dtype=np.float32),
        "feat_1": np.random.randn(num_rows).astype(np.float32),
        "feat_2": np.random.randn(num_rows).astype(np.float32),
    }
    for i in range(1, order_book_depth + 1):
        data[f"bid{i}_price"] = data["mark_price"] - i * 0.1
        data[f"ask{i}_price"] = data["mark_price"] + i * 0.1
        data[f"bid{i}_size"] = np.ones(num_rows, dtype=np.float32) * 10.0
        data[f"ask{i}_size"] = np.ones(num_rows, dtype=np.float32) * 10.0
    return pd.DataFrame(data)


def _get_test_env_kwargs(order_book_depth: int = 5) -> dict:
    return {
        "feature_list": ["feat_1", "feat_2"],
        "order_book_depth": order_book_depth,
        "max_holding_number": 1,
        "position_choices": 3,
        "leverage_choices": [1],
        "long_estimated_rate": 0.0,
        "short_estimated_rate": 0.0,
        "commission_rate": 0.0002,
        "maintenance_margin_ratio_dict": {"50000": [0.004, 0]},
        "early_stop": 0,
        "gamma": 0.99,
        "enable_limit_reward": False,
    }


def test_shared_qtable_in_shared_market_data_pack():
    df = _create_sample_df(num_rows=20, order_book_depth=5)
    env_kwargs = _get_test_env_kwargs(order_book_depth=5)

    shared_pack = SharedMarketDataPack.from_dataframes({0: df}, env_kwargs)
    assert 0 in shared_pack
    pack = shared_pack[0]

    assert pack.q_table_tensor is not None
    assert pack.q_table_tensor.is_shared()
    assert pack.q_table_array is not None
    assert np.shares_memory(pack.q_table_tensor.numpy(), pack.q_table_array)


def test_demo_env_bypasses_qtable_computation_when_supplied(monkeypatch):
    df = _create_sample_df(num_rows=20, order_book_depth=5)
    env_kwargs = _get_test_env_kwargs(order_book_depth=5)

    shared_pack = SharedMarketDataPack.from_dataframes({0: df}, env_kwargs)
    pack = shared_pack[0]
    precomputed_q = pack.q_table_array

    call_count = [0]

    def spy_create_optimal_q_table(*args, **kwargs):
        call_count[0] += 1
        return np.zeros((20, 3, 3), dtype=np.float32)

    import env.env_class.demo_env as demo_env_module
    monkeypatch.setattr(demo_env_module, "create_optimal_q_table", spy_create_optimal_q_table)

    env = create_demo_env_from_pack(pack, env_kwargs)

    assert call_count[0] == 0
    assert np.shares_memory(env.q_table, pack.q_table_tensor.numpy())
    np.testing.assert_array_equal(env.q_table, precomputed_q)


def test_demo_env_qtable_identical_to_inplace_computation():
    df = _create_sample_df(num_rows=20, order_book_depth=5)
    env_kwargs = _get_test_env_kwargs(order_book_depth=5)

    shared_pack = SharedMarketDataPack.from_dataframes({0: df}, env_kwargs)
    pack = shared_pack[0]

    env_with_shared = create_demo_env_from_pack(pack, env_kwargs)

    from env.env_class.demo_env import Demo_Env
    arrays = pack.to_env_kwargs()
    env_computed_inplace = Demo_Env(
        state_array=arrays["state_array"],
        ask_prices_array=arrays["ask_prices_array"],
        bid_prices_array=arrays["bid_prices_array"],
        ask_qtys_array=arrays["ask_qtys_array"],
        bid_qtys_array=arrays["bid_qtys_array"],
        markprice_array=arrays["markprice_array"],
        timestamp_array=arrays["timestamp_array"],
        funding_rate_array=arrays["funding_rate_array"],
        funding_timestamp_array=arrays["funding_timestamp_array"],
        max_holding_number=env_kwargs["max_holding_number"],
        position_choices=env_kwargs["position_choices"],
        leverage_choice=env_kwargs["leverage_choices"],
        long_estimated_rate=env_kwargs["long_estimated_rate"],
        short_estimated_rate=env_kwargs["short_estimated_rate"],
        commission_rate=env_kwargs["commission_rate"],
        maintenance_margin_ratio_dict=env_kwargs["maintenance_margin_ratio_dict"],
        early_stop=env_kwargs["early_stop"],
        initial_state=(1e5, 0.0, 0.0, 0.0, 1.0),
        gamma=env_kwargs["gamma"],
        max_punishment=1e10,
    )

    np.testing.assert_allclose(env_with_shared.q_table, env_computed_inplace.q_table, rtol=1e-5, atol=1e-5)


def test_default_diverse_num_workers_is_40():
    import argparse
    from RL.DiHFT.low_level.parallel_weight_advantage_pretrain import parser

    actions = [a for a in parser._actions if "--diverse_num_workers" in a.option_strings]
    assert len(actions) == 1
    assert actions[0].default == 40

    with open("FineFT/script/train/train_commodity_fu_10.sh", "r", encoding="utf-8") as f:
        sh_content = f.read()
    assert "--diverse_num_workers 40" in sh_content
