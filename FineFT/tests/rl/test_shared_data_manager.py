import os
import glob
import tempfile
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest
import torch

from env.env_class.futures_util import map_action_to_position_leverage
from RL.DiHFT.low_level.shared_data_manager import (
    EnvTensorPack,
    SharedMarketDataPack,
    create_demo_env_from_pack,
)
from RL.DiHFT.low_level.persistent_pool import PersistentRolloutPool
from RL.DiHFT.low_level.pretrain_qtable_diagnostics import (
    build_initial_state,
    create_demo_env,
)


def _create_sample_df(
    num_rows: int = 30,
    order_book_depth: int = 5,
    include_limits: bool = False,
    include_regime: bool = False,
) -> pd.DataFrame:
    data = {
        "mark_price": np.linspace(100.0, 115.0, num_rows, dtype=np.float32),
        "timestamp": pd.date_range("2026-01-01", periods=num_rows, freq="15min"),
        "funding_rate": np.full(num_rows, 0.0001, dtype=np.float32),
        "funding_timestamp": pd.date_range("2026-01-01 01:00", periods=num_rows, freq="15min"),
        "feat_1": np.arange(num_rows, dtype=np.float32),
        "feat_2": np.arange(num_rows, dtype=np.float32) * 1.5,
    }
    for i in range(1, order_book_depth + 1):
        data[f"bid{i}_price"] = data["mark_price"] - (0.1 * i)
        data[f"ask{i}_price"] = data["mark_price"] + (0.1 * i)
        data[f"bid{i}_size"] = np.full(num_rows, 10.0, dtype=np.float32)
        data[f"ask{i}_size"] = np.full(num_rows, 10.0, dtype=np.float32)

    if include_limits:
        data["limit_up_single_sided_ratio"] = np.zeros(num_rows, dtype=np.float32)
        data["limit_down_single_sided_ratio"] = np.zeros(num_rows, dtype=np.float32)
        data["limit_up_ask_depth_ratio_5"] = np.zeros(num_rows, dtype=np.float32)
        data["limit_down_bid_depth_ratio_5"] = np.zeros(num_rows, dtype=np.float32)
        data["UpperLimitPrice"] = data["mark_price"] * 1.1
        data["LowerLimitPrice"] = data["mark_price"] * 0.9

    if include_regime:
        data["regime_grid_id"] = np.random.randint(0, 9, size=num_rows, dtype=np.int64)

    return pd.DataFrame(data)


def _get_test_env_kwargs(order_book_depth: int = 5, enable_limit_reward: bool = False) -> dict:
    return {
        "feature_list": ["feat_1", "feat_2"],
        "order_book_depth": order_book_depth,
        "max_holding_number": 8,
        "position_choices": 5,
        "leverage_choices": [5],
        "long_estimated_rate": 0.0005,
        "short_estimated_rate": 0.0,
        "commission_rate": 0.0002,
        "maintenance_margin_ratio_dict": {"50000": [0.004, 0]},
        "early_stop": 0,
        "gamma": 0.99,
        "enable_limit_reward": enable_limit_reward,
    }


def test_shared_market_data_pack_creation_and_zero_copy_views():
    df = _create_sample_df(num_rows=25, order_book_depth=5)
    env_kwargs = _get_test_env_kwargs(order_book_depth=5)

    shared_pack = SharedMarketDataPack.from_dataframes({0: df}, env_kwargs)
    assert len(shared_pack) == 1
    assert 0 in shared_pack

    pack = shared_pack[0]
    # Assert POSIX shared memory flags
    assert pack.state_tensor.is_shared()
    assert pack.markprice_tensor.is_shared()
    assert pack.ask_prices_tensor.is_shared()
    assert pack.bid_prices_tensor.is_shared()
    assert pack.ask_qtys_tensor.is_shared()
    assert pack.bid_qtys_tensor.is_shared()
    assert pack.timestamp_tensor.is_shared()
    assert pack.funding_rate_tensor.is_shared()
    assert pack.funding_timestamp_tensor.is_shared()

    # Assert zero-copy views share underlying memory pointers
    assert np.shares_memory(pack.state_tensor.numpy(), pack.state_array)
    assert np.shares_memory(pack.markprice_tensor.numpy(), pack.markprice_array)
    assert np.shares_memory(pack.ask_prices_tensor.numpy(), pack.ask_prices_array)
    assert np.shares_memory(pack.timestamp_tensor.numpy(), pack.timestamp_array)

    # Verify dtypes and shapes
    assert pack.state_array.shape == (25, 2)
    assert pack.ask_prices_array.shape == (25, 5)
    assert pack.markprice_array.shape == (25,)
    assert pack.timestamp_array.dtype == np.dtype("datetime64[ns]")


def test_shared_market_data_pack_fail_fast_on_missing_limit_columns():
    # DataFrame without limit columns
    df = _create_sample_df(num_rows=20, include_limits=False)
    env_kwargs = _get_test_env_kwargs(enable_limit_reward=True)

    with pytest.raises(ValueError, match="enable_limit_reward=True 但 df_index=0 缺少必须的涨跌停列"):
        SharedMarketDataPack.from_dataframes({0: df}, env_kwargs)


def test_shared_market_data_pack_with_limit_columns():
    df = _create_sample_df(num_rows=20, include_limits=True, include_regime=True)
    env_kwargs = _get_test_env_kwargs(enable_limit_reward=True)

    shared_pack = SharedMarketDataPack.from_dataframes({0: df}, env_kwargs)
    pack = shared_pack[0]

    assert pack.is_limit_up_tensor is not None
    assert pack.is_limit_up_tensor.is_shared()
    assert pack.upper_limit_prices_tensor is not None
    assert pack.upper_limit_prices_tensor.is_shared()
    assert pack.regime_grid_ids_tensor is not None
    assert pack.regime_grid_ids_tensor.is_shared()


def test_in_place_environment_reset_numerical_invariants():
    df = _create_sample_df(num_rows=25, order_book_depth=5)
    env_kwargs = _get_test_env_kwargs(order_book_depth=5)
    shared_pack = SharedMarketDataPack.from_dataframes({0: df}, env_kwargs)
    pack = shared_pack[0]

    position_list = [-8.0, -4.0, 0.0, 4.0, 8.0]
    _, _, _, s1 = build_initial_state(df, 0, [5], position_list, 10000.0, 0.0)
    _, _, _, s2 = build_initial_state(df, 3, [5], position_list, 10000.0, 0.0)

    # 1. Instantiate persistent environment with s1
    env = create_demo_env_from_pack(pack, env_kwargs, initial_state=s1)
    # Memory pointer check: env arrays must share memory with pack tensors
    assert np.shares_memory(pack.state_tensor.numpy(), env.state_array)
    assert np.shares_memory(pack.markprice_tensor.numpy(), env.markprice_array)

    state1, info1 = env.reset()
    assert env.position == s1[3]
    # Execute a few steps to dirty internal state
    for action in [0, 1, 2]:
        env.step(action)
    assert env.day > 0

    # 2. In-place reset with s2
    state2_inplace, info2_inplace = env.reset(initial_state=s2)
    assert env.day == 0
    assert env.position == s2[3]

    # 3. Fresh environment with s2
    env_fresh = create_demo_env_from_pack(pack, env_kwargs, initial_state=s2)
    state2_fresh, info2_fresh = env_fresh.reset()

    # Compare exact numerical equivalence
    assert np.allclose(state2_inplace, state2_fresh)
    for k in info2_fresh:
        v1 = info2_inplace[k]
        v2 = info2_fresh[k]
        if isinstance(v1, np.ndarray):
            assert np.allclose(v1, v2), f"Field {k} differs in array value"
        else:
            assert v1 == v2, f"Field {k} differs: {v1} vs {v2}"

    # Step both environments forward and assert exact trajectory equivalence
    for step_idx in range(5):
        action = info2_fresh["avaiable_action_list"][0]
        s_in, r_in, d_in, i_in = env.step(action)
        s_fr, r_fr, d_fr, i_fr = env_fresh.step(action)
        assert np.allclose(s_in, s_fr)
        assert r_in == r_fr
        assert d_in == d_fr
        assert d_in is False


def test_90_worker_shared_memory_data_pointer_invariance():
    """Verify that 90 worker runners share identical memory addresses without duplication."""
    df = _create_sample_df(num_rows=20, order_book_depth=5)
    env_kwargs = _get_test_env_kwargs(order_book_depth=5)
    shared_pack = SharedMarketDataPack.from_dataframes({0: df}, env_kwargs)
    master_state_ptr = shared_pack[0].state_tensor.data_ptr()
    master_markprice_ptr = shared_pack[0].markprice_tensor.data_ptr()

    # Emulate 90 worker configurations
    worker_pointers = []
    for worker_id in range(90):
        pack = shared_pack[0]
        worker_pointers.append((pack.state_tensor.data_ptr(), pack.markprice_tensor.data_ptr()))

    assert len(worker_pointers) == 90
    for state_ptr, markprice_ptr in worker_pointers:
        assert state_ptr == master_state_ptr
        assert markprice_ptr == master_markprice_ptr


def test_zero_disk_artifacts_and_clean_pool_shutdown():
    """Verify that SharedMarketDataPack + PersistentRolloutPool creates 0 .pkl files in /dev/shm."""
    df = _create_sample_df(num_rows=15, order_book_depth=5)
    env_kwargs = _get_test_env_kwargs(order_book_depth=5)
    shared_pack = SharedMarketDataPack.from_dataframes({0: df}, env_kwargs)

    shm_dir = "/dev/shm" if os.path.exists("/dev/shm") else tempfile.gettempdir()
    shm_before = set(glob.glob(os.path.join(shm_dir, "fineft_train_df_*.pkl")))

    trainer = MagicMock()
    trainer.diverse_num_workers = 2
    trainer.dataset_name = "test_zerodisk"
    trainer.experiment_name = "test_exp"
    trainer.total_df_index_length = 1
    trainer.worker_processes = []
    trainer.leverage_choices = [5]
    trainer.position_list = [0.0]
    trainer.initial_wallet_balance = 10000.0
    trainer.initial_unrealized_pnL = 0.0
    trainer.tech_indicator_list = ["feat_1", "feat_2"]
    trainer.N_ACTIONS = 3
    trainer.hidden_nodes = 16
    trainer.time_info_dim = 2
    trainer.N = 1
    trainer.gamma = 0.99
    trainer.n_step = 1
    trainer.action_persistence = 1

    with PersistentRolloutPool(
        trainer=trainer,
        train_df_cache=None,
        env_kwargs=env_kwargs,
        shared_manager=None,
        shared_market_data=shared_pack,
    ) as pool:
        assert trainer.shm_df_cache_path is None
        shm_during = set(glob.glob(os.path.join(shm_dir, "fineft_train_df_*.pkl")))
        assert shm_during == shm_before, "Found unexpected .pkl file in shared memory directory!"

    assert pool.is_shutdown is True
    assert pool.shared_market_data is None
    shm_after = set(glob.glob(os.path.join(shm_dir, "fineft_train_df_*.pkl")))
    assert shm_after == shm_before
