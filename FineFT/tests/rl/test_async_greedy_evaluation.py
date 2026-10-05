from __future__ import annotations

import time
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest
import torch

from model.low_level import ensemble_Qnet
from RL.DiHFT.low_level.parallel_diverse_train import (
    AsyncGreedyEvaluator,
    EvaluationTask,
)
from RL.DiHFT.low_level.shared_data_manager import SharedMarketDataPack


def _create_sample_df(num_rows: int = 15, order_book_depth: int = 5) -> pd.DataFrame:
    data = {
        "timestamp": pd.date_range("2026-01-01", periods=num_rows, freq="10min"),
        "funding_timestamp": pd.date_range("2026-01-01", periods=num_rows, freq="8h"),
        "mark_price": np.linspace(100.0, 105.0, num_rows, dtype=np.float32),
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


def test_async_evaluator_lifecycle_and_non_blocking():
    df = _create_sample_df(num_rows=15, order_book_depth=5)
    env_kwargs = _get_test_env_kwargs(order_book_depth=5)
    shared_pack = SharedMarketDataPack.from_dataframes({0: df}, env_kwargs)

    model = ensemble_Qnet(
        N_STATES=2,
        N_ACTIONS=3,
        hidden_nodes=16,
        TIME_INFO_DIM=2,
        ensemble_number=2,
        TRADING_INFO_DIM=4,
    )

    trainer = MagicMock()
    trainer.eval_net = model
    trainer.device = "cpu"
    trainer.tech_indicator_list = ["feat_1", "feat_2"]
    trainer.N_ACTIONS = 3
    trainer.hidden_nodes = 16
    trainer.time_info_dim = 2
    trainer.N = 2
    trainer.eval_dfs = "0"
    trainer.position_list = [-1.0, 0.0, 1.0]
    trainer.leverage_choices = [1]
    trainer.initial_wallet_balance = 10000.0
    trainer.initial_unrealized_pnL = 0.0
    trainer.early_stop = 0
    trainer.writer = MagicMock()

    evaluator = AsyncGreedyEvaluator(
        trainer=trainer,
        train_df_cache={0: df},
        env_kwargs=env_kwargs,
        shared_market_data=shared_pack,
    )

    start_time = time.perf_counter()
    evaluator.submit_evaluation(epoch_index=0, model=model)
    dispatch_time = time.perf_counter() - start_time

    assert dispatch_time < 0.1

    evaluator.wait_all()
    evaluator.shutdown()

    assert "mean_return_rate" in evaluator.latest_metrics
    assert "profit_ratio" in evaluator.latest_metrics
    assert "mean_trades" in evaluator.latest_metrics
    assert trainer.writer.add_scalar.call_count >= 3


def test_async_evaluator_thread_safe_early_stopping():
    df = _create_sample_df(num_rows=15, order_book_depth=5)
    env_kwargs = _get_test_env_kwargs(order_book_depth=5)
    shared_pack = SharedMarketDataPack.from_dataframes({0: df}, env_kwargs)

    model = ensemble_Qnet(
        N_STATES=2,
        N_ACTIONS=3,
        hidden_nodes=16,
        TIME_INFO_DIM=2,
        ensemble_number=2,
        TRADING_INFO_DIM=4,
    )

    trainer = MagicMock()
    trainer.eval_net = model
    trainer.device = "cpu"
    trainer.writer = MagicMock()

    evaluator = AsyncGreedyEvaluator(
        trainer=trainer,
        train_df_cache={0: df},
        env_kwargs=env_kwargs,
        shared_market_data=shared_pack,
    )

    assert not evaluator.should_early_stop()
    evaluator.trigger_early_stop()
    assert evaluator.should_early_stop()

    evaluator.shutdown()
