import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class FakeEnv:
    def __init__(self):
        self.slippage_sum = 0.0
        self.position = 0.0
        self.leverage = 1
        self.commission_rate = 0.0002
        self._step = 0

    def reset(self):
        return None, {"previous_action": 0}

    def step(self, action):
        self._step += 1
        self.slippage_sum += 0.5
        self.position = float(action)
        reward = 10.0 * self._step
        done = self._step == 2
        return None, reward, done, {"previous_action": action}


def test_diagnostics_cache_qtables_once_and_export_one_csv_per_sample(
    monkeypatch, tmp_path
):
    from RL.DiHFT.low_level import pretrain_qtable_diagnostics as diag

    train_data_path = tmp_path / "train"
    train_data_path.mkdir()
    df = pd.DataFrame(
        {
            "timestamp": [1, 2, 3],
            "open": [10.0, 11.0, 12.0],
            "high": [11.0, 12.0, 13.0],
            "low": [9.0, 10.0, 11.0],
            "close": [10.5, 11.5, 12.5],
            "volume": [100.0, 110.0, 120.0],
            "mark_price": [10.2, 11.2, 12.2],
        }
    )
    df.to_feather(train_data_path / "df_0.feather")

    calls = []

    def fake_create_q_table_from_df(df, **kwargs):
        calls.append(kwargs)
        return np.zeros((len(df), 3, 3))

    monkeypatch.setattr(
        diag, "create_optimal_q_table_from_df", fake_create_q_table_from_df
    )
    monkeypatch.setattr(
        diag, "get_dp_action_from_qtable", lambda q_table, initial_action: [1, 2]
    )
    monkeypatch.setattr(diag, "initiate_demo_env", lambda **kwargs: FakeEnv())
    monkeypatch.setattr(
        diag, "build_sample_plan", lambda num_sample, total, choices: [(0, 0), (0, 1)]
    )

    sample_plan, q_table_cache, train_df_cache, diagnostics = (
        diag.prepare_pretrain_qtable_diagnostics(
            num_sample=2,
            total_df_index_length=1,
            position_choices=3,
            train_data_path=str(train_data_path),
            qtable_kwargs={
                "max_holding_number": 8,
                "order_book_depth": 25,
                "position_choices": 3,
                "leverage_choice": [1],
                "long_estimated_rate": 0.0005,
                "short_estimated_rate": 0,
                "commission_rate": 0.0002,
                "max_punishment": 1e10,
                "gamma": 1,
            },
            env_kwargs={
                "feature_list": [],
                "max_holding_number": 8,
                "order_book_depth": 25,
                "position_choices": 3,
                "leverage_choices": [1],
                "position_list": [-8.0, 0, 8.0],
                "long_estimated_rate": 0.0005,
                "short_estimated_rate": 0,
                "commission_rate": 0.0002,
                "maintenance_margin_ratio_dict": {},
                "early_stop": 0,
                "gamma": 1,
                "initial_wallet_balance": 100000,
                "initial_unrealized_pnl": 0,
            },
            output_dir=str(tmp_path / "diagnostics"),
            process_count=1,
        )
    )

    assert sample_plan == [(0, 0), (0, 1)]
    assert list(q_table_cache) == [0]
    assert list(train_df_cache) == [0]
    assert len(calls) == 1
    assert len(diagnostics) == 2

    csv_files = sorted((tmp_path / "diagnostics").glob("sample_*.csv"))
    assert len(csv_files) == 2
    first = pd.read_csv(csv_files[0])
    required_columns = {
        "sample_index",
        "df_index",
        "initial_action",
        "step_index",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "mark_price",
        "action",
        "previous_action",
        "position",
        "leverage",
        "commission_rate",
        "step_slippage",
        "step_reward",
        "cumulative_profit",
        "profitable",
    }
    assert required_columns.issubset(first.columns)
    assert first["cumulative_profit"].tolist() == [10.0, 30.0]
    assert first["step_slippage"].tolist() == [0.5, 0.5]
