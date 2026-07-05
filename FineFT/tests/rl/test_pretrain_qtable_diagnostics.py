import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


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


def test_build_sample_plan_traverses_every_df_and_position():
    from RL.DiHFT.low_level import pretrain_qtable_diagnostics as diag

    sample_plan = diag.build_sample_plan(
        total_df_index_length=2,
        position_choices=3,
    )

    assert sample_plan == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
    ]


def test_select_sample_from_plan_randomly_picks_one_combination(monkeypatch):
    from RL.DiHFT.low_level import pretrain_qtable_diagnostics as diag

    sample_plan = [(0, 0), (0, 1), (1, 0), (1, 1)]
    monkeypatch.setattr(diag.random, "choice", lambda choices: choices[3])

    assert diag.select_sample_from_plan(sample_plan) == (1, 1)


def test_get_sample_action_from_cache_raises_when_combination_missing():
    from RL.DiHFT.low_level import pretrain_qtable_diagnostics as diag

    sample_action_cache_by_plan = {(0, 0): [1, 2]}

    assert diag.get_sample_action_from_cache(
        sample_action_cache_by_plan, 0, 0
    ) == [1, 2]
    with pytest.raises(KeyError, match="df_index=0.*initial_action=1"):
        diag.get_sample_action_from_cache(sample_action_cache_by_plan, 0, 1)


def test_diagnostics_cache_qtables_once_and_export_one_csv_per_df_action(
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
        diag, "build_sample_plan", lambda total, choices: [(0, 0), (0, 1)]
    )

    sample_plan, q_table_cache, train_df_cache, diagnostics, sample_action_cache = (
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
    assert sample_action_cache == {(0, 0): [1, 2], (0, 1): [1, 2]}

    csv_files = sorted((tmp_path / "diagnostics").glob("df_*_initial_action_*.csv"))
    assert len(csv_files) == 2
    assert [path.name for path in csv_files] == [
        "df_0_initial_action_0.csv",
        "df_0_initial_action_1.csv",
    ]
    first = pd.read_csv(csv_files[0])
    required_columns = {
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
    assert "sample_index" not in first.columns
    assert first["cumulative_profit"].tolist() == [10.0, 30.0]
    assert first["step_slippage"].tolist() == [0.5, 0.5]


def test_diagnostics_read_existing_csvs_without_recomputing(monkeypatch, tmp_path):
    from RL.DiHFT.low_level import pretrain_qtable_diagnostics as diag

    train_data_path = tmp_path / "train"
    train_data_path.mkdir()
    pd.DataFrame({"mark_price": [10.0]}).to_feather(train_data_path / "df_0.feather")
    pd.DataFrame({"mark_price": [20.0]}).to_feather(train_data_path / "df_1.feather")

    output_dir = tmp_path / "diagnostics"
    output_dir.mkdir()
    pd.DataFrame(
        {
            "action": [1, 2],
            "step_reward": [10.0, 20.0],
            "cumulative_profit": [10.0, 30.0],
        }
    ).to_csv(output_dir / "df_0_initial_action_0.csv", index=False)
    pd.DataFrame(
        {
            "action": [3],
            "step_reward": [-5.0],
            "cumulative_profit": [-5.0],
        }
    ).to_csv(output_dir / "df_1_initial_action_0.csv", index=False)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "diagnostic_count": 2,
                "total_df_index_length": 2,
                "position_choices": 1,
                "qtable_kwargs": {},
                "env_kwargs": {},
            }
        )
    )

    monkeypatch.setattr(
        diag,
        "build_sample_plan",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("sample plan should be loaded from CSV")
        ),
    )
    monkeypatch.setattr(
        diag,
        "create_optimal_q_table_from_df",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("qtable should not be recomputed")
        ),
    )

    sample_plan, q_table_cache, train_df_cache, diagnostics, sample_action_cache = (
        diag.prepare_pretrain_qtable_diagnostics(
            num_sample=2,
            total_df_index_length=2,
            position_choices=1,
            train_data_path=str(train_data_path),
            qtable_kwargs={},
            env_kwargs={},
            output_dir=str(output_dir),
            process_count=1,
        )
    )

    assert sample_plan == [(0, 0), (1, 0)]
    assert q_table_cache == {}
    assert sorted(train_df_cache) == [0, 1]
    assert sample_action_cache == {(0, 0): [1, 2], (1, 0): [3]}
    assert [item["episode_reward_sum"] for item in diagnostics] == [30.0, -5.0]
    assert [item["profitable"] for item in diagnostics] == [True, False]


def test_diagnostics_ignore_existing_csvs_when_manifest_does_not_match(
    monkeypatch, tmp_path
):
    from RL.DiHFT.low_level import pretrain_qtable_diagnostics as diag

    train_data_path = tmp_path / "train"
    train_data_path.mkdir()
    pd.DataFrame({"mark_price": [10.0]}).to_feather(train_data_path / "df_0.feather")

    output_dir = tmp_path / "diagnostics"
    output_dir.mkdir()
    pd.DataFrame(
        {
            "action": [1],
            "step_reward": [10.0],
            "cumulative_profit": [10.0],
        }
    ).to_csv(output_dir / "df_0_initial_action_0.csv", index=False)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "diagnostic_count": 1,
                "total_df_index_length": 1,
                "position_choices": 3,
                "qtable_kwargs": {"commission_rate": 0.0002},
                "env_kwargs": {"commission_rate": 0.0002},
            }
        )
    )

    recomputed = {"called": False}

    def fake_build_q_table_cache(
        sample_plan,
        train_data_path,
        qtable_kwargs,
        env_kwargs=None,
        output_dir=None,
        process_count=None,
    ):
        recomputed["called"] = True
        return (
            {0: np.zeros((1, 3, 3))},
            {0: pd.DataFrame({"mark_price": [10.0]})},
            [
                {
                    "df_index": 0,
                    "initial_action": 0,
                    "episode_reward_sum": 20.0,
                    "profitable": True,
                    "csv_path": str(
                        Path(output_dir) / "df_0_initial_action_0.csv"
                    ),
                    "action_list": [2],
                }
            ],
    )

    monkeypatch.setattr(
        diag, "build_sample_plan", lambda total, choices: [(0, 0)]
    )
    monkeypatch.setattr(diag, "build_q_table_cache", fake_build_q_table_cache)
    monkeypatch.setattr(
        diag,
        "evaluate_and_export_sample",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("diagnostics should be produced by build_q_table_cache")
        ),
    )

    _, _, _, diagnostics, sample_action_cache = diag.prepare_pretrain_qtable_diagnostics(
        num_sample=1,
        total_df_index_length=1,
        position_choices=3,
        train_data_path=str(train_data_path),
        qtable_kwargs={"commission_rate": 0},
        env_kwargs={"commission_rate": 0},
        output_dir=str(output_dir),
        process_count=1,
    )

    assert recomputed["called"] is True
    assert diagnostics[0]["episode_reward_sum"] == 20.0
    assert sample_action_cache == {(0, 0): [2]}


def test_prepare_uses_qtable_cache_builder_diagnostics(monkeypatch, tmp_path):
    from RL.DiHFT.low_level import pretrain_qtable_diagnostics as diag

    output_dir = tmp_path / "diagnostics"
    env_kwargs = {"initial_wallet_balance": 100000}

    monkeypatch.setattr(
        diag,
        "build_sample_plan",
        lambda total, choices: [(2, 1), (3, 0)],
    )

    def fake_build_q_table_cache(
        sample_plan,
        train_data_path,
        qtable_kwargs,
        env_kwargs=None,
        output_dir=None,
        process_count=None,
    ):
        assert sample_plan == [(2, 1), (3, 0)]
        assert env_kwargs == {"initial_wallet_balance": 100000}
        assert output_dir == str(output_dir_path)
        return (
            {2: "q-table"},
            {2: "train-df"},
            [
                {
                    "df_index": 3,
                    "initial_action": 0,
                    "episode_reward_sum": -1.0,
                    "profitable": False,
                    "csv_path": str(
                        output_dir_path / "df_3_initial_action_0.csv"
                    ),
                    "action_list": [7],
                },
                {
                    "df_index": 2,
                    "initial_action": 1,
                    "episode_reward_sum": 3.0,
                    "profitable": True,
                    "csv_path": str(
                        output_dir_path / "df_2_initial_action_1.csv"
                    ),
                    "action_list": [5, 6],
                },
            ],
        )

    output_dir_path = output_dir
    monkeypatch.setattr(diag, "build_q_table_cache", fake_build_q_table_cache)
    monkeypatch.setattr(
        diag,
        "evaluate_and_export_sample",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("diagnostics should be produced by build_q_table_cache")
        ),
    )

    sample_plan, q_table_cache, train_df_cache, diagnostics, sample_action_cache = (
        diag.prepare_pretrain_qtable_diagnostics(
            num_sample=2,
            total_df_index_length=4,
            position_choices=3,
            train_data_path=str(tmp_path / "train"),
            qtable_kwargs={},
            env_kwargs=env_kwargs,
            output_dir=str(output_dir),
            process_count=1,
        )
    )

    assert sample_plan == [(2, 1), (3, 0)]
    assert q_table_cache == {2: "q-table"}
    assert train_df_cache == {2: "train-df"}
    assert [(item["df_index"], item["initial_action"]) for item in diagnostics] == [
        (2, 1),
        (3, 0),
    ]
    assert sample_action_cache == {(2, 1): [5, 6], (3, 0): [7]}


def test_create_demo_env_passes_order_book_depth(monkeypatch):
    from RL.DiHFT.low_level import pretrain_qtable_diagnostics as diag

    captured_kwargs = {}

    def fake_initiate_demo_env(**kwargs):
        captured_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr(diag, "initiate_demo_env", fake_initiate_demo_env)

    env_kwargs = {
        "feature_list": [],
        "max_holding_number": 1,
        "order_book_depth": 5,
        "position_choices": 3,
        "leverage_choices": [1],
        "long_estimated_rate": 0,
        "short_estimated_rate": 0,
        "commission_rate": 0,
        "maintenance_margin_ratio_dict": {},
        "early_stop": 0,
        "gamma": 1,
    }

    diag.create_demo_env(pd.DataFrame(), env_kwargs, (100000, 0, 0, 0, 1))

    assert captured_kwargs["order_book_depth"] == 5


def test_extend_q_table_cache_only_computes_missing_df_indices(monkeypatch, tmp_path):
    from RL.DiHFT.low_level import pretrain_qtable_diagnostics as diag

    train_data_path = tmp_path / "train"
    train_data_path.mkdir()
    pd.DataFrame({"mark_price": [10.0]}).to_feather(train_data_path / "df_0.feather")
    pd.DataFrame({"mark_price": [20.0]}).to_feather(train_data_path / "df_1.feather")
    pd.DataFrame({"mark_price": [30.0]}).to_feather(train_data_path / "df_2.feather")

    existing_q_table = np.ones((1, 3, 3))
    existing_df = pd.DataFrame({"mark_price": [10.0]})
    computed = []

    def fake_create_q_table_from_df(df, **kwargs):
        computed.append(float(df["mark_price"].iloc[0]))
        return np.zeros((len(df), 3, 3)) + float(df["mark_price"].iloc[0])

    monkeypatch.setattr(
        diag, "create_optimal_q_table_from_df", fake_create_q_table_from_df
    )

    q_table_cache, train_df_cache = diag.extend_q_table_cache(
        df_indices=[0, 1, 2],
        train_data_path=str(train_data_path),
        qtable_kwargs={},
        q_table_cache={0: existing_q_table},
        train_df_cache={0: existing_df},
        process_count=1,
    )

    assert q_table_cache[0] is existing_q_table
    assert train_df_cache[0] is existing_df
    assert sorted(q_table_cache) == [0, 1, 2]
    assert sorted(train_df_cache) == [0, 1, 2]
    assert computed == [20.0, 30.0]
