import logging
import sys
from pathlib import Path


FINEFT_ROOT = Path(__file__).resolve().parents[2]
if str(FINEFT_ROOT) not in sys.path:
    sys.path.insert(0, str(FINEFT_ROOT))

from RL.DiHFT.low_level.weight_advantage_pretrain import (
    Weighted_Contexts_DQN,
    summarize_rollout_diagnostics,
    logger,
    summarize_rollout_metrics,
)


class DummyLargeObject:
    pass


def test_log_internal_parameters_logs_values_and_summarizes_large_objects(caplog):
    trainer = Weighted_Contexts_DQN.__new__(Weighted_Contexts_DQN)
    trainer.dataset_name = "BTCUSDT"
    trainer.batch_size = 64
    trainer.position_list = [-8.0, 0, 8.0]
    trainer.eval_net = DummyLargeObject()

    with caplog.at_level(logging.INFO, logger=logger.name):
        trainer._log_internal_parameters("train_start")

    assert "Weighted_Contexts_DQN internal parameters | stage=train_start" in caplog.text
    assert "dataset_name=BTCUSDT" in caplog.text
    assert "batch_size=64" in caplog.text
    assert "position_list=[-8.0, 0, 8.0]" in caplog.text
    assert "eval_net=<DummyLargeObject>" in caplog.text


def test_summarize_rollout_metrics_uses_all_rollouts():
    metrics = [
        {"reward_sum": 100.0, "return_rate": 1.1, "final_balance": 110000.0},
        {"reward_sum": -20.0, "return_rate": 0.99, "final_balance": 99000.0},
    ]

    summary = summarize_rollout_metrics(metrics)

    assert summary["mean_reward_sum"] == 40.0
    assert summary["mean_return_rate"] == 1.045
    assert summary["mean_final_balance"] == 104500.0


def test_summarize_rollout_diagnostics_counts_actions_and_positions():
    summary = summarize_rollout_diagnostics(
        actions=[3, 1, 3, 2],
        positions=[0.0, -1.0, -1.0, 1.0],
        preview_limit=3,
    )

    assert summary["action_counts"] == [(1, 1), (2, 1), (3, 2)]
    assert summary["position_counts"] == [(-1.0, 2), (0.0, 1), (1.0, 1)]
    assert summary["first_actions"] == [3, 1, 3]
    assert summary["first_positions"] == [0.0, -1.0, -1.0]
    assert summary["position_switches"] == 2


def test_build_loss_nan_diagnostics_identifies_nonfinite_training_data():
    import numpy as np
    import torch
    from RL.DiHFT.low_level import loss_nan_diagnostics

    diagnostics = loss_nan_diagnostics.build_loss_nan_diagnostics(
        numeric_values={
            "loss": torch.tensor(float("nan")),
            "td_loss": torch.tensor(3.0),
            "states": torch.tensor([[1.0, float("inf")]]),
        },
        info_values={
            "info": {
                "q_value": torch.tensor([[1.0, float("nan")]]),
                "funding_count_down_hour": np.array([1.0, float("-inf")]),
                "safe_value": [1.0, 2.0],
            }
        },
    )

    assert isinstance(diagnostics, loss_nan_diagnostics.LossNanDiagnostics)
    assert diagnostics.numeric["loss"].nan_count == 1
    assert diagnostics.numeric["td_loss"].finite_count == 1
    assert diagnostics.numeric["states"].inf_count == 1
    assert diagnostics.info_nonfinite == [
        loss_nan_diagnostics.NonfiniteLocation(
            path="info.q_value",
            shape=[1, 2],
            dtype="torch.float32",
            nan_count=1,
            inf_count=0,
            first_nonfinite_indices=[[0, 1]],
        ),
        loss_nan_diagnostics.NonfiniteLocation(
            path="info.funding_count_down_hour",
            shape=[2],
            dtype="float64",
            nan_count=0,
            inf_count=1,
            first_nonfinite_indices=[[1]],
        ),
    ]
    assert diagnostics.to_dict()["numeric"]["loss"]["nan_count"] == 1
    assert diagnostics.to_dict()["info_nonfinite"][1]["inf_count"] == 1


def test_record_diverse_rollout_latest_metric_overwrites_existing_key():
    from RL.DiHFT.low_level import weight_advantage_pretrain as wap

    metrics_by_df = {}

    wap.record_diverse_rollout_latest_metric(
        metrics_by_df,
        df_index=2,
        rollout_index=1,
        reward_sum=100.0,
        final_balance=101000.0,
        return_rate=0.01,
    )
    wap.record_diverse_rollout_latest_metric(
        metrics_by_df,
        df_index=2,
        rollout_index=1,
        reward_sum=-50.0,
        final_balance=99500.0,
        return_rate=-0.005,
    )

    assert metrics_by_df == {
        2: {
            1: {
                "reward_sum": -50.0,
                "final_balance": 99500.0,
                "return_rate": -0.005,
            }
        }
    }


def test_log_diverse_rollout_latest_metrics_sorts_and_labels_profit_loss(caplog):
    from RL.DiHFT.low_level import weight_advantage_pretrain as wap

    metrics_by_df = {
        2: {
            1: {
                "reward_sum": 20.0,
                "final_balance": 100100.0,
                "return_rate": 0.001,
            }
        },
        1: {
            3: {
                "reward_sum": 0.0,
                "final_balance": 100000.0,
                "return_rate": 0.0,
            },
            0: {
                "reward_sum": -30.0,
                "final_balance": 99900.0,
                "return_rate": -0.001,
            },
        },
    }

    with caplog.at_level(logging.INFO, logger=logger.name):
        wap.log_diverse_rollout_latest_metrics(7, metrics_by_df)

    messages = [
        record.message
        for record in caplog.records
        if "多样化训练最新明细" in record.message
    ]
    assert messages == [
        "第 7 轮 epoch 训练完成 | 多样化训练最新明细 | df_index=1 | rollout_index=0 | 累计奖励=-30.0000 | 最终余额=99900.0000 | 收益率=-0.001000 | 亏损",
        "第 7 轮 epoch 训练完成 | 多样化训练最新明细 | df_index=1 | rollout_index=3 | 累计奖励=0.0000 | 最终余额=100000.0000 | 收益率=0.000000 | 亏损",
        "第 7 轮 epoch 训练完成 | 多样化训练最新明细 | df_index=2 | rollout_index=1 | 累计奖励=20.0000 | 最终余额=100100.0000 | 收益率=0.001000 | 盈利",
    ]


def test_log_diverse_rollout_latest_metrics_skips_empty_cache(caplog):
    from RL.DiHFT.low_level import weight_advantage_pretrain as wap

    with caplog.at_level(logging.INFO, logger=logger.name):
        wap.log_diverse_rollout_latest_metrics(3, {})

    assert "多样化训练最新明细" not in caplog.text


def test_train_wires_diverse_rollout_latest_metrics_before_epoch_summary():
    import inspect

    source = inspect.getsource(Weighted_Contexts_DQN.train)

    assert "diverse_rollout_latest_metrics_by_df = {}" in source
    assert "record_diverse_rollout_latest_metric(" in source
    assert "log_diverse_rollout_latest_metrics(" in source
    diverse_branch_index = source.index("else:\n                for index in range(self.N):")
    diverse_return_rate_index = source.index("diverse_return_rate =")
    record_metric_index = source.index("record_diverse_rollout_latest_metric(")
    assert diverse_branch_index < diverse_return_rate_index < record_metric_index
    assert source.index("log_diverse_rollout_latest_metrics(") < source.index(
        '"第 %d 轮 epoch 训练完成 | 平均收益率=%.6f'
    )


def test_parser_defaults_enable_full_df_warmup_and_zero_sample_pretrain():
    from RL.DiHFT.low_level import weight_advantage_pretrain as wap

    args = wap.parser.parse_args([])

    assert args.full_df_warmup is True
    assert args.pretrain_epoch == 0


def test_parser_can_disable_full_df_warmup():
    from RL.DiHFT.low_level import weight_advantage_pretrain as wap

    args = wap.parser.parse_args(["--no_full_df_warmup"])

    assert args.full_df_warmup is False


def test_resolve_empty_initial_action_uses_action_mapping():
    trainer = Weighted_Contexts_DQN.__new__(Weighted_Contexts_DQN)
    trainer.position_choices = 5
    trainer.leverage_choices = [1, 2]
    trainer.position_list = [-8.0, -4.0, 0, 4.0, 8.0]

    assert trainer._resolve_empty_initial_action() == 4


def test_resolve_empty_initial_action_raises_when_no_empty_position():
    trainer = Weighted_Contexts_DQN.__new__(Weighted_Contexts_DQN)
    trainer.position_choices = 2
    trainer.leverage_choices = [1]
    trainer.position_list = [-8.0, 8.0]

    try:
        trainer._resolve_empty_initial_action()
    except ValueError as exc:
        assert "empty position action" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_run_full_df_warmup_updates_once_per_df(monkeypatch):
    import pandas as pd
    from RL.DiHFT.low_level import weight_advantage_pretrain as wap

    class TinyBuffer:
        def __init__(self):
            self.added = []

        def add(self, s, info, a, r, s_, info_, done):
            self.added.append((a, r, done))

        def sample(self):
            raise AssertionError("sample should not be called with large batch threshold")

    class TinyEnv:
        def __init__(self):
            self.wallet_balance = 100000.0
            self.unrealized_pnl = 0.0
            self._step = 0

        def reset(self):
            self._step = 0
            return "s0", {
                "avaiable_action_list": [0, 1, 2],
                "previous_action": 0,
                "personal_state": [0, 0, 0, 0, 0.0],
            }

        def step(self, action):
            self._step += 1
            done = self._step == 1
            return "s1", 1.0, done, {
                "avaiable_action_list": [0, 1, 2],
                "previous_action": action,
                "personal_state": [0, 0, 0, 0, 0.0],
            }

    trainer = Weighted_Contexts_DQN.__new__(Weighted_Contexts_DQN)
    trainer.full_df_warmup = True
    trainer.total_df_index_length = 3
    trainer.position_choices = 3
    trainer.leverage_choices = [1]
    trainer.position_list = [-1.0, 0, 1.0]
    trainer.N_ACTIONS = 3
    trainer.initial_wallet_balance = 100000.0
    trainer.initial_unrealized_pnL = 0.0
    trainer.batch_size = 999
    trainer.update_times = 1
    trainer.n_step = 1
    trainer.rollout_steps = 1024
    trainer.tech_indicator_list = ["mark_price"]
    trainer.writer = type("Writer", (), {"add_scalar": lambda *args, **kwargs: None})()
    trainer.update_counter = 0
    trainer._set_initial_state_from_action = lambda train_df, action: setattr(
        trainer, "initial_state", ("state", int(action))
    )
    trainer.act_multi_styles_pretrain = (
        lambda info, optimal_step_counter, rollout_index: rollout_index
    )
    trainer.update_pretrain = lambda *args, **kwargs: (0.0, 0.0, 0.0)

    created_envs = []

    def fake_create_demo_env(train_df, env_kwargs, initial_state):
        created_envs.append((float(train_df["mark_price"].iloc[0]), initial_state))
        return TinyEnv()

    monkeypatch.setattr(wap, "create_demo_env", fake_create_demo_env)
    monkeypatch.setattr(
        wap, "get_dp_action_from_qtable", lambda q_table, initial_action: [initial_action]
    )

    train_df_cache = {
        0: pd.DataFrame({"mark_price": [10.0]}),
        1: pd.DataFrame({"mark_price": [20.0]}),
        2: pd.DataFrame({"mark_price": [30.0]}),
    }
    q_table_cache = {0: "q0", 1: "q1", 2: "q2"}
    buffer_pretrain = TinyBuffer()

    summary, step_counter = trainer._run_full_df_warmup(
        q_table_cache=q_table_cache,
        train_df_cache=train_df_cache,
        env_kwargs={},
        buffer_pretrain=buffer_pretrain,
        step_counter_pretrain=0,
    )

    assert [item[0] for item in created_envs] == [10.0, 20.0, 30.0]
    assert [item[1][1] for item in created_envs] == [1, 1, 1]
    assert step_counter == 12
    assert len(buffer_pretrain.added) == 12
    assert summary["df_count"] == 3


def test_run_full_df_warmup_logs_first_row_tech_indicators(monkeypatch, caplog):
    import pandas as pd
    from RL.DiHFT.low_level import weight_advantage_pretrain as wap

    class TinyBuffer:
        def add(self, s, info, a, r, s_, info_, done):
            pass

        def sample(self):
            raise AssertionError("sample should not be called with large batch threshold")

    class TinyEnv:
        wallet_balance = 100000.0
        unrealized_pnl = 0.0

        def reset(self):
            return "s0", {
                "avaiable_action_list": [0, 1, 2],
                "previous_action": 0,
                "personal_state": [0, 0, 0, 0, 0.0],
            }

        def step(self, action):
            return "s1", 1.0, True, {
                "avaiable_action_list": [0, 1, 2],
                "previous_action": action,
                "personal_state": [0, 0, 0, 0, 0.0],
            }

    trainer = Weighted_Contexts_DQN.__new__(Weighted_Contexts_DQN)
    trainer.full_df_warmup = True
    trainer.total_df_index_length = 2
    trainer.position_choices = 3
    trainer.leverage_choices = [1]
    trainer.position_list = [-1.0, 0, 1.0]
    trainer.N_ACTIONS = 3
    trainer.initial_wallet_balance = 100000.0
    trainer.initial_unrealized_pnL = 0.0
    trainer.batch_size = 999
    trainer.update_times = 1
    trainer.n_step = 1
    trainer.rollout_steps = 1024
    trainer.writer = type("Writer", (), {"add_scalar": lambda *args, **kwargs: None})()
    trainer.update_counter = 0
    trainer.tech_indicator_list = ["rsi", "macd"]
    trainer._set_initial_state_from_action = lambda train_df, action: setattr(
        trainer, "initial_state", ("state", int(action))
    )
    trainer.act_multi_styles_pretrain = (
        lambda info, optimal_step_counter, rollout_index: rollout_index
    )
    trainer.update_pretrain = lambda *args, **kwargs: (0.0, 0.0, 0.0)

    monkeypatch.setattr(
        wap, "create_demo_env", lambda train_df, env_kwargs, initial_state: TinyEnv()
    )
    monkeypatch.setattr(
        wap, "get_dp_action_from_qtable", lambda q_table, initial_action: [initial_action]
    )

    with caplog.at_level(logging.INFO, logger=logger.name):
        trainer._run_full_df_warmup(
            q_table_cache={0: "q0", 1: "q1"},
            train_df_cache={
                0: pd.DataFrame({"rsi": [12.5], "macd": [-0.1]}),
                1: pd.DataFrame({"rsi": [20.0], "macd": [0.3]}),
            },
            env_kwargs={},
            buffer_pretrain=TinyBuffer(),
            step_counter_pretrain=0,
        )

    messages = [
        record.message
        for record in caplog.records
        if record.message.startswith("full-df warmup first row")
    ]
    assert messages == [
        "full-df warmup first row | df_index=0 | rsi=12.5, macd=-0.1",
        "full-df warmup first row | df_index=1 | rsi=20.0, macd=0.3",
    ]


def test_full_df_warmup_logs_rollout_balances_without_df_final_balance(
    monkeypatch, caplog
):
    import pandas as pd
    from RL.DiHFT.low_level import weight_advantage_pretrain as wap

    class TinyBuffer:
        def add(self, s, info, a, r, s_, info_, done):
            pass

        def sample(self):
            raise AssertionError("sample should not be called with large batch threshold")

    class ResettingEnv:
        def __init__(self):
            self.wallet_balance = 100000.0
            self.unrealized_pnl = 0.0
            self._rollout_index = -1

        def reset(self):
            self._rollout_index += 1
            self.wallet_balance = 100000.0
            self.unrealized_pnl = 0.0
            return "s0", {
                "avaiable_action_list": [0, 1, 2],
                "previous_action": 0,
                "personal_state": [0, 0, 0, 0, 0.0],
            }

        def step(self, action):
            if self._rollout_index == 0:
                self.wallet_balance = 100500.0
                reward = 500.0
            elif self._rollout_index == 3:
                self.wallet_balance = 100000.0
                reward = 0.0
            else:
                self.wallet_balance = 100100.0
                reward = 100.0
            return "s1", reward, True, {
                "avaiable_action_list": [0, 1, 2],
                "previous_action": action,
                "personal_state": [0, 0, 0, 0, 0.0],
            }

    trainer = Weighted_Contexts_DQN.__new__(Weighted_Contexts_DQN)
    trainer.full_df_warmup = True
    trainer.total_df_index_length = 1
    trainer.position_choices = 3
    trainer.leverage_choices = [1]
    trainer.position_list = [-1.0, 0, 1.0]
    trainer.N_ACTIONS = 3
    trainer.initial_wallet_balance = 100000.0
    trainer.initial_unrealized_pnL = 0.0
    trainer.batch_size = 999
    trainer.update_times = 1
    trainer.n_step = 1
    trainer.rollout_steps = 1024
    trainer.tech_indicator_list = ["mark_price"]
    trainer.writer = type("Writer", (), {"add_scalar": lambda *args, **kwargs: None})()
    trainer.update_counter = 0
    trainer._set_initial_state_from_action = lambda train_df, action: setattr(
        trainer, "initial_state", ("state", int(action))
    )
    trainer.act_multi_styles_pretrain = (
        lambda info, optimal_step_counter, rollout_index: rollout_index
    )
    trainer.update_pretrain = lambda *args, **kwargs: (0.0, 0.0, 0.0)

    monkeypatch.setattr(
        wap, "create_demo_env", lambda train_df, env_kwargs, initial_state: ResettingEnv()
    )
    monkeypatch.setattr(
        wap, "get_dp_action_from_qtable", lambda q_table, initial_action: [initial_action]
    )

    with caplog.at_level(logging.INFO, logger=logger.name):
        summary, _ = trainer._run_full_df_warmup(
            q_table_cache={0: "q0"},
            train_df_cache={0: pd.DataFrame({"mark_price": [10.0]})},
            env_kwargs={},
            buffer_pretrain=TinyBuffer(),
            step_counter_pretrain=0,
        )

    assert summary["reward_sum"] == 700.0
    assert "full-df warmup rollout complete | df_index=0 | rollout_index=0" in caplog.text
    assert "final_balance=100500.0000" in caplog.text
    assert "full-df warmup df complete | df_index=0 | reward_sum=700.0000" in caplog.text
    df_summary = [
        record.message
        for record in caplog.records
        if record.message.startswith("full-df warmup df complete")
    ][0]
    assert "final_balance" not in df_summary


def test_parser_defaults_experiment_name_to_default():
    from RL.DiHFT.low_level import weight_advantage_pretrain as wap

    args = wap.parser.parse_args([])

    assert args.experiment_name == "default"


def test_parser_accepts_explicit_experiment_name():
    from RL.DiHFT.low_level import weight_advantage_pretrain as wap

    args = wap.parser.parse_args(["--experiment_name", "5min_gamma097"])

    assert args.experiment_name == "5min_gamma097"


def test_build_serial_model_path_includes_experiment_name():
    from RL.DiHFT.low_level import weight_advantage_pretrain as wap

    assert wap.build_serial_model_path(
        result_path="result/DiHFT/low_level",
        dataset_name="fu",
        experiment_name="5min_gamma097",
    ) == "result/DiHFT/low_level/fu/5min_gamma097/weights_advantage_pretrain"


def test_build_train_data_paths_keep_base_path_dataset_name_semantics():
    from RL.DiHFT.low_level import weight_advantage_pretrain as wap

    paths = wap.build_training_data_paths(base_path="dataset_5min", dataset_name="fu")

    assert paths == {
        "train_data_path": "dataset_5min/fu/train",
        "state_features_path": "dataset_5min/fu/state_features.npy",
        "maintenance_margin_ratio_path": "dataset_5min/fu/maintenance_margin_ratio_dict.npy",
    }


def test_build_train_data_paths_prefers_train_slice_when_present(tmp_path):
    from RL.DiHFT.low_level import weight_advantage_pretrain as wap

    dataset_root = tmp_path / "dataset_5min" / "fu"
    (dataset_root / "train" / "slice").mkdir(parents=True)

    paths = wap.build_training_data_paths(
        base_path=str(tmp_path / "dataset_5min"),
        dataset_name="fu",
    )

    assert paths["train_data_path"] == str(dataset_root / "train" / "slice")


def test_count_training_data_files_includes_last_df_file(tmp_path):
    from RL.DiHFT.low_level import weight_advantage_pretrain as wap

    train_data_path = tmp_path / "train"
    train_data_path.mkdir()
    for df_index in range(3):
        (train_data_path / f"df_{df_index}.feather").touch()

    assert wap.count_training_data_files(str(train_data_path)) == 3


def test_build_file_log_path_includes_experiment_name():
    from RL.DiHFT.low_level import weight_advantage_pretrain as wap

    assert wap.build_train_log_path(
        dataset_name="fu",
        experiment_name="5min_gamma097",
    ) == "log_futures/fu/low_level/train/5min_gamma097/advantage.log"


def test_parser_allow_reverse_position_default_and_flag():
    from RL.DiHFT.low_level import weight_advantage_pretrain as wap

    args_default = wap.parser.parse_args([])
    assert args_default.allow_reverse_position is False

    args_flag = wap.parser.parse_args(["--allow_reverse_position"])
    assert args_flag.allow_reverse_position is True
