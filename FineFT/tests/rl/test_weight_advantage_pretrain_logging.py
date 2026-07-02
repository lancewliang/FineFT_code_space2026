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
