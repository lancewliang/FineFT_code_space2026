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
