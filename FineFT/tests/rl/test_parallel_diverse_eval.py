import os
from pathlib import Path
import types
from unittest.mock import MagicMock
import numpy as np
import pytest
import torch

from RL.DiHFT.low_level import parallel_diverse_train as pdt
from RL.DiHFT.low_level import parallel_weight_advantage_pretrain as pwap


def test_pwap_module_exports_run_parallel_diverse_training():
    """Verify run_parallel_diverse_training is accessible from parallel_weight_advantage_pretrain."""
    assert hasattr(pwap, "run_parallel_diverse_training")
    assert pwap.run_parallel_diverse_training is pdt.run_parallel_diverse_training
    assert hasattr(pwap, "evaluate_parallel_diverse_model")
    assert pwap.evaluate_parallel_diverse_model is pdt.evaluate_parallel_diverse_model


def test_run_parallel_diverse_training_selects_min_total_loss_model(tmp_path: Path, monkeypatch):
    """3 epochs with losses [2.5, 0.4, 1.8]: eval must be called with epoch 2 (1-based, index 1) model."""
    events = []

    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True)

    # Mock trainer
    trainer = MagicMock()
    trainer.total_df_index_length = 1
    trainer.num_epoch = 3
    trainer.model_path = str(model_dir)
    trainer.update_times = 1
    trainer.writer = MagicMock()

    # Create dummy model files for each epoch
    for epoch_idx in range(3):
        epoch_dir = model_dir / f"epoch_{epoch_idx + 1}"
        epoch_dir.mkdir(parents=True)
        dummy_model_file = epoch_dir / "trained_model.pkl"
        torch.save({"epoch": epoch_idx + 1}, str(dummy_model_file))

    # Loss values for the 3 epochs: epoch 1 has the minimum loss (0.4)
    epoch_losses = [
        (2.5, 0.5, 2.0),
        (0.4, 0.1, 0.3),
        (1.8, 0.3, 1.5),
    ]
    loss_iter = iter(epoch_losses)

    # Monkeypatch exploration and training internals
    monkeypatch.setattr(pdt, "apply_epoch_training_params", lambda *args: None)
    monkeypatch.setattr(pdt, "is_buffer_full", lambda *args: False)
    monkeypatch.setattr(
        pdt,
        "run_epoch_exploration",
        lambda *args, **kwargs: ([], 10, 1),
    )
    monkeypatch.setattr(pdt, "save_diverse_buffer", lambda *args: None)
    monkeypatch.setattr(pdt, "write_epoch_rollout_scalars", lambda *args: None)
    monkeypatch.setattr(pdt, "save_parallel_epoch_model", lambda *args: None)
    monkeypatch.setattr(pdt, "log_diverse_rollout_latest_metrics", lambda *args: None)

    def fake_training_phase(tr, source, update_times, epoch_index):
        losses = next(loss_iter)
        events.append(("training_phase", epoch_index, losses[0]))
        return losses

    monkeypatch.setattr(pdt, "run_diverse_training_phase", fake_training_phase)

    eval_called_with = []

    def fake_evaluate(tr, model_path):
        events.append(("evaluate", model_path))
        eval_called_with.append(model_path)
        return [{"metric": 1.0}]

    monkeypatch.setattr(pdt, "evaluate_parallel_diverse_model", fake_evaluate)

    buffer_diverse = MagicMock()
    step_counter = pdt.run_parallel_diverse_training(
        trainer=trainer,
        train_df_cache={},
        env_kwargs={},
        buffer_diverse=buffer_diverse,
        step_counter_diverse=0,
        diverse_rollout_latest_metrics_by_df={},
    )

    # Training for all 3 epochs occurred
    assert len([ev for ev in events if ev[0] == "training_phase"]) == 3

    # Evaluation occurred exactly once
    assert len(eval_called_with) == 1

    # Selected model must be epoch 2 (index 1), where total_loss was 0.4 (minimum)
    expected_model = str(model_dir / "epoch_2" / "trained_model.pkl")
    assert eval_called_with[0] == expected_model
    assert trainer.best_model_path == expected_model
    assert trainer.best_loss == pytest.approx(0.4)
    assert trainer.best_epoch_index == 1
    assert trainer.diverse_eval_metrics == [{"metric": 1.0}]

    # Evaluation happened strictly AFTER all 3 training epochs
    eval_event_idx = events.index(("evaluate", expected_model))
    training_event_indices = [i for i, ev in enumerate(events) if ev[0] == "training_phase"]
    assert eval_event_idx > max(training_event_indices)


def test_run_parallel_diverse_training_waits_for_eval_completion(tmp_path: Path, monkeypatch):
    """Verify that training returns only after evaluation has completely finished."""
    events = []

    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True)
    epoch_dir = model_dir / "epoch_1"
    epoch_dir.mkdir(parents=True)
    dummy_model_file = epoch_dir / "trained_model.pkl"
    torch.save({"w": 1}, str(dummy_model_file))

    trainer = MagicMock()
    trainer.total_df_index_length = 1
    trainer.num_epoch = 1
    trainer.model_path = str(model_dir)
    trainer.update_times = 1
    trainer.writer = MagicMock()

    monkeypatch.setattr(pdt, "apply_epoch_training_params", lambda *args: None)
    monkeypatch.setattr(pdt, "is_buffer_full", lambda *args: False)
    monkeypatch.setattr(pdt, "run_epoch_exploration", lambda *args, **kwargs: ([], 5, 1))
    monkeypatch.setattr(pdt, "save_diverse_buffer", lambda *args: None)
    monkeypatch.setattr(pdt, "write_epoch_rollout_scalars", lambda *args: None)
    monkeypatch.setattr(pdt, "save_parallel_epoch_model", lambda *args: None)
    monkeypatch.setattr(pdt, "log_diverse_rollout_latest_metrics", lambda *args: None)
    monkeypatch.setattr(pdt, "run_diverse_training_phase", lambda *args: (0.5, 0.1, 0.4))

    def fake_evaluate(tr, model_path):
        events.append("eval_started")
        # simulate evaluation work
        events.append("eval_finished")
        return [{"status": "done"}]

    monkeypatch.setattr(pdt, "evaluate_parallel_diverse_model", fake_evaluate)

    pdt.run_parallel_diverse_training(
        trainer=trainer,
        train_df_cache={},
        env_kwargs={},
        buffer_diverse=MagicMock(),
        step_counter_diverse=0,
        diverse_rollout_latest_metrics_by_df={},
    )
    events.append("function_returned")

    assert events == ["eval_started", "eval_finished", "function_returned"]


def test_evaluate_parallel_diverse_model_calls_evaluates_with_expected_args(tmp_path: Path, monkeypatch):
    """Verify evaluate_parallel_diverse_model passes all configuration args to evaluate_sub_agents.evaluates."""
    model_file = str(tmp_path / "epoch_1" / "trained_model.pkl")
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    torch.save({"w": 1}, model_file)

    train_data_dir = str(tmp_path / "data")
    os.makedirs(train_data_dir, exist_ok=True)
    df_file = os.path.join(train_data_dir, "df_0.feather")
    with open(df_file, "w") as f:
        f.write("dummy")

    trainer = MagicMock()
    trainer.model_path = str(tmp_path)
    trainer.train_data_path = train_data_dir
    trainer.tech_indicator_list_path = "tech.npy"
    trainer.maintenance_margin_ratio_dict_path = "margin.npy"
    trainer.transcation_cost = 0.0003
    trainer.max_holding_number = 10
    trainer.position_choices = 5
    trainer.N = 3
    trainer.time_info_dim = 2
    trainer.hidden_nodes = 64
    trainer.leverage_choices = [1, 2]
    trainer.initial_leverage = 1
    trainer.initial_position = 0
    trainer.initial_wallet_balance = 20000
    trainer.order_book_depth = 10
    trainer.early_stop = 100
    trainer.enable_limit_reward = True
    trainer.limit_hold_bonus = 2.0
    trainer.limit_stay_bonus = 1.0
    trainer.limit_reverse_penalty = 2.0
    trainer.near_limit_threshold = 0.01
    trainer.allow_reverse_position = True

    called_kwargs = {}

    def mock_evaluates(**kwargs):
        called_kwargs.update(kwargs)
        return [{"result": "success"}]

    monkeypatch.setattr(pdt, "evaluates", mock_evaluates)

    metrics = pdt.evaluate_parallel_diverse_model(trainer, model_file)
    assert metrics == [{"result": "success"}]
    assert called_kwargs["model_path"] == model_file
    assert called_kwargs["data_file_paths"] == [df_file]
    assert called_kwargs["transcation_cost"] == 0.0003
    assert called_kwargs["max_holding_number"] == 10
    assert called_kwargs["position_choices"] == 5
    assert called_kwargs["N"] == 3
    assert called_kwargs["hidden_nodes"] == 64
    assert called_kwargs["leverage_choices"] == [1, 2]
    assert called_kwargs["allow_reverse_position"] is True
    assert called_kwargs["logg_file_path"] == str(tmp_path / "diverse_evaluation.log")


def test_evaluate_parallel_diverse_model_safe_on_missing_model_or_data(tmp_path: Path):
    """Verify evaluate_parallel_diverse_model safely returns [] without crashing if model or data is absent."""
    trainer = MagicMock()
    trainer.train_data_path = str(tmp_path / "nonexistent_data")
    trainer.model_path = str(tmp_path / "nonexistent_model")

    # Nonexistent model
    res1 = pdt.evaluate_parallel_diverse_model(trainer, str(tmp_path / "no_model.pkl"))
    assert res1 == []

    # None model
    res2 = pdt.evaluate_parallel_diverse_model(trainer, None)
    assert res2 == []

    # Model exists, but no data files
    model_file = str(tmp_path / "model.pkl")
    torch.save({"w": 1}, model_file)
    res3 = pdt.evaluate_parallel_diverse_model(trainer, model_file)
    assert res3 == []


def test_evaluate_parallel_diverse_model_fails_fast_on_missing_attribute(tmp_path: Path):
    """Verify that evaluate_parallel_diverse_model raises AttributeError if a required field is missing."""
    model_file = str(tmp_path / "epoch_1" / "trained_model.pkl")
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    torch.save({"w": 1}, model_file)

    train_data_dir = str(tmp_path / "data")
    os.makedirs(train_data_dir, exist_ok=True)
    df_file = os.path.join(train_data_dir, "df_0.feather")
    with open(df_file, "w") as f:
        f.write("dummy")

    class IncompleteTrainer:
        model_path = str(tmp_path)
        train_data_path = train_data_dir
        # deliberately missing tech_indicator_list_path, transcation_cost, etc.

    with pytest.raises(AttributeError):
        pdt.evaluate_parallel_diverse_model(IncompleteTrainer(), model_file)
