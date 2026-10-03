import pytest
from unittest.mock import MagicMock
import torch
from torch import nn
from RL.DiHFT.low_level.persistent_pool import PersistentRolloutPool
from RL.DiHFT.low_level.shared_model_manager import SharedInferenceManager


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


def test_persistent_pool_lifecycle_and_idempotent_shutdown(monkeypatch):
    import RL.DiHFT.low_level.persistent_pool as pp_mod

    mock_start = MagicMock()
    mock_shutdown = MagicMock()
    monkeypatch.setattr("RL.DiHFT.low_level.parallel_diverse_train.start_parallel_workers", mock_start)
    monkeypatch.setattr("RL.DiHFT.low_level.parallel_diverse_train.shutdown_exploration_workers", mock_shutdown)

    trainer = MagicMock()
    trainer.diverse_num_workers = 4
    shared_manager = MagicMock()
    shared_manager.get_shared_model.return_value = ToyModel()

    with PersistentRolloutPool(
        trainer=trainer,
        train_df_cache={},
        env_kwargs={},
        shared_manager=shared_manager,
    ) as pool:
        assert not pool.is_shutdown
        assert mock_start.called
        assert not mock_shutdown.called

        # Test sync_model_weights
        dummy_net = ToyModel()
        pool.sync_model_weights(dummy_net)
        shared_manager.sync_weights_from_gpu.assert_called_with(dummy_net)

        # Early manual shutdown
        pool.shutdown()
        assert pool.is_shutdown
        assert mock_shutdown.call_count == 1

    # On context exit, shutdown should be idempotent
    assert mock_shutdown.call_count == 1


def test_persistent_pool_reuses_workers_across_epochs(monkeypatch):
    from RL.DiHFT.low_level import parallel_diverse_train as pdt

    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid

        def is_alive(self):
            return True

        def terminate(self):
            pass

        def join(self, timeout=None):
            pass

    fake_procs = [FakeProcess(pid=1001), FakeProcess(pid=1002)]

    def fake_start(trainer, train_df_cache, env_kwargs, shared_model=None, shared_market_data=None):
        trainer.worker_processes = fake_procs
        trainer.worker_task_queue = MagicMock()
        trainer.worker_result_queue = MagicMock()
        trainer.worker_input_queues = {0: trainer.worker_task_queue}

    monkeypatch.setattr(pdt, "start_parallel_workers", fake_start)
    monkeypatch.setattr(pdt, "shutdown_exploration_workers", lambda tr: tr.worker_processes.clear())

    trainer = MagicMock()
    trainer.diverse_num_workers = 2
    trainer.worker_processes = []
    shared_manager = MagicMock()
    shared_manager.get_shared_model.return_value = ToyModel()

    with PersistentRolloutPool(
        trainer=trainer,
        train_df_cache={},
        env_kwargs={},
        shared_manager=shared_manager,
    ) as pool:
        pids_epoch1 = [p.pid for p in trainer.worker_processes]
        assert pids_epoch1 == [1001, 1002]

        # Simulate second epoch: processes remain alive and untouched
        pids_epoch2 = [p.pid for p in trainer.worker_processes]
        assert pids_epoch2 == pids_epoch1
