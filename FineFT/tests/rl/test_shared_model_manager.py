import torch
from torch import nn
from RL.DiHFT.low_level.shared_model_manager import SharedInferenceManager
from RL.DiHFT.low_level.parallel_diverse_train import DfRolloutWorkerRunner


class SimpleToyModel(nn.Module):
    def __init__(self, in_features=4, out_features=2):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.fc(x)


def test_shared_inference_manager_shares_memory_and_eval():
    manager = SharedInferenceManager(
        model_factory=SimpleToyModel,
        model_kwargs={"in_features": 4, "out_features": 2},
    )
    shared_model = manager.get_shared_model()

    assert not shared_model.training
    for param in shared_model.parameters():
        assert param.is_shared()
    for buf in shared_model.buffers():
        assert buf.is_shared()


def test_shared_inference_manager_in_place_sync():
    manager = SharedInferenceManager(
        model_factory=SimpleToyModel,
        model_kwargs={"in_features": 4, "out_features": 2},
    )
    shared_model = manager.get_shared_model()

    # Create a source model with different weights
    source_model = SimpleToyModel(in_features=4, out_features=2)
    with torch.no_grad():
        source_model.fc.weight.fill_(7.5)
        source_model.fc.bias.fill_(3.2)

    # Perform in-place sync
    manager.sync_weights_from_gpu(source_model)

    # Verify weights are identical and memory is still shared
    assert torch.allclose(shared_model.fc.weight, source_model.fc.weight)
    assert torch.allclose(shared_model.fc.bias, source_model.fc.bias)
    for param in shared_model.parameters():
        assert param.is_shared()


def test_df_rollout_worker_runner_uses_shared_model():
    shared_model = SimpleToyModel()
    shared_model.share_memory()

    worker_config = {
        "df_indices": [0],
        "train_df_by_df": {0: None},
        "env_kwargs": {},
        "leverage_choices": [1.0],
        "position_list": [0],
        "initial_wallet_balance": 10000.0,
        "initial_unrealized_pnL": 0.0,
        "gamma": 0.99,
        "n_step": 1,
        "shared_model": shared_model,
    }

    runner = DfRolloutWorkerRunner(worker_config)
    assert runner.model is shared_model
    assert not runner.model.training


def _child_eval_worker(shared_model, in_queue, out_queue):
    while True:
        msg = in_queue.get()
        if msg == "STOP":
            break
        with torch.no_grad():
            inp = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
            out = shared_model(inp)
            out_queue.put(out[0, 0].item())


def test_shared_inference_visibility_across_multiprocess():
    import torch.multiprocessing as tmp
    ctx = tmp.get_context("spawn")

    manager = SharedInferenceManager(
        model_factory=SimpleToyModel,
        model_kwargs={"in_features": 4, "out_features": 2},
    )
    shared_model = manager.get_shared_model()

    source = SimpleToyModel(in_features=4, out_features=2)
    with torch.no_grad():
        source.fc.weight.fill_(1.0)
        source.fc.bias.zero_()
    manager.sync_weights_from_gpu(source)

    in_q = ctx.Queue()
    out_q = ctx.Queue()

    p = ctx.Process(target=_child_eval_worker, args=(shared_model, in_q, out_q))
    p.start()

    try:
        # First query: weight = 1.0, input sum = 4.0 -> output = 4.0
        in_q.put("EVAL")
        res1 = out_q.get(timeout=5)
        assert abs(res1 - 4.0) < 1e-4

        # In-place update from parent: weight = 2.5 -> output should become 10.0
        with torch.no_grad():
            source.fc.weight.fill_(2.5)
        manager.sync_weights_from_gpu(source)

        in_q.put("EVAL")
        res2 = out_q.get(timeout=5)
        assert abs(res2 - 10.0) < 1e-4

        in_q.put("STOP")
    finally:
        in_q.cancel_join_thread()
        out_q.cancel_join_thread()
        p.join(timeout=3)
        if p.is_alive():
            p.terminate()


def _high_concurrency_worker(worker_id, shared_model, in_queue, out_queue):
    try:
        task = in_queue.get()
        if task == "STOP":
            return
        all_shared = all(p.is_shared() for p in shared_model.parameters())
        with torch.no_grad():
            inp = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
            out1 = shared_model(inp)[0, 0].item()
        out_queue.put({"worker_id": worker_id, "all_shared": all_shared, "out1": out1})

        task2 = in_queue.get()
        if task2 == "STOP":
            return
        with torch.no_grad():
            out2 = shared_model(inp)[0, 0].item()
        out_queue.put({"worker_id": worker_id, "out2": out2})
    except Exception as e:
        out_queue.put({"worker_id": worker_id, "error": str(e)})


def test_high_concurrency_90_workers_shared_memory_and_no_fd_leak():
    """90-worker high concurrency test verifying shared memory and zero Errno 24 leak."""
    import torch.multiprocessing as tmp
    from RL.DiHFT.low_level.parallel_weight_advantage_pretrain import shutdown_workers

    num_workers = 90
    ctx = tmp.get_context("spawn")

    manager = SharedInferenceManager(
        model_factory=SimpleToyModel,
        model_kwargs={"in_features": 4, "out_features": 2},
    )
    shared_model = manager.get_shared_model()

    source = SimpleToyModel(in_features=4, out_features=2)
    with torch.no_grad():
        source.fc.weight.fill_(1.0)
        source.fc.bias.zero_()
    manager.sync_weights_from_gpu(source)

    in_queues = [ctx.Queue() for _ in range(num_workers)]
    out_queue = ctx.Queue()

    processes = [
        ctx.Process(
            target=_high_concurrency_worker,
            args=(i, shared_model, in_queues[i], out_queue),
        )
        for i in range(num_workers)
    ]

    for p in processes:
        p.start()

    try:
        # Phase 1: All 90 workers receive signal, assert is_shared, evaluate
        for q in in_queues:
            q.put("EVAL")

        phase1_results = [out_queue.get(timeout=25) for _ in range(num_workers)]
        assert len(phase1_results) == num_workers
        for r in phase1_results:
            err = r.get("error")
            wid = r.get("worker_id")
            assert "error" not in r, f"Worker {wid} failed: {err}"
            assert r["all_shared"] is True
            assert abs(r["out1"] - 4.0) < 1e-4

        # Phase 2: In-place update from parent without process recreation
        with torch.no_grad():
            source.fc.weight.fill_(3.0)
        manager.sync_weights_from_gpu(source)

        for q in in_queues:
            q.put("EVAL")

        phase2_results = [out_queue.get(timeout=25) for _ in range(num_workers)]
        assert len(phase2_results) == num_workers
        for r in phase2_results:
            err = r.get("error")
            wid = r.get("worker_id")
            assert "error" not in r, f"Worker {wid} failed: {err}"
            assert abs(r["out2"] - 12.0) < 1e-4

        # Phase 3: Send STOP signal
        for q in in_queues:
            q.put("STOP")

    finally:
        for q in in_queues:
            q.cancel_join_thread()
        out_queue.cancel_join_thread()
        shutdown_workers(in_queues, processes, timeout=10.0, grace_period=2.0)
        assert all(not p.is_alive() for p in processes)
