from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from model.low_level import ensemble_Qnet
from RL.DiHFT.low_level import parallel_pretrain as pretrain

DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    ),
]


def payload(n=37):
    rows = torch.arange(n, dtype=torch.float64)
    info = {
        "previous_action": rows.clone(),
        "avaliable_action": torch.ones(n, 3),
        "funding_count_down_hour": rows + 1,
        "funding_count_down_minute": rows + 2,
        "trading_info": rows[:, None].repeat(1, 4),
        "q_value": rows[:, None].repeat(1, 3),
    }
    return {
        "states": rows[:, None].repeat(1, 4),
        "actions": rows.long().reshape(-1, 1),
        "rewards": rows.reshape(-1, 1),
        "next_states": (rows + 1)[:, None].repeat(1, 4),
        "dones": torch.zeros(n, 1),
        "infos": info,
        "next_infos": {k: v + 1 for k, v in info.items()},
        "buffer_size": n,
    }


@pytest.mark.parametrize("device", DEVICES)
def test_sampler_residency_alignment_and_without_replacement(device, monkeypatch):
    source = payload()
    sampler = pretrain.StackedTransitionSampler(source, 13, device)
    assert sampler.states.device.type == device
    assert sampler.states.dtype == torch.float32
    assert all(t.device.type == device for t in sampler.infos.values())
    assert source["states"].dtype == torch.float64
    if device == "cuda":

        def reject_cpu_sampling(*args, **kwargs):
            raise AssertionError("CUDA sampler must generate indices on device")

        monkeypatch.setattr(np.random, "choice", reject_cpu_sampling)
    for _ in range(3):
        states, infos, actions, rewards, next_states, next_infos, dones = (
            sampler.sample()
        )
        ids = actions.flatten().cpu()
        assert len(ids.unique()) == 13
        assert actions.dtype == torch.int64
        for actual, expected in [
            (states, source["states"]),
            (actions, source["actions"]),
            (rewards, source["rewards"]),
            (next_states, source["next_states"]),
            (dones, source["dones"]),
        ]:
            assert actual.device.type == device
            torch.testing.assert_close(actual.cpu(), expected[ids].to(actual.dtype))
        for name in infos:
            torch.testing.assert_close(
                infos[name].cpu(), source["infos"][name][ids].float()
            )
            torch.testing.assert_close(
                next_infos[name].cpu(), source["next_infos"][name][ids].float()
            )


@pytest.mark.parametrize("device", DEVICES)
def test_sampler_seed_repeatability_and_full_population(device):
    source = payload()
    sampler = pretrain.StackedTransitionSampler(source, 37, device)
    np.random.seed(42)
    torch.manual_seed(42)
    first = sampler.sample()[2]
    np.random.seed(42)
    torch.manual_seed(42)
    second = sampler.sample()[2]
    assert torch.equal(first, second)
    assert torch.equal(first.flatten().sort().values.cpu(), torch.arange(37))
    with pytest.raises(ValueError, match="smaller than batch_size"):
        pretrain.StackedTransitionSampler(source, 38, device)


def test_cpu_sampler_preserves_numpy_index_sequence_for_legacy_transitions():
    source = payload()
    transitions = [
        (
            source["states"][i].numpy(),
            {k: v[i].numpy() for k, v in source["infos"].items()},
            source["actions"][i].numpy(),
            source["rewards"][i].numpy(),
            source["next_states"][i].numpy(),
            source["dones"][i].numpy(),
            {k: v[i].numpy() for k, v in source["next_infos"].items()},
        )
        for i in range(37)
    ]
    sampler = pretrain.StackedTransitionSampler(transitions, 13, "cpu")
    np.random.seed(7)
    expected = np.random.choice(37, size=13, replace=False)
    np.random.seed(7)
    actual = sampler.sample()[2].flatten().numpy()
    np.testing.assert_array_equal(actual, expected)


def make_trainer(device):
    model = ensemble_Qnet(4, 3, 16, 2, 2).to(device)
    return SimpleNamespace(
        device=device,
        batch_size=7,
        N=2,
        N_ACTIONS=3,
        gamma=0.992,
        n_step=18,
        eval_net=model,
        target_net=deepcopy(model),
        ada=0.7,
        tau=0.01,
        grad_clip=1.0,
        update_counter=0,
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        loss_func_pretrain=torch.nn.SmoothL1Loss(reduction="none"),
    )


def network_inputs(states, info):
    return {
        "state": states,
        "previous_action": info["previous_action"].unsqueeze(1),
        "time": torch.stack(
            [info["funding_count_down_hour"], info["funding_count_down_minute"]], 1
        ),
        "avaliable_action": info["avaliable_action"],
        "trading_info": info["trading_info"],
    }


def reference_double_forward_update(trainer, batch):
    states, infos, actions, rewards, next_states, next_infos, dones = batch
    inputs = network_inputs(states, infos)
    chosen = pretrain.evaluate_quantile_at_action(
        trainer.eval_net(**inputs), actions
    ).squeeze(-1)
    with torch.no_grad():
        target = rewards + (1 - dones) * trainer.gamma**trainer.n_step * (
            trainer.target_net.get_best_q(**network_inputs(next_states, next_infos))
        )
    td = trainer.loss_func_pretrain(chosen, target).sum(1).mean()
    prediction = trainer.eval_net(**inputs)
    teacher = pretrain.recalculate_q_demonstration(
        infos["q_value"], infos["avaliable_action"]
    )
    kl = pretrain.calculate_paper_supervisor_kl_loss(
        prediction,
        teacher,
        torch.ones(trainer.batch_size, trainer.N, device=trainer.device),
    )
    loss = td + trainer.ada * kl
    pretrain.update_params(
        trainer.optimizer, loss, trainer.eval_net, grad_cliping=trainer.grad_clip
    )
    pretrain.soft_copy_params(trainer.eval_net, trainer.target_net, trainer.tau)
    return loss.item(), kl.item(), td.item()


@pytest.mark.parametrize("device", DEVICES)
def test_pretrain_single_forward_preserves_losses_gradients_and_updates(device):
    torch.manual_seed(42)
    trainer = make_trainer(device)
    reference = deepcopy(trainer)
    source = payload()
    source["actions"] %= 3
    source["dones"][::3] = 1
    source["infos"]["q_value"] = torch.randn(37, 3)
    source["next_infos"]["avaliable_action"] = torch.ones(37, 3)
    sampler = pretrain.StackedTransitionSampler(source, 7, device)
    calls = []
    hook = trainer.eval_net.register_forward_hook(lambda *args: calls.append(1))
    try:
        for step in range(3):
            batch = sampler.sample()
            expected = reference_double_forward_update(reference, batch)
            actual = pretrain.update_pretrain(trainer, *batch)
            assert actual == pytest.approx(expected, rel=1e-5, abs=1e-6)
            assert len(calls) == step + 1
            for actual_param, expected_param in zip(
                trainer.eval_net.parameters(), reference.eval_net.parameters()
            ):
                torch.testing.assert_close(
                    actual_param.grad, expected_param.grad, rtol=1e-5, atol=1e-6
                )
                torch.testing.assert_close(
                    actual_param, expected_param, rtol=1e-5, atol=1e-6
                )
            for actual_param, expected_param in zip(
                trainer.target_net.parameters(), reference.target_net.parameters()
            ):
                torch.testing.assert_close(
                    actual_param, expected_param, rtol=1e-5, atol=1e-6
                )
    finally:
        hook.remove()
