from __future__ import annotations

import numpy as np
import torch

from RL.DiHFT.VAE.vae import MLP_VAE
from RL.DiHFT.high_level.tri_module.ood_gate import (
    ExtremeOODGate,
    InputValidity,
)


def test_ood_gate_enters_on_either_axis_and_recovers_after_k_clean_steps() -> None:
    gate = ExtremeOODGate(
        enter_threshold=0.005,
        recover_threshold=0.01,
        recovery_steps=2,
    )

    entered = gate.evaluate(
        volatility_weights=[0.004, 0.003],
        slope_weights=[0.2, 0.1],
        input_validity=InputValidity.valid(),
    )
    pending = gate.evaluate(
        volatility_weights=[0.02, 0.01],
        slope_weights=[0.03, 0.02],
        input_validity=InputValidity.valid(),
    )
    recovered = gate.evaluate(
        volatility_weights=[0.02, 0.01],
        slope_weights=[0.03, 0.02],
        input_validity=InputValidity.valid(),
    )

    assert entered.degraded is True
    assert entered.lifecycle == "enter"
    assert entered.reason == "vae_ood:volatility"
    assert pending.degraded is True
    assert pending.lifecycle == "stay"
    assert pending.reason == "recovery_pending"
    assert recovered.degraded is False
    assert recovered.lifecycle == "recover"


def test_data_invalid_reason_is_distinct_and_resets_recovery() -> None:
    gate = ExtremeOODGate(recovery_steps=2)
    gate.evaluate(
        [0.0, 0.0],
        [0.2, 0.1],
        InputValidity.valid(),
    )
    gate.evaluate(
        [0.2, 0.1],
        [0.2, 0.1],
        InputValidity.valid(),
    )

    invalid = gate.evaluate(
        [0.2, 0.1],
        [0.2, 0.1],
        InputValidity.invalid("missing_mark_price"),
    )
    still_pending = gate.evaluate(
        [0.2, 0.1],
        [0.2, 0.1],
        InputValidity.valid(),
    )

    assert invalid.degraded is True
    assert invalid.reason == "data_invalid:missing_mark_price"
    assert invalid.recovery_streak == 0
    assert still_pending.degraded is True
    assert still_pending.recovery_streak == 1


def test_deterministic_vae_forward_uses_latent_mean() -> None:
    torch.manual_seed(7)
    model = MLP_VAE(INPUT_DIM=3, Z_DIM=2, hidden_dims=[4], loss_func="NLL")
    sample = torch.tensor([[0.2, -0.1, 0.4]], dtype=torch.float32)

    first = model.forward_deterministic(sample)
    second = model.forward_deterministic(sample)
    encoded_mu, _ = model.encode(sample)
    expected_reconstruction = model.decode(encoded_mu)

    for first_value, second_value in zip(first, second):
        assert np.array_equal(
            first_value.detach().numpy(), second_value.detach().numpy()
        )
    assert torch.equal(first[0], expected_reconstruction[0])
    assert torch.equal(first[1], expected_reconstruction[1])
