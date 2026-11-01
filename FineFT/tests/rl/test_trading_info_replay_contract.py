import numpy as np
import pytest
import torch

from common.trading_observation import TradingObservationConfig, TradingObservationError
from RL.DiHFT.low_level.parallel_pretrain import (
    load_pretrain_buffer_file,
    save_pretrain_buffer_file,
)
from RL.util.replay_buffer_DQN import Multi_step_ReplayBuffer_multi_info
from RL.util.update import update_params
from model.low_level import Qnet


def _info(value=0.0):
    return {
        "avaliable_action": np.ones(3),
        "previous_action": 0,
        "q_value": 0.0,
        "high_level_state": np.zeros(2),
        "funding_count_down_hour": 0.0,
        "funding_count_down_minute": 0.0,
        "trading_info": np.full(11, value, dtype=np.float32),
    }


def test_replay_rejects_missing_and_legacy_trading_observations():
    buffer = Multi_step_ReplayBuffer_multi_info(
        buffer_size=8, batch_size=1, device="cpu", n_step=1, gamma=0.99, seed=1
    )
    with pytest.raises(TradingObservationError, match="missing"):
        buffer.add("s", {}, 0, 1.0, "s2", _info(), False)
    legacy = _info()
    legacy["trading_info"] = np.zeros(4, dtype=np.float32)
    with pytest.raises(TradingObservationError, match="width 11"):
        buffer.add("s", legacy, 0, 1.0, "s2", _info(), False)


def test_replay_preserves_current_and_next_eleven_dimensional_values():
    buffer = Multi_step_ReplayBuffer_multi_info(
        buffer_size=8, batch_size=1, device="cpu", n_step=1, gamma=0.99, seed=1
    )
    buffer.add("s", _info(1.0), 0, 1.0, "s2", _info(2.0), True)
    transition = buffer.memory[0]
    assert transition.info["trading_info"].shape == (11,)
    assert transition.next_info["trading_info"].shape == (11,)
    assert transition.info["trading_info"][10] == 1.0
    assert transition.next_info["trading_info"][10] == 2.0


def test_real_qnet_update_consumes_replayed_eleven_dimensional_context():
    buffer = Multi_step_ReplayBuffer_multi_info(
        buffer_size=8, batch_size=2, device="cpu", n_step=1, gamma=0.99, seed=1
    )
    for index in range(3):
        info = _info(float(index))
        info["previous_action"] = 0
        info["funding_count_down_hour"] = 1.0
        info["funding_count_down_minute"] = 2.0
        next_info = _info(float(index + 1))
        next_info["previous_action"] = 1
        next_info["funding_count_down_hour"] = 1.0
        next_info["funding_count_down_minute"] = 1.0
        buffer.add(np.zeros(3, dtype=np.float32), info, 0, 1.0, np.ones(3, dtype=np.float32), next_info, False)
    states, infos, actions, rewards, _, _, _ = buffer.sample()
    model = Qnet(3, 3, 8, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.fc_trading.weight.detach().clone()
    q_values = model(
        states,
        torch.stack((infos["funding_count_down_hour"], infos["funding_count_down_minute"]), dim=1),
        infos["previous_action"].reshape(-1, 1),
        infos["avaliable_action"],
        infos["trading_info"],
    )
    loss = (q_values.gather(1, actions) - rewards).pow(2).mean()
    assert torch.isfinite(loss)
    update_params(optimizer, loss, model)
    assert not torch.equal(before, model.fc_trading.weight.detach())


def test_pretrain_replay_snapshot_requires_schema_metadata_and_round_trips(tmp_path):
    config = TradingObservationConfig(exit_price_scale_floor=5.0)
    buffer = Multi_step_ReplayBuffer_multi_info(
        buffer_size=8, batch_size=1, device="cpu", n_step=1, gamma=0.99, seed=1
    )
    buffer.add("s", _info(1.0), 0, 1.0, "s2", _info(2.0), True)
    path = tmp_path / "pretrain_buffer.pt"

    save_pretrain_buffer_file(
        buffer,
        str(path),
        1,
        trading_observation_config=config,
    )
    restored = Multi_step_ReplayBuffer_multi_info(
        buffer_size=8, batch_size=1, device="cpu", n_step=1, gamma=0.99, seed=1
    )
    assert load_pretrain_buffer_file(
        restored,
        str(path),
        trading_observation_config=config,
    ) == 1
    assert restored.memory[0].info["trading_info"].shape == (11,)

    torch.save({"transitions": []}, path)
    with pytest.raises(TradingObservationError, match="unversioned"):
        load_pretrain_buffer_file(
            restored,
            str(path),
            trading_observation_config=config,
        )
