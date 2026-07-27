import sys
from pathlib import Path
import torch
import pytest

FINEFT_ROOT = Path(__file__).resolve().parents[2]
if str(FINEFT_ROOT) not in sys.path:
    sys.path.insert(0, str(FINEFT_ROOT))

from model.low_level import Qnet, ensemble_Qnet


def test_qnet_forward_with_default_trading_info():
    batch_size = 4
    n_states = 10
    n_actions = 5
    hidden_nodes = 64
    time_info_dim = 2

    model = Qnet(n_states, n_actions, hidden_nodes, time_info_dim)
    assert model.fc_trading.in_features == 4
    state = torch.randn(batch_size, n_states)
    time_tensor = torch.randn(batch_size, time_info_dim)
    previous_action = torch.zeros(batch_size, 1)
    avaliable_action = torch.ones(batch_size, n_actions)
    trading_info = torch.randn(batch_size, 4)

    out = model(state, time_tensor, previous_action, avaliable_action, trading_info)
    assert out.shape == (batch_size, n_actions)


def test_ensemble_qnet_forward_with_default_trading_info():
    batch_size = 4
    n_states = 10
    n_actions = 5
    hidden_nodes = 64
    time_info_dim = 2
    ensemble_number = 3

    model = ensemble_Qnet(
        n_states, n_actions, hidden_nodes, time_info_dim, ensemble_number
    )
    assert model.qnet_list[0].fc_trading.in_features == 4
    state = torch.randn(batch_size, n_states)
    time_tensor = torch.randn(batch_size, time_info_dim)
    previous_action = torch.zeros(batch_size, 1)
    avaliable_action = torch.ones(batch_size, n_actions)
    trading_info = torch.randn(batch_size, 4)

    q_vals = model(state, time_tensor, previous_action, avaliable_action, trading_info)
    assert q_vals.shape == (batch_size, ensemble_number, n_actions)

    best_q = model.get_best_q(state, time_tensor, previous_action, avaliable_action, trading_info)
    assert best_q.shape == (batch_size, ensemble_number)


def test_qnet_raises_without_trading_info():
    model = Qnet(10, 5, 64, 2)
    state = torch.randn(2, 10)
    time_tensor = torch.randn(2, 2)
    previous_action = torch.zeros(2, 1)
    avaliable_action = torch.ones(2, 5)

    with pytest.raises(TypeError):
        model(state, time_tensor, previous_action, avaliable_action)


def test_qnet_state_dict_contains_fc_trading():
    model = Qnet(10, 5, 64, 2)
    state_dict = model.state_dict()
    assert "fc_trading.weight" in state_dict
    assert "fc_trading.bias" in state_dict
    assert state_dict["fc_trading.weight"].shape[1] == 4
