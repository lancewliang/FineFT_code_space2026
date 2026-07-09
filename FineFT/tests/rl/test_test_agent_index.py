import sys
from pathlib import Path

import pandas as pd
import torch


FINEFT_ROOT = Path(__file__).resolve().parents[2]
if str(FINEFT_ROOT) not in sys.path:
    sys.path.insert(0, str(FINEFT_ROOT))


class FakeNet:
    def eval(self):
        pass


class FakeEnv:
    initial_margin_history = []
    wallet_balance_history = []
    unrealized_pnl_history = []
    maintain_marigine_history = []
    new_position_required_money_history = []

    def reset(self):
        return [0.0], {"previous_action": 0}

    def step(self, action):
        return [0.0], 1.0, True, {"previous_action": action}


class CountingQNet:
    def __init__(self, values):
        self.values = values
        self.calls = 0

    def __call__(self, state, time, previous_action, avaliable_action):
        self.calls += 1
        return torch.tensor([self.values], dtype=torch.float32, device=state.device)


class FakeEnsemble:
    def __init__(self):
        self.qnet_list = [
            CountingQNet([1.0, 2.0, 3.0]),
            CountingQNet([4.0, 9.0, 5.0]),
            CountingQNet([7.0, 6.0, 8.0]),
        ]

    def __call__(self, *args, **kwargs):
        raise AssertionError("act_test should not evaluate the full ensemble")


def test_act_test_only_evaluates_selected_context_qnet():
    from RL.DiHFT.low_level import test_agent_index as tai

    trader = tai.weighted_trader.__new__(tai.weighted_trader)
    trader.device = "cpu"
    trader.N = 3
    trader.eval_net = FakeEnsemble()

    action = trader.act_test(
        state=[0.0, 1.0],
        info={
            "previous_action": 0,
            "avaliable_action": [1, 1, 1],
            "funding_count_down_hour": 1,
            "funding_count_down_minute": 30,
        },
        context_index=1,
    )

    assert action == 1
    assert [qnet.calls for qnet in trader.eval_net.qnet_list] == [0, 1, 0]


def test_weighted_trader_passes_order_book_depth_to_base_env(monkeypatch, tmp_path):
    from RL.DiHFT.low_level import test_agent_index as tai

    valid_dir = tmp_path / "valid" / "label"
    valid_dir.mkdir(parents=True)
    pd.DataFrame({"mark_price": [100.0]}).to_feather(valid_dir / "df_0.feather")

    captured_kwargs = {}

    def fake_initiate_base_env(**kwargs):
        captured_kwargs.update(kwargs)
        return FakeEnv()

    monkeypatch.setattr(tai, "initiate_base_env", fake_initiate_base_env)
    monkeypatch.setattr(tai, "map_action_to_position_leverage", lambda *args: (0, 1))

    trader = tai.weighted_trader.__new__(tai.weighted_trader)
    trader.eval_net = FakeNet()
    trader.valid_data_path = str(tmp_path / "valid")
    trader.initial_action_list = [0]
    trader.N = 1
    trader.leverage_choices = [1]
    trader.position_list = [0]
    trader.initial_wallet_balance = 100000
    trader.initial_unrealized_pnL = 0
    trader.max_holding_number = 1
    trader.position_choices = 3
    trader.order_book_depth = 5
    trader.long_estimated_rate = 0
    trader.short_estimated_rate = 0
    trader.transcation_cost = 0
    trader.maintenance_margin_ratio_dict = {}
    trader.tech_indicator_list = []
    trader.epoch_path = str(tmp_path)
    trader.act_test = lambda state, info, bin_index: 0

    trader.test()

    assert captured_kwargs["order_book_depth"] == 5
