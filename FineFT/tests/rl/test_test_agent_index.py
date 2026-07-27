import json
import sys
from pathlib import Path

import numpy as np
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
        return [0.0], 1.0, True, {
            "previous_action": action,
            "commission_fee_step": 0.0,
            "realized_pnl_step": 0.0,
            "slippage_step": 0.0,
            "cumulative_commission_fee": 0.0,
            "cumulative_realized_pnl": 0.0,
            "cumulative_slippage": 0.0,
        }


class DetailFakeEnv:
    initial_margin_history = []
    wallet_balance_history = []
    unrealized_pnl_history = []
    maintain_marigine_history = []
    new_position_required_money_history = []

    def __init__(self):
        self.step_index = 0
        self.position = 0
        self.leverage = 1
        self.wallet_balance = 1000.0
        self.unrealized_pnl = 0.0

    def reset(self):
        return [0.0], {
            "previous_action": 0,
            "avaliable_action": [1, 1, 1],
            "funding_count_down_hour": 0,
            "funding_count_down_minute": 0,
        }

    def step(self, action):
        if self.step_index == 0:
            self.step_index += 1
            return [0.0], 2.0, False, {
                "previous_action": action,
                "avaliable_action": [1, 1, 1],
                "funding_count_down_hour": 0,
                "funding_count_down_minute": 0,
                "commission_fee_step": 0.5,
                "realized_pnl_step": 0.0,
                "slippage_step": 0.1,
                "cumulative_commission_fee": 0.5,
                "cumulative_realized_pnl": 0.0,
                "cumulative_slippage": 0.1,
            }
        self.position = 1
        self.wallet_balance = 1001.0
        self.unrealized_pnl = 3.0
        self.step_index += 1
        return [0.0], 4.0, True, {
            "previous_action": action,
            "avaliable_action": [1, 1, 1],
            "funding_count_down_hour": 0,
            "funding_count_down_minute": 0,
            "commission_fee_step": 0.7,
            "realized_pnl_step": 5.0,
            "slippage_step": 0.2,
            "cumulative_commission_fee": 1.2,
            "cumulative_realized_pnl": 5.0,
            "cumulative_slippage": 0.3,
        }


class CountingQNet:
    def __init__(self, values):
        self.values = values
        self.calls = 0

    def __call__(self, state, time, previous_action, avaliable_action, trading_info=None):
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


def _write_valid_slice(
    tmp_path, contract, label, filename="df_0.feather", mark_prices=None
):
    if mark_prices is None:
        mark_prices = [100.0]
    valid_dir = tmp_path / "valid" / contract / label
    valid_dir.mkdir(parents=True, exist_ok=True)
    df_path = valid_dir / filename
    if not df_path.exists():
        pd.DataFrame({"mark_price": mark_prices}).to_feather(df_path)
    return df_path


def _make_test_trader(tai, tmp_path, save_trading_detail_csv=False):
    _write_valid_slice(tmp_path, "fu2507", "label_0")

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
    trader.epoch_num = 1
    trader.save_trading_detail_csv = save_trading_detail_csv
    trader.act_test = lambda state, info, bin_index: 0
    return trader


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
            "trading_info": [0.0, 0.0, 0.0],
        },
        context_index=1,
    )

    assert action == 1
    assert [qnet.calls for qnet in trader.eval_net.qnet_list] == [0, 1, 0]


def test_weighted_trader_passes_order_book_depth_to_base_env(monkeypatch, tmp_path):
    from RL.DiHFT.low_level import test_agent_index as tai

    captured_kwargs = {}

    def fake_initiate_base_env(**kwargs):
        captured_kwargs.update(kwargs)
        return FakeEnv()

    monkeypatch.setattr(tai, "initiate_base_env", fake_initiate_base_env)
    monkeypatch.setattr(tai, "map_action_to_position_leverage", lambda *args: (0, 1))

    trader = _make_test_trader(tai, tmp_path)

    trader.test()

    assert captured_kwargs["order_book_depth"] == 5
    npy_path = tmp_path / "analysis_result.npy"
    csv_path = tmp_path / "analysis_result.csv"

    assert npy_path.exists()
    assert csv_path.exists()

    result = np.load(npy_path, allow_pickle=True).tolist()
    assert result[0]["label"] == "label_0"
    assert result[0]["contract"] == ["fu2507"]
    assert result[0]["df_path"] == ["fu2507/label_0/df_0.feather"]

    csv_df = pd.read_csv(csv_path)
    assert list(csv_df.columns) == [
        "标签",
        "初始动作",
        "分箱索引",
        "合约",
        "数据文件",
        "奖励总和",
        "数据长度",
        "换手率",
    ]
    assert json.loads(csv_df.loc[0, "合约"]) == ["fu2507"]
    assert json.loads(csv_df.loc[0, "数据文件"]) == ["fu2507/label_0/df_0.feather"]
    assert json.loads(csv_df.loc[0, "奖励总和"]) == [1.0]
    assert json.loads(csv_df.loc[0, "数据长度"]) == [1]
    assert json.loads(csv_df.loc[0, "换手率"]) == [0.0]


def test_trading_detail_csv_is_disabled_by_default(monkeypatch, tmp_path):
    from RL.DiHFT.low_level import test_agent_index as tai

    monkeypatch.setattr(tai, "initiate_base_env", lambda **kwargs: FakeEnv())
    monkeypatch.setattr(tai, "map_action_to_position_leverage", lambda *args: (0, 1))

    trader = _make_test_trader(tai, tmp_path, save_trading_detail_csv=False)
    trader.test()

    assert not (tmp_path / "trading_action_detail_epoch_1.csv").exists()


def test_trading_detail_csv_is_written_when_enabled(monkeypatch, tmp_path):
    from RL.DiHFT.low_level import test_agent_index as tai

    monkeypatch.setattr(tai, "initiate_base_env", lambda **kwargs: FakeEnv())
    monkeypatch.setattr(tai, "map_action_to_position_leverage", lambda *args: (0, 1))

    trader = _make_test_trader(tai, tmp_path, save_trading_detail_csv=True)
    trader.test()

    assert (tmp_path / "trading_action_detail_epoch_1.csv").exists()


def test_trading_detail_csv_records_actions_trades_and_execution_metrics(
    monkeypatch, tmp_path
):
    from RL.DiHFT.low_level import test_agent_index as tai

    valid_dir = tmp_path / "valid" / "fu2507" / "label_0"
    valid_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="min"),
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [10.0, 11.0],
            "mark_price": [100.0, 101.0],
        }
    ).to_feather(valid_dir / "df_0.feather")

    monkeypatch.setattr(tai, "initiate_base_env", lambda **kwargs: DetailFakeEnv())
    monkeypatch.setattr(
        tai,
        "map_action_to_position_leverage",
        lambda action, leverage_choices, position_list: (1, 1)
        if action == 1
        else (0, 1),
    )

    trader = _make_test_trader(tai, tmp_path, save_trading_detail_csv=True)
    trader.act_test = lambda state, info, bin_index: 1
    trader.test()

    detail_df = pd.read_csv(tmp_path / "trading_action_detail_epoch_1.csv")
    assert len(detail_df) == 2
    for column in [
        "标签",
        "数据文件",
        "时间戳",
        "目标仓位",
        "目标杠杆",
        "单步实现盈亏",
        "累计已实现盈亏",
        "单步手续费",
        "累计手续费",
        "单步滑点",
        "累计滑点",
    ]:
        assert column in detail_df.columns
    assert detail_df.loc[0, "标签"] == "label_0"
    assert detail_df.loc[0, "数据文件"] == "fu2507/label_0/df_0.feather"
    assert detail_df.loc[0, "目标仓位"] == 1
    assert detail_df.loc[0, "目标杠杆"] == 1
    assert detail_df.loc[0, "动作变化"] == 1
    assert detail_df.loc[0, "交易计数"] == 0
    assert detail_df.loc[1, "动作变化"] == 0
    assert detail_df.loc[1, "交易计数"] == 1
    assert detail_df.loc[1, "累计动作变化次数"] == 1
    assert detail_df.loc[1, "累计交易次数"] == 1
    assert detail_df.loc[1, "单步手续费"] == 0.7
    assert detail_df.loc[1, "累计手续费"] == 1.2
    assert detail_df.loc[1, "单步实现盈亏"] == 5.0
    assert detail_df.loc[1, "累计已实现盈亏"] == 5.0
    assert detail_df.loc[1, "单步滑点"] == 0.2
    assert detail_df.loc[1, "累计滑点"] == 0.3
    assert detail_df.loc[1, "保证金余额"] == 1004.0
    assert detail_df.loc[1, "浮动总价值"] == 1004.0
    assert detail_df.loc[1, "持仓资产"] == 101.0


def test_weighted_trader_handles_nested_contract_label_directories(
    monkeypatch, tmp_path
):
    from RL.DiHFT.low_level import test_agent_index as tai

    _write_valid_slice(tmp_path, "fu2507", "label_2")
    (tmp_path / "valid" / "processed").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"mark_price": [100.0]}).to_feather(
        tmp_path / "valid" / "processed" / "valid_processed_fu2507.feather"
    )
    pd.DataFrame({"mark_price": [100.0]}).to_feather(
        tmp_path / "valid" / "fu2507.feather"
    )

    monkeypatch.setattr(tai, "initiate_base_env", lambda **kwargs: FakeEnv())
    monkeypatch.setattr(tai, "map_action_to_position_leverage", lambda *args: (0, 1))

    trader = _make_test_trader(tai, tmp_path)
    trader.test()

    result = np.load(tmp_path / "analysis_result.npy", allow_pickle=True).tolist()
    assert result[0]["label"] == "label_0"
    assert result[0]["contract"] == ["fu2507"]
    assert result[0]["df_path"] == ["fu2507/label_0/df_0.feather"]
    label_2_records = [row for row in result if row["label"] == "label_2"]
    assert label_2_records[0]["contract"] == ["fu2507"]
    assert label_2_records[0]["df_path"] == ["fu2507/label_2/df_0.feather"]
