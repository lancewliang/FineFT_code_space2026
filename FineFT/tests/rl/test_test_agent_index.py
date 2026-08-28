import json
import os
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
    tmp_path, contract, label, filename="df_0.feather", mark_prices=None, label_type="slope"
):
    if mark_prices is None:
        mark_prices = [100.0]
    valid_dir = tmp_path / "valid" / label_type / contract / label
    valid_dir.mkdir(parents=True, exist_ok=True)
    df_path = valid_dir / filename
    if not df_path.exists():
        pd.DataFrame({"mark_price": mark_prices}).to_feather(df_path)
    return df_path


def _make_test_trader(tai, tmp_path, save_trading_detail_csv=False, label_type="slope"):
    _write_valid_slice(tmp_path, "fu2507", "label_0", label_type=label_type)

    trader = tai.weighted_trader.__new__(tai.weighted_trader)
    trader.eval_net = FakeNet()
    trader.label_type = label_type
    trader.valid_data_path = str(tmp_path / "valid" / label_type)
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
            "trading_info": [0.0, 0.0, 0.0, 0.0],
        },
        context_index=1,
    )

    assert action == 1
    assert [qnet.calls for qnet in trader.eval_net.qnet_list] == [0, 1, 0]


def test_act_test_selected_context_accepts_real_four_field_ensemble():
    from model.low_level import ensemble_Qnet
    from RL.DiHFT.low_level import test_agent_index as tai

    trader = tai.weighted_trader.__new__(tai.weighted_trader)
    trader.device = "cpu"
    trader.N = 2
    trader.eval_net = ensemble_Qnet(
        N_STATES=2,
        N_ACTIONS=3,
        hidden_nodes=16,
        TIME_INFO_DIM=2,
        ensemble_number=2,
    )

    action = trader.act_test(
        state=[0.0, 1.0],
        info={
            "previous_action": 0,
            "avaliable_action": [1, 1, 1],
            "funding_count_down_hour": 1,
            "funding_count_down_minute": 30,
            "trading_info": [0.0, 0.0, 0.0, 0.0],
        },
        context_index=1,
    )

    assert action in [0, 1, 2]


def test_average_act_test_accepts_real_four_field_ensemble():
    from model.low_level import ensemble_Qnet
    from RL.DiHFT.low_level import test_agent_average as taa

    trader = taa.weighted_trader.__new__(taa.weighted_trader)
    trader.device = "cpu"
    trader.eval_net = ensemble_Qnet(
        N_STATES=2,
        N_ACTIONS=3,
        hidden_nodes=16,
        TIME_INFO_DIM=2,
        ensemble_number=2,
    )

    action = trader.act_test(
        state=[0.0, 1.0],
        info={
            "previous_action": 0,
            "avaliable_action": [1, 1, 1],
            "funding_count_down_hour": 1,
            "funding_count_down_minute": 30,
            "trading_info": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        },
    )

    assert action in [0, 1, 2]


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
    npy_path = tmp_path / "slope" / "analysis_result.npy"
    csv_path = tmp_path / "slope" / "analysis_result.csv"

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
        "平均仓位",
        "平均绝对仓位",
        "多头步数占比",
        "空头步数占比",
        "空仓步数占比",
        "多头奖励总和",
            "空头奖励总和",
            "空仓奖励总和",
            "净仓位敞口",
            "仓位与下一期收益相关",
            "仓位换向率",
            "平均持仓时长",
            "多头下一期平均收益",
            "空头下一期平均收益",
            "涨停步数占比",
        "跌停步数占比",
        "涨停多头奖励总和",
        "跌停空头奖励总和",
        "涨停反向空头占比",
        "跌停反向多头占比",
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

    assert not (tmp_path / "slope" / "trading_action_detail_epoch_1.csv").exists()


def test_trading_detail_csv_is_written_when_enabled(monkeypatch, tmp_path):
    from RL.DiHFT.low_level import test_agent_index as tai

    monkeypatch.setattr(tai, "initiate_base_env", lambda **kwargs: FakeEnv())
    monkeypatch.setattr(tai, "map_action_to_position_leverage", lambda *args: (0, 1))

    trader = _make_test_trader(tai, tmp_path, save_trading_detail_csv=True)
    trader.test()

    assert (tmp_path / "slope" / "trading_action_detail_epoch_1.csv").exists()


def test_trading_detail_csv_records_actions_trades_and_execution_metrics(
    monkeypatch, tmp_path
):
    from RL.DiHFT.low_level import test_agent_index as tai

    valid_dir = tmp_path / "valid" / "slope" / "fu2507" / "label_0"
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

    detail_df = pd.read_csv(tmp_path / "slope" / "trading_action_detail_epoch_1.csv")
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

    result = np.load(tmp_path / "slope" / "analysis_result.npy", allow_pickle=True).tolist()
    assert result[0]["label"] == "label_0"
    assert result[0]["contract"] == ["fu2507"]
    assert result[0]["df_path"] == ["fu2507/label_0/df_0.feather"]
    label_2_records = [row for row in result if row["label"] == "label_2"]
    assert label_2_records[0]["contract"] == ["fu2507"]
    assert label_2_records[0]["df_path"] == ["fu2507/label_2/df_0.feather"]



def test_parser_allow_reverse_position_default_and_flag():
    from RL.DiHFT.low_level import test_agent_index as tai

    args_default = tai.parser.parse_args(["--label_type", "slope"])
    assert args_default.allow_reverse_position is False

    args_flag = tai.parser.parse_args(["--allow_reverse_position", "--label_type", "slope"])
    assert args_flag.allow_reverse_position is True


def test_parser_label_type_required():
    from RL.DiHFT.low_level import test_agent_index as tai
    import pytest

    with pytest.raises(SystemExit):
        tai.parser.parse_args(["--allow_reverse_position"])

    args_slope = tai.parser.parse_args(["--label_type", "slope"])
    assert args_slope.label_type == "slope"

    args_vol = tai.parser.parse_args(["--label_type", "volatility"])
    assert args_vol.label_type == "volatility"


def test_weighted_trader_init_constructs_valid_data_path_with_label_type(monkeypatch):
    from RL.DiHFT.low_level import test_agent_index as tai

    monkeypatch.setattr(tai, "build_serial_model_path", lambda *args: "/fake/model/path")
    monkeypatch.setattr(np, "load", lambda path, **kwargs: (
        type("DummyDict", (), {"item": lambda self: {}})()
        if "maintenance_margin" in str(path)
        else np.array([])
    ))
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {})
    monkeypatch.setattr(tai.ensemble_Qnet, "load_state_dict", lambda self, sd: None)
    monkeypatch.setattr(tai.ensemble_Qnet, "eval", lambda self: None)

    args_slope = tai.parser.parse_args([
        "--base_path", "dataset/10min",
        "--dataset_name", "fu",
        "--label_type", "slope",
    ])
    trader = tai.weighted_trader(args_slope)
    assert trader.label_type == "slope"
    assert trader.valid_data_path == os.path.join("dataset/10min", "fu", "valid", "slope")

    args_vol = tai.parser.parse_args([
        "--base_path", "dataset/10min",
        "--dataset_name", "fu",
        "--label_type", "volatility",
    ])
    trader_vol = tai.weighted_trader(args_vol)
    assert trader_vol.label_type == "volatility"
    assert trader_vol.valid_data_path == os.path.join("dataset/10min", "fu", "valid", "volatility")


def test_analysis_result_includes_directional_and_limit_behavior_metrics(monkeypatch, tmp_path):
    from RL.DiHFT.low_level import test_agent_index as tai

    monkeypatch.setattr(tai, "initiate_base_env", lambda **kwargs: DetailFakeEnv())
    monkeypatch.setattr(
        tai,
        "map_action_to_position_leverage",
        lambda action, leverage_choices, position_list: (1, 1) if action == 1 else (0, 1),
    )

    trader = _make_test_trader(tai, tmp_path, save_trading_detail_csv=False)
    trader.act_test = lambda state, info, bin_index: 1
    trader.test()

    result = np.load(tmp_path / "slope" / "analysis_result.npy", allow_pickle=True).tolist()
    record = result[0]
    expected_fields = [
        "mean_position",
        "mean_abs_position",
        "long_step_ratio",
        "short_step_ratio",
        "flat_step_ratio",
        "long_reward_sum",
        "short_reward_sum",
        "flat_reward_sum",
        "net_position_exposure",
        "limit_up_step_ratio",
        "limit_down_step_ratio",
        "limit_up_long_reward_sum",
        "limit_down_short_reward_sum",
        "limit_up_reverse_short_ratio",
        "limit_down_reverse_long_ratio",
    ]
    for field in expected_fields:
        assert field in record, f"missing field {field} in npy output"
        assert len(record[field]) == len(record["contract"])

    df = pd.read_csv(tmp_path / "slope" / "analysis_result.csv")
    for field in expected_fields:
        col = tai.CSV_HEADER_LABELS.get(field, field)
        assert col in df.columns, f"missing column {col} in csv output"
        val = json.loads(df.loc[0, col])
        assert isinstance(val, list)


def test_seed_torch_enables_float32_matmul_precision_and_allow_tf32():
    from RL.DiHFT.low_level import test_agent_index as tai

    tai.seed_torch(42)
    if hasattr(torch, "get_float32_matmul_precision"):
        assert torch.get_float32_matmul_precision() == "high"
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        assert torch.backends.cuda.matmul.allow_tf32 is True
    if hasattr(torch.backends, "cudnn"):
        assert torch.backends.cudnn.allow_tf32 is True


def test_weighted_trader_init_enables_matmul_precision_and_allow_tf32(monkeypatch):
    from RL.DiHFT.low_level import test_agent_index as tai

    monkeypatch.setattr(tai, "build_serial_model_path", lambda *args: "/fake/model/path")
    monkeypatch.setattr(np, "load", lambda path, **kwargs: (
        type("DummyDict", (), {"item": lambda self: {}})()
        if "maintenance_margin" in str(path)
        else np.array([])
    ))
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {})
    monkeypatch.setattr(tai.ensemble_Qnet, "load_state_dict", lambda self, sd: None)
    monkeypatch.setattr(tai.ensemble_Qnet, "eval", lambda self: None)

    args = tai.parser.parse_args([
        "--base_path", "dataset/10min",
        "--dataset_name", "fu",
        "--label_type", "slope",
    ])
    trader = tai.weighted_trader(args)
    if trader.device == "cuda":
        if hasattr(torch, "get_float32_matmul_precision"):
            assert torch.get_float32_matmul_precision() == "high"
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            assert torch.backends.cuda.matmul.allow_tf32 is True

