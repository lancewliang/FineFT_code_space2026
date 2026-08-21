# Code reference: https://github.com/Lizhi-sjtu/DRL-code-pytorch/tree/main/3.Rainbow_DQN

import sys

sys.path.append(".")
import os
import random
import argparse
import json
import re
import numpy as np
import torch
from torch import nn
import pandas as pd

# RL util
import torch.nn.functional as F


# model
from model.low_level import ensemble_Qnet

# env
from env.env_initiate.base_initiate import initiate_base_env
from env.env_class.futures_util import (
    create_optimal_q_table_from_df,
    get_dp_action_from_qtable,
    map_action_to_position_leverage,
)
from env.env_class.policy_util import get_close_element
from RL.DiHFT.low_level.policy_diagnostics import calculate_policy_direction_metrics
import copy


os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"


parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser()

# * Env setting
parser.add_argument(
    "--base_path",
    type=str,
    default="dataset",
    help="the number of action we have in the training and testing env",
)
parser.add_argument(
    "--dataset_name",
    type=str,
    default="BTCUSDT",
    help="training data chunk",
)
parser.add_argument(
    "--max_holding_number",
    type=float,
    default=8,
    help="the transcation cost of not holding the same action as before",
)
parser.add_argument(
    "--position_choices",
    type=int,
    default=9,
    help="the transcation cost of not holding the same action as before",
)
parser.add_argument(
    "--leverage_choices",
    action="append",
    type=int,
    default=[1],
    help="the transaction cost of not holding the same action as before",
)
parser.add_argument(
    "--long_estimated_rate",
    type=float,
    default=0.0005,
    help="the transcation cost of not holding the same action as before",
)
parser.add_argument(
    "--short_estimated_rate",
    type=float,
    default=0,
    help="the transcation cost of not holding the same action as before",
)
parser.add_argument(
    "--transcation_cost",
    type=float,
    default=0.0002,
    help="the transcation cost of not holding the same action as before",
)

parser.add_argument(
    "--early_stop",
    type=int,
    default=2160,
    help="the transcation cost of not holding the same action as before",
)
parser.add_argument(
    "--initial_wallet_balance",
    type=float,
    default=1e5,
    help="wallet balance",
)
parser.add_argument(
    "--initial_margin",
    type=float,
    default=0,
    help="initial margin",
)
parser.add_argument(
    "--initial_unrealized_pnL",
    type=float,
    default=0,
    help="unrealized pnL",
)
parser.add_argument(
    "--initial_position",
    type=float,
    default=0,
    help="unrealized pnL",
)
parser.add_argument(
    "--initial_leverage",
    type=float,
    default=5,
    help="initial leverage",
)
parser.add_argument(
    "--order_book_depth",
    type=int,
    default=25,
    help="number of bid/ask price levels available in the order book",
)
# network setting
parser.add_argument(
    "--hidden_nodes",
    type=int,
    default=128,
    help="the number of the hidden nodes",
)
parser.add_argument(
    "--N",
    type=int,
    default=7,
    help="context number",
)
parser.add_argument(
    "--time_info_dim",
    type=int,
    default=2,
    help="context number",
)
# model setting
parser.add_argument(
    "--epoch_num",
    type=int,
    default=1,
    help="the path for storing the test result",
)
parser.add_argument(
    "--result_path",
    type=str,
    default="result/DiHFT/low_level",
    help="the path for storing the test result",
)
parser.add_argument(
    "--experiment_name",
    type=str,
    default="default",
    help="experiment name used to namespace serial training outputs",
)
parser.add_argument(
    "--save_trading_detail_csv",
    default=True,
    action="store_true",
    help="write per-step trading detail CSV for the tested epoch",
)
parser.add_argument(
    "--allow_reverse_position",
    action="store_true",
    help="allow direct position reversal from long to short or vice versa",
)

def build_serial_model_path(result_path, dataset_name, experiment_name):
    return os.path.join(
        result_path,
        dataset_name,
        experiment_name,
        "weights_advantage_pretrain",
    )



def _detect_step_limit_states(test_df, step_index):
    if step_index >= len(test_df):
        idx = len(test_df) - 1
    else:
        idx = step_index

    is_limit_up = False
    is_limit_down = False

    if "limit_up_single_sided_ratio" in test_df.columns:
        if test_df["limit_up_single_sided_ratio"].iloc[idx] > 0:
            is_limit_up = True
    if "limit_down_single_sided_ratio" in test_df.columns:
        if test_df["limit_down_single_sided_ratio"].iloc[idx] > 0:
            is_limit_down = True

    if "is_limit_up" in test_df.columns and bool(test_df["is_limit_up"].iloc[idx]):
        is_limit_up = True
    if "is_limit_down" in test_df.columns and bool(test_df["is_limit_down"].iloc[idx]):
        is_limit_down = True

    if "UpperLimitPrice" in test_df.columns:
        upper_price = test_df["UpperLimitPrice"].iloc[idx]
        if pd.notna(upper_price) and upper_price > 0:
            p = test_df["mark_price"].iloc[idx] if "mark_price" in test_df.columns else test_df["close"].iloc[idx]
            if p >= upper_price:
                is_limit_up = True

    if "LowerLimitPrice" in test_df.columns:
        lower_price = test_df["LowerLimitPrice"].iloc[idx]
        if pd.notna(lower_price) and lower_price > 0:
            p = test_df["mark_price"].iloc[idx] if "mark_price" in test_df.columns else test_df["close"].iloc[idx]
            if p <= lower_price:
                is_limit_down = True

    return is_limit_up, is_limit_down

AGGREGATE_JSON_COLUMNS = [
    "contract",
    "df_path",
    "reward_sum",
    "df_length",
    "turnover",
    "mean_position",
    "mean_abs_position",
    "long_step_ratio",
    "short_step_ratio",
    "flat_step_ratio",
    "long_reward_sum",
    "short_reward_sum",
    "flat_reward_sum",
    "net_position_exposure",
    "position_forward_return_corr",
    "position_flip_rate",
    "mean_holding_duration",
    "long_forward_return_mean",
    "short_forward_return_mean",
    "limit_up_step_ratio",
    "limit_down_step_ratio",
    "limit_up_long_reward_sum",
    "limit_down_short_reward_sum",
    "limit_up_reverse_short_ratio",
    "limit_down_reverse_long_ratio",
]
LABEL_DIR_PATTERN = re.compile(r"^label_\d+$")

CSV_HEADER_LABELS = {
    "mean_position": "平均仓位",
    "mean_abs_position": "平均绝对仓位",
    "long_step_ratio": "多头步数占比",
    "short_step_ratio": "空头步数占比",
    "flat_step_ratio": "空仓步数占比",
    "long_reward_sum": "多头奖励总和",
    "short_reward_sum": "空头奖励总和",
    "flat_reward_sum": "空仓奖励总和",
    "net_position_exposure": "净仓位敞口",
    "position_forward_return_corr": "仓位与下一期收益相关",
    "position_flip_rate": "仓位换向率",
    "mean_holding_duration": "平均持仓时长",
    "long_forward_return_mean": "多头下一期平均收益",
    "short_forward_return_mean": "空头下一期平均收益",
    "limit_up_step_ratio": "涨停步数占比",
    "limit_down_step_ratio": "跌停步数占比",
    "limit_up_long_reward_sum": "涨停多头奖励总和",
    "limit_down_short_reward_sum": "跌停空头奖励总和",
    "limit_up_reverse_short_ratio": "涨停反向空头占比",
    "limit_down_reverse_long_ratio": "跌停反向多头占比",
    "label": "标签",
    "initial_action": "初始动作",
    "bin_index": "分箱索引",
    "contract": "合约",
    "df_path": "数据文件",
    "reward_sum": "奖励总和",
    "df_length": "数据长度",
    "turnover": "换手率",
    "timestep": "时间步",
    "timestamp": "时间戳",
    "open": "开盘价",
    "high": "最高价",
    "low": "最低价",
    "close": "收盘价",
    "volume": "成交量",
    "mark_price": "标记价格",
    "action": "动作",
    "target_position": "目标仓位",
    "target_leverage": "目标杠杆",
    "position_before": "执行前仓位",
    "leverage_before": "执行前杠杆",
    "position_after": "执行后仓位",
    "leverage_after": "执行后杠杆",
    "action_change_step": "动作变化",
    "trade_count_step": "交易计数",
    "cumulative_action_change_count": "累计动作变化次数",
    "cumulative_trade_count": "累计交易次数",
    "step_reward": "单步奖励",
    "realized_pnl_step": "单步实现盈亏",
    "cumulative_realized_pnl": "累计已实现盈亏",
    "commission_fee_step": "单步手续费",
    "cumulative_commission_fee": "累计手续费",
    "slippage_step": "单步滑点",
    "cumulative_slippage": "累计滑点",
    "wallet_balance": "结算总价值",
    "unrealized_pnl": "浮动盈亏",
    "margin_balance": "保证金余额",
    "notional_asset_value": "持仓资产",
    "cash_balance": "结算总价值",
    "total_value": "浮动总价值",
}


def _bilingual_csv_columns(df):
    return df.rename(columns=CSV_HEADER_LABELS)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_array(value):
    return json.dumps(list(value), default=_json_default)


def write_analysis_csv(overall_result, csv_path):
    analysis_df = pd.DataFrame(overall_result)
    for column in AGGREGATE_JSON_COLUMNS:
        analysis_df[column] = analysis_df[column].apply(_json_array)
    _bilingual_csv_columns(analysis_df).to_csv(csv_path, index=False)


def trading_detail_csv_path(epoch_path, epoch_num):
    return os.path.join(epoch_path, f"trading_action_detail_epoch_{epoch_num}.csv")


def write_trading_detail_csv(detail_rows, csv_path):
    _bilingual_csv_columns(pd.DataFrame(detail_rows)).to_csv(csv_path, index=False)


def _iter_valid_feather_files(root_dir):
    entries = []
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"valid data path does not exist: {root_dir}")

    for contract in sorted(os.listdir(root_dir)):
        contract_dir = os.path.join(root_dir, contract)
        if contract == "processed" or not os.path.isdir(contract_dir):
            continue
        for label in sorted(os.listdir(contract_dir)):
            label_dir = os.path.join(contract_dir, label)
            if not os.path.isdir(label_dir) or not LABEL_DIR_PATTERN.fullmatch(label):
                continue
            for filename in sorted(os.listdir(label_dir)):
                if filename.startswith("df_") and filename.endswith(".feather"):
                    rel_path = os.path.join(contract, label, filename)
                    entries.append(
                        {
                            "contract": contract,
                            "label": label,
                            "df_path": rel_path,
                            "abs_path": os.path.join(root_dir, rel_path),
                        }
                    )

    if not entries:
        raise FileNotFoundError(
            f"no validation label slices found under {root_dir}; expected "
            "valid/<contract>/label_*/df_*.feather"
        )
    return entries


DETAIL_REQUIRED_MARKET_COLUMNS = ["timestamp", "close", "volume", "mark_price"]
DETAIL_MARKET_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "mark_price"]


def _market_fields(test_df, timestep):
    row = test_df.iloc[timestep]
    return {
        column: row.get(column, np.nan)
        for column in DETAIL_MARKET_COLUMNS
        if column in DETAIL_REQUIRED_MARKET_COLUMNS or column in test_df.columns
    }


def _personal_state_from_env(test_env):
    return {
        "wallet_balance": getattr(test_env, "wallet_balance", np.nan),
        "unrealized_pnl": getattr(test_env, "unrealized_pnl", np.nan),
        "position": getattr(test_env, "position", np.nan),
        "leverage": getattr(test_env, "leverage", np.nan),
    }


def build_trading_detail_row(
    *,
    label,
    df_path,
    initial_action,
    bin_index,
    timestep,
    test_df,
    action,
    target_position,
    target_leverage,
    position_before,
    leverage_before,
    test_env,
    info,
    step_reward,
    action_change_step,
    trade_count_step,
    cumulative_action_change_count,
    cumulative_trade_count,
):
    state_after = _personal_state_from_env(test_env)
    market_fields = _market_fields(test_df, timestep)
    mark_price = market_fields.get("mark_price", np.nan)
    wallet_balance = state_after["wallet_balance"]
    unrealized_pnl = state_after["unrealized_pnl"]
    position_after = state_after["position"]
    margin_balance = wallet_balance + unrealized_pnl
    row = {
        "label": label,
        "df_path": df_path,
        "initial_action": initial_action,
        "bin_index": bin_index,
        "timestep": timestep,
        **market_fields,
        "action": action,
        "target_position": target_position,
        "target_leverage": target_leverage,
        "position_before": position_before,
        "leverage_before": leverage_before,
        "position_after": position_after,
        "leverage_after": state_after["leverage"],
        "action_change_step": action_change_step,
        "trade_count_step": trade_count_step,
        "cumulative_action_change_count": cumulative_action_change_count,
        "cumulative_trade_count": cumulative_trade_count,
        "step_reward": step_reward,
        "realized_pnl_step": info["realized_pnl_step"],
        "cumulative_realized_pnl": info["cumulative_realized_pnl"],
        "commission_fee_step": info["commission_fee_step"],
        "cumulative_commission_fee": info["cumulative_commission_fee"],
        "slippage_step": info["slippage_step"],
        "cumulative_slippage": info["cumulative_slippage"],
        "wallet_balance": wallet_balance,
        "unrealized_pnl": unrealized_pnl,
        "margin_balance": margin_balance,
        "notional_asset_value": mark_price * position_after,
        "cash_balance": wallet_balance,
        "total_value": margin_balance,
    }
    return row


def seed_torch(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class weighted_trader:
    def __init__(self, args):

        # device
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        # log path
        self.model_path = build_serial_model_path(
            args.result_path,
            args.dataset_name,
            args.experiment_name,
        )
  
        # trading environment setting
        self.base_path = args.base_path
        self.dataset_name = args.dataset_name
        self.valid_data_path = os.path.join(self.base_path, self.dataset_name, "valid")
        self.tech_indicator_list = np.load(
            os.path.join(self.base_path, self.dataset_name, "state_features.npy")
        )
        self.maintenance_margin_ratio_dict = np.load(
            os.path.join(
                self.base_path, self.dataset_name, "maintenance_margin_ratio_dict.npy"
            ),
            allow_pickle=True,
        ).item()
        self.max_holding_number = args.max_holding_number
        self.order_book_depth = args.order_book_depth
        self.position_choices = args.position_choices
        self.single_side_action_num = int((self.position_choices - 1) / 2)
        self.position_list = (
            [
                self.max_holding_number / self.single_side_action_num * i
                for i in range(1, self.single_side_action_num + 1)
            ]
            + [0]
            + [
                self.max_holding_number / self.single_side_action_num * -i
                for i in range(1, self.single_side_action_num + 1)
            ]
        )
        self.position_list.sort()
        self.leverage_choices = args.leverage_choices
        self.long_estimated_rate = args.long_estimated_rate
        self.short_estimated_rate = args.short_estimated_rate
        self.transcation_cost = args.transcation_cost
        self.allow_reverse_position = getattr(args, "allow_reverse_position", False)
        self.early_stop = args.early_stop
        self.initial_wallet_balance = args.initial_wallet_balance
        self.initial_margin = args.initial_margin
        self.initial_unrealized_pnL = args.initial_unrealized_pnL
        self.initial_position = args.initial_position
        self.initial_leverage = args.initial_leverage
        self.initial_state = (
            self.initial_wallet_balance,
            self.initial_margin,
            self.initial_unrealized_pnL,
            self.initial_position,
            self.initial_leverage,
        )

        # network
        self.time_info_dim = args.time_info_dim
        self.hidden_nodes = args.hidden_nodes
        self.N = args.N
        self.N_ACTIONS = (self.position_choices - 1) * len(self.leverage_choices) + 1
        self.eval_net = ensemble_Qnet(
            N_STATES=len(self.tech_indicator_list),
            N_ACTIONS=self.N_ACTIONS,
            hidden_nodes=self.hidden_nodes,
            TIME_INFO_DIM=self.time_info_dim,
            ensemble_number=self.N,
        ).to(self.device)

        self.epoch_num = args.epoch_num
        self.save_trading_detail_csv = args.save_trading_detail_csv
        self.epoch_path = os.path.join(
            self.model_path,
            "epoch_" + str(self.epoch_num),
        )
        self.eval_net.load_state_dict(
            torch.load(
                os.path.join(self.epoch_path, "trained_model.pkl"),
                map_location=self.device,
            )
        )
        self.eval_net.eval()
        self.initial_action_list = range(
            (self.position_choices - 1) * len(self.leverage_choices) + 1
        )

    def act_test(self, state, info, context_index):
        assert context_index in range(self.N)
        state = torch.unsqueeze(torch.FloatTensor(state).reshape(-1), 0).to(self.device)
        previous_action = torch.unsqueeze(
            torch.tensor([info["previous_action"]]).float().to(self.device), 0
        ).to(self.device)
        avaliable_action = torch.unsqueeze(
            torch.tensor(info["avaliable_action"]).to(self.device), 0
        ).to(self.device)
        hour_count_down = (
            torch.unsqueeze(torch.tensor([info["funding_count_down_hour"]]), 0)
            .to(self.device)
            .float()
        )
        minute_count_down = (
            torch.unsqueeze(torch.tensor([info["funding_count_down_minute"]]), 0)
            .to(self.device)
            .float()
        )
        time_input = torch.cat([hour_count_down, minute_count_down], dim=1).to(
            self.device
        )
        trading_info = torch.unsqueeze(
            torch.tensor(info["trading_info"]).float().to(self.device), 0
        )
        with torch.inference_mode():
            action_value_chosen_index = self.eval_net.qnet_list[context_index](
                state=state,
                time=time_input,
                previous_action=previous_action,
                avaliable_action=avaliable_action,
                trading_info=trading_info,
            )
            action = torch.max(action_value_chosen_index, 1)[1].data.cpu().numpy()
        action = action[0]

        return action

    def test(self):
        print('start')
        overall_result = []
        trading_detail_rows = []
        self.eval_net.eval()
        df_entries = _iter_valid_feather_files(self.valid_data_path)
        label_list = sorted({entry["label"] for entry in df_entries})
        for label in label_list:
            print('start label {}'.format(label))
            label_entries = [
                entry for entry in df_entries if entry["label"] == label
            ]
            for initial_action in self.initial_action_list:
                for bin_index in range(self.N):
                    single_label_initial_action_bin_index_reward_sum_result = []
                    single_label_initial_action_bin_index_df_length_result = []
                    single_label_initial_action_bin_index_turnover_result = []
                    single_label_initial_action_bin_index_contract_result = []
                    single_label_initial_action_bin_index_df_path_result = []
                    single_label_initial_action_bin_index_mean_position_result = []
                    single_label_initial_action_bin_index_mean_abs_position_result = []
                    single_label_initial_action_bin_index_long_step_ratio_result = []
                    single_label_initial_action_bin_index_short_step_ratio_result = []
                    single_label_initial_action_bin_index_flat_step_ratio_result = []
                    single_label_initial_action_bin_index_long_reward_sum_result = []
                    single_label_initial_action_bin_index_short_reward_sum_result = []
                    single_label_initial_action_bin_index_flat_reward_sum_result = []
                    single_label_initial_action_bin_index_net_position_exposure_result = []
                    single_label_initial_action_bin_index_position_forward_return_corr_result = []
                    single_label_initial_action_bin_index_position_flip_rate_result = []
                    single_label_initial_action_bin_index_mean_holding_duration_result = []
                    single_label_initial_action_bin_index_long_forward_return_mean_result = []
                    single_label_initial_action_bin_index_short_forward_return_mean_result = []
                    single_label_initial_action_bin_index_limit_up_step_ratio_result = []
                    single_label_initial_action_bin_index_limit_down_step_ratio_result = []
                    single_label_initial_action_bin_index_limit_up_long_reward_sum_result = []
                    single_label_initial_action_bin_index_limit_down_short_reward_sum_result = []
                    single_label_initial_action_bin_index_limit_up_reverse_short_ratio_result = []
                    single_label_initial_action_bin_index_limit_down_reverse_long_ratio_result = []
                    for entry in label_entries:
                        contract = entry["contract"]
                        df_path = entry["df_path"]
                        initial_position, initial_leverage = (
                            map_action_to_position_leverage(
                                initial_action,
                                self.leverage_choices,
                                self.position_list,
                            )
                        )
                        self.test_df = pd.read_feather(entry["abs_path"])
                        current_markprice = self.test_df["mark_price"].values[0]
                        self.initial_margin = np.abs(
                            initial_position * current_markprice / initial_leverage
                        )
                        self.initial_state = (
                            self.initial_wallet_balance,
                            self.initial_margin,
                            self.initial_unrealized_pnL,
                            initial_position,
                            initial_leverage,
                        )
                        test_env = initiate_base_env(
                            df=self.test_df,
                            feature_list=self.tech_indicator_list,
                            max_holding_number=self.max_holding_number,
                            order_book_depth=self.order_book_depth,
                            position_choices=self.position_choices,  # (must be an odd number, the minum of trading equals to (max_holder_number)/((action_dim-1)/2)s))
                            leverage_choice=self.leverage_choices,  # recommend only use one leverage choice, because the leverage does not influence the return directly, the position
                            # itself is enough to show the risk preference
                            long_estimated_rate=self.long_estimated_rate,
                            short_estimated_rate=self.short_estimated_rate,
                            commission_rate=self.transcation_cost,
                            # maten_mar_ratio_dict varies among different perpertual contracts, need to perform a config file for different perpertual
                            # the default is for btcusdt perpetual contract
                            maintenance_margin_ratio_dict=self.maintenance_margin_ratio_dict,
                            early_stop=0,
                            # initial_personal_state
                            initial_state=self.initial_state,
                            allow_reverse_position=getattr(self, "allow_reverse_position", False),
                        )
                        position_after_list = []
                        limit_up_list = []
                        limit_down_list = []
                        s, info = test_env.reset()
                        done = False
                        reward_sum = 0
                        reward_list = []
                        action_list = []
                        turn_over = 0
                        previous_action = initial_action
                        cumulative_action_change_count = 0
                        cumulative_trade_count = 0
                        while not done:
                            timestep = len(action_list)
                            position_before = getattr(
                                test_env, "position", initial_position
                            )
                            leverage_before = getattr(
                                test_env, "leverage", initial_leverage
                            )
                            a = self.act_test(s, info, bin_index)
                            target_position, target_leverage = (
                                map_action_to_position_leverage(
                                    a,
                                    self.leverage_choices,
                                    self.position_list,
                                )
                            )
                            action_change_step = int(a != previous_action)
                            turn_over += np.abs(a - previous_action) / 4
                            s_, r, done, info = test_env.step(a)
                            position_after = getattr(
                                test_env, "position", position_before
                            )
                            leverage_after = getattr(
                                test_env, "leverage", leverage_before
                            )
                            trade_count_step = int(
                                position_after != position_before
                                or leverage_after != leverage_before
                            )
                            cumulative_action_change_count += action_change_step
                            cumulative_trade_count += trade_count_step
                            if self.save_trading_detail_csv:
                                trading_detail_rows.append(
                                    build_trading_detail_row(
                                        label=label,
                                        df_path=df_path,
                                        initial_action=initial_action,
                                        bin_index=bin_index,
                                        timestep=timestep,
                                        test_df=self.test_df,
                                        action=a,
                                        target_position=target_position,
                                        target_leverage=target_leverage,
                                        position_before=position_before,
                                        leverage_before=leverage_before,
                                        test_env=test_env,
                                        info=info,
                                        step_reward=r,
                                        action_change_step=action_change_step,
                                        trade_count_step=trade_count_step,
                                        cumulative_action_change_count=cumulative_action_change_count,
                                        cumulative_trade_count=cumulative_trade_count,
                                    )
                                )
                            action_list.append(a)
                            reward_list.append(r)
                            position_after_list.append(position_after)
                            is_limit_up, is_limit_down = _detect_step_limit_states(self.test_df, timestep)
                            limit_up_list.append(is_limit_up)
                            limit_down_list.append(is_limit_down)
                            s = s_
                            reward_sum += r
                            previous_action = a
                        initial_margin_history = test_env.initial_margin_history
                        wallet_balance_history = test_env.wallet_balance_history
                        unrealized_pnl_history = test_env.unrealized_pnl_history
                        maintain_marigine_history = test_env.maintain_marigine_history
                        new_position_required_money_history = (
                            test_env.new_position_required_money_history
                        )
                        # single_epoch_single_index_single_df_path = os.path.join(
                        #     self.epoch_path,
                        #     "test_dynamics",
                        #     "bin_index_{}".format(bin_index),
                        #     label,
                        #     "initial_action_{}".format(initial_action),
                        #     df_path.removesuffix(".feather"),
                        # )
                        # if not os.path.exists(single_epoch_single_index_single_df_path):
                        #     os.makedirs(single_epoch_single_index_single_df_path)
                        # np.save(
                        #     os.path.join(
                        #         single_epoch_single_index_single_df_path,
                        #         "micro_action_history.npy",
                        #     ),
                        #     action_list,
                        # )
                        # np.save(
                        #     os.path.join(
                        #         single_epoch_single_index_single_df_path,
                        #         "reward_history.npy",
                        #     ),
                        #     reward_list,
                        # )
                        # np.save(
                        #     os.path.join(
                        #         single_epoch_single_index_single_df_path,
                        #         "initial_margin_history.npy",
                        #     ),
                        #     initial_margin_history,
                        # )
                        # np.save(
                        #     os.path.join(
                        #         single_epoch_single_index_single_df_path,
                        #         "wallet_balance_history.npy",
                        #     ),
                        #     wallet_balance_history,
                        # )
                        # np.save(
                        #     os.path.join(
                        #         single_epoch_single_index_single_df_path,
                        #         "unrealized_pnl_history.npy",
                        #     ),
                        #     unrealized_pnl_history,
                        # )
                        # np.save(
                        #     os.path.join(
                        #         single_epoch_single_index_single_df_path,
                        #         "maintain_marigine_history.npy",
                        #     ),
                        #     maintain_marigine_history,
                        # )
                        # np.save(
                        #     os.path.join(
                        #         single_epoch_single_index_single_df_path,
                        #         "new_position_required_money_history.npy",
                        #     ),
                        #     new_position_required_money_history,
                        # )

                        single_label_initial_action_bin_index_reward_sum_result.append(
                            reward_sum
                        )
                        single_label_initial_action_bin_index_df_length_result.append(
                            len(self.test_df)
                        )
                        single_label_initial_action_bin_index_turnover_result.append(
                            turn_over
                        )
                        single_label_initial_action_bin_index_contract_result.append(
                            contract
                        )
                        single_label_initial_action_bin_index_df_path_result.append(
                            df_path
                        )

                        total_steps = len(position_after_list)
                        if total_steps > 0:
                            pos_arr = np.array(position_after_list, dtype=float)
                            rew_arr = np.array(reward_list, dtype=float)
                            up_arr = np.array(limit_up_list, dtype=bool)
                            down_arr = np.array(limit_down_list, dtype=bool)

                            mean_pos = float(np.mean(pos_arr))
                            mean_abs_pos = float(np.mean(np.abs(pos_arr)))
                            long_mask = pos_arr > 0
                            short_mask = pos_arr < 0
                            flat_mask = pos_arr == 0

                            long_step_ratio = float(np.mean(long_mask))
                            short_step_ratio = float(np.mean(short_mask))
                            flat_step_ratio = float(np.mean(flat_mask))

                            long_reward_sum = float(np.sum(rew_arr[long_mask]))
                            short_reward_sum = float(np.sum(rew_arr[short_mask]))
                            flat_reward_sum = float(np.sum(rew_arr[flat_mask]))

                            max_hold = float(getattr(self, "max_holding_number", 1.0))
                            if max_hold <= 0:
                                max_hold = 1.0
                            net_position_exposure = float(mean_pos / max_hold)
                            direction_metrics = calculate_policy_direction_metrics(
                                pos_arr,
                                self.test_df["mark_price"].to_numpy(),
                            )

                            limit_up_step_ratio = float(np.mean(up_arr))
                            limit_down_step_ratio = float(np.mean(down_arr))

                            limit_up_long_reward_sum = float(np.sum(rew_arr[up_arr & long_mask]))
                            limit_down_short_reward_sum = float(np.sum(rew_arr[down_arr & short_mask]))

                            up_count = np.sum(up_arr)
                            down_count = np.sum(down_arr)

                            limit_up_reverse_short_ratio = (
                                float(np.sum(up_arr & short_mask) / up_count) if up_count > 0 else 0.0
                            )
                            limit_down_reverse_long_ratio = (
                                float(np.sum(down_arr & long_mask) / down_count) if down_count > 0 else 0.0
                            )
                        else:
                            mean_pos = 0.0
                            mean_abs_pos = 0.0
                            long_step_ratio = 0.0
                            short_step_ratio = 0.0
                            flat_step_ratio = 0.0
                            long_reward_sum = 0.0
                            short_reward_sum = 0.0
                            flat_reward_sum = 0.0
                            net_position_exposure = 0.0
                            limit_up_step_ratio = 0.0
                            limit_down_step_ratio = 0.0
                            limit_up_long_reward_sum = 0.0
                            limit_down_short_reward_sum = 0.0
                            limit_up_reverse_short_ratio = 0.0
                            limit_down_reverse_long_ratio = 0.0
                            direction_metrics = {
                                "position_forward_return_corr": 0.0,
                                "position_flip_rate": 0.0,
                                "mean_holding_duration": 0.0,
                                "long_forward_return_mean": 0.0,
                                "short_forward_return_mean": 0.0,
                            }

                        single_label_initial_action_bin_index_mean_position_result.append(mean_pos)
                        single_label_initial_action_bin_index_mean_abs_position_result.append(mean_abs_pos)
                        single_label_initial_action_bin_index_long_step_ratio_result.append(long_step_ratio)
                        single_label_initial_action_bin_index_short_step_ratio_result.append(short_step_ratio)
                        single_label_initial_action_bin_index_flat_step_ratio_result.append(flat_step_ratio)
                        single_label_initial_action_bin_index_long_reward_sum_result.append(long_reward_sum)
                        single_label_initial_action_bin_index_short_reward_sum_result.append(short_reward_sum)
                        single_label_initial_action_bin_index_flat_reward_sum_result.append(flat_reward_sum)
                        single_label_initial_action_bin_index_net_position_exposure_result.append(net_position_exposure)
                        single_label_initial_action_bin_index_position_forward_return_corr_result.append(
                            direction_metrics["position_forward_return_corr"]
                        )
                        single_label_initial_action_bin_index_position_flip_rate_result.append(
                            direction_metrics["position_flip_rate"]
                        )
                        single_label_initial_action_bin_index_mean_holding_duration_result.append(
                            direction_metrics["mean_holding_duration"]
                        )
                        single_label_initial_action_bin_index_long_forward_return_mean_result.append(
                            direction_metrics["long_forward_return_mean"]
                        )
                        single_label_initial_action_bin_index_short_forward_return_mean_result.append(
                            direction_metrics["short_forward_return_mean"]
                        )
                        single_label_initial_action_bin_index_limit_up_step_ratio_result.append(limit_up_step_ratio)
                        single_label_initial_action_bin_index_limit_down_step_ratio_result.append(limit_down_step_ratio)
                        single_label_initial_action_bin_index_limit_up_long_reward_sum_result.append(limit_up_long_reward_sum)
                        single_label_initial_action_bin_index_limit_down_short_reward_sum_result.append(limit_down_short_reward_sum)
                        single_label_initial_action_bin_index_limit_up_reverse_short_ratio_result.append(limit_up_reverse_short_ratio)
                        single_label_initial_action_bin_index_limit_down_reverse_long_ratio_result.append(limit_down_reverse_long_ratio)
                    _overall_result = {
                            "label": label,
                            "initial_action": initial_action,
                            "bin_index": bin_index,
                            "contract": single_label_initial_action_bin_index_contract_result,
                            "df_path": single_label_initial_action_bin_index_df_path_result,
                            "reward_sum": single_label_initial_action_bin_index_reward_sum_result,
                            "df_length": single_label_initial_action_bin_index_df_length_result,
                            "turnover": single_label_initial_action_bin_index_turnover_result,
                            "mean_position": single_label_initial_action_bin_index_mean_position_result,
                            "mean_abs_position": single_label_initial_action_bin_index_mean_abs_position_result,
                            "long_step_ratio": single_label_initial_action_bin_index_long_step_ratio_result,
                            "short_step_ratio": single_label_initial_action_bin_index_short_step_ratio_result,
                            "flat_step_ratio": single_label_initial_action_bin_index_flat_step_ratio_result,
                            "long_reward_sum": single_label_initial_action_bin_index_long_reward_sum_result,
                            "short_reward_sum": single_label_initial_action_bin_index_short_reward_sum_result,
                            "flat_reward_sum": single_label_initial_action_bin_index_flat_reward_sum_result,
                            "net_position_exposure": single_label_initial_action_bin_index_net_position_exposure_result,
                            "position_forward_return_corr": single_label_initial_action_bin_index_position_forward_return_corr_result,
                            "position_flip_rate": single_label_initial_action_bin_index_position_flip_rate_result,
                            "mean_holding_duration": single_label_initial_action_bin_index_mean_holding_duration_result,
                            "long_forward_return_mean": single_label_initial_action_bin_index_long_forward_return_mean_result,
                            "short_forward_return_mean": single_label_initial_action_bin_index_short_forward_return_mean_result,
                            "limit_up_step_ratio": single_label_initial_action_bin_index_limit_up_step_ratio_result,
                            "limit_down_step_ratio": single_label_initial_action_bin_index_limit_down_step_ratio_result,
                            "limit_up_long_reward_sum": single_label_initial_action_bin_index_limit_up_long_reward_sum_result,
                            "limit_down_short_reward_sum": single_label_initial_action_bin_index_limit_down_short_reward_sum_result,
                            "limit_up_reverse_short_ratio": single_label_initial_action_bin_index_limit_up_reverse_short_ratio_result,
                            "limit_down_reverse_long_ratio": single_label_initial_action_bin_index_limit_down_reverse_long_ratio_result,
                    }    
                    print(_overall_result)
                    overall_result.append(
                        _overall_result
                    )
        
        np.save(os.path.join(self.epoch_path, "analysis_result.npy"), overall_result)
        write_analysis_csv(
            overall_result,
            os.path.join(self.epoch_path, "analysis_result.csv"),
        )
        if self.save_trading_detail_csv:
            write_trading_detail_csv(
                trading_detail_rows,
                trading_detail_csv_path(self.epoch_path, self.epoch_num),
            )


if __name__ == "__main__":
    args = parser.parse_args()
    trader = weighted_trader(args)
    trader.test()
