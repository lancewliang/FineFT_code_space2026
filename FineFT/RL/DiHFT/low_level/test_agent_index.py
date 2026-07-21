# Code reference: https://github.com/Lizhi-sjtu/DRL-code-pytorch/tree/main/3.Rainbow_DQN

import sys

sys.path.append(".")
import os
import random
import argparse
import json
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

def build_serial_model_path(result_path, dataset_name, experiment_name):
    return os.path.join(
        result_path,
        dataset_name,
        experiment_name,
        "weights_advantage_pretrain",
    )


AGGREGATE_JSON_COLUMNS = ["df_path", "reward_sum", "df_length", "turnover"]

CSV_HEADER_LABELS = {
    "label": "标签",
    "initial_action": "初始动作",
    "bin_index": "分箱索引",
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
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames.sort()
        rel_dir = os.path.relpath(dirpath, root_dir)
        if rel_dir == ".":
            continue
        for filename in sorted(filenames):
            if filename.startswith("df_") and filename.endswith(".feather"):
                yield rel_dir, filename


DETAIL_MARKET_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "mark_price"]


def _market_fields(test_df, timestep):
    row = test_df.iloc[timestep]
    return {
        column: row[column]
        for column in DETAIL_MARKET_COLUMNS
        if column in test_df.columns
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
        with torch.inference_mode():
            action_value_chosen_index = self.eval_net.qnet_list[context_index](
                state=state,
                time=time_input,
                previous_action=previous_action,
                avaliable_action=avaliable_action,
            )
            action = torch.max(action_value_chosen_index, 1)[1].data.cpu().numpy()
        action = action[0]

        return action

    def test(self):
        print('start')
        overall_result = []
        trading_detail_rows = []
        self.eval_net.eval()
        df_entries = list(_iter_valid_feather_files(self.valid_data_path))
        label_list = sorted({label for label, _ in df_entries})
        for label in label_list:
            print('start label {}'.format(label))
            df_list = [
                df_path
                for label_path, df_path in df_entries
                if label_path == label
            ]
            for initial_action in self.initial_action_list:
                for bin_index in range(self.N):
                    single_label_initial_action_bin_index_reward_sum_result = []
                    single_label_initial_action_bin_index_df_length_result = []
                    single_label_initial_action_bin_index_turnover_result = []
                    single_label_initial_action_bin_index_df_path_result = []
                    for df_path in df_list:
                        initial_position, initial_leverage = (
                            map_action_to_position_leverage(
                                initial_action,
                                self.leverage_choices,
                                self.position_list,
                            )
                        )
                        self.test_df = pd.read_feather(
                            os.path.join(self.valid_data_path, label, df_path)
                        )
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
                        )
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
                        single_label_initial_action_bin_index_df_path_result.append(
                            df_path
                        )
                    _overall_result = {
                            "label": label,
                            "initial_action": initial_action,
                            "bin_index": bin_index,
                            "df_path": single_label_initial_action_bin_index_df_path_result,
                            "reward_sum": single_label_initial_action_bin_index_reward_sum_result,
                            "df_length": single_label_initial_action_bin_index_df_length_result,
                            "turnover": single_label_initial_action_bin_index_turnover_result,
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
