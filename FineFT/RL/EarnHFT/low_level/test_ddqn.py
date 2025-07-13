# Code reference: https://github.com/Lizhi-sjtu/DRL-code-pytorch/tree/main/3.Rainbow_DQN

import sys

sys.path.append(".")
import os
from torch.utils.tensorboard import SummaryWriter
from RL.util.replay_buffer_DQN import Multi_step_ReplayBuffer_multi_info
import random
from tqdm import tqdm
import argparse
from model.low_level import Qnet
import numpy as np
import torch
from torch import nn
import yaml
import pandas as pd
from env.env_initiate.base_initiate import initiate_base_env
import re

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"

parser = argparse.ArgumentParser()
# path
parser.add_argument(
    "--beta",
    type=int,
    default=-20,
    help="the path of test model",
)

parser.add_argument(
    "--model_path",
    type=str,
    default="result/EarnHFT/low_level",
    help="the path of test model",
)
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
    default=[5],
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
# network
parser.add_argument(
    "--hidden_nodes",
    type=int,
    default=128,
    help="the number of the hidden nodes",
)
parser.add_argument(
    "--time_info_dim",
    type=int,
    default=2,
    help="context number",
)
parser.add_argument(
    "--epoch_number",
    type=int,
    default=1,
    help="the number of the hidden nodes",
)


def find_matching_path(root_dir, target_beta_value):
    # 编译一个正则表达式，用于解析目录名中的beta值
    beta_pattern = re.compile(r"beta_([-\d\.]+)")

    # 遍历目录树，查找所有目录名
    for dirpath, dirnames, _ in os.walk(root_dir):
        for dirname in dirnames:
            match = beta_pattern.search(dirname)
            if match:
                # 解析出的beta值
                beta_value = float(match.group(1))
                # 检查解析出的beta值是否与给定的beta值匹配
                if int(beta_value) == target_beta_value:
                    return os.path.join(dirpath, dirname)
    print("wrong, no matching path found")


class trader(object):
    def __init__(self, args) -> None:
        # device
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
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

        # network setting
        self.time_info_dim = args.time_info_dim
        self.hidden_nodes = args.hidden_nodes
        self.N_ACTIONS = (self.position_choices - 1) * len(self.leverage_choices) + 1
        self.eval_net = Qnet(
            N_STATES=len(self.tech_indicator_list),
            N_ACTIONS=self.N_ACTIONS,
            hidden_nodes=self.hidden_nodes,
            TIME_INFO_DIM=self.time_info_dim,
        ).to(self.device)
        model_base_path = os.path.join(args.model_path, args.dataset_name)
        beta_path = find_matching_path(model_base_path, args.beta)
        self.epoch_path = os.path.join(
            beta_path, "seed_12345", "epoch_{}".format(args.epoch_number)
        )
        model_path = os.path.join(self.epoch_path, "trained_model.pkl")
        self.eval_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.initial_action_list = range(
            (self.position_choices - 1) * len(self.leverage_choices) + 1
        )

    def act_test(self, state, info):
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
        actions_value = self.eval_net(
            state=state,
            time=time_input,
            previous_action=previous_action,
            avaliable_action=avaliable_action,
        )
        action = torch.max(actions_value, 1)[1].data.cpu().numpy()
        action = action[0]
        return action

    def comprehensive_test(self):
        overall_result = []
        label_list = os.listdir(self.valid_data_path)
        for label in label_list:
            df_list = os.listdir(os.path.join(self.valid_data_path, label))
            for initial_action in self.initial_action_list:
                single_label_initial_action_reward_sum_result = []
                single_label_initial_action_df_length_result = []
                for df_path in df_list:
                    initial_position, initial_leverage = (
                        self.map_action_to_position_leverage(initial_action)
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
                    while not done:
                        a = self.act_test(s, info)
                        s_, r, done, info = test_env.step(a)
                        s = s_
                        reward_sum += r
                    single_label_initial_action_reward_sum_result.append(reward_sum)
                    single_label_initial_action_df_length_result.append(
                        len(self.test_df)
                    )
                overall_result.append(
                    {
                        "label": label,
                        "initial_action": initial_action,
                        "reward_sum": single_label_initial_action_reward_sum_result,
                        "df_length": single_label_initial_action_df_length_result,
                    }
                )
        np.save(os.path.join(self.epoch_path, "analysis_result.npy"), overall_result)

    def map_action_to_position_leverage(self, action):
        leverage_length = len(self.leverage_choices)
        position_length = len(self.position_list)
        zero_position_action = leverage_length * (position_length // 2)
        if action == zero_position_action:
            return 0, self.leverage_choices[0]
        elif action > zero_position_action:
            action = action + leverage_length - 1
        else:
            action = action
        # 返回对应的仓位和杠杆倍率
        position_index = action // len(self.leverage_choices)
        leverage_index = action % len(self.leverage_choices)
        position = self.position_list[position_index]
        leverage = self.leverage_choices[leverage_index]
        return position, leverage


if __name__ == "__main__":
    args = parser.parse_args()
    test_trader = trader(args)
    test_trader.comprehensive_test()
