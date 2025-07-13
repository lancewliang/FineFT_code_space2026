# Code reference: https://github.com/Lizhi-sjtu/DRL-code-pytorch/tree/main/3.Rainbow_DQN

import sys

sys.path.append(".")
import os
from torch.utils.tensorboard import SummaryWriter
from RL.util.replay_buffer_DQN import Multi_step_ReplayBuffer_multi_info
from env.env_initiate.agg_initiate import initiate_high_level_earnhft_env
from analysis.calculate_metric.calculate_metric import calculate_differences
import random
from tqdm import tqdm
import argparse
from model.high_level import Qnet_high_level_position
import numpy as np
import torch
from torch import nn
import yaml
import pandas as pd
import copy
import re

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"

parser = argparse.ArgumentParser()
# * Env setting
# high level unique

parser.add_argument(
    "--dynamics_num",
    type=int,
    default=5,
    help="the number of action we have in the training and testing env",
)
parser.add_argument(
    "--adjust_len",
    type=int,
    default=12,
    help="the number of action we have in the training and testing env",
)
parser.add_argument(
    "--low_level_hidden_nodes",
    type=int,
    default=128,
    help="the number of action we have in the training and testing env",
)
# low level setting
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
# network setting
parser.add_argument(
    "--hidden_nodes",
    type=int,
    default=128,
    help="the number of the hidden nodes",
)
parser.add_argument(
    "--epoch_num",
    type=int,
    default=1,
    help="the number of the hidden nodes",
)


def seed_torch(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class DQN(object):
    def __init__(self, args):
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # env
        self.base_path = args.base_path
        self.dataset_name = args.dataset_name
        self.test_data_path = os.path.join(
            self.base_path, self.dataset_name, "test.feather"
        )
        self.test_df = pd.read_feather(self.test_data_path)
        self.tech_indicator_list = np.load(
            os.path.join(self.base_path, self.dataset_name, "state_features.npy")
        )
        self.high_level_feature_list = np.load(
            os.path.join(self.base_path, self.dataset_name, "high_level_state_features.npy")
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
        # high level unique
        self.dynamics_num = args.dynamics_num
        self.adjust_len = args.adjust_len
        self.low_level_hidden_nodes = args.low_level_hidden_nodes
        self.potential_model_path = os.path.join(
            "result",
            "EarnHFT",
            "potential_model",
            args.dataset_name,
        )
        self.n_state = len(self.high_level_feature_list)

        self.hidden_nodes = args.hidden_nodes
        self.eval_net = Qnet_high_level_position(
            self.n_state,
            self.dynamics_num,
            self.hidden_nodes,
        ).to(self.device)
        self.test_path = os.path.join(
            "result",
            "EarnHFT",
            "high_level",
            args.dataset_name,
            'seed_12345',
            "epoch_{}".format(args.epoch_num),
        )
        self.eval_net.load_state_dict(
            torch.load(
                os.path.join(
                    self.test_path,
                    "trained_model.pkl",
                )
            )
        )

    def act_test(self, info):
        x = (
            torch.tensor(info["high_level_state"])
            .to(self.device)
            .to(torch.float32)
            .unsqueeze(0)
        )
        position = (
            torch.tensor([info["previous_action"]])
            .to(self.device)
            .to(torch.float32)
            .unsqueeze(1)
        )

        actions_value = self.eval_net.forward(x, position)
        action = torch.max(actions_value, 1)[1].data.cpu().numpy()
        action = action[0]
        
        return action

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

    def test(self):
        test_env = initiate_high_level_earnhft_env(
            df=self.test_df,
            adjust_len=self.adjust_len,
            potential_model_path=self.potential_model_path,
            dynamics_num=self.dynamics_num,
            low_level_hidden_nodes=self.low_level_hidden_nodes,
            high_level_feature_list=self.high_level_feature_list,
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
            device=self.device,
        )
        s, info = test_env.reset()
        episode_reward_sum = 0
        while True:
            a = self.act_test(info)
            s_, r, done, info = test_env.step(a)
            s = s_
            episode_reward_sum += r
            if done:
                break
        total_asset_history = test_env.margine_balance_history
        reward_history = calculate_differences(total_asset_history)
        micro_action_history = test_env.micro_action_history
        trading_info = {
            "return rate": total_asset_history[-1] / self.initial_wallet_balance
        }
        np.save(os.path.join(self.test_path, "reward_history.npy"), reward_history)
        np.save(
            os.path.join(self.test_path, "total_asset_history.npy"), total_asset_history
        )
        np.save(
            os.path.join(self.test_path, "micro_action_history.npy"),
            micro_action_history,
        )
        np.save(os.path.join(self.test_path, "trading_info.npy"), trading_info)
        np.save(
            os.path.join(self.test_path, "initial_margin_history.npy"),
            test_env.initial_margin_history,
        )
        np.save(
            os.path.join(self.test_path, "wallet_balance_history.npy"),
            test_env.wallet_balance_history,
        )
        np.save(
            os.path.join(self.test_path, "unrealized_pnl_history.npy"),
            test_env.unrealized_pnl_history,
        )
        np.save(
            os.path.join(self.test_path, "maintain_marigine_history.npy"),
            test_env.maintain_marigine_history,
        )
        np.save(
            os.path.join(self.test_path, "new_position_required_money_history.npy"),
            test_env.new_position_required_money_history,
        )


if __name__ == "__main__":
    args = parser.parse_args()
    dqn = DQN(args)
    dqn.test()
