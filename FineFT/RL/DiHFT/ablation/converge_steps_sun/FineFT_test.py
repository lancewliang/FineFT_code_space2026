# Code reference: https://github.com/Lizhi-sjtu/DRL-code-pytorch/tree/main/3.Rainbow_DQN

import sys

sys.path.append(".")
import os
import random
import argparse
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
    default="dataset/ablation",
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
    default="result/DiHFT/ablation/FineFT",
    help="the path for storing the test result",
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


class weighted_trader:
    def __init__(self, args):

        # device
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        # log path
        self.model_path = os.path.join(
            args.result_path, args.dataset_name, "FineFT"
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
        actions_value = self.eval_net(
            state=state,
            time=time_input,
            previous_action=previous_action,
            avaliable_action=avaliable_action,
        )
        action_value_chosen_index = actions_value[:, context_index, :]
        action = torch.max(action_value_chosen_index, 1)[1].data.cpu().numpy()
        action = action[0]

        return action

    def test(self):
        overall_result = []
        self.eval_net.eval()
        label_list = os.listdir(self.valid_data_path)
        for label in label_list:
            df_list = os.listdir(os.path.join(self.valid_data_path, label))
            for initial_action in self.initial_action_list:
                for bin_index in range(self.N):
                    single_label_initial_action_bin_index_reward_sum_result = []
                    single_label_initial_action_bin_index_df_length_result = []
                    single_label_initial_action_bin_index_turnover_result = []
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
                        while not done:
                            a = self.act_test(s, info, bin_index)
                            turn_over += np.abs(a - previous_action) / 4
                            s_, r, done, info = test_env.step(a)
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

                        single_label_initial_action_bin_index_reward_sum_result.append(
                            reward_sum
                        )
                        single_label_initial_action_bin_index_df_length_result.append(
                            len(self.test_df)
                        )
                        single_label_initial_action_bin_index_turnover_result.append(
                            turn_over
                        )
                    overall_result.append(
                        {
                            "label": label,
                            "initial_action": initial_action,
                            "bin_index": bin_index,
                            "reward_sum": single_label_initial_action_bin_index_reward_sum_result,
                            "df_length": single_label_initial_action_bin_index_df_length_result,
                            "turnover": single_label_initial_action_bin_index_turnover_result,
                        }
                    )
        np.save(os.path.join(self.epoch_path, "analysis_result.npy"), overall_result)


if __name__ == "__main__":
    args = parser.parse_args()
    trader = weighted_trader(args)
    trader.test()
