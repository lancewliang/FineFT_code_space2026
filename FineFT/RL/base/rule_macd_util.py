import torch
import sys
import random
import os

sys.path.append(".")
import re
from model.low_level import Actor
from torch.utils.tensorboard import SummaryWriter
from RL.base.util.replay_buffer_PPO import ReplayBuffer_lstm
from env.env_initiate.simple_initiate import initiate_simple_env
import argparse
import numpy as np
from torch.distributions import Categorical
from env.env_class.futures_util import (
    map_action_to_position_leverage,
)
from torch.utils.data.sampler import (
    BatchSampler,
    SubsetRandomSampler,
    SequentialSampler,
)
import pandas as pd
from analysis.calculate_metric.calculate_metric import (
    calculate_differences,
    calculate_required_money,
)


parser = argparse.ArgumentParser()
# basic setting
parser.add_argument(
    "--result_path",
    type=str,
    default="result/base",
    help="the path for storing the test result",
)

# trading setting

parser.add_argument(
    "--seed",
    type=int,
    default=12345,
    help="the random seed for training and sample",
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
    "--early_stop",
    type=int,
    default=0,
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
    "--long_term",
    type=int,
    default=26,
    help="the stop win for trading",
)

parser.add_argument(
    "--mid_term",
    type=int,
    default=12,
    help="the stop lose for trading",
)


parser.add_argument(
    "--short_term",
    type=int,
    default=9,
    help="the stop lose for trading",
)


def sort_list(lst: list):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    lst.sort(key=alphanum_key)


def seed_torch(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class macd_trader:
    def __init__(self, args):

        self.test_path = os.path.join(
            args.result_path,
            "macd",
            args.dataset_name,
            "{}_{}_{}".format(
                args.long_term,
                args.mid_term,
                args.short_term,
            ),
        )
        if not os.path.exists(self.test_path):
            os.makedirs(self.test_path)
        # trading setting
        self.base_path = args.base_path
        self.dataset_name = args.dataset_name
        self.train_data_path = os.path.join(self.base_path, self.dataset_name, "train")
        self.total_df_index_length = len(os.listdir(self.train_data_path)) - 1
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
        self.valid_data_path = os.path.join(
            "dataset", self.dataset_name, "valid.feather"
        )
        self.test_data_path = os.path.join("dataset", self.dataset_name, "test.feather")
        # trading parameter
        self.long_term = args.long_term
        self.mid_term = args.mid_term
        self.short_term = args.short_term

    def test(self):
        for name, data_path in zip(
            ["valid", "test"], [self.valid_data_path, self.test_data_path]
        ):
            self.test_df = pd.read_feather(data_path)
            import_fea = "mark_price"
            df_singnal = pd.DataFrame()
            DIF = (
                self.test_df[import_fea].ewm(span=self.mid_term).mean()
                - self.test_df[import_fea].ewm(span=self.long_term).mean()
            )
            df_singnal["DIF"] = DIF
            DEA = df_singnal["DIF"].ewm(span=self.short_term).mean()
            df_singnal["DEA"] = DEA
            MACD = DIF - DEA
            df_singnal["MACD"] = MACD
            action_list = []
            true_action_list = []
            reward_list = []
            for macd, dif in zip(MACD, DIF):
                if macd > 0 and dif > 0:
                    action_list.append("buy")
                elif macd * dif <= 0:
                    action_list.append("hold")
                else:
                    action_list.append("sell")
            self.test_env_instance = initiate_simple_env(
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
                early_stop=self.early_stop,
                # initial_personal_state
                initial_state=self.initial_state,
                # window length
            )
            s, info = self.test_env_instance.reset()
            done = False
            previous_true_action = 4
            for action in action_list:
                avaliable_index = info["avaiable_action_list"]
                if action == "buy":
                    true_action = max(avaliable_index)
                if action == "hold":
                    true_action = previous_true_action
                if action == "sell":
                    true_action = min(avaliable_index)
                s_, r, done, info_ = self.test_env_instance.step(true_action)
                reward_list.append(r)
                true_action_list.append(true_action)
                s = s_
                info = info_
                previous_true_action = true_action
                if done:
                    break
            action_list = np.array(action_list)
            if not os.path.exists(os.path.join(self.test_path, name)):
                os.makedirs(os.path.join(self.test_path, name))
            np.save(os.path.join(self.test_path, name, "action.npy"), true_action_list)
            total_asset_history = self.test_env_instance.margine_balance_history
            reward_history = calculate_differences(total_asset_history)
            trading_info = {
                "return rate": total_asset_history[-1] / self.initial_wallet_balance
            }
            np.save(
                os.path.join(self.test_path, name, "total_asset_history.npy"),
                total_asset_history,
            )

            np.save(
                os.path.join(self.test_path, name, "trading_info.npy"), trading_info
            )
            np.save(
                os.path.join(self.test_path, name, "initial_margin_history.npy"),
                self.test_env_instance.initial_margin_history,
            )
            np.save(
                os.path.join(self.test_path, name, "wallet_balance_history.npy"),
                self.test_env_instance.wallet_balance_history,
            )
            np.save(
                os.path.join(self.test_path, name, "unrealized_pnl_history.npy"),
                self.test_env_instance.unrealized_pnl_history,
            )
            np.save(
                os.path.join(self.test_path, name, "maintain_marigine_history.npy"),
                self.test_env_instance.maintain_marigine_history,
            )
            np.save(
                os.path.join(
                    self.test_path, name, "new_position_required_money_history.npy"
                ),
                self.test_env_instance.new_position_required_money_history,
            )
            # TODO check the length
            require_money = calculate_required_money(
                np.array(self.test_env_instance.initial_margin_history),
                np.array(self.test_env_instance.maintain_marigine_history),
                np.array(self.test_env_instance.new_position_required_money_history),
                np.array(self.test_env_instance.unrealized_pnl_history),
                np.array(self.test_env_instance.wallet_balance_history),
            )
            reward_sum = np.sum(reward_history)
            self.return_rate = reward_sum / (require_money + 1e-12)
            trading_info = {
                "return rate": self.return_rate,
                "required_money": require_money,
                "reward_seq": reward_history,
            }
            np.save(
                os.path.join(self.test_path, name, "trading_info.npy"), trading_info
            )
        return self.return_rate


if __name__ == "__main__":
    args = parser.parse_args()
    seed_torch(args.seed)
    rule_base_trader_instance = macd_trader(args)
    rule_base_trader_instance.test()
    print(f"return rate: {rule_base_trader_instance.return_rate}")
