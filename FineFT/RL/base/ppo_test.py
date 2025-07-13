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

# network
parser.add_argument(
    "--hidden_nodes",
    type=int,
    default=128,
    help="The number of neurons in hidden layers of the neural network",
)
parser.add_argument(
    "--epoch_number",
    type=int,
    default=1,
    help="The number of neurons in hidden layers of the neural network",
)

# RL & trading setting

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


class PPO_discrete_RNN:
    def __init__(self, args):
        self.seed = args.seed
        seed_torch(self.seed)
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # log path
        self.model_path = os.path.join(
            args.result_path,
            "ppo",
            args.dataset_name,
            "seed_{}".format(self.seed),
        )

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

        # network
        self.hidden_nodes = args.hidden_nodes
        self.n_state = len(self.tech_indicator_list)
        self.N_ACTIONS = (self.position_choices - 1) * len(self.leverage_choices) + 1
        self.actor = Actor(
            N_STATES=self.n_state,
            hidden_nodes=self.hidden_nodes,
            N_ACTIONS=self.N_ACTIONS,
        ).to(self.device)
        self.test_path = os.path.join(
            self.model_path, "epoch_{}".format(args.epoch_number)
        )
        self.load_model(self.test_path)
        # data
        self.valid_data_path = os.path.join(
            "dataset", self.dataset_name, "valid.feather"
        )
        self.test_data_path = os.path.join("dataset", self.dataset_name, "test.feather")

    def reset_rnn_hidden(self):
        self.ac.actor_rnn_hidden = None
        self.ac.critic_rnn_hidden = None

    def choose_action(self, s, info, evaluate=True):
        with torch.no_grad():
            s = torch.tensor(s, dtype=torch.float).unsqueeze(0).to(self.device)
            previous_action = torch.unsqueeze(
                torch.tensor([info["previous_action"]]).float().to(self.device), 0
            ).to(self.device)
            avaliable_action = torch.unsqueeze(
                torch.tensor(info["avaliable_action"]).to(self.device), 0
            ).to(self.device)
            hour_count_down = torch.unsqueeze(
                torch.tensor([info["funding_count_down_hour"]]).float().to(self.device),
                0,
            ).to(self.device)
            minute_count_down = torch.unsqueeze(
                torch.tensor([info["funding_count_down_minute"]])
                .float()
                .to(self.device),
                0,
            ).to(self.device)

            time_input = torch.cat([hour_count_down, minute_count_down], dim=1).to(
                self.device
            )
            logit = self.actor(s, time_input, previous_action, avaliable_action)
            if evaluate:
                a = torch.argmax(logit)
                return a.item(), None
            else:
                dist = Categorical(logits=logit)
                a = dist.sample()
                a_logprob = dist.log_prob(a)
                return a.item(), a_logprob.item()

    def load_model(self, epoch_path):
        self.actor.load_state_dict(
            torch.load(os.path.join(epoch_path, "trained_model.pkl"))
        )

    def test(self):
        for name, data_path in zip(
            ["valid", "test"], [self.valid_data_path, self.test_data_path]
        ):
            action_list = []
            self.test_df = pd.read_feather(data_path)
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
            action_list = []
            while not done:
                a, _ = self.choose_action(s, info)
                s_, r, done, info_ = self.test_env_instance.step(a)
                s = s_
                info = info_
                action_list.append(a)
            action_list = np.array(action_list)
            if not os.path.exists(os.path.join(self.test_path, name)):
                os.makedirs(os.path.join(self.test_path, name))
            np.save(os.path.join(self.test_path, name, "action.npy"), action_list)
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
    agent = PPO_discrete_RNN(args)
    agent.test()
