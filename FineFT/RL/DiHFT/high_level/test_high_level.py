# the frequency of the high level agent is the same as the low level agent
# based on a sequency of high levelimport pandas as pd
import numpy as np
import pandas as pd
import os
import sys
import random
import argparse
import torch
import sys
from torch import nn

sys.path.append(".")
import copy
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Categorical
import torch.nn.functional as F
from env.env_initiate.base_initiate import initiate_base_env
from collections import deque
from env.env_class.futures_util import (
    map_action_to_position_leverage,
    map_position_leverage_to_action,
)
import os
from model.low_level import ensemble_Qnet
from model.high_level import RankBasedQNetwork
from RL.util.update import disable_gradients, get_rank
from analysis.calculate_metric.calculate_metric import calculate_differences

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"
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
# low level network setting
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
parser.add_argument(
    "--low_level_path",
    type=str,
    default="result/DiHFT/potential_model",
    help="context number",
)
# high level network setting
parser.add_argument(
    "--TCN_channels",
    type=list,
    default=[1, 2],
    help="context number",
)
parser.add_argument(
    "--seq_length",
    type=int,
    default=288,
    help="context number",
)
parser.add_argument(
    "--epoch_number",
    type=int,
    default=1,
    help="epoch number of the high level agent",
)
parser.add_argument(
    "--result_path",
    type=str,
    default="result/DiHFT/high_level",
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


class high_level_trader:
    def __init__(self, args) -> None:
        # device
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.model_path = os.path.join(
            args.result_path, args.dataset_name, "td_indicator"
        )
        self.high_level_path = os.path.join(
            self.model_path, "epoch_{}/trained_model.pkl".format(args.epoch_number)
        )
        self.test_path = os.path.join(
            self.model_path, "epoch_{}".format(args.epoch_number), "test"
        )
        if not os.path.exists(self.test_path):
            os.makedirs(self.test_path)
        # high level input sequence length
        self.seq_length = args.seq_length
        # trading environment setting
        self.base_path = args.base_path
        self.dataset_name = args.dataset_name
        self.test_data_path = os.path.join(
            self.base_path, self.dataset_name, "test.feather"
        )
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
        self.initial_action = map_position_leverage_to_action(
            self.initial_position,
            self.initial_leverage,
            self.leverage_choices,
            self.position_list,
        )

        # low-level network
        self.time_info_dim = args.time_info_dim
        self.hidden_nodes = args.hidden_nodes
        self.N = args.N
        self.N_ACTIONS = (self.position_choices - 1) * len(self.leverage_choices) + 1
        self.low_level_network = ensemble_Qnet(
            N_STATES=len(self.tech_indicator_list),
            N_ACTIONS=self.N_ACTIONS,
            hidden_nodes=self.hidden_nodes,
            TIME_INFO_DIM=self.time_info_dim,
            ensemble_number=self.N,
        ).to(self.device)
        self.low_level_path = args.low_level_path
        self.low_level_network.load_state_dict(
            torch.load(
                os.path.join(
                    self.low_level_path,
                    self.dataset_name,
                    "single_agent_selection",
                    "model.pth",
                )
            )
        )
        self.low_level_network.to(self.device)
        self.low_level_network.eval()
        disable_gradients(self.low_level_network)
        # high-level network
        self.TCN_channels = args.TCN_channels
        self.high_level_network_eval = RankBasedQNetwork(
            input_dim=self.N,
            num_channels=self.TCN_channels,
            output_dim=self.N,
        ).to(self.device)
        self.high_level_network_eval.load_state_dict(torch.load(self.high_level_path))
        disable_gradients(self.high_level_network_eval)
        self.high_level_network_eval.eval()
        # state setting
        self.seq_rank = deque(maxlen=self.seq_length)

    def calculate_rollout_transition_rank(self, s, info, a, r, s_, info_):
        # calculate single step transition rank
        evaluate_q = self.calculate_low_level_q_value(s, info)
        evaluate_q = evaluate_q[:, :, a]
        target_q = self.calculate_low_level_q_value(s_, info_) + r
        target_q = torch.max(target_q, dim=2)[0]
        assert evaluate_q.shape == target_q.shape
        td_error = torch.abs(evaluate_q - target_q).detach()
        assert td_error.shape == (1, self.N)
        rank = get_rank(td_error, td_error.device)
        return rank

    def store_seq_length_rank(self, state):
        self.seq_rank.append(state)
        if len(self.seq_rank) == self.seq_length:
            return True
        else:
            return False

    def reset_seq_length_rank(self):
        self.seq_rank.clear()
        return self.seq_rank

    def initial_rollout(self, initial_action, env, s, info):
        reward_sum = 0
        for i in range(self.seq_length):
            trading_action = initial_action
            s_, r, done, info_ = env.step(trading_action)
            rank = self.calculate_rollout_transition_rank(
                s, info, trading_action, r, s_, info_
            )
            self.store_seq_length_rank(rank.detach().cpu().numpy())
            s, info = s_, info_
            reward_sum += r
        return rank, s, info, reward_sum

    def get_seq_length_rank(self):
        seq_rank = np.array(self.seq_rank).transpose(1, 0, 2)
        return seq_rank

    def calculate_low_level_q_value(self, state, info):
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
        actions_value = self.low_level_network(
            state=state,
            time=time_input,
            previous_action=previous_action,
            avaliable_action=avaliable_action,
        )

        return actions_value

    def low_level_act_test(self, state, info, context_index):
        assert context_index in range(self.N)
        actions_value = self.calculate_low_level_q_value(state, info)
        action_value_chosen_index = actions_value[:, context_index, :]
        action = torch.max(action_value_chosen_index, 1)[1].data.cpu().numpy()
        action = action[0]
        return action

    def choose_context_test(self, rank_seq):
        rank_seq = torch.tensor(rank_seq).to(self.device)
        q_values = self.high_level_network_eval(rank_seq.float())
        context_index = torch.max(q_values, 1)[1].data.cpu().numpy()
        context_index = context_index[0]

        return context_index

    def test(self):
        self.df = pd.read_feather(self.test_data_path)
        env = initiate_base_env(
            df=self.df,
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
        )
        s, info = env.reset()
        episode_reward_sum = 0
        rank, s, info, r = self.initial_rollout(self.initial_action, env, s, info)
        while True:
            episode_reward_sum += r
            high_level_state = np.array(self.get_seq_length_rank())
            context_index = self.choose_context_test(high_level_state)
            # the high level state is (1, seq_length, N)
            info["high_level_state"] = high_level_state[0]
            trading_action = self.low_level_act_test(s, info, context_index)
            s_, r, done, info_ = env.step(trading_action)
            rank = self.calculate_rollout_transition_rank(
                s, info, trading_action, r, s_, info_
            )
            self.store_seq_length_rank(rank.detach().cpu().numpy())
            high_level_state_ = np.array(self.get_seq_length_rank())
            s, info = s_, info_
            if done:
                break
        total_asset_history = env.margine_balance_history
        reward_history = calculate_differences(total_asset_history)
        micro_action_history = env.micro_action_history
        micro_action_history = env.micro_action_history
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
            env.initial_margin_history,
        )
        np.save(
            os.path.join(self.test_path, "wallet_balance_history.npy"),
            env.wallet_balance_history,
        )
        np.save(
            os.path.join(self.test_path, "unrealized_pnl_history.npy"),
            env.unrealized_pnl_history,
        )
        np.save(
            os.path.join(self.test_path, "maintain_marigine_history.npy"),
            env.maintain_marigine_history,
        )
        np.save(
            os.path.join(self.test_path, "new_position_required_money_history.npy"),
            env.new_position_required_money_history,
        )


if __name__ == "__main__":
    args = parser.parse_args()
    ppo = high_level_trader(args)
    ppo.test()
