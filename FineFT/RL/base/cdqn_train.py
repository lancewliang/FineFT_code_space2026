# Code reference: https://github.com/Lizhi-sjtu/DRL-code-pytorch/tree/main/3.Rainbow_DQN

import sys

sys.path.append(".")
import os
from torch.utils.tensorboard import SummaryWriter
from RL.util.replay_buffer_DQN import Multi_step_ReplayBuffer_multi_info
import random
from tqdm import tqdm
import argparse
from model.low_level import CQnet
import numpy as np
import torch
from torch import nn
import yaml
import pandas as pd
from env.env_initiate.simple_initiate import initiate_rolling_env
import copy
import math
from env.env_class.futures_util import (
    map_action_to_position_leverage,
)

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"

parser = argparse.ArgumentParser()

parser.add_argument(
    "--buffer_size",
    type=int,
    default=100000,
    help="the number of transcation we store in one memory",
)
parser.add_argument(
    "--n_step",
    type=int,
    default=1,
    help="the number of step we have in the td error and replay buffer",
)
# RL & trading setting
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

parser.add_argument(
    "--rolling_window_length",
    type=int,
    default=30,
    help="window length",
)

# network setting
parser.add_argument(
    "--hidden_nodes",
    type=int,
    default=128,
    help="the number of the hidden nodes",
)
# RL training coffient

parser.add_argument(
    "--batch_size",
    type=int,
    default=64,
    help="the number of transcation we learn at a time",
)
parser.add_argument("--update_times", type=int, default=1, help="the update times")
parser.add_argument("--gamma", type=float, default=1, help="the learning rate")
parser.add_argument(
    "--epsilon_init",
    type=float,
    default=1,
    help="the coffient for decay",
)
parser.add_argument(
    "--epsilon_min",
    type=float,
    default=0.01,
    help="the coffient for decay",
)
parser.add_argument(
    "--epsilon_decay_rate",
    type=float,
    default=0.00025,
    help="the coffient for decay",
)
parser.add_argument(
    "--update_frequency",
    type=int,
    default=128,
    help="the coffient for decay",
)

# general learning setting
parser.add_argument("--lr", type=float, default=0.00025, help="the learning rate")

parser.add_argument(
    "--num_sample",
    type=int,
    default=200,
    help="the overall number of sampling",
)

# log setting
parser.add_argument(
    "--result_path",
    type=str,
    default="result/base",
    help="the path for storing the test result",
)
parser.add_argument(
    "--seed",
    type=int,
    default=12345,
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


class CDQN_rp(object):
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
            "cdqn_rp",
            args.dataset_name,
            "seed_{}".format(self.seed),
        )
        self.log_path = os.path.join(self.model_path, "log")
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.writer = SummaryWriter(self.log_path)
        self.grad_clip = 0.1

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
        self.rolling_window_length = args.rolling_window_length
        # RL setting
        self.update_counter = 0
        self.batch_size = args.batch_size
        self.update_times = args.update_times
        self.gamma = args.gamma
        self.epsilon_init = args.epsilon_init
        self.epsilon_min = args.epsilon_min
        self.epsilon_decay = args.epsilon_decay_rate
        self.epsilon = self.epsilon_init
        self.target_freq_list = [5000, 8000, 10000, 20000]
        self.update_freq = args.update_frequency
        # replay buffer setting
        self.n_step = args.n_step
        self.buffer_size = args.buffer_size

        # general learning setting

        self.lr = args.lr
        self.num_sample = args.num_sample

        # network & loss function
        self.hidden_nodes = args.hidden_nodes
        self.N_ACTIONS = (self.position_choices - 1) * len(self.leverage_choices) + 1
        self.eval_net = CQnet(
            N_STATES=len(self.tech_indicator_list),
            N_ACTIONS=self.N_ACTIONS,
            hidden_nodes=self.hidden_nodes,
            kernel_size=self.rolling_window_length,
            TIME_INFO_DIM=2,
        ).to(
            self.device
        )  # 利用Net创建两个神经网络: 评估网络和目标网络
        self.target_net = copy.deepcopy(self.eval_net)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()
        self.target_freq = self.target_freq_list[0]

    def update(
        self,
        states: torch.tensor,
        info: dict,
        actions: torch.tensor,
        rewards: torch.tensor,
        next_states: torch.tensor,
        info_: dict,
        dones: torch.tensor,
    ):
        # TD error
        b = states.shape[0]
        # input
        previous_action = info["previous_action"].float().unsqueeze(1)
        avaliable_action = info["avaliable_action"]
        hour_count_down = info["funding_count_down_hour"].float().unsqueeze(1)
        minute_count_down = info["funding_count_down_minute"].float().unsqueeze(1)
        time_input = torch.cat([hour_count_down, minute_count_down], dim=1).to(
            self.device
        )

        # next input
        states_ = next_states
        previous_action_ = info_["previous_action"].float().unsqueeze(1)
        avaliable_action_ = info_["avaliable_action"]
        hour_count_down_ = info_["funding_count_down_hour"].float().unsqueeze(1)
        minute_count_down_ = info_["funding_count_down_minute"].float().unsqueeze(1)
        time_input_ = torch.cat([hour_count_down_, minute_count_down_], dim=1).to(
            self.device
        )

        q_eval = self.eval_net(
            state=states,
            time=time_input,
            previous_action=previous_action,
            available_action=avaliable_action,
        ).gather(1, actions)
        q_next = self.target_net(
            state=states_,
            time=time_input_,
            previous_action=previous_action_,
            available_action=avaliable_action_,
        ).detach()
        # since investigating is a open end problem, we do not use the done here to update
        q_target = rewards + torch.max(q_next, 1)[0].view(self.batch_size, 1) * (
            1 - dones
        )

        td_error = self.loss_func(q_eval, q_target)
        # KL divergence

        loss = td_error
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), self.grad_clip)
        self.optimizer.step()
        if self.update_counter > self.target_freq:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            self.update_counter = 0
            self.target_freq = self.target_freq_list[
                random.choices(
                    range(len(self.probability_list)),
                    weights=self.probability_list,
                    k=1,
                )[0]
            ]

        return (
            td_error.cpu(),
            torch.mean(q_eval.cpu()),
            torch.mean(q_target.cpu()),
            torch.mean(rewards.cpu()),
            torch.std(rewards.cpu()),
        )

    def act(self, state, info, epsilon):

        if np.random.uniform() > epsilon:
            state = np.copy(state)  #
            state.setflags(write=True)
            state = torch.unsqueeze(
                torch.FloatTensor(state).reshape(self.rolling_window_length, -1), 0
            ).to(self.device)
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
                available_action=avaliable_action,
            )
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            action = action[0]
        else:
            action = np.random.choice(info["avaiable_action_list"])
        return action

    def act_test(self, state, info):

        state = np.copy(state)  #
        state.setflags(write=True)
        state = torch.unsqueeze(
            torch.FloatTensor(state).reshape(self.rolling_window_length, -1), 0
        ).to(self.device)
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

    def train(self):
        step_counter_normal = 0
        c = 5000
        epoch_return_rate_train_list = []
        epoch_final_balance_train_list = []
        epoch_required_money_train_list = []
        epoch_reward_sum_train_list = []
        # epoch_number = int(len(self.train_df) / self.chunk_length)
        epoch_number = 4
        random_position_list = random.choices(range(self.N_ACTIONS), k=self.num_sample)
        df_number = len(os.listdir(self.train_data_path)) - 1
        df_index_list = random.choices(range(df_number), k=self.num_sample)

        replay_buffer = Multi_step_ReplayBuffer_multi_info(
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            device=self.device,
            seed=self.seed,
            gamma=self.gamma,
            n_step=self.n_step,
        )
        reversed_list = self.target_freq_list[::-1]
        self.probability_list = np.exp(np.array(reversed_list) / c) / np.sum(
            np.exp(np.array(reversed_list) / c)
        )
        self.target_freq = self.target_freq_list[
            random.choices(
                range(len(self.probability_list)), weights=self.probability_list, k=1
            )[0]
        ]

        for sample in range(self.num_sample):
            if sample < math.ceil(self.num_sample / 3):
                reversed_list = self.target_freq_list[::-1]
                self.probability_list = np.exp(np.array(reversed_list) / c) / np.sum(
                    np.exp(np.array(reversed_list) / c)
                )
            if sample >= math.ceil(self.num_sample / 3) and sample < math.ceil(
                self.num_sample / 3 * 2
            ):
                self.probability_list = [1 / len(self.target_freq_list)] * len(
                    self.target_freq_list
                )
            if sample >= math.ceil(self.num_sample / 3 * 2):
                self.probability_list = np.exp(
                    np.array(self.target_freq_list) / c
                ) / np.sum(np.exp(np.array(self.target_freq_list) / c))

            df_index = df_index_list[sample]
            print("we are training with ", df_index)
            self.train_df = pd.read_feather(
                os.path.join(self.train_data_path, "df_{}.feather".format(df_index))
            )
            initial_action = random_position_list[sample]
            self.initial_position, self.initial_leverage = (
                map_action_to_position_leverage(
                    initial_action, self.leverage_choices, self.position_list
                )
            )
            print(
                "the initial position and leverage are {} and {}".format(
                    self.initial_position, self.initial_leverage
                )
            )
            current_markprice = self.train_df["mark_price"].values[
                self.rolling_window_length-1
            ]
            self.initial_margin = np.abs(
                self.initial_position * current_markprice / self.initial_leverage
            )
            self.initial_state = (
                self.initial_wallet_balance,
                self.initial_margin,
                self.initial_unrealized_pnL,
                self.initial_position,
                self.initial_leverage,
            )
            train_env = initiate_rolling_env(
                df=self.train_df,
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
                window_size=self.rolling_window_length,
            )
            s, info = train_env.reset()

            episode_reward_sum = 0
            while True:
                a = self.act(s, info, self.epsilon)

                s_, r, done, info_ = train_env.step(a)

                replay_buffer.add(s, info, a, r, s_, info_, done)
                episode_reward_sum += r
                step_counter_normal += 1
                self.update_counter += 1
                s, info = s_, info_

                if (
                    step_counter_normal > (self.batch_size + self.n_step)
                    and step_counter_normal % self.update_freq == 1
                ):
                    self.epsilon = (
                        self.epsilon * (1 - self.epsilon_decay)
                        if self.epsilon * (1 - self.epsilon_decay) > self.epsilon_min
                        else self.epsilon_min
                    )
                    for _ in range(self.update_times):
                        (
                            states,
                            infos,
                            actions,
                            rewards,
                            next_states,
                            next_infos,
                            dones,
                        ) = replay_buffer.sample()
                        (
                            td_error,
                            q_eval,
                            q_target,
                            rewards_mean,
                            rewards_std,
                        ) = self.update(
                            states,
                            infos,
                            actions,
                            rewards,
                            next_states,
                            next_infos,
                            dones,
                        )

                        self.writer.add_scalar(
                            tag="td_error",
                            scalar_value=td_error,
                            global_step=self.update_counter,
                            walltime=None,
                        )
                        self.writer.add_scalar(
                            tag="q_eval",
                            scalar_value=q_eval,
                            global_step=self.update_counter,
                            walltime=None,
                        )
                        self.writer.add_scalar(
                            tag="q_target",
                            scalar_value=q_target,
                            global_step=self.update_counter,
                            walltime=None,
                        )
                        self.writer.add_scalar(
                            tag="rewards_mean",
                            scalar_value=rewards_mean,
                            global_step=self.update_counter,
                            walltime=None,
                        )
                        self.writer.add_scalar(
                            tag="rewards_std",
                            scalar_value=rewards_std,
                            global_step=self.update_counter,
                            walltime=None,
                        )
                if done:
                    break

            final_balance = train_env.unrealized_pnl + train_env.wallet_balance
            required_money = self.initial_wallet_balance
            self.writer.add_scalar(
                tag="return_rate_train",
                scalar_value=final_balance / (required_money + 1e-12) - 1,
                global_step=sample,
                walltime=None,
            )

            self.writer.add_scalar(
                tag="reward_sum_train",
                scalar_value=episode_reward_sum,
                global_step=sample,
                walltime=None,
            )
            epoch_return_rate_train_list.append(
                final_balance / (required_money + 1e-12)
            )
            epoch_final_balance_train_list.append(final_balance)
            epoch_required_money_train_list.append(required_money)
            epoch_reward_sum_train_list.append(episode_reward_sum)
            if len(epoch_final_balance_train_list) == epoch_number:
                epoch_index = int((sample + 1) / epoch_number)
                mean_return_rate_train = np.mean(epoch_return_rate_train_list)
                mean_final_balance_train = np.mean(epoch_final_balance_train_list)
                mean_required_money_train = np.mean(epoch_required_money_train_list)
                mean_reward_sum_train = np.mean(epoch_reward_sum_train_list)
                self.writer.add_scalar(
                    tag="epoch_return_rate_train",
                    scalar_value=mean_return_rate_train,
                    global_step=epoch_index,
                    walltime=None,
                )
                self.writer.add_scalar(
                    tag="epoch_final_balance_train",
                    scalar_value=mean_final_balance_train,
                    global_step=epoch_index,
                    walltime=None,
                )
                self.writer.add_scalar(
                    tag="epoch_required_money_train",
                    scalar_value=mean_required_money_train,
                    global_step=epoch_index,
                    walltime=None,
                )
                self.writer.add_scalar(
                    tag="epoch_reward_sum_train",
                    scalar_value=mean_reward_sum_train,
                    global_step=epoch_index,
                    walltime=None,
                )
                epoch_path = os.path.join(
                    self.model_path, "epoch_{}".format(epoch_index)
                )
                if not os.path.exists(epoch_path):
                    os.makedirs(epoch_path)
                torch.save(
                    self.eval_net.state_dict(),
                    os.path.join(epoch_path, "trained_model.pkl"),
                )
                epoch_return_rate_train_list = []
                epoch_final_balance_train_list = []
                epoch_required_money_train_list = []
                epoch_reward_sum_train_list = []


if __name__ == "__main__":
    args = parser.parse_args()
    agent = CDQN_rp(args)
    agent.train()
