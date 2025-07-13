# Code reference: https://github.com/Lizhi-sjtu/DRL-code-pytorch/tree/main/3.Rainbow_DQN

import sys

sys.path.append(".")
import os
from torch.utils.tensorboard import SummaryWriter
from RL.util.replay_buffer_DQN import Multi_step_ReplayBuffer_multi_info
from env.env_initiate.agg_initiate import initiate_high_level_earnhft_env
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
# replay buffer coffient
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
# RL training
parser.add_argument(
    "--seed",
    type=int,
    default=12345,
    help="the random seed for training and sample",
)

parser.add_argument(
    "--reward_scale",
    type=float,
    default=1,
    help="the scale factor we put in reward",
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
    "--tau", type=float, default=0.005, help="soft update the target network"
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=512,
    help="the number of transcation we learn at a time",
)
parser.add_argument("--update_times", type=int, default=1, help="the update times")
parser.add_argument("--gamma", type=float, default=1, help="the learning rate")
parser.add_argument(
    "--epsilon_init",
    type=float,
    default=0.9,
    help="the coffient for decay",
)
parser.add_argument(
    "--epsilon_min",
    type=float,
    default=0.1,
    help="the coffient for decay",
)
parser.add_argument(
    "--epsilon_step",
    type=float,
    default=5e3,
    help="the coffient for decay",
)
parser.add_argument(
    "--target_freq",
    type=int,
    default=512,
    help="the number of sampling during one epoch",
)
# general learning setting
parser.add_argument("--lr_init", type=float, default=2e-3, help="the learning rate")
parser.add_argument("--lr_min", type=float, default=1e-4, help="the learning rate")
parser.add_argument("--lr_step", type=float, default=1e3, help="the learning rate")
parser.add_argument(
    "--num_sample",
    type=int,
    default=100,
    help="the overall number of sampling",
)
parser.add_argument(
    "--result_path",
    type=str,
    default="result/EarnHFT/high_level",
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


class DQN(object):
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
            args.dataset_name,
            "seed_{}".format(self.seed),
        )
        self.log_path = os.path.join(self.model_path, "log")
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.writer = SummaryWriter(self.log_path)
        # env
        self.base_path = args.base_path
        self.dataset_name = args.dataset_name
        self.train_data_path = os.path.join(
            self.base_path, self.dataset_name, "train.feather"
        )
        self.tech_indicator_list = np.load(
            os.path.join(self.base_path, self.dataset_name, "state_features.npy")
        )
        self.high_level_feature_list = np.load(
            os.path.join(
                self.base_path, self.dataset_name, "high_level_state_features.npy"
            )
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
        # RL setting
        self.update_counter = 0
        self.grad_clip = 0.01
        self.tau = args.tau
        self.batch_size = args.batch_size
        self.update_times = args.update_times
        self.gamma = args.gamma
        self.epsilon_init = args.epsilon_init
        self.epsilon_min = args.epsilon_min
        self.epsilon_step = args.epsilon_step
        self.epsilon_decay = (self.epsilon_init - self.epsilon_min) / self.epsilon_step
        self.epsilon = self.epsilon_init
        self.target_freq = args.target_freq
        # replay buffer setting
        self.n_step = args.n_step
        self.buffer_size = args.buffer_size

        # general learning setting
        self.lr_init = args.lr_init
        self.lr_min = args.lr_min
        self.lr_step = args.lr_step
        self.lr_decay = (self.lr_init - self.lr_min) / self.lr_step
        self.lr = self.lr_init
        self.num_sample = args.num_sample
        # data

        self.n_state = len(self.high_level_feature_list)
        # network & loss function
        self.hidden_nodes = args.hidden_nodes
        self.eval_net = Qnet_high_level_position(
            self.n_state,
            self.dynamics_num,
            self.hidden_nodes,
        ).to(
            self.device
        )  # 利用Net创建两个神经网络: 评估网络和目标网络
        self.target_net = copy.deepcopy(self.eval_net)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

    def update(
        self,
        info: dict,
        actions: torch.tensor,
        rewards: torch.tensor,
        info_: dict,
        dones: torch.tensor,
    ):
        print("udpate")
        # TD error
        q_eval = self.eval_net(
            torch.squeeze(info["high_level_state"]),
            info["previous_action"].float().unsqueeze(1),
        ).gather(1, actions)
        q_next = self.target_net(
            torch.squeeze(info_["high_level_state"]),
            info["previous_action"].float().unsqueeze(1),
        ).detach()
        # since investigating is a open end problem, we do not use the done here to update
        q_target = (
            rewards
            + torch.max(q_next, 1)[0].view(self.batch_size, 1)
            * (1 - dones)
            * self.gamma
        )

        td_error = self.loss_func(q_eval, q_target)

        # final loss function
        loss = td_error
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), self.grad_clip)
        self.optimizer.step()
        for param, target_param in zip(
            self.eval_net.parameters(), self.target_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        self.update_counter += 1

        return (
            td_error.cpu(),
            torch.mean(q_eval.cpu()),
            torch.mean(q_target.cpu()),
            torch.mean(rewards.cpu()),
            torch.std(rewards.cpu()),
        )

    def act(self, info, epsilon):
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

        if np.random.uniform() > epsilon:
            actions_value = self.eval_net.forward(x, position)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            action = action[0]
        else:
            action = random.choice(range(self.dynamics_num))
        return action

    def act_test(self, info):
        x = torch.tensor(info["high_level_state"]).to(self.device).to(torch.float32)
        position = (
            torch.tensor(info["previous_action"]).to(self.device).to(torch.float32)
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

    def train(self):
        epoch_return_rate_train_list = []
        epoch_final_balance_train_list = []
        epoch_required_money_train_list = []
        epoch_reward_sum_train_list = []
        # epoch_number = int(len(self.train_df) / self.chunk_length)
        epoch_number = 1
        return_rate_list = []
        replay_buffer = Multi_step_ReplayBuffer_multi_info(
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            device=self.device,
            seed=self.seed,
            gamma=self.gamma,
            n_step=self.n_step,
        )
        step_counter = 0
        initial_action_list = random.choices(
            range((len(self.position_list) - 1) * len(self.leverage_choices) + 1),
            k=self.num_sample,
        )
        for sample in range(self.num_sample):
            self.train_df = pd.read_feather(self.train_data_path)
            initial_action = initial_action_list[sample]
            self.initial_position, self.initial_leverage = (
                self.map_action_to_position_leverage(initial_action)
            )
            print(
                "initial_position is ",
                self.initial_position,
                "self.initial_leverage is ",
                self.initial_leverage,
            )
            current_markprice = self.train_df["mark_price"].values[0]
            self.initial_margin = np.abs(
                self.initial_position * current_markprice / self.initial_leverage
            )
            train_env = initiate_high_level_earnhft_env(
                df=self.train_df,
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
                time_info_dim=2,
            )
            s, info = train_env.reset()
            episode_reward_sum = 0
            while True:
                a = self.act(info, self.epsilon)
                self.epsilon = (
                    self.epsilon - self.epsilon_decay
                    if self.epsilon - self.epsilon_decay > self.epsilon_min
                    else self.epsilon_min
                )
                self.lr = (
                    self.lr - self.lr_decay
                    if self.lr - self.lr_decay > self.lr_min
                    else self.lr_min
                )
                for p in self.optimizer.param_groups:
                    p["lr"] = self.lr

                s_, r, done, info_ = train_env.step(a)
                if not done:
                    replay_buffer.add(s, info, a, r, s_, info_, done)
                episode_reward_sum += r

                s, info = s_, info_
                step_counter += 1
                if (
                    step_counter > (self.batch_size + self.n_step)
                    and step_counter % self.target_freq == 1
                ):
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
                            infos,
                            actions,
                            rewards,
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
            final_balance, required_money = (
                train_env.unrealized_pnl + train_env.wallet_balance,
                1e5,
            )
            self.writer.add_scalar(
                tag="return_rate_train",
                scalar_value=final_balance / (required_money + 1e-12) - 1,
                global_step=sample,
                walltime=None,
            )
            self.writer.add_scalar(
                tag="final_balance_train",
                scalar_value=final_balance,
                global_step=sample,
                walltime=None,
            )
            self.writer.add_scalar(
                tag="required_money_train",
                scalar_value=required_money,
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
                # self.test(epoch_path)
                epoch_return_rate_train_list = []
                epoch_final_balance_train_list = []
                epoch_required_money_train_list = []
                epoch_reward_sum_train_list = []


if __name__ == "__main__":
    args = parser.parse_args()
    agent = DQN(args)
    agent.train()
