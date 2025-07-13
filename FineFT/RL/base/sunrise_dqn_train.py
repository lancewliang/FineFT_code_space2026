# Code reference: https://github.com/Lizhi-sjtu/DRL-code-pytorch/tree/main/3.Rainbow_DQN

import sys

sys.path.append(".")
import os
from torch.utils.tensorboard import SummaryWriter
from RL.util.replay_buffer_DQN import Multi_step_ReplayBuffer_multi_info_sunrise
import random
from tqdm import tqdm
import argparse
from model.low_level import Qnet
import numpy as np
import torch
from torch import nn
import yaml
import pandas as pd
from env.env_initiate.simple_initiate import initiate_simple_env
import copy
from analysis.calculate_metric.calculate_metric import (
    calculate_differences,
    calculate_required_money,
)
from env.env_class.futures_util import (
    map_action_to_position_leverage,
)

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"

parser = argparse.ArgumentParser()
# path
parser.add_argument(
    "--beta",
    type=float,
    default=0.5,
    help="the number of transcation we store in one memory",
)
parser.add_argument(
    "--seed",
    type=int,
    default=12345,
    help="the random seed for training and sample",
)
parser.add_argument(
    "--buffer_size",
    type=int,
    default=1000000,
    help="the number of transcation we store in one memory",
)
parser.add_argument(
    "--n_step",
    type=int,
    default=1,
    help="the number of step we have in the td error and replay buffer",
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


# network setting
parser.add_argument(
    "--ensemble_num",
    type=int,
    default=5,
    help="the number of the hidden nodes",
)
parser.add_argument(
    "--lamda",
    type=float,
    default=0.1,
    help="the number of the hidden nodes",
)
parser.add_argument(
    "--hidden_nodes",
    type=int,
    default=128,
    help="the number of the hidden nodes",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=512,
    help="the number of transcation we learn at a time",
)
parser.add_argument("--update_times", type=int, default=1, help="the update times")
parser.add_argument(
    "--gamma", type=float, default=0.9, help="the gamma for decay reward"
)

parser.add_argument(
    "--target_freq",
    type=int,
    default=512,
    help="the number of sampling during one epoch",
)
# general learning setting
parser.add_argument("--temperature", type=float, default=2, help="the learning rate")

parser.add_argument("--lr", type=float, default=5e-4, help="the learning rate")
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
    "--tau", type=float, default=0.005, help="soft update the target network"
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


class Sunrise_DQN(object):
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
            "sunrise_dqn",
            args.dataset_name,
            "seed_{}".format(self.seed),
        )
        self.log_path = os.path.join(self.model_path, "log")
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.writer = SummaryWriter(self.log_path)
        self.grad_clip = 5
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
        # RL setting
        self.update_counter = 0
        self.batch_size = args.batch_size
        self.update_times = args.update_times
        self.gamma = args.gamma

        # replay buffer setting
        self.n_step = args.n_step
        self.buffer_size = args.buffer_size

        # general learning setting

        self.lr = args.lr

        self.num_sample = args.num_sample

        # network & loss function
        self.ensemble_num = args.ensemble_num
        self.hidden_nodes = args.hidden_nodes
        self.N_ACTIONS = (self.position_choices - 1) * len(self.leverage_choices) + 1
        self.eval_net_list = [
            Qnet(
                N_STATES=len(self.tech_indicator_list),
                N_ACTIONS=self.N_ACTIONS,
                hidden_nodes=self.hidden_nodes,
                TIME_INFO_DIM=2,
            ).to(self.device)
            for _ in range(self.ensemble_num)
        ]  # 利用Net创建两个神经网络: 评估网络和目标网络
        self.target_net_list = copy.deepcopy(self.eval_net_list)
        self.optimizer_list = [
            torch.optim.Adam(eval_net.parameters(), lr=self.lr)
            for eval_net in self.eval_net_list
        ]
        self.loss_func = nn.MSELoss(reduction="none")
        self.target_freq = args.target_freq
        self.beta = args.beta
        self.lamda = args.lamda
        self.temperature = args.temperature
        self.tau = args.tau

    def ucb_act(self, state, info):
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
        actions_value_list = [
            eval_net(
                state=state,
                time=time_input,
                previous_action=previous_action,
                avaliable_action=avaliable_action,
            )
            for eval_net in self.eval_net_list
        ]

        actions_value = torch.stack(actions_value_list, dim=1)
        action_value_ucb = actions_value.mean(dim=1) + self.lamda * actions_value.std(
            dim=1
        )
        # print('avaliable_action', avaliable_action)
        # print('actions_value', actions_value)
        action = torch.max(action_value_ucb, 1)[1].data.cpu().numpy()
        action = action[0]
        return action

    def update(
        self,
        mask: torch.tensor,
        states: torch.tensor,
        info: dict,
        actions: torch.tensor,
        rewards: torch.tensor,
        next_states: torch.tensor,
        info_: dict,
        dones: torch.tensor,
    ):
        # TD error
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
        q_eval_list = [
            target_net(
                state=states,
                time=time_input,
                previous_action=previous_action,
                avaliable_action=avaliable_action,
            ).gather(1, actions)
            for target_net in self.target_net_list
        ]
        q_eval_list_tensor = torch.stack(q_eval_list, dim=1)
        std_q_eval = torch.std(q_eval_list_tensor, dim=1)
        weights = (torch.sigmoid(-std_q_eval * self.temperature) + 0.5).detach()
        for index, eval_net, target_net, optimizer in zip(
            range(self.ensemble_num),
            self.eval_net_list,
            self.target_net_list,
            self.optimizer_list,
        ):

            mask_index = mask[index].to(self.device)
            if mask_index.sum() == 0:
                continue
            q_eval = eval_net(
                state=states,
                time=time_input,
                previous_action=previous_action,
                avaliable_action=avaliable_action,
            ).gather(1, actions)
            q_next = target_net(
                state=states_,
                time=time_input_,
                previous_action=previous_action_,
                avaliable_action=avaliable_action_,
            ).detach()
            q_target = (
                rewards + torch.max(q_next, 1)[0].view(self.batch_size, 1) * self.gamma
            )
            td_error = (
                self.loss_func(q_eval, q_target) * weights * mask_index
            ).sum() / mask_index.sum()
            td_error_mean = self.loss_func(q_eval, q_target).mean()

            optimizer.zero_grad()
            td_error.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(eval_net.parameters(), self.grad_clip)
            optimizer.step()
            for param, target_param in zip(
                eval_net.parameters(), target_net.parameters()
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

    def train(self):
        epoch_return_rate_train_list = []
        epoch_final_balance_train_list = []
        epoch_required_money_train_list = []
        epoch_reward_sum_train_list = []
        # epoch_number = int(len(self.train_df) / self.chunk_length)
        epoch_number = 4
        random_position_list = random.choices(range(self.N_ACTIONS), k=self.num_sample)
        df_number = len(os.listdir(self.train_data_path)) - 1
        df_index_list = random.choices(range(df_number), k=self.num_sample)

        replay_buffer_normal = Multi_step_ReplayBuffer_multi_info_sunrise(
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            device=self.device,
            seed=self.seed,
            gamma=self.gamma,
            n_step=self.n_step,
            parallel_env=1,
            beta=self.beta,
            num_ensemble=self.ensemble_num,
        )
        step_counter_normal = 0
        for sample in range(self.num_sample):
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

            train_env = initiate_simple_env(
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
            )
            s, info = train_env.reset()
            episode_reward_sum = 0
            while True:
                a = self.ucb_act(s, info)
                s_, r, done, info_ = train_env.step(a)

                replay_buffer_normal.add(s, info, a, r, s_, info_, done)
                episode_reward_sum += r

                s, info = s_, info_
                step_counter_normal += 1
                if (
                    step_counter_normal > (self.batch_size + self.n_step)
                    and step_counter_normal % self.target_freq == 1
                ):
                    for _ in range(self.update_times):
                        (
                            mask,
                            states,
                            infos,
                            actions,
                            rewards,
                            next_states,
                            next_infos,
                            dones,
                        ) = replay_buffer_normal.sample()
                        (
                            td_error,
                            q_eval,
                            q_target,
                            rewards_mean,
                            rewards_std,
                        ) = self.update(
                            mask,
                            states,
                            infos,
                            actions,
                            rewards,
                            next_states,
                            next_infos,
                            dones,
                        )
                if done:
                    break
            final_balance = train_env.unrealized_pnl + train_env.wallet_balance
            required_money = self.initial_wallet_balance
            self.writer.add_scalar(
                tag="return_rate_train",
                scalar_value=final_balance / (required_money + 1e-12),
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
            print("length", len(epoch_final_balance_train_list))
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
                for i in range(self.ensemble_num):
                    torch.save(
                        self.eval_net_list[i].state_dict(),
                        os.path.join(epoch_path, "trained_model_{}.pkl".format(i)),
                    )
                epoch_return_rate_train_list = []
                epoch_final_balance_train_list = []
                epoch_required_money_train_list = []
                epoch_reward_sum_train_list = []


if __name__ == "__main__":
    args = parser.parse_args()
    sunrise = Sunrise_DQN(args)
    sunrise.train()
