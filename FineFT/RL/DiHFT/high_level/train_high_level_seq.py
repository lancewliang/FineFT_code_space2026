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
from env.env_initiate.demo_initiate import initiate_demo_env
from collections import deque

from RL.util.update import (
    disable_gradients,
    evaluate_quantile_at_action,
    update_params,
    soft_copy_params,
    recalculate_q_demonstration,
    get_rank,
)
from env.env_class.futures_util import map_action_to_position_leverage

from RL.util.replay_buffer_DQN import Multi_step_ReplayBuffer_multi_info
import os
from model.low_level import ensemble_Qnet
from model.high_level import RankBasedQNetwork

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"
parser = argparse.ArgumentParser()
# replay buffer coffient
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
# * RL setting
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
parser.add_argument(
    "--gamma", type=float, default=0.9, help="the gamma for decay reward"
)
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
    "--epsilon_step",
    type=float,
    default=1e3,
    help="the coffient for decay",
)
parser.add_argument(
    "--rollout_steps",
    type=int,
    default=512,
    help="the number of sampling during one epoch",
)
# general learning setting
parser.add_argument("--lr_init", type=float, default=1e-2, help="the learning rate")
parser.add_argument("--lr_min", type=float, default=5e-4, help="the learning rate")
parser.add_argument("--lr_step", type=float, default=2e3, help="the learning rate")
parser.add_argument(
    "--num_sample",
    type=int,
    default=100,
    help="the overall number of sampling",
)
parser.add_argument(
    "--seed",
    type=int,
    default=12345,
    help="the overall number of sampling",
)
# log setting
parser.add_argument(
    "--result_path",
    type=str,
    default="result/DiHFT/high_level",
    help="the path for storing the test result",
)
# supervisor
parser.add_argument(
    "--ada_init",
    type=float,
    default=128,
    help="the coffient for decay",
)
parser.add_argument(
    "--ada_min",
    type=float,
    default=0,
    help="the coffient for decay",
)
parser.add_argument(
    "--ada_step",
    type=float,
    default=1e3,
    help="the coffient for decay",
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


class high_level_agent:
    def __init__(self, args):
        # seed
        self.seed = args.seed
        seed_torch(self.seed)
        # device
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        # log path
        self.model_path = os.path.join(
            args.result_path, args.dataset_name, "td_indicator_seq_ada_final_0"
        )
        self.log_path = os.path.join(self.model_path, "log")
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.writer = SummaryWriter(self.log_path)
        # sampling method
        # RL setting
        self.update_counter = 0
        self.grad_clip = 5
        self.tau = args.tau
        self.batch_size = args.batch_size
        self.update_times = args.update_times
        self.gamma = args.gamma
        self.epsilon_init = args.epsilon_init
        self.epsilon_min = args.epsilon_min
        self.epsilon_step = args.epsilon_step
        self.epsilon_decay = (self.epsilon_init - self.epsilon_min) / self.epsilon_step
        self.epsilon = self.epsilon_init
        self.rollout_steps = args.rollout_steps
        # high level input sequence length
        self.seq_length = args.seq_length
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
        # trading environment setting
        self.base_path = args.base_path
        self.dataset_name = args.dataset_name
        self.train_data_path = os.path.join(
            self.base_path, self.dataset_name, "train.feather"
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
        self.high_level_network_target = copy.deepcopy(self.high_level_network_eval)
        disable_gradients(self.high_level_network_target)
        self.optimizer = torch.optim.Adam(
            self.high_level_network_eval.parameters(), lr=self.lr
        )
        # supervisor setting
        self.ada_init = args.ada_init
        self.ada_min = args.ada_min
        self.ada_step = args.ada_step
        self.ada_decay = (self.ada_init - self.ada_min) / self.ada_step
        self.ada = self.ada_init
        # state setting
        self.seq_rank = deque(maxlen=self.seq_length)
        # loss func
        self.loss_func = nn.MSELoss()

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
        print("initial_rollout")
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

    def micro_rollout(self, context_index, env, s, info):
        print("micro_rollout")
        reward_sum = 0
        for i in range(self.seq_length):
            trading_action = self.low_level_act_test(s, info, context_index)
            s_, r, done, info_ = env.step(trading_action)
            rank = self.calculate_rollout_transition_rank(
                s, info, trading_action, r, s_, info_
            )
            self.store_seq_length_rank(rank.detach().cpu().numpy())
            s, info = s_, info_
            reward_sum += r
            if done:
                break
        return s, info, reward_sum, done

    def get_seq_length_rank(self):
        seq_rank = np.array(self.seq_rank).transpose(1, 0, 2)
        return seq_rank

    def update(
        self,
        states: torch.tensor,
        info: dict,
        context_indexes: torch.tensor,
        rewards: torch.tensor,
        next_states: torch.tensor,
        info_: dict,
        dones: torch.tensor,
    ):
        print("update")
        # low level
        # current input
        bs = states.shape[0]
        states = states.reshape(bs, -1)
        previous_action = info["previous_action"].float().unsqueeze(1)
        avaliable_action = info["avaliable_action"]
        hour_count_down = info["funding_count_down_hour"].float().unsqueeze(1)
        minute_count_down = info["funding_count_down_minute"].float().unsqueeze(1)
        time_input = torch.cat([hour_count_down, minute_count_down], dim=1).to(
            self.device
        )
        predict_action_distrbution = self.low_level_network(
            state=states,
            time=time_input,
            previous_action=previous_action,
            avaliable_action=avaliable_action,
        )

        # high level
        high_level_state = info["high_level_state"]
        high_level_state_ = info_["high_level_state"]

        context_q_values_distribution = self.high_level_network_eval(
            high_level_state.float()
        )
        context_q_values = context_q_values_distribution.gather(
            dim=1, index=context_indexes
        )
        next_q = self.high_level_network_target(high_level_state_.float()).detach()
        q_target = rewards + self.gamma * torch.max(next_q, 1)[0].view(
            self.batch_size, 1
        )

        td_error = self.loss_func(context_q_values, q_target)

        weighted_trading_action_value_distribution = (
            predict_action_distrbution * context_q_values_distribution.unsqueeze(-1)
        )

        weighted_trading_action_value_distribution = (
            weighted_trading_action_value_distribution.sum(dim=1)
        )
        q_value_demonstration = recalculate_q_demonstration(
            info["q_value"], info["avaliable_action"]
        )
        KL_loss = F.kl_div(
            (weighted_trading_action_value_distribution.softmax(dim=-1) + 1e-8).log(),
            (q_value_demonstration.softmax(dim=-1) + 1e-8),
            reduction="batchmean",
        )
        loss = td_error + KL_loss * self.ada
        update_params(
            self.optimizer,
            loss,
            self.high_level_network_eval,
            retain_graph=False,
            grad_cliping=self.grad_clip,
        )
        soft_copy_params(
            self.high_level_network_eval, self.high_level_network_target, self.tau
        )
        self.update_counter += 1
        return loss.item(), KL_loss.item(), td_error.item()

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

    def choose_context(self, rank_seq, epsilon):
        if np.random.uniform() > epsilon:
            rank_seq = torch.tensor(rank_seq).to(self.device)
            q_values = self.high_level_network_eval(rank_seq.float())
            context_index = torch.max(q_values, 1)[1].data.cpu().numpy()
            context_index = context_index[0]
        else:
            context_index = np.random.choice(range(self.N))

        return context_index

    def train(self):
        epoch_return_rate_train_list = []
        epoch_final_balance_train_list = []
        epoch_reward_sum_train_list = []
        # epoch_number = int(len(self.train_df) / self.chunk_length)
        epoch_number = 1
        step_counter = 0
        return_rate_list = []

        # perfect experience
        buffer = Multi_step_ReplayBuffer_multi_info(
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            device=self.device,
            seed=self.seed,
            gamma=self.gamma,
            n_step=self.n_step,
        )
        for sample in range(self.num_sample):
            self.train_df = pd.read_feather(self.train_data_path)
            initial_action = random.choices(range(self.position_choices), k=1)[0]

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
            current_markprice = self.train_df["mark_price"].values[0]
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
            env = initiate_demo_env(
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
                gamma=self.gamma,
                max_punishment=1e10,
            )
            s, info = env.reset()
            episode_reward_sum = 0
            rank, s, info, r = self.initial_rollout(initial_action, env, s, info)
            while True:
                episode_reward_sum += r
                high_level_state = np.array(self.get_seq_length_rank())
                context_index = self.choose_context(high_level_state, self.epsilon)
                # the high level state is (1, seq_length, N)
                info["high_level_state"] = high_level_state[0]

                s_, info_, r, done = self.micro_rollout(context_index, env, s, info)
                # trading_action = self.low_level_act_test(s, info, context_index)
                # s_, r, done, info_ = env.step(trading_action)
                # rank = self.calculate_rollout_transition_rank(
                #     s, info, trading_action, r, s_, info_
                # )
                # self.store_seq_length_rank(rank.detach().cpu().numpy())
                high_level_state_ = np.array(self.get_seq_length_rank())
                # the high level state is (1, seq_length, N)
                info_["high_level_state"] = high_level_state_[0]
                buffer.add(
                    s,
                    info,
                    context_index,
                    r,
                    s_,
                    info_,
                    done,
                )
                step_counter += 1
                s, info = s_, info_
                self.epsilon = (
                    self.epsilon - self.epsilon_decay
                    if self.epsilon - self.epsilon_decay > self.epsilon_min
                    else self.epsilon_min
                )
                self.ada = (
                    self.ada - self.ada_decay
                    if self.ada - self.ada_decay > self.ada_min
                    else self.ada_min
                )
                self.lr = (
                    self.lr - self.lr_decay
                    if self.lr - self.lr_decay > self.lr_min
                    else self.lr_min
                )
                for p in self.optimizer.param_groups:
                    p["lr"] = self.lr
                if done:
                    self.reset_seq_length_rank()
                    break
                print('step_counter', step_counter)
                print("update_freq", self.batch_size * self.update_times + self.n_step)
                if (
                    step_counter > (self.batch_size * self.update_times + self.n_step)
                    and step_counter % self.rollout_steps == 1
                ):
                    for _ in range(self.update_times):
                        (
                            states,
                            infos,
                            context_indexes,
                            rewards,
                            next_states,
                            next_infos,
                            dones,
                        ) = buffer.sample()
                        total_loss, KL_loss, td_loss = self.update(
                            states,
                            infos,
                            context_indexes,
                            rewards,
                            next_states,
                            next_infos,
                            dones,
                        )
                        self.writer.add_scalar(
                            tag="total_loss",
                            scalar_value=total_loss,
                            global_step=self.update_counter,
                            walltime=None,
                        )
                        self.writer.add_scalar(
                            tag="KL_loss",
                            scalar_value=KL_loss,
                            global_step=self.update_counter,
                            walltime=None,
                        )
                        self.writer.add_scalar(
                            tag="td_loss",
                            scalar_value=td_loss,
                            global_step=self.update_counter,
                            walltime=None,
                        )

            final_balance = env.unrealized_pnl + env.wallet_balance
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
            epoch_reward_sum_train_list.append(episode_reward_sum)
            if len(epoch_reward_sum_train_list) == epoch_number:
                epoch_index = int((sample + 1) / epoch_number)
                mean_return_rate_train = np.mean(epoch_return_rate_train_list)
                mean_final_balance_train = np.mean(epoch_final_balance_train_list)
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
                    self.high_level_network_eval.state_dict(),
                    os.path.join(epoch_path, "trained_model.pkl"),
                )
                # self.test(epoch_path)
                epoch_return_rate_train_list = []
                epoch_final_balance_train_list = []
                epoch_reward_sum_train_list = []


if __name__ == "__main__":
    args = parser.parse_args()
    td_agent = high_level_agent(args)
    td_agent.train()
