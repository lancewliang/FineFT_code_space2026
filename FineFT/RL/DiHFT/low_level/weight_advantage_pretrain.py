# Code reference: https://github.com/Lizhi-sjtu/DRL-code-pytorch/tree/main/3.Rainbow_DQN

import sys

sys.path.append(".")
import os
import random
import argparse
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def configure_logger(dataset_name):
    log_dir = os.path.join("log_futures", dataset_name, "low_level", "train")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "advantage.log")
    abs_log_path = os.path.abspath(log_path)

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == abs_log_path:
            return abs_log_path

    file_handler = logging.FileHandler(abs_log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return abs_log_path

# RL util
from RL.util.replay_buffer_DQN import Multi_step_ReplayBuffer_multi_info
import torch.nn.functional as F
from RL.util.update import (
    calculate_huber_loss,
    disable_gradients,
    update_params,
    soft_copy_params,
    calculate_partial_loss,
    recalculate_q_demonstration,
    evaluate_quantile_at_action,
)
from RL.util.episode_selector import get_transformation_even_risk

# model
from model.low_level import ensemble_Qnet

# env
from env.env_class.futures_util import get_dp_action_from_qtable
from env.env_class.policy_util import get_close_element
from RL.DiHFT.low_level.pretrain_qtable_diagnostics import (
    build_initial_state,
    create_demo_env,
    prepare_pretrain_qtable_diagnostics,
)
import copy


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
    "--order_book_depth",
    type=int,
    default=25,
    help="number of bid/ask price levels available in the order book",
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
    default=[1],
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
    default=1,
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
# * RL training coffient need to change if the dataset become larger
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
    default=0.1,
    help="the coffient for decay",
)
parser.add_argument(
    "--epsilon_step",
    type=float,
    default=1e5,
    help="the coffient for decay",
)
parser.add_argument(
    "--rollout_steps",
    type=int,
    default=1024,
    help="the number of sampling during one epoch",
)
# general learning setting
parser.add_argument("--lr_init", type=float, default=5e-3, help="the learning rate")
parser.add_argument("--lr_min", type=float, default=1e-4, help="the learning rate")
parser.add_argument("--lr_step", type=float, default=2e4, help="the learning rate")
parser.add_argument(
    "--num_sample",
    type=int,
    default=400,
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
    default="result/DiHFT/low_level",
    help="the path for storing the test result",
)
# loss setting
parser.add_argument(
    "--outer_bond",
    type=float,
    default=4,
    help="the path for storing the test result",
)
parser.add_argument(
    "--reachout_index",
    type=int,
    default=1,
    help="the path for storing the test result",
)
parser.add_argument(
    "--if_use_hubber_loss",
    type=bool,
    default=True,
    help="whether use hubber loss for td error",
)
# supervisor
parser.add_argument(
    "--ada_init",
    type=float,
    default=256,
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
    default=5e5,
    help="the coffient for decay",
)
# pretrain
parser.add_argument(
    "--pretrain_epoch",
    type=int,
    default=2,
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


class Weighted_Contexts_DQN:
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
            args.result_path, args.dataset_name, "weights_advantage_pretrain"
        )
        self.log_path = os.path.join(self.model_path, "log")
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.writer = SummaryWriter(self.log_path)

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
        # replay buffer setting
        self.n_step = args.n_step
        self.buffer_size = args.buffer_size
        # resample method
        self.priority_transformation = get_transformation_even_risk
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
        self.order_book_depth = args.order_book_depth
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
        self.target_net = copy.deepcopy(self.eval_net)
        disable_gradients(self.target_net)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        # loss
        self.outer_bond = args.outer_bond
        self.reachout_index = args.reachout_index
        self.if_use_hubber_loss = args.if_use_hubber_loss
        # supervisor setting
        self.ada_init = args.ada_init
        self.ada_min = args.ada_min
        self.ada_step = args.ada_step
        self.ada_decay = (self.ada_init - self.ada_min) / self.ada_step
        self.ada = self.ada_init
        # loss function
        self.loss_func_pretrain = nn.SmoothL1Loss(reduction="none")
        # pretrain
        self.pretrain_epoch = args.pretrain_epoch
        self._log_internal_parameters("init_end")

    def _format_internal_parameter_value(self, value):
        if value is None or isinstance(value, (bool, int, float, str)):
            return str(value)
        if isinstance(value, np.ndarray):
            return "ndarray(shape={}, dtype={})".format(value.shape, value.dtype)
        if torch.is_tensor(value):
            return "tensor(shape={}, dtype={}, device={})".format(
                tuple(value.shape), value.dtype, value.device
            )
        if isinstance(value, dict):
            if len(value) <= 10:
                return repr(value)
            keys = list(value.keys())[:10]
            return "dict(len={}, sample_keys={})".format(len(value), repr(keys))
        if isinstance(value, (list, tuple)):
            if len(value) <= 20:
                return repr(value)
            return "{}(len={}, sample={})".format(
                type(value).__name__, len(value), repr(list(value[:10]))
            )
        return "<{}>".format(type(value).__name__)

    def _log_internal_parameters(self, stage):
        logger.info("Weighted_Contexts_DQN internal parameters | stage=%s", stage)
        for name, value in self.__dict__.items():
            logger.info("%s=%s", name, self._format_internal_parameter_value(value))

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
        # next input
        states_ = next_states.reshape(bs, -1)
        previous_action_ = info_["previous_action"].float().unsqueeze(1)
        avaliable_action_ = info_["avaliable_action"]
        hour_count_down_ = info_["funding_count_down_hour"].float().unsqueeze(1)
        minute_count_down_ = info_["funding_count_down_minute"].float().unsqueeze(1)
        time_input_ = torch.cat([hour_count_down_, minute_count_down_], dim=1).to(
            self.device
        )

        current_sa_quantiles = evaluate_quantile_at_action(
            self.eval_net(
                state=states,
                time=time_input,
                previous_action=previous_action,
                avaliable_action=avaliable_action,
            ),
            actions,
        )
        assert current_sa_quantiles.shape == (bs, self.N, 1)
        with torch.no_grad():
            next_q = self.target_net.get_best_q(
                state=states_,
                time=time_input_,
                previous_action=previous_action_,
                avaliable_action=avaliable_action_,
            )
            next_sa_quantiles = next_q.unsqueeze(1)
            assert next_sa_quantiles.shape == (self.batch_size, 1, self.N)
            target_sa_quantiles = (
                rewards[..., None]
                + (1.0 - dones[..., None]) * self.gamma * next_sa_quantiles
            )
            assert target_sa_quantiles.shape == (self.batch_size, 1, self.N)
        td_errors = target_sa_quantiles - current_sa_quantiles
        # logger.info("td_errors %s", td_errors)
        assert td_errors.shape == (self.batch_size, self.N, self.N)
        if self.if_use_hubber_loss:
            td_errors = calculate_huber_loss(td_errors)
        batch_weights, partial_td_error_loss = calculate_partial_loss(
            td_errors=td_errors,
            outer_bond=self.outer_bond,
            reach_out_index=self.reachout_index,
        )
        predict_action_distrbution = self.eval_net(
            state=states,
            time=time_input,
            previous_action=previous_action,
            avaliable_action=avaliable_action,
        )
        assert predict_action_distrbution.shape == (
            self.batch_size,
            self.N,
            self.N_ACTIONS,
        )
        assert batch_weights.shape == (self.batch_size, self.N)

        weighted_action_distribution = torch.einsum(
            "ijk,ij->ik", predict_action_distrbution, batch_weights
        )
        q_value = recalculate_q_demonstration(info["q_value"], info["avaliable_action"])
        KL_div = F.kl_div(
            (weighted_action_distribution.softmax(dim=-1) + 1e-8).log(),
            (q_value.softmax(dim=-1) + 1e-8),
            reduction="batchmean",
        )
        loss = partial_td_error_loss + KL_div * self.ada
        update_params(
            self.optimizer,
            loss,
            self.eval_net,
            retain_graph=False,
            grad_cliping=self.grad_clip,
        )
        soft_copy_params(self.eval_net, self.target_net, self.tau)
        self.update_counter += 1
        return loss.item(), KL_div.item(), partial_td_error_loss.item()

    def update_pretrain(
        self,
        states: torch.tensor,
        info: dict,
        actions: torch.tensor,
        rewards: torch.tensor,
        next_states: torch.tensor,
        info_: dict,
        dones: torch.tensor,
    ):
        bs = states.shape[0]
        states = states.reshape(bs, -1)
        previous_action = info["previous_action"].float().unsqueeze(1)
        avaliable_action = info["avaliable_action"]
        hour_count_down = info["funding_count_down_hour"].float().unsqueeze(1)
        minute_count_down = info["funding_count_down_minute"].float().unsqueeze(1)
        time_input = torch.cat([hour_count_down, minute_count_down], dim=1).to(
            self.device
        )
        # next input
        states_ = next_states.reshape(bs, -1)
        previous_action_ = info_["previous_action"].float().unsqueeze(1)
        avaliable_action_ = info_["avaliable_action"]
        hour_count_down_ = info_["funding_count_down_hour"].float().unsqueeze(1)
        minute_count_down_ = info_["funding_count_down_minute"].float().unsqueeze(1)
        time_input_ = torch.cat([hour_count_down_, minute_count_down_], dim=1).to(
            self.device
        )

        current_sa_quantiles = evaluate_quantile_at_action(
            self.eval_net(
                state=states,
                time=time_input,
                previous_action=previous_action,
                avaliable_action=avaliable_action,
            ),
            actions,
        )
        assert current_sa_quantiles.shape == (bs, self.N, 1)
        current_sa_quantiles = current_sa_quantiles.squeeze(-1)
        with torch.no_grad():
            next_q = self.target_net.get_best_q(
                state=states_,
                time=time_input_,
                previous_action=previous_action_,
                avaliable_action=avaliable_action_,
            )
            next_sa_quantiles = next_q.unsqueeze(1)
            assert next_sa_quantiles.shape == (self.batch_size, 1, self.N)
            target_sa_quantiles = (
                rewards[..., None]
                + (1.0 - dones[..., None]) * self.gamma * next_sa_quantiles
            )
            target_sa_quantiles = target_sa_quantiles.permute(0, 2, 1)
            assert target_sa_quantiles.shape == (
                self.batch_size,
                self.N,
                1,
            )
        target_sa_quantiles = target_sa_quantiles.squeeze(-1)
        td_loss = self.loss_func_pretrain(current_sa_quantiles, target_sa_quantiles)
        td_loss = td_loss.sum(dim=1)
        td_loss = td_loss.mean()

        batch_weights = torch.ones(self.batch_size, self.N).to(self.device)
        predict_action_distrbution = self.eval_net(
            state=states,
            time=time_input,
            previous_action=previous_action,
            avaliable_action=avaliable_action,
        )
        assert predict_action_distrbution.shape == (
            self.batch_size,
            self.N,
            self.N_ACTIONS,
        )
        assert batch_weights.shape == (self.batch_size, self.N)

        weighted_action_distribution = torch.einsum(
            "ijk,ij->ik", predict_action_distrbution, batch_weights
        )
        q_value = recalculate_q_demonstration(info["q_value"], info["avaliable_action"])
        KL_div = F.kl_div(
            (weighted_action_distribution.softmax(dim=-1) + 1e-8).log(),
            (q_value.softmax(dim=-1) + 1e-8),
            reduction="batchmean",
        )
        loss = td_loss + KL_div * self.ada
        update_params(
            self.optimizer,
            loss,
            self.eval_net,
            retain_graph=False,
            grad_cliping=self.grad_clip,
        )
        soft_copy_params(self.eval_net, self.target_net, self.tau)
        self.update_counter += 1
        return loss.item(), KL_div.item(), td_loss.item()

    def act_single_context(self, state, info, context_index, epsilon):
        if np.random.uniform() > epsilon:
            state = torch.unsqueeze(torch.FloatTensor(state).reshape(-1), 0).to(
                self.device
            )
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
        else:
            action = np.random.choice(info["avaiable_action_list"])

        return action

    def act_multi_styles_pretrain(self, info, optimal_step_counter, rollout_index):
        assert rollout_index in range(4)
        avaliable_action_list = info["avaiable_action_list"]
        # 0 perfect 1 buy and hold 2 sell and keep 3 empty position 4..N different preference
        if rollout_index == 0:
            return self.perfection_action_list[optimal_step_counter]
        elif rollout_index == 1:
            action = (self.position_choices - 1) * len(self.leverage_choices) + 1 - 1
            action = get_close_element(action, avaliable_action_list)
            return action
        elif rollout_index == 2:
            action = len(self.leverage_choices) - 1
            action = get_close_element(action, avaliable_action_list)
            return action
        elif rollout_index == 3:
            action = (self.position_choices // 2) * len(self.leverage_choices)
            action = get_close_element(action, avaliable_action_list)
            return action

    def _set_initial_state_from_action(self, train_df, initial_action):
        (
            self.initial_position,
            self.initial_leverage,
            self.initial_margin,
            self.initial_state,
        ) = build_initial_state(
            train_df,
            initial_action,
            self.leverage_choices,
            self.position_list,
            self.initial_wallet_balance,
            self.initial_unrealized_pnL,
        )

    def act_multi_styles(self, state, info, epsilon, rollout_index):
        assert rollout_index in range(self.N)
        action = self.act_single_context(state, info, rollout_index, epsilon)
        return action

    def train(self):
        self._log_internal_parameters("train_start")
        logger.info(
            "开始训练 | 数据集=%s | 总采样数=%d | 预训练轮数=%d | 设备=%s",
            self.dataset_name,
            self.num_sample,
            self.pretrain_epoch,
            self.device,
        )
        epoch_return_rate_train_list = []
        epoch_final_balance_train_list = []
        epoch_reward_sum_train_list = []
        # epoch_number = int(len(self.train_df) / self.chunk_length)
        epoch_number = 4
        group_number = self.N
        # perfect experience
        buffer_pretrain = Multi_step_ReplayBuffer_multi_info(
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            device=self.device,
            seed=self.seed,
            gamma=self.gamma,
            n_step=self.n_step,
        )
        buffer_diverse = Multi_step_ReplayBuffer_multi_info(
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            device=self.device,
            seed=self.seed,
            gamma=self.gamma,
            n_step=self.n_step,
        )
        step_counter_pretrain = 0
        step_counter_diverse = 0
        qtable_diagnostics_dir = os.path.join(self.model_path, "qtable_diagnostics")
        qtable_kwargs = {
            "max_holding_number": self.max_holding_number,
            "order_book_depth": self.order_book_depth,
            "position_choices": self.position_choices,
            "leverage_choice": self.leverage_choices,
            "long_estimated_rate": self.long_estimated_rate,
            "short_estimated_rate": self.short_estimated_rate,
            "commission_rate": self.transcation_cost,
            "max_punishment": 1e10,
            "gamma": 1,
        }
        env_kwargs = {
            "feature_list": self.tech_indicator_list,
            "max_holding_number": self.max_holding_number,
            "order_book_depth": self.order_book_depth,
            "position_choices": self.position_choices,
            "leverage_choices": self.leverage_choices,
            "position_list": self.position_list,
            "long_estimated_rate": self.long_estimated_rate,
            "short_estimated_rate": self.short_estimated_rate,
            "commission_rate": self.transcation_cost,
            "maintenance_margin_ratio_dict": self.maintenance_margin_ratio_dict,
            "early_stop": self.early_stop,
            "gamma": self.gamma,
            "initial_wallet_balance": self.initial_wallet_balance,
            "initial_unrealized_pnl": self.initial_unrealized_pnL,
        }
        sample_plan, q_table_cache, train_df_cache, _ = (
            prepare_pretrain_qtable_diagnostics(
                num_sample=self.num_sample,
                total_df_index_length=self.total_df_index_length,
                position_choices=self.position_choices,
                train_data_path=self.train_data_path,
                qtable_kwargs=qtable_kwargs,
                env_kwargs=env_kwargs,
                output_dir=qtable_diagnostics_dir,
                logger=logger,
            )
        )
        for sample in range(self.num_sample):
            logger.info("===== 第 %d/%d 轮采样 =====", sample + 1, self.num_sample)
            pretrain = sample < self.pretrain_epoch
            logger.info("当前阶段: %s", "预训练" if pretrain else "多样化训练")
            df_index, initial_action = sample_plan[sample]
            logger.info(
                "正在使用 df_%d 进行训练, 初始动作=%d",
                df_index,
                initial_action,
            )
            self.train_df = train_df_cache[df_index]
            self._set_initial_state_from_action(self.train_df, initial_action)
            logger.info(
                "初始仓位=%s, 初始杠杆=%s",
                self.initial_position,
                self.initial_leverage,
            )
            env = create_demo_env(self.train_df, env_kwargs, self.initial_state)
            if pretrain:
                q_table = q_table_cache[df_index]
                self.perfection_action_list = get_dp_action_from_qtable(
                    q_table, initial_action
                )

                for index in range(4):
                    s, info = env.reset()
                    optimal_step_counter = 0
                    episode_reward_sum = 0
                    logger.info("预训练阶段: 使用基于规则策略 index=%d 采数", index)
                    while True:
                        a = self.act_multi_styles_pretrain(
                            info, optimal_step_counter, index
                        )
                        optimal_step_counter += 1
                        s_, r, done, info_ = env.step(a)

                        step_counter_pretrain += 1
                        buffer_pretrain.add(s, info, a, r, s_, info_, done)
                        episode_reward_sum += r

                        s, info = s_, info_
                        if done:
                            break
                        if (
                            step_counter_pretrain
                            > (self.batch_size * self.update_times + self.n_step)
                            and step_counter_pretrain % self.rollout_steps == 1
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
                                ) = buffer_pretrain.sample()
                                total_loss, KL_loss, td_loss = self.update_pretrain(
                                    states,
                                    infos,
                                    actions,
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
                                logger.info(
                                    "预训练更新 | 步数=%d | 累计更新次数=%d | 总损失=%.6f | KL损失=%.6f | TD损失=%.6f",
                                    step_counter_pretrain,
                                    self.update_counter,
                                    total_loss,
                                    KL_loss,
                                    td_loss,
                                )
                    final_balance = env.unrealized_pnl + env.wallet_balance
                    required_money = self.initial_wallet_balance
                    self.writer.add_scalar(
                        tag="return_rate_train_{}".format(index),
                        scalar_value=final_balance / (required_money + 1e-12) - 1,
                        global_step=sample,
                        walltime=None,
                    )

                    self.writer.add_scalar(
                        tag="reward_sum_train_{}".format(index),
                        scalar_value=episode_reward_sum,
                        global_step=sample,
                        walltime=None,
                    )
                    logger.info(
                        "预训练回合结束 | 规则索引=%d | 累计奖励=%.4f | 最终余额=%.4f | 收益率=%.6f",
                        index,
                        episode_reward_sum,
                        final_balance,
                        final_balance / (required_money + 1e-12) - 1,
                    )
            else:
                for index in range(self.N):
                    s, info = env.reset()
                    episode_reward_sum = 0
                    logger.info("多样化训练: 使用上下文索引 index=%d 采数", index)
                    while True:
                        a = self.act_multi_styles(s, info, self.epsilon, index)
                        if index == 0:
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
                        s_, r, done, info_ = env.step(a)

                        step_counter_diverse += 1
                        buffer_diverse.add(s, info, a, r, s_, info_, done)
                        episode_reward_sum += r

                        s, info = s_, info_
                        if done:
                            break
                        if (
                            step_counter_diverse
                            > (self.batch_size * self.update_times + self.n_step)
                            and step_counter_diverse % self.rollout_steps == 1
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
                                ) = buffer_diverse.sample()
                                total_loss, KL_loss, td_loss = self.update(
                                    states,
                                    infos,
                                    actions,
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
                                logger.info(
                                    "多样化训练更新 | 步数=%d | 累计更新次数=%d | 总损失=%.6f | KL损失=%.6f | TD损失=%.6f | 探索率=%.4f | 适配系数=%.4f | 学习率=%.6f",
                                    step_counter_diverse,
                                    self.update_counter,
                                    total_loss,
                                    KL_loss,
                                    td_loss,
                                    self.epsilon,
                                    self.ada,
                                    self.lr,
                                )

                    final_balance = env.unrealized_pnl + env.wallet_balance
                    required_money = self.initial_wallet_balance
                    self.writer.add_scalar(
                        tag="return_rate_train_{}".format(index),
                        scalar_value=final_balance / (required_money + 1e-12) - 1,
                        global_step=sample,
                        walltime=None,
                    )

                    self.writer.add_scalar(
                        tag="reward_sum_train_{}".format(index),
                        scalar_value=episode_reward_sum,
                        global_step=sample,
                        walltime=None,
                    )
                    logger.info(
                        "多样化回合结束 | 上下文索引=%d | 累计奖励=%.4f | 最终余额=%.4f | 收益率=%.6f",
                        index,
                        episode_reward_sum,
                        final_balance,
                        final_balance / (required_money + 1e-12) - 1,
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
                    self.eval_net.state_dict(),
                    os.path.join(epoch_path, "trained_model.pkl"),
                )
                logger.info(
                    "第 %d 轮 epoch 训练完成 | 平均收益率=%.6f | 平均最终余额=%.4f | 平均累计奖励=%.4f | 模型已保存至=%s",
                    epoch_index,
                    mean_return_rate_train,
                    mean_final_balance_train,
                    mean_reward_sum_train,
                    epoch_path,
                )
                # self.test(epoch_path)
                epoch_return_rate_train_list = []
                epoch_final_balance_train_list = []
                epoch_reward_sum_train_list = []


if __name__ == "__main__":
    args = parser.parse_args()
    configure_logger(args.dataset_name)
    logger.info('start')
    trainer = Weighted_Contexts_DQN(args)
    trainer.train()
