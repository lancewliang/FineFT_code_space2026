# Code reference: https://github.com/Lizhi-sjtu/DRL-code-pytorch/tree/main/3.Rainbow_DQN

import copy
import os
import random
import argparse
import logging
import sys
import traceback
from dataclasses import dataclass
import numpy as np
import torch
import torch.multiprocessing as tmp
from torch import nn
from torch.utils.tensorboard import SummaryWriter

sys.path.append(".")

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


@dataclass(frozen=True)
class ShutdownWorker:
    pass


@dataclass(frozen=True)
class WorkerErrorMessage:
    df_index: int
    epoch_index: int
    context_index: int
    initial_action: int
    round_counter: int
    traceback: str


def build_serial_model_path(result_path, dataset_name, experiment_name):
    return os.path.join(
        result_path,
        dataset_name,
        experiment_name,
        "weights_advantage_pretrain",
    )


def build_training_data_paths(base_path, dataset_name):
    dataset_root = os.path.join(base_path, dataset_name)
    train_root = os.path.join(dataset_root, "train")
    train_slice_root = os.path.join(train_root, "slice")
    train_data_path = train_slice_root if os.path.isdir(train_slice_root) else train_root
    return {
        "train_data_path": train_data_path,
        "state_features_path": os.path.join(dataset_root, "state_features.npy"),
        "maintenance_margin_ratio_path": os.path.join(
            dataset_root,
            "maintenance_margin_ratio_dict.npy",
        ),
    }


def count_training_data_files(train_data_path: str) -> int:
    return len(
        [
            file_name
            for file_name in os.listdir(train_data_path)
            if file_name.startswith("df_") and file_name.endswith(".feather")
        ]
    )


def configure_logger(dataset_name, experiment_name):
    log_dir = os.path.join(
        "log/DiHFT", dataset_name, "low_level", "train", experiment_name
    )
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "advantage.log")
    abs_log_path = os.path.abspath(log_path)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == abs_log_path:
            return abs_log_path

    file_handler = logging.FileHandler(abs_log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    return abs_log_path


# RL util
from RL.util.replay_buffer_DQN import Multi_step_ReplayBuffer_multi_info
from RL.util.update import disable_gradients
from RL.util.episode_selector import get_transformation_even_risk

# model
from model.low_level import ensemble_Qnet

# env
from RL.DiHFT.low_level.pretrain_qtable_diagnostics import (
    extend_q_table_cache,
    prepare_pretrain_qtable_diagnostics,
)
from RL.DiHFT.low_level.qtable_config import build_optimal_qtable_kwargs
from RL.DiHFT.low_level.parallel_pretrain import (
    CollectPretrainEpisode,
    run_exhaustive_warmup,
)
from RL.DiHFT.low_level.parallel_diverse_train import (
    DfRolloutWorkerRunner,
    ResetWorkerTask,
    ExploreWorkerRound,
    run_parallel_diverse_training,
)


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
    "--experiment_name",
    type=str,
    default="default",
    help="experiment name used to namespace parallel training outputs",
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
parser.add_argument(
    "--allow_reverse_position",
    action="store_true",
    help="allow reverse position in single step",
)
parser.add_argument(
    "--enable_limit_reward",
    action="store_true",
    default=True,
    help="enable limit up/down reward shaping in DP teacher and environment",
)
parser.add_argument(
    "--no_enable_limit_reward",
    dest="enable_limit_reward",
    action="store_false",
    help="disable limit up/down reward shaping",
)
parser.add_argument(
    "--limit_hold_bonus",
    type=float,
    default=1.0,
    help="bonus for holding position in limit direction",
)
parser.add_argument(
    "--limit_stay_bonus",
    type=float,
    default=0.5,
    help="bonus for maintaining unchanged position during limit",
)
parser.add_argument(
    "--limit_reverse_penalty",
    type=float,
    default=1.5,
    help="penalty for taking position opposite to limit direction",
)
parser.add_argument(
    "--near_limit_threshold",
    type=float,
    default=0.003,
    help="relative threshold for near-limit shaping",
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
    default=128,
    help="the number of transcation we learn at a time",
)
parser.add_argument("--update_times", type=int, default=20, help="the update times")
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
    "--num_epoch",
    type=int,
    default=None,
    help="number of parallel diverse-training epochs; one epoch explores every effective df once",
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
    default="result/DiHFT/low_level/parallel",
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
    default=0,
    help="number of exhaustive-warmup training rounds over the collected pretrain buffer",
)
parser.add_argument(
    "--neighbor_size",
    type=int,
    default=1,
    help="fixed learner neighbor count from FineFT Algorithm 2",
)
parser.add_argument(
    "--pretrain_num_workers",
    "--pretrain_workers",
    dest="pretrain_num_workers",
    type=int,
    default=150,
    help="number of parallel worker processes for pretrain exploration/collection",
)
parser.add_argument(
    "--eval_num_workers",
    "--eval_workers",
    "--pretrain_eval_num_workers",
    "--pretrain_eval_workers",
    dest="eval_num_workers",
    type=int,
    default=150,
    help="number of parallel worker processes for sub-agent evaluation process pool",
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
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True


def build_effective_df_indices(total_df_index_length):
    return list(range(total_df_index_length))


def raise_for_worker_error(message):
    if not isinstance(message, WorkerErrorMessage):
        return
    raise RuntimeError(
        "worker_error df_index={} epoch_index={} context_index={} "
        "initial_action={} round_counter={}: {}".format(
            message.df_index,
            message.epoch_index,
            message.context_index,
            message.initial_action,
            message.round_counter,
            message.traceback,
        )
    )


def df_rollout_worker(worker_config, input_queue, result_queue):
    df_index = worker_config["df_index"]
    message = None
    try:
        runner_factory = worker_config.get("runner_factory", DfRolloutWorkerRunner)
        runner = runner_factory(worker_config)
        while True:
            message = input_queue.get()
            if isinstance(message, ShutdownWorker):
                return
            if isinstance(message, ResetWorkerTask):
                runner.reset_task(message)
                continue
            if isinstance(message, ExploreWorkerRound):
                result_queue.put(runner.explore_round(message))
                continue
            if isinstance(message, CollectPretrainEpisode):
                result_queue.put(runner.collect_episode(message))
                continue
            raise ValueError(
                "unknown worker message type: {}".format(type(message).__name__)
            )
    except Exception:
        result_queue.put(
            WorkerErrorMessage(
                df_index=df_index,
                epoch_index=getattr(message, "epoch_index", -1),
                context_index=getattr(message, "context_index", -1),
                initial_action=getattr(message, "initial_action", -1),
                round_counter=getattr(message, "round_counter", -1),
                traceback=traceback.format_exc(),
            )
        )


def create_worker_context():
    return tmp.get_context("spawn")


def shutdown_workers(input_queues, processes):
    seen = set()
    unique_queues = []
    for queue in input_queues:
        if id(queue) not in seen:
            seen.add(id(queue))
            unique_queues.append(queue)
    for queue in unique_queues:
        queue.put(ShutdownWorker())
    for process in processes:
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()


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
        # self.device = "cpu"
        # log path
        self.experiment_name = args.experiment_name
        self.model_path = build_serial_model_path(
            args.result_path,
            args.dataset_name,
            args.experiment_name,
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
        self.num_epoch = args.num_epoch if args.num_epoch is not None else args.num_sample
        # trading environment setting
        self.base_path = args.base_path
        self.dataset_name = args.dataset_name
        training_data_paths = build_training_data_paths(
            self.base_path,
            self.dataset_name,
        )
        self.train_data_path = training_data_paths["train_data_path"]
        self.total_df_index_length = count_training_data_files(self.train_data_path)
        self.tech_indicator_list = np.load(training_data_paths["state_features_path"])
        self.maintenance_margin_ratio_dict = np.load(
            training_data_paths["maintenance_margin_ratio_path"],
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
        self.neighbor_size = args.neighbor_size
        if self.neighbor_size < 0:
            raise ValueError("neighbor_size must be non-negative")
        self.pretrain_num_workers = getattr(args, "pretrain_num_workers", 150)
        if self.pretrain_num_workers <= 0:
            raise ValueError("pretrain_num_workers must be positive")
        self.eval_num_workers = getattr(args, "eval_num_workers", 150)
        self.pretrain_eval_num_workers = self.eval_num_workers
        if self.eval_num_workers <= 0:
            raise ValueError("eval_num_workers must be positive")
        self.allow_reverse_position = getattr(args, "allow_reverse_position", False)
        self.enable_limit_reward = getattr(args, "enable_limit_reward", True)
        self.limit_hold_bonus = getattr(args, "limit_hold_bonus", 1.0)
        self.limit_stay_bonus = getattr(args, "limit_stay_bonus", 0.5)
        self.limit_reverse_penalty = getattr(args, "limit_reverse_penalty", 1.5)
        self.near_limit_threshold = getattr(args, "near_limit_threshold", 0.003)
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

    def _shutdown_parallel_workers(self):
        shutdown_workers(
            self.worker_input_queues.values(),
            self.worker_processes,
        )
        self.worker_input_queues = {}
        self.worker_processes = []
        self.worker_result_queue = None

    def train(self):
        self._log_internal_parameters("train_start")
        logger.info(
            "开始训练 | 数据集=%s | 总采样数=%d | 预训练轮数=%d | 设备=%s",
            self.dataset_name,
            self.num_sample,
            self.pretrain_epoch,
            self.device,
        )
        diverse_rollout_latest_metrics_by_df = {}
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
        qtable_kwargs = build_optimal_qtable_kwargs(
            max_holding_number=self.max_holding_number,
            order_book_depth=self.order_book_depth,
            position_choices=self.position_choices,
            leverage_choice=self.leverage_choices,
            long_estimated_rate=self.long_estimated_rate,
            short_estimated_rate=self.short_estimated_rate,
            commission_rate=self.transcation_cost,
            gamma=self.gamma,
            allow_reverse_position=self.allow_reverse_position,
            enable_limit_reward=self.enable_limit_reward,
            limit_hold_bonus=self.limit_hold_bonus,
            limit_stay_bonus=self.limit_stay_bonus,
            limit_reverse_penalty=self.limit_reverse_penalty,
            near_limit_threshold=self.near_limit_threshold,
        )
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
            "allow_reverse_position": self.allow_reverse_position,
            "enable_limit_reward": self.enable_limit_reward,
            "limit_hold_bonus": self.limit_hold_bonus,
            "limit_stay_bonus": self.limit_stay_bonus,
            "limit_reverse_penalty": self.limit_reverse_penalty,
            "near_limit_threshold": self.near_limit_threshold,
        }
        diagnostics_result = prepare_pretrain_qtable_diagnostics(
            total_df_index_length=self.total_df_index_length,
            position_choices=self.position_choices,
            train_data_path=self.train_data_path,
            qtable_kwargs=qtable_kwargs,
            env_kwargs=env_kwargs,
            output_dir=qtable_diagnostics_dir,
            logger=logger,
        )
        q_table_cache = diagnostics_result.q_table_cache
        train_df_cache = diagnostics_result.train_df_cache
        q_table_cache, train_df_cache = extend_q_table_cache(
            df_indices=range(self.total_df_index_length),
            train_data_path=self.train_data_path,
            qtable_kwargs=qtable_kwargs,
            q_table_cache=q_table_cache,
            train_df_cache=train_df_cache,
        )
        _, step_counter_pretrain = run_exhaustive_warmup(
            trainer=self,
            q_table_cache=q_table_cache,
            train_df_cache=train_df_cache,
            env_kwargs=env_kwargs,
            buffer_pretrain=buffer_pretrain,
            step_counter_pretrain=step_counter_pretrain,
        )
        # step_counter_diverse = run_parallel_diverse_training(
        #     trainer=self,
        #     train_df_cache=train_df_cache,
        #     env_kwargs=env_kwargs,
        #     buffer_diverse=buffer_diverse,
        #     step_counter_diverse=step_counter_diverse,
        #     diverse_rollout_latest_metrics_by_df=diverse_rollout_latest_metrics_by_df,
        # )


if __name__ == "__main__":
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
    args = parser.parse_args()
    configure_logger(args.dataset_name, args.experiment_name)
    logger.info('start')
    trainer = Weighted_Contexts_DQN(args)
    trainer.train()
