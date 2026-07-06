# Code reference: https://github.com/Lizhi-sjtu/DRL-code-pytorch/tree/main/3.Rainbow_DQN

import sys

sys.path.append(".")
import os
import random
import argparse
import logging
import traceback
import numpy as np
import torch
import torch.multiprocessing as tmp
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


def summarize_rollout_metrics(metrics):
    return {
        "mean_return_rate": float(np.mean([item["return_rate"] for item in metrics])),
        "mean_final_balance": float(
            np.mean([item["final_balance"] for item in metrics])
        ),
        "mean_reward_sum": float(np.mean([item["reward_sum"] for item in metrics])),
    }


def record_diverse_rollout_latest_metric(
    metrics_by_df,
    df_index,
    rollout_index,
    reward_sum,
    final_balance,
    return_rate,
):
    df_metrics = metrics_by_df.setdefault(int(df_index), {})
    df_metrics[int(rollout_index)] = {
        "reward_sum": float(reward_sum),
        "final_balance": float(final_balance),
        "return_rate": float(return_rate),
    }


def log_diverse_rollout_latest_metrics(epoch_index, metrics_by_df):
    for df_index in sorted(metrics_by_df):
        for rollout_index in sorted(metrics_by_df[df_index]):
            metrics = metrics_by_df[df_index][rollout_index]
            profit_label = "盈利" if metrics["return_rate"] > 0 else "亏损"
            logger.info(
                "第 %d 轮 epoch 训练完成 | 多样化训练最新明细 | "
                "df_index=%d | rollout_index=%d | 累计奖励=%.4f | "
                "最终余额=%.4f | 收益率=%.6f | %s",
                epoch_index,
                df_index,
                rollout_index,
                metrics["reward_sum"],
                metrics["final_balance"],
                metrics["return_rate"],
                profit_label,
            )


def summarize_rollout_diagnostics(actions, positions, preview_limit=20):
    action_values, action_counts = np.unique(actions, return_counts=True)
    position_values, position_counts = np.unique(positions, return_counts=True)
    position_switches = sum(
        1
        for previous_position, current_position in zip(positions, positions[1:])
        if current_position != previous_position
    )
    return {
        "action_counts": [
            (int(action), int(count))
            for action, count in zip(action_values.tolist(), action_counts.tolist())
        ],
        "position_counts": [
            (float(position), int(count))
            for position, count in zip(position_values.tolist(), position_counts.tolist())
        ],
        "first_actions": [int(action) for action in actions[:preview_limit]],
        "first_positions": [
            float(position) for position in positions[:preview_limit]
        ],
        "position_switches": int(position_switches),
    }


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
from env.env_class.futures_util import (
    get_dp_action_from_qtable,
    map_action_to_position_leverage,
)
from env.env_class.policy_util import get_close_element
from RL.DiHFT.low_level.pretrain_qtable_diagnostics import (
    build_initial_state,
    create_demo_env,
    extend_q_table_cache,
    get_sample_action_from_cache,
    prepare_pretrain_qtable_diagnostics,
    select_sample_from_plan,
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
    default=0,
    help="the number of sample-level pretrain epochs after full df warmup",
)
parser.add_argument(
    "--full_df_warmup",
    dest="full_df_warmup",
    action="store_true",
    default=True,
    help="run one empty-position pretrain warmup for every training df before sample loop",
)
parser.add_argument(
    "--no_full_df_warmup",
    dest="full_df_warmup",
    action="store_false",
    help="disable full df warmup before sample loop",
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


def build_effective_df_indices(total_df_index_length):
    return list(range(total_df_index_length))


def iter_parallel_rollout_tasks(num_epoch, context_count, position_choices):
    for epoch_index in range(num_epoch):
        for context_index in range(context_count):
            for initial_action in range(position_choices):
                yield {
                    "epoch_index": epoch_index,
                    "context_index": context_index,
                    "initial_action": initial_action,
                }


def _linear_value(start, end, index, total_count):
    if total_count <= 1:
        return float(start)
    progress = min(max(index, 0), total_count - 1) / float(total_count - 1)
    return float(max(end, start - (start - end) * progress))


def _held_then_linear_value(start, end, epoch_index, num_epoch):
    if num_epoch <= 1:
        return float(start)
    hold_epochs = num_epoch // 2
    if epoch_index < hold_epochs:
        return float(start)
    decay_epochs = max(num_epoch - hold_epochs - 1, 1)
    decay_index = min(max(epoch_index - hold_epochs, 0), decay_epochs)
    return float(max(end, start - (start - end) * decay_index / float(decay_epochs)))


def compute_epoch_training_params(
    epoch_index,
    num_epoch,
    epsilon_init,
    epsilon_min,
    ada_init,
    ada_min,
    lr_init,
    lr_min,
):
    return {
        "epsilon": _linear_value(epsilon_init, epsilon_min, epoch_index, num_epoch),
        "ada": _held_then_linear_value(ada_init, ada_min, epoch_index, num_epoch),
        "lr": _held_then_linear_value(lr_init, lr_min, epoch_index, num_epoch),
    }


def make_cpu_state_dict(module):
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
    }


def sort_round_transitions(round_results):
    ordered = []
    for result in sorted(round_results, key=lambda item: item["df_index"]):
        ordered.extend(
            item["transition"]
            for item in sorted(
                result.get("transitions", []),
                key=lambda transition: transition["step_index"],
            )
        )
    return ordered


def raise_for_worker_error(message):
    if message.get("type") != "worker_error":
        return
    raise RuntimeError(
        "worker_error df_index={df_index} epoch_index={epoch_index} "
        "context_index={context_index} initial_action={initial_action} "
        "round_counter={round_counter}: {traceback}".format(**message)
    )


def df_rollout_worker(worker_config, input_queue, result_queue):
    df_index = worker_config["df_index"]
    message = {}
    try:
        runner_factory = worker_config.get("runner_factory", DfRolloutWorkerRunner)
        runner = runner_factory(worker_config)
        while True:
            message = input_queue.get()
            message_type = message["type"]
            if message_type == "shutdown":
                return
            if message_type == "reset_task":
                runner.reset_task(message)
                continue
            if message_type == "explore_round":
                result_queue.put(runner.explore_round(message))
                continue
            raise ValueError("unknown worker message type: {}".format(message_type))
    except Exception:
        result_queue.put(
            {
                "type": "worker_error",
                "df_index": df_index,
                "epoch_index": message.get("epoch_index", -1),
                "context_index": message.get("context_index", -1),
                "initial_action": message.get("initial_action", -1),
                "round_counter": message.get("round_counter", -1),
                "traceback": traceback.format_exc(),
            }
        )


class DfRolloutWorkerRunner:
    def __init__(self, worker_config):
        self.df_index = worker_config["df_index"]
        self.train_df = worker_config["train_df"]
        self.env_kwargs = worker_config["env_kwargs"]
        self.model_factory = worker_config.get("model_factory", create_parallel_worker_model)
        self.device = worker_config["device"]
        self.leverage_choices = worker_config["leverage_choices"]
        self.position_list = worker_config["position_list"]
        self.initial_wallet_balance = worker_config["initial_wallet_balance"]
        self.initial_unrealized_pnL = worker_config["initial_unrealized_pnL"]
        if self.model_factory is create_parallel_worker_model:
            self.model = self.model_factory(worker_config).to(self.device)
        else:
            self.model = self.model_factory().to(self.device)
        self.env = None
        self.state = None
        self.info = None
        self.done = True
        self.reward_sum = 0.0
        self.transition_count = 0

    def reset_task(self, message):
        _, _, _, initial_state = build_initial_state(
            self.train_df,
            message["initial_action"],
            self.leverage_choices,
            self.position_list,
            self.initial_wallet_balance,
            self.initial_unrealized_pnL,
        )
        self.env = create_demo_env(self.train_df, self.env_kwargs, initial_state)
        self.state, self.info = self.env.reset()
        self.done = False
        self.reward_sum = 0.0
        self.transition_count = 0

    def _act(self, state, info, context_index, epsilon):
        if np.random.uniform() <= epsilon:
            return np.random.choice(info["avaiable_action_list"])
        with torch.no_grad():
            state_tensor = torch.unsqueeze(torch.FloatTensor(state).reshape(-1), 0).to(
                self.device
            )
            previous_action = torch.unsqueeze(
                torch.tensor([info["previous_action"]]).float().to(self.device), 0
            )
            avaliable_action = torch.unsqueeze(
                torch.tensor(info["avaliable_action"]).to(self.device), 0
            )
            hour_count_down = torch.unsqueeze(
                torch.tensor([info["funding_count_down_hour"]]).float().to(self.device),
                0,
            )
            minute_count_down = torch.unsqueeze(
                torch.tensor([info["funding_count_down_minute"]]).float().to(self.device),
                0,
            )
            time_input = torch.cat([hour_count_down, minute_count_down], dim=1)
            q_values = self.model(
                state=state_tensor,
                time=time_input,
                previous_action=previous_action,
                avaliable_action=avaliable_action,
            )
            return int(torch.max(q_values[:, context_index, :], 1)[1].data.cpu().numpy()[0])

    def explore_round(self, message):
        self.model.load_state_dict(message["state_dict"])
        self.model.eval()
        transitions = []
        step_index = 0
        while not self.done and step_index < message["rollout_steps"]:
            action = self._act(
                self.state,
                self.info,
                message["context_index"],
                message["epsilon"],
            )
            next_state, reward, done, next_info = self.env.step(action)
            transitions.append(
                {
                    "step_index": self.transition_count,
                    "transition": (
                        self.state,
                        self.info,
                        action,
                        reward,
                        next_state,
                        next_info,
                        done,
                    ),
                }
            )
            self.reward_sum += reward
            self.transition_count += 1
            step_index += 1
            self.state, self.info, self.done = next_state, next_info, done
        final_balance = self.env.unrealized_pnl + self.env.wallet_balance
        return {
            "type": "round_result",
            "df_index": self.df_index,
            "epoch_index": message["epoch_index"],
            "context_index": message["context_index"],
            "initial_action": message["initial_action"],
            "round_counter": message["round_counter"],
            "worker_steps": len(transitions),
            "transitions": transitions,
            "rollout_metrics": [
                {
                    "epoch_index": message["epoch_index"],
                    "context_index": message["context_index"],
                    "initial_action": message["initial_action"],
                    "df_index": self.df_index,
                    "transition_count": self.transition_count,
                    "reward_sum": float(self.reward_sum),
                    "final_balance": float(final_balance),
                    "return_rate": float(
                        final_balance / (self.initial_wallet_balance + 1e-12) - 1
                    ),
                }
            ],
            "done": self.done,
            "progress": {"transition_count": self.transition_count},
        }


def create_worker_context():
    return tmp.get_context("spawn")

runner
def shutdown_workers(input_queues, processes):
    for queue in input_queues:
        queue.put({"type": "shutdown"})
    for process in processes:
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()


def write_round_transitions_to_buffer(buffer_diverse, round_results):
    for transition in sort_round_transitions(round_results):
        buffer_diverse.add(*transition)


def run_fixed_diverse_updates(trainer, buffer_diverse, update_times, round_counter):
    last_losses = None
    for _ in range(update_times):
        (
            states,
            infos,
            actions,
            rewards,
            next_states,
            next_infos,
            dones,
        ) = buffer_diverse.sample()
        last_losses = trainer.update(
            states,
            infos,
            actions,
            rewards,
            next_states,
            next_infos,
            dones,
        )
        total_loss, KL_loss, td_loss = last_losses
        trainer.writer.add_scalar("total_loss", total_loss, round_counter)
        trainer.writer.add_scalar("KL_loss", KL_loss, round_counter)
        trainer.writer.add_scalar("td_loss", td_loss, round_counter)
    return last_losses


def create_parallel_worker_model(worker_config):
    return ensemble_Qnet(
        N_STATES=worker_config["state_dim"],
        N_ACTIONS=worker_config["action_count"],
        hidden_nodes=worker_config["hidden_nodes"],
        TIME_INFO_DIM=worker_config["time_info_dim"],
        ensemble_number=worker_config["ensemble_number"],
    )


def summarize_parallel_round(
    round_counter,
    epoch_index,
    context_index,
    initial_action,
    round_results,
    buffer_size,
    update_count,
):
    return {
        "round_counter": int(round_counter),
        "epoch_index": int(epoch_index),
        "context_index": int(context_index),
        "initial_action": int(initial_action),
        "round_steps": int(sum(result["worker_steps"] for result in round_results)),
        "active_worker_count": int(len(round_results)),
        "buffer_size": int(buffer_size),
        "update_count": int(update_count),
    }


def build_epoch_model_path(model_path, epoch_index):
    return os.path.join(model_path, "epoch_{}".format(epoch_index + 1))


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
        self.num_epoch = args.num_epoch if args.num_epoch is not None else args.num_sample
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
        self.full_df_warmup = args.full_df_warmup
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

    def _resolve_empty_initial_action(self):
        if 0 not in self.position_list:
            raise ValueError(
                "Unable to resolve empty position action from position_list={} and "
                "leverage_choices={}".format(self.position_list, self.leverage_choices)
            )
        action_count = getattr(
            self,
            "N_ACTIONS",
            (self.position_choices - 1) * len(self.leverage_choices) + 1,
        )
        for action in range(action_count):
            position, _ = map_action_to_position_leverage(
                action,
                self.leverage_choices,
                self.position_list,
            )
            if position == 0:
                return action
        raise ValueError(
            "Unable to resolve empty position action from position_list={} and "
            "leverage_choices={}".format(self.position_list, self.leverage_choices)
        )

    def _write_pretrain_loss_scalars(self, total_loss, KL_loss, td_loss):
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

    def _run_pretrain_updates_if_ready(self, buffer_pretrain, step_counter_pretrain):
        if not (
            step_counter_pretrain > (self.batch_size * self.update_times + self.n_step)
            and step_counter_pretrain % self.rollout_steps == 1
        ):
            return None
        last_losses = None
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
            last_losses = self.update_pretrain(
                states,
                infos,
                actions,
                rewards,
                next_states,
                next_infos,
                dones,
            )
            self._write_pretrain_loss_scalars(*last_losses)
        return last_losses

    def _run_full_df_warmup(
        self,
        q_table_cache,
        train_df_cache,
        env_kwargs,
        buffer_pretrain,
        step_counter_pretrain,
    ):
        if not self.full_df_warmup:
            logger.info("full-df warmup disabled")
            return {"df_count": 0, "reward_sum": 0.0, "update_count": 0}, step_counter_pretrain
        if self.total_df_index_length <= 0:
            raise ValueError("full-df warmup requires total_df_index_length > 0")

        empty_initial_action = self._resolve_empty_initial_action()
        logger.info(
            "full-df warmup start | df_count=%d | empty_initial_action=%d",
            self.total_df_index_length,
            empty_initial_action,
        )
        total_reward_sum = 0.0
        update_count_before = self.update_counter

        for df_index in range(self.total_df_index_length):
            train_df = train_df_cache[df_index]
            q_table = q_table_cache[df_index]
            first_row_indicators = ", ".join(
                f"{column}={train_df[column].iloc[0]}"
                for column in self.tech_indicator_list
            )
            logger.info(
                "full-df warmup first row | df_index=%d | %s",
                df_index,
                first_row_indicators,
            )
            self._set_initial_state_from_action(train_df, empty_initial_action)
            env = create_demo_env(train_df, env_kwargs, self.initial_state)
            self.perfection_action_list = get_dp_action_from_qtable(
                q_table,
                empty_initial_action,
            )
            df_reward_sum = 0.0
            last_losses = None
            for rollout_index in range(4):
                s, info = env.reset()
                optimal_step_counter = 0
                rollout_reward_sum = 0.0
                losses = None
                while True:
                    a = self.act_multi_styles_pretrain(
                        info,
                        optimal_step_counter,
                        rollout_index,
                    )
                    optimal_step_counter += 1
                    s_, r, done, info_ = env.step(a)
                    step_counter_pretrain += 1
                    buffer_pretrain.add(s, info, a, r, s_, info_, done)
                    rollout_reward_sum += r
                    s, info = s_, info_
                    if done:
                        break
                    losses = self._run_pretrain_updates_if_ready(
                        buffer_pretrain,
                        step_counter_pretrain,
                    )
                    if losses is not None:
                        last_losses = losses
                if last_losses is not None:
                    logger.info(
                        "full-df warmup update | df_index=%d | step=%d | "
                        "total_loss=%.6f | KL_loss=%.6f | td_loss=%.6f",
                        df_index,
                        step_counter_pretrain,
                        last_losses[0],
                        last_losses[1],
                        last_losses[2],
                    )
                rollout_final_balance = env.unrealized_pnl + env.wallet_balance
                rollout_return_rate = rollout_final_balance / self.initial_wallet_balance
                logger.info(
                    "full-df warmup rollout complete | df_index=%d | rollout_index=%d | "
                    "reward_sum=%.4f | final_balance=%.4f | return_rate=%.6f",
                    df_index,
                    rollout_index,
                    rollout_reward_sum,
                    rollout_final_balance,
                    rollout_return_rate,
                )
                df_reward_sum += rollout_reward_sum

            df_update_count = self.update_counter - update_count_before
            if df_reward_sum <= 0:
                logger.warning(
                    "full-df warmup unprofitable | df_index=%d | reward_sum=%.4f | "
                    "update_count=%d",
                    df_index,
                    df_reward_sum,
                    df_update_count,
                )
            else:
                logger.info(
                    "full-df warmup df complete | df_index=%d | reward_sum=%.4f | "
                    "update_count=%d",
                    df_index,
                    df_reward_sum,
                    df_update_count,
                )
            total_reward_sum += df_reward_sum

        update_count = self.update_counter - update_count_before
        logger.info(
            "full-df warmup complete | df_count=%d | reward_sum=%.4f | update_count=%d",
            self.total_df_index_length,
            total_reward_sum,
            update_count,
        )
        return {
            "df_count": self.total_df_index_length,
            "reward_sum": total_reward_sum,
            "update_count": update_count,
        }, step_counter_pretrain


    def _start_parallel_workers(self, train_df_cache, env_kwargs):
        worker_context = create_worker_context()
        self.worker_result_queue = worker_context.Queue()
        self.worker_input_queues = {}
        self.worker_processes = []
        for df_index in build_effective_df_indices(self.total_df_index_length):
            input_queue = worker_context.Queue()
            worker_config = {
                "df_index": df_index,
                "train_df": train_df_cache[df_index],
                "env_kwargs": env_kwargs,
                "device": self.device,
                "leverage_choices": self.leverage_choices,
                "position_list": self.position_list,
                "initial_wallet_balance": self.initial_wallet_balance,
                "initial_unrealized_pnL": self.initial_unrealized_pnL,
                "state_dim": len(self.tech_indicator_list),
                "action_count": self.N_ACTIONS,
                "hidden_nodes": self.hidden_nodes,
                "time_info_dim": self.time_info_dim,
                "ensemble_number": self.N,
            }
            process = worker_context.Process(
                target=df_rollout_worker,
                args=(worker_config, input_queue, self.worker_result_queue),
            )
            process.start()
            self.worker_input_queues[df_index] = input_queue
            self.worker_processes.append(process)

    def _shutdown_parallel_workers(self):
        shutdown_workers(
            self.worker_input_queues.values(),
            self.worker_processes,
        )
        self.worker_input_queues = {}
        self.worker_processes = []
        self.worker_result_queue = None

    def _reset_worker_task(
        self,
        epoch_index,
        context_index,
        initial_action,
        active_df_indices,
    ):
        for df_index in sorted(active_df_indices):
            self.worker_input_queues[df_index].put(
                {
                    "type": "reset_task",
                    "epoch_index": epoch_index,
                    "context_index": context_index,
                    "initial_action": initial_action,
                }
            )

    def _send_worker_rounds(
        self,
        active_df_indices,
        epoch_index,
        context_index,
        initial_action,
        round_counter,
        state_dict,
    ):
        for df_index in sorted(active_df_indices):
            self.worker_input_queues[df_index].put(
                {
                    "type": "explore_round",
                    "epoch_index": epoch_index,
                    "context_index": context_index,
                    "initial_action": initial_action,
                    "round_counter": round_counter,
                    "state_dict": state_dict,
                    "epsilon": self.epsilon,
                    "rollout_steps": self.rollout_steps,
                }
            )

    def _collect_worker_rounds(self, active_df_indices, round_counter):
        expected_count = len(active_df_indices)
        results = []
        while len(results) < expected_count:
            message = self.worker_result_queue.get()
            if message.get("type") == "worker_error":
                return [message]
            if message.get("round_counter") != round_counter:
                raise RuntimeError(
                    "unexpected worker round_counter={} expected={}".format(
                        message.get("round_counter"),
                        round_counter,
                    )
                )
            if message.get("df_index") not in active_df_indices:
                raise RuntimeError(
                    "unexpected worker df_index={} active={}".format(
                        message.get("df_index"),
                        sorted(active_df_indices),
                    )
                )
            results.append(message)
        return sorted(results, key=lambda result: result["df_index"])

    def _run_parallel_rollout_task(
        self,
        epoch_index,
        context_index,
        initial_action,
        train_df_cache,
        env_kwargs,
        buffer_diverse,
        step_counter_diverse,
        round_counter,
    ):
        active_df_indices = set(build_effective_df_indices(self.total_df_index_length))
        self._reset_worker_task(
            epoch_index,
            context_index,
            initial_action,
            active_df_indices,
        )
        while active_df_indices:
            self._send_worker_rounds(
                active_df_indices=active_df_indices,
                epoch_index=epoch_index,
                context_index=context_index,
                initial_action=initial_action,
                round_counter=round_counter,
                state_dict=make_cpu_state_dict(self.eval_net),
            )
            round_results = self._collect_worker_rounds(active_df_indices, round_counter)
            for result in round_results:
                raise_for_worker_error(result)
            write_round_transitions_to_buffer(buffer_diverse, round_results)
            round_steps = sum(result["worker_steps"] for result in round_results)
            step_counter_diverse += round_steps
            update_count = 0
            if step_counter_diverse > (self.batch_size * self.update_times + self.n_step):
                run_fixed_diverse_updates(
                    self,
                    buffer_diverse,
                    self.update_times,
                    round_counter,
                )
                update_count = self.update_times
            round_summary = summarize_parallel_round(
                round_counter=round_counter,
                epoch_index=epoch_index,
                context_index=context_index,
                initial_action=initial_action,
                round_results=round_results,
                buffer_size=len(buffer_diverse),
                update_count=update_count,
            )
            logger.info(
                "parallel rollout round complete | round_counter=%d | epoch_index=%d | "
                "context_index=%d | initial_action=%d | round_steps=%d | "
                "active_worker_count=%d | buffer_size=%d | update_count=%d",
                round_summary["round_counter"],
                round_summary["epoch_index"],
                round_summary["context_index"],
                round_summary["initial_action"],
                round_summary["round_steps"],
                round_summary["active_worker_count"],
                round_summary["buffer_size"],
                round_summary["update_count"],
            )
            for result in round_results:
                for metrics in result.get("rollout_metrics", []):
                    logger.info(
                        "parallel rollout metrics | epoch_index=%d | context_index=%d | "
                        "initial_action=%d | df_index=%d | transition_count=%d | "
                        "reward_sum=%.4f | final_balance=%.4f | return_rate=%.6f",
                        metrics["epoch_index"],
                        metrics["context_index"],
                        metrics["initial_action"],
                        metrics["df_index"],
                        metrics["transition_count"],
                        metrics["reward_sum"],
                        metrics["final_balance"],
                        metrics["return_rate"],
                    )
            active_df_indices = {
                result["df_index"]
                for result in round_results
                if not result.get("done", False)
            }
            round_counter += 1
        return round_counter, step_counter_diverse

    def _save_parallel_epoch_model(self, epoch_index):
        epoch_path = build_epoch_model_path(self.model_path, epoch_index)
        if not os.path.exists(epoch_path):
            os.makedirs(epoch_path)
        torch.save(
            self.eval_net.state_dict(),
            os.path.join(epoch_path, "trained_model.pkl"),
        )
        logger.info(
            "第 %d 轮 epoch 训练完成 | 模型已保存至=%s",
            epoch_index + 1,
            epoch_path,
        )

    def _run_parallel_diverse_training(
        self,
        train_df_cache,
        env_kwargs,
        buffer_diverse,
        step_counter_diverse,
    ):
        if self.total_df_index_length <= 0:
            raise ValueError("parallel diverse training requires total_df_index_length > 0")
        round_counter = 0
        self._start_parallel_workers(train_df_cache, env_kwargs)
        try:
            for epoch_index in range(self.num_epoch):
                params = compute_epoch_training_params(
                    epoch_index=epoch_index,
                    num_epoch=self.num_epoch,
                    epsilon_init=self.epsilon_init,
                    epsilon_min=self.epsilon_min,
                    ada_init=self.ada_init,
                    ada_min=self.ada_min,
                    lr_init=self.lr_init,
                    lr_min=self.lr_min,
                )
                self.epsilon = params["epsilon"]
                self.ada = params["ada"]
                self.lr = params["lr"]
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr

                for context_index in range(self.N):
                    for initial_action in range(self.position_choices):
                        round_counter, step_counter_diverse = self._run_parallel_rollout_task(
                            epoch_index=epoch_index,
                            context_index=context_index,
                            initial_action=initial_action,
                            train_df_cache=train_df_cache,
                            env_kwargs=env_kwargs,
                            buffer_diverse=buffer_diverse,
                            step_counter_diverse=step_counter_diverse,
                            round_counter=round_counter,
                        )
                self._save_parallel_epoch_model(epoch_index)
        finally:
            self._shutdown_parallel_workers()
        return step_counter_diverse

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
        sample_plan, q_table_cache, train_df_cache, _, sample_action_cache = (
            prepare_pretrain_qtable_diagnostics(             
                total_df_index_length=self.total_df_index_length,
                position_choices=self.position_choices,
                train_data_path=self.train_data_path,
                qtable_kwargs=qtable_kwargs,
                env_kwargs=env_kwargs,
                output_dir=qtable_diagnostics_dir,
                logger=logger,
            )
        )
        if self.full_df_warmup:
            q_table_cache, train_df_cache = extend_q_table_cache(
                df_indices=range(self.total_df_index_length),
                train_data_path=self.train_data_path,
                qtable_kwargs=qtable_kwargs,
                q_table_cache=q_table_cache,
                train_df_cache=train_df_cache,
            )
        _, step_counter_pretrain = self._run_full_df_warmup(
            q_table_cache=q_table_cache,
            train_df_cache=train_df_cache,
            env_kwargs=env_kwargs,
            buffer_pretrain=buffer_pretrain,
            step_counter_pretrain=step_counter_pretrain,
        )
        if not self.full_df_warmup:
            q_table_cache, train_df_cache = extend_q_table_cache(
                df_indices=range(self.total_df_index_length),
                train_data_path=self.train_data_path,
                qtable_kwargs=qtable_kwargs,
                q_table_cache=q_table_cache,
                train_df_cache=train_df_cache,
            )
        step_counter_diverse = self._run_parallel_diverse_training(
            train_df_cache=train_df_cache,
            env_kwargs=env_kwargs,
            buffer_diverse=buffer_diverse,
            step_counter_diverse=step_counter_diverse,
        )


if __name__ == "__main__":
    args = parser.parse_args()
    configure_logger(args.dataset_name)
    logger.info('start')
    trainer = Weighted_Contexts_DQN(args)
    trainer.train()
