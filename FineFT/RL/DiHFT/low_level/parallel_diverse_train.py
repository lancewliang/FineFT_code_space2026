# Parallel diverse-training collection + orchestration components extracted
# from parallel_weight_advantage_pretrain.py for easier review.
#
# This module holds the model-driven rollout runner, rollout metrics/diagnostics
# dataclasses, the pure helpers and the parallel diverse-training orchestration
# used by the diverse-training stage. It does NOT import the orchestrator module
# at top level to keep the dependency graph acyclic; shared infrastructure is
# imported lazily inside the functions that need it.

import logging
import os
import torch
import numpy as np
from dataclasses import dataclass

import torch.nn.functional as F
from model.low_level import ensemble_Qnet
from RL.DiHFT.low_level.pretrain_qtable_diagnostics import (
    build_initial_state,
    create_demo_env,
)
from RL.util.update import (
    evaluate_quantile_at_action,
    calculate_huber_loss,
    recalculate_q_demonstration,
    update_params,
    soft_copy_params,
)
from RL.DiHFT.low_level.weight_advantage_pretrain import (
    calculate_paper_partial_loss,
    calculate_paper_supervisor_kl_loss,
)

# Reuse the orchestrator's configured logger so all log output flows through
# the same file handler set up by configure_logger() in
# parallel_weight_advantage_pretrain.
logger = logging.getLogger(
    "RL.DiHFT.low_level.parallel_weight_advantage_pretrain"
)


@dataclass(frozen=True)
class RolloutMetrics:
    epoch_index: int
    context_index: int
    initial_action: int
    df_index: int
    transition_count: int
    reward_sum: float
    final_balance: float
    return_rate: float

    def to_dict(self):
        return {
            "epoch_index": self.epoch_index,
            "context_index": self.context_index,
            "initial_action": self.initial_action,
            "df_index": self.df_index,
            "transition_count": self.transition_count,
            "reward_sum": self.reward_sum,
            "final_balance": self.final_balance,
            "return_rate": self.return_rate,
        }


@dataclass(frozen=True)
class RolloutMetricsSummary:
    mean_return_rate: float
    mean_final_balance: float
    mean_reward_sum: float

    def to_dict(self):
        return {
            "mean_return_rate": self.mean_return_rate,
            "mean_final_balance": self.mean_final_balance,
            "mean_reward_sum": self.mean_reward_sum,
        }


@dataclass(frozen=True)
class RolloutDiagnosticsSummary:
    action_counts: list[tuple[int, int]]
    position_counts: list[tuple[float, int]]
    first_actions: list[int]
    first_positions: list[float]
    position_switches: int

    def to_dict(self):
        return {
            "action_counts": self.action_counts,
            "position_counts": self.position_counts,
            "first_actions": self.first_actions,
            "first_positions": self.first_positions,
            "position_switches": self.position_switches,
        }


@dataclass(frozen=True)
class ParallelRolloutTask:
    epoch_index: int
    context_index: int
    initial_action: int

    def to_dict(self):
        return {
            "epoch_index": self.epoch_index,
            "context_index": self.context_index,
            "initial_action": self.initial_action,
        }


@dataclass(frozen=True)
class EpochTrainingParams:
    epsilon: float
    ada: float
    lr: float

    def to_dict(self):
        return {"epsilon": self.epsilon, "ada": self.ada, "lr": self.lr}


@dataclass(frozen=True)
class ResetWorkerTask:
    epoch_index: int
    context_index: int
    initial_action: int


@dataclass(frozen=True)
class ExploreWorkerRound:
    epoch_index: int
    context_index: int
    initial_action: int
    round_counter: int
    state_dict: dict
    epsilon: float
    rollout_steps: int


@dataclass(frozen=True)
class WorkerTransitionRecord:
    step_index: int
    transition: object


@dataclass(frozen=True)
class WorkerRoundResult:
    df_index: int
    epoch_index: int
    context_index: int
    initial_action: int
    round_counter: int
    worker_steps: int
    transitions: list[WorkerTransitionRecord]
    rollout_metrics: list[RolloutMetrics]
    done: bool
    progress: dict | None = None


@dataclass(frozen=True)
class ParallelRoundSummary:
    round_counter: int
    epoch_index: int
    context_index: int
    initial_action: int
    round_steps: int
    active_worker_count: int
    buffer_size: int
    update_count: int

    def to_dict(self):
        return {
            "round_counter": self.round_counter,
            "epoch_index": self.epoch_index,
            "context_index": self.context_index,
            "initial_action": self.initial_action,
            "round_steps": self.round_steps,
            "active_worker_count": self.active_worker_count,
            "buffer_size": self.buffer_size,
            "update_count": self.update_count,
        }


def summarize_rollout_metrics(metrics):
    return RolloutMetricsSummary(
        mean_return_rate=float(np.mean([item.return_rate for item in metrics])),
        mean_final_balance=float(np.mean([item.final_balance for item in metrics])),
        mean_reward_sum=float(np.mean([item.reward_sum for item in metrics])),
    )


def record_diverse_rollout_latest_metric(
    metrics_by_df,
    df_index,
    rollout_index,
    reward_sum,
    final_balance,
    return_rate,
):
    df_metrics = metrics_by_df.setdefault(int(df_index), {})
    df_metrics[int(rollout_index)] = RolloutMetrics(
        epoch_index=-1,
        context_index=int(rollout_index),
        initial_action=-1,
        df_index=int(df_index),
        transition_count=0,
        reward_sum=float(reward_sum),
        final_balance=float(final_balance),
        return_rate=float(return_rate),
    )


def log_diverse_rollout_latest_metrics(epoch_index, metrics_by_df, logger):
    for df_index in sorted(metrics_by_df):
        for rollout_index in sorted(metrics_by_df[df_index]):
            metrics = metrics_by_df[df_index][rollout_index]
            profit_label = "盈利" if metrics.return_rate > 0 else "亏损"
            logger.info(
                "第 %d 轮 epoch 训练完成 | 多样化训练最新明细 | "
                "df_index=%d | rollout_index=%d | 累计奖励=%.4f | "
                "最终余额=%.4f | 收益率=%.6f | %s",
                epoch_index,
                df_index,
                rollout_index,
                metrics.reward_sum,
                metrics.final_balance,
                metrics.return_rate,
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
    return RolloutDiagnosticsSummary(
        action_counts=[
            (int(action), int(count))
            for action, count in zip(action_values.tolist(), action_counts.tolist())
        ],
        position_counts=[
            (float(position), int(count))
            for position, count in zip(position_values.tolist(), position_counts.tolist())
        ],
        first_actions=[int(action) for action in actions[:preview_limit]],
        first_positions=[float(position) for position in positions[:preview_limit]],
        position_switches=int(position_switches),
    )


def iter_parallel_rollout_tasks(num_epoch, context_count, position_choices):
    for epoch_index in range(num_epoch):
        for context_index in range(context_count):
            for initial_action in range(position_choices):
                yield ParallelRolloutTask(
                    epoch_index=epoch_index,
                    context_index=context_index,
                    initial_action=initial_action,
                )


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
    return EpochTrainingParams(
        epsilon=_linear_value(epsilon_init, epsilon_min, epoch_index, num_epoch),
        ada=_held_then_linear_value(ada_init, ada_min, epoch_index, num_epoch),
        lr=_held_then_linear_value(lr_init, lr_min, epoch_index, num_epoch),
    )


def make_cpu_state_dict(module):
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
    }


def sort_round_transitions(round_results):
    ordered = []
    for result in sorted(round_results, key=lambda item: item.df_index):
        ordered.extend(
            item.transition
            for item in sorted(
                result.transitions,
                key=lambda transition: transition.step_index,
            )
        )
    return ordered


def write_round_transitions_to_buffer(buffer_diverse, round_results):
    for transition in sort_round_transitions(round_results):
        buffer_diverse.add(*transition)


def count_update_windows_crossed(
    previous_step_counter,
    current_step_counter,
    rollout_steps,
    warmup_steps,
):
    if rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")
    if current_step_counter <= previous_step_counter:
        return 0
    if current_step_counter <= warmup_steps:
        return 0

    previous_effective_step = max(previous_step_counter, warmup_steps)
    if previous_effective_step <= 0:
        previous_window = -1
    else:
        previous_window = (previous_effective_step - 1) // rollout_steps
    current_window = (current_step_counter - 1) // rollout_steps
    return max(0, current_window - previous_window)


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
        last_losses = update(
            trainer,
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
    return ParallelRoundSummary(
        round_counter=int(round_counter),
        epoch_index=int(epoch_index),
        context_index=int(context_index),
        initial_action=int(initial_action),
        round_steps=int(sum(result.worker_steps for result in round_results)),
        active_worker_count=int(len(round_results)),
        buffer_size=int(buffer_size),
        update_count=int(update_count),
    )


def build_epoch_model_path(model_path, epoch_index):
    return os.path.join(model_path, "epoch_{}".format(epoch_index + 1))


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
            message.initial_action,
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
            trading_info = torch.from_numpy(info["trading_info"]).float().reshape(1, -1).to(
                self.device
            )
            q_values = self.model(
                state=state_tensor,
                time=time_input,
                previous_action=previous_action,
                avaliable_action=avaliable_action,
                trading_info=trading_info,
            )
            return int(torch.max(q_values[:, context_index, :], 1)[1].data.cpu().numpy()[0])

    def explore_round(self, message):
        self.model.load_state_dict(message.state_dict)
        self.model.eval()
        transitions = []
        step_index = 0
        while not self.done and step_index < message.rollout_steps:
            action = self._act(
                self.state,
                self.info,
                message.context_index,
                message.epsilon,
            )
            next_state, reward, done, next_info = self.env.step(action)
            transitions.append(
                WorkerTransitionRecord(
                    step_index=self.transition_count,
                    transition=(
                        self.state,
                        self.info,
                        action,
                        reward,
                        next_state,
                        next_info,
                        done,
                    ),
                )
            )
            self.reward_sum += reward
            self.transition_count += 1
            step_index += 1
            self.state, self.info, self.done = next_state, next_info, done
        final_balance = self.env.unrealized_pnl + self.env.wallet_balance
        return WorkerRoundResult(
            df_index=self.df_index,
            epoch_index=message.epoch_index,
            context_index=message.context_index,
            initial_action=message.initial_action,
            round_counter=message.round_counter,
            worker_steps=len(transitions),
            transitions=transitions,
            rollout_metrics=[
                RolloutMetrics(
                    epoch_index=message.epoch_index,
                    context_index=message.context_index,
                    initial_action=message.initial_action,
                    df_index=self.df_index,
                    transition_count=self.transition_count,
                    reward_sum=float(self.reward_sum),
                    final_balance=float(final_balance),
                    return_rate=float(
                        final_balance / (self.initial_wallet_balance + 1e-12) - 1
                    ),
                )
            ],
            done=self.done,
            progress={"transition_count": self.transition_count},
        )


def start_parallel_workers(trainer, train_df_cache, env_kwargs):
    from RL.DiHFT.low_level.parallel_weight_advantage_pretrain import (
        build_effective_df_indices,
        create_worker_context,
        df_rollout_worker,
    )

    worker_context = create_worker_context()
    trainer.worker_result_queue = worker_context.Queue()
    trainer.worker_input_queues = {}
    trainer.worker_processes = []
    for df_index in build_effective_df_indices(trainer.total_df_index_length):
        input_queue = worker_context.Queue()
        worker_config = {
            "df_index": df_index,
            "train_df": train_df_cache[df_index],
            "env_kwargs": env_kwargs,
            "device": trainer.device,
            "leverage_choices": trainer.leverage_choices,
            "position_list": trainer.position_list,
            "initial_wallet_balance": trainer.initial_wallet_balance,
            "initial_unrealized_pnL": trainer.initial_unrealized_pnL,
            "state_dim": len(trainer.tech_indicator_list),
            "action_count": trainer.N_ACTIONS,
            "hidden_nodes": trainer.hidden_nodes,
            "time_info_dim": trainer.time_info_dim,
            "ensemble_number": trainer.N,
        }
        process = worker_context.Process(
            target=df_rollout_worker,
            args=(worker_config, input_queue, trainer.worker_result_queue),
        )
        process.start()
        trainer.worker_input_queues[df_index] = input_queue
        trainer.worker_processes.append(process)


def reset_worker_task(
    trainer,
    epoch_index,
    context_index,
    initial_action,
    active_df_indices,
):
    for df_index in sorted(active_df_indices):
        trainer.worker_input_queues[df_index].put(
            ResetWorkerTask(
                epoch_index=epoch_index,
                context_index=context_index,
                initial_action=initial_action,
            )
        )


def send_worker_rounds(
    trainer,
    active_df_indices,
    epoch_index,
    context_index,
    initial_action,
    round_counter,
    state_dict,
):
    for df_index in sorted(active_df_indices):
        trainer.worker_input_queues[df_index].put(
            ExploreWorkerRound(
                epoch_index=epoch_index,
                context_index=context_index,
                initial_action=initial_action,
                round_counter=round_counter,
                state_dict=state_dict,
                epsilon=trainer.epsilon,
                rollout_steps=trainer.rollout_steps,
            )
        )


def collect_worker_rounds(trainer, active_df_indices, round_counter):
    from RL.DiHFT.low_level.parallel_weight_advantage_pretrain import (
        WorkerErrorMessage,
    )

    expected_count = len(active_df_indices)
    results = []
    while len(results) < expected_count:
        message = trainer.worker_result_queue.get()
        if isinstance(message, WorkerErrorMessage):
            return [message]
        if not isinstance(message, WorkerRoundResult):
            raise ValueError(
                "unknown worker result message type: {}".format(
                    type(message).__name__
                )
            )
        if message.round_counter != round_counter:
            raise RuntimeError(
                "unexpected worker round_counter={} expected={}".format(
                    message.round_counter,
                    round_counter,
                )
            )
        if message.df_index not in active_df_indices:
            raise RuntimeError(
                "unexpected worker df_index={} active={}".format(
                    message.df_index,
                    sorted(active_df_indices),
                )
            )
        results.append(message)
    return sorted(results, key=lambda result: result.df_index)


def run_parallel_rollout_task(
    trainer,
    epoch_index,
    context_index,
    initial_action,
    train_df_cache,
    env_kwargs,
    buffer_diverse,
    step_counter_diverse,
    round_counter,
):
    from RL.DiHFT.low_level.parallel_weight_advantage_pretrain import (
        build_effective_df_indices,
        raise_for_worker_error,
    )

    active_df_indices = set(build_effective_df_indices(trainer.total_df_index_length))
    reset_worker_task(
        trainer,
        epoch_index,
        context_index,
        initial_action,
        active_df_indices,
    )
    task_metrics = []
    while active_df_indices:
        send_worker_rounds(
            trainer,
            active_df_indices=active_df_indices,
            epoch_index=epoch_index,
            context_index=context_index,
            initial_action=initial_action,
            round_counter=round_counter,
            state_dict=make_cpu_state_dict(trainer.eval_net),
        )
        round_results = collect_worker_rounds(trainer, active_df_indices, round_counter)
        for result in round_results:
            raise_for_worker_error(result)
        write_round_transitions_to_buffer(buffer_diverse, round_results)
        task_metrics.extend(
            metrics for result in round_results for metrics in result.rollout_metrics
        )
        round_steps = sum(result.worker_steps for result in round_results)
        previous_step_counter = step_counter_diverse
        step_counter_diverse += round_steps
        update_count = 0
        warmup_steps = trainer.batch_size * trainer.update_times + trainer.n_step
        update_windows = count_update_windows_crossed(
            previous_step_counter=previous_step_counter,
            current_step_counter=step_counter_diverse,
            rollout_steps=trainer.rollout_steps,
            warmup_steps=warmup_steps,
        )
        if update_windows:
            update_count = trainer.update_times * update_windows
            run_fixed_diverse_updates(
                trainer,
                buffer_diverse,
                update_count,
                round_counter,
            )
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
            round_summary.round_counter,
            round_summary.epoch_index,
            round_summary.context_index,
            round_summary.initial_action,
            round_summary.round_steps,
            round_summary.active_worker_count,
            round_summary.buffer_size,
            round_summary.update_count,
        )
        for result in round_results:
            for metrics in result.rollout_metrics:
                logger.info(
                    "parallel rollout metrics | epoch_index=%d | context_index=%d | "
                    "initial_action=%d | df_index=%d | transition_count=%d | "
                    "reward_sum=%.4f | final_balance=%.4f | return_rate=%.6f",
                    metrics.epoch_index,
                    metrics.context_index,
                    metrics.initial_action,
                    metrics.df_index,
                    metrics.transition_count,
                    metrics.reward_sum,
                    metrics.final_balance,
                    metrics.return_rate,
                )
        active_df_indices = {
            result.df_index
            for result in round_results
            if not result.done
        }
        round_counter += 1
    return round_counter, step_counter_diverse, task_metrics


def save_parallel_epoch_model(trainer, epoch_index):
    epoch_path = build_epoch_model_path(trainer.model_path, epoch_index)
    if not os.path.exists(epoch_path):
        os.makedirs(epoch_path)
    torch.save(
        trainer.eval_net.state_dict(),
        os.path.join(epoch_path, "trained_model.pkl"),
    )
    logger.info(
        "第 %d 轮 epoch 训练完成 | 模型已保存至=%s",
        epoch_index + 1,
        epoch_path,
    )


def run_parallel_diverse_training(
    trainer,
    train_df_cache,
    env_kwargs,
    buffer_diverse,
    step_counter_diverse,
    diverse_rollout_latest_metrics_by_df,
):
    if trainer.total_df_index_length <= 0:
        raise ValueError("parallel diverse training requires total_df_index_length > 0")
    round_counter = 0
    start_parallel_workers(trainer, train_df_cache, env_kwargs)
    try:
        for epoch_index in range(trainer.num_epoch):
            params = compute_epoch_training_params(
                epoch_index=epoch_index,
                num_epoch=trainer.num_epoch,
                epsilon_init=trainer.epsilon_init,
                epsilon_min=trainer.epsilon_min,
                ada_init=trainer.ada_init,
                ada_min=trainer.ada_min,
                lr_init=trainer.lr_init,
                lr_min=trainer.lr_min,
            )
            trainer.epsilon = params.epsilon
            trainer.ada = params.ada
            trainer.lr = params.lr
            for param_group in trainer.optimizer.param_groups:
                param_group["lr"] = trainer.lr

            epoch_metrics = []
            for context_index in range(trainer.N):
                context_metrics = []
                for initial_action in range(trainer.position_choices):
                    (
                        round_counter,
                        step_counter_diverse,
                        task_metrics,
                    ) = run_parallel_rollout_task(
                        trainer,
                        epoch_index=epoch_index,
                        context_index=context_index,
                        initial_action=initial_action,
                        train_df_cache=train_df_cache,
                        env_kwargs=env_kwargs,
                        buffer_diverse=buffer_diverse,
                        step_counter_diverse=step_counter_diverse,
                        round_counter=round_counter,
                    )
                    for metrics in task_metrics:
                        record_diverse_rollout_latest_metric(
                            diverse_rollout_latest_metrics_by_df,
                            metrics.df_index,
                            context_index,
                            metrics.reward_sum,
                            metrics.final_balance,
                            metrics.return_rate,
                        )
                    context_metrics.extend(task_metrics)
                    epoch_metrics.extend(task_metrics)
                if context_metrics:
                    context_summary = summarize_rollout_metrics(context_metrics)
                    trainer.writer.add_scalar(
                        tag="return_rate_train_{}".format(context_index),
                        scalar_value=context_summary.mean_return_rate,
                        global_step=epoch_index + 1,
                        walltime=None,
                    )
                    trainer.writer.add_scalar(
                        tag="reward_sum_train_{}".format(context_index),
                        scalar_value=context_summary.mean_reward_sum,
                        global_step=epoch_index + 1,
                        walltime=None,
                    )
            if epoch_metrics:
                epoch_summary = summarize_rollout_metrics(epoch_metrics)
                trainer.writer.add_scalar(
                    tag="epoch_return_rate_train",
                    scalar_value=epoch_summary.mean_return_rate,
                    global_step=epoch_index + 1,
                    walltime=None,
                )
                trainer.writer.add_scalar(
                    tag="epoch_final_balance_train",
                    scalar_value=epoch_summary.mean_final_balance,
                    global_step=epoch_index + 1,
                    walltime=None,
                )
                trainer.writer.add_scalar(
                    tag="epoch_reward_sum_train",
                    scalar_value=epoch_summary.mean_reward_sum,
                    global_step=epoch_index + 1,
                    walltime=None,
                )
            log_diverse_rollout_latest_metrics(
                epoch_index + 1,
                diverse_rollout_latest_metrics_by_df,
                logger,
            )
            save_parallel_epoch_model(trainer, epoch_index)
    finally:
        trainer._shutdown_parallel_workers()
    return step_counter_diverse


def update(
    trainer,
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
        trainer.device
    )
    trading_info = info["trading_info"].float().to(trainer.device)
    # next input
    states_ = next_states.reshape(bs, -1)
    previous_action_ = info_["previous_action"].float().unsqueeze(1)
    avaliable_action_ = info_["avaliable_action"]
    hour_count_down_ = info_["funding_count_down_hour"].float().unsqueeze(1)
    minute_count_down_ = info_["funding_count_down_minute"].float().unsqueeze(1)
    time_input_ = torch.cat([hour_count_down_, minute_count_down_], dim=1).to(
        trainer.device
    )
    trading_info_ = info_["trading_info"].float().to(trainer.device)

    current_sa_quantiles = evaluate_quantile_at_action(
        trainer.eval_net(
            state=states,
            time=time_input,
            previous_action=previous_action,
            avaliable_action=avaliable_action,
            trading_info=trading_info,
        ),
        actions,
    )
    assert current_sa_quantiles.shape == (bs, trainer.N, 1)
    with torch.no_grad():
        next_q = trainer.target_net.get_best_q(
            state=states_,
            time=time_input_,
            previous_action=previous_action_,
            avaliable_action=avaliable_action_,
            trading_info=trading_info_,
        )
        next_sa_quantiles = next_q.unsqueeze(1)
        assert next_sa_quantiles.shape == (trainer.batch_size, 1, trainer.N)
        target_sa_quantiles = (
            rewards[..., None]
            + (1.0 - dones[..., None]) * trainer.gamma * next_sa_quantiles
        )
        assert target_sa_quantiles.shape == (trainer.batch_size, 1, trainer.N)
    td_errors = target_sa_quantiles - current_sa_quantiles
    # logger.info("td_errors %s", td_errors)
    assert td_errors.shape == (trainer.batch_size, trainer.N, trainer.N)
    if trainer.if_use_hubber_loss:
        td_errors = calculate_huber_loss(td_errors)
    batch_weights, partial_td_error_loss = calculate_paper_partial_loss(
        td_errors,
        trainer.neighbor_size,
    )
    predict_action_distrbution = trainer.eval_net(
        state=states,
        time=time_input,
        previous_action=previous_action,
        avaliable_action=avaliable_action,
        trading_info=trading_info,
    )
    assert predict_action_distrbution.shape == (
        trainer.batch_size,
        trainer.N,
        trainer.N_ACTIONS,
    )
    assert batch_weights.shape == (trainer.batch_size, trainer.N)

    q_value = recalculate_q_demonstration(
        info["q_value"],
        info["avaliable_action"],
    )
    KL_div = calculate_paper_supervisor_kl_loss(
        predict_action_distrbution,
        q_value,
        batch_weights,
    )
    loss = partial_td_error_loss + KL_div * trainer.ada
    update_params(
        trainer.optimizer,
        loss,
        trainer.eval_net,
        retain_graph=False,
        grad_cliping=trainer.grad_clip,
    )
    soft_copy_params(trainer.eval_net, trainer.target_net, trainer.tau)
    trainer.update_counter += 1
    return loss.item(), KL_div.item(), partial_td_error_loss.item()
