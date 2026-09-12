# Parallel diverse-training collection + orchestration components extracted
# from parallel_weight_advantage_pretrain.py for easier review.
#
# This module holds the model-driven rollout runner, rollout metrics dataclasses,
# the pure helpers and the parallel diverse-training orchestration.
#
# 训练流程（每个 epoch 严格遵循）：
#   1. 完整探索 —— 每轮探索创建全新的探索子进程（数据处理方式与 pretrain
#      阶段完全一致：所有 df 以 round-robin 分配给子进程，子进程数量上限
#      MAX_EXPLORATION_WORKERS=20），覆盖全部 context × initial_action 任务；
#      worker 侧的 explore_round 单条消息一次性探索至回合结束（整个 df），
#      主进程每个任务只需一轮派发/收集，无多轮重派发机制。
#   2. 彻底关闭 —— 探索全部完成后，先关闭所有探索子进程并逐一确认退出，
#      之后才允许进入训练阶段（避免子进程与训练争抢 GPU/CPU 资源）。
#   3. 保存经验池 —— 探索结束后将经验池快照保存到
#      model_path/buffer_diverse.pkl（覆盖式保存最新快照）。
#   4. 完整训练 —— 对已冻结的经验池统一执行本轮全部参数更新
#      （StackedTransitionSampler 预堆叠采样，避免逐元素 np.stack）。
#   5. 训练结束后进入下一轮探索（回到 1）。
# 经验唯一性由内容指纹保证：已存在相同指纹的经验不会重复写入经验池。
# 连续 MAX_CONSECUTIVE_NO_NEW_EXPERIENCE_EPOCHS 个 epoch 探索均未新增任何
# 经验后，后续 epoch 跳过探索阶段，仅执行训练。
#
# 依赖方向：本模块在顶层导入编排模块 parallel_weight_advantage_pretrain 的
# 共享基础设施；编排模块改为在 df_rollout_worker 函数内部延迟导入本模块，
# 以保持模块加载依赖图无环。

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    import pandas as pd
    from RL.DiHFT.low_level.parallel_weight_advantage_pretrain import (
        Weighted_Contexts_DQN,
    )
    from RL.util.replay_buffer_DQN import Multi_step_ReplayBuffer_multi_info
    from RL.util.regime_stratified_replay_buffer import (
        RegimeStratifiedReplayBuffer,
        accumulate_trajectory_n_step,
    )

from model.low_level import ensemble_Qnet
from RL.DiHFT.low_level.pretrain_qtable_diagnostics import (
    build_initial_state,
    create_demo_env,
)
from RL.DiHFT.low_level.parallel_pretrain import (
    StackedTransitionSampler,
    extract_stacked_tensor_dict,
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
from RL.util.regime_stratified_replay_buffer import (
    RegimeStratifiedReplayBuffer,
    accumulate_trajectory_n_step,
)
from RL.DiHFT.low_level.parallel_weight_advantage_pretrain import (
    WorkerErrorMessage,
    build_effective_df_indices,
    create_worker_context,
    df_rollout_worker,
    raise_for_worker_error,
    shutdown_workers,
)
from RL.DiHFT.low_level.evaluate_sub_agents import evaluates

# 探索子进程数量上限（严格控制为 20）：df 数量更多时按 round-robin 分配给子进程
MAX_EXPLORATION_WORKERS = 20
# 每个 epoch 训练阶段的更新窗口数（每窗口执行 trainer.update_times 次参数更新）
UPDATE_WINDOWS_PER_EPOCH = 30
# 连续多少个 epoch 探索未新增任何经验后，后续 epoch 不再探索（仅训练）
MAX_CONSECUTIVE_NO_NEW_EXPERIENCE_EPOCHS = 5
# 关闭子进程时 join 的超时秒数；超时未退出的进程以 terminate 兜底
WORKER_JOIN_TIMEOUT_SECONDS = 10

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
    df_index: int
    epoch_index: int
    context_index: int
    initial_action: int


@dataclass(frozen=True)
class ExploreWorkerRound:
    df_index: int
    epoch_index: int
    context_index: int
    initial_action: int
    round_counter: int
    state_dict: dict
    epsilon: float


@dataclass(frozen=True)
class WorkerTransitionRecord:
    step_index: int
    transition: object
    td_error: float = 0.0


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


@dataclass(frozen=True)
class ParallelRoundSummary:
    round_counter: int
    epoch_index: int
    context_index: int
    initial_action: int
    round_steps: int
    active_worker_count: int
    buffer_size: int

    def to_dict(self):
        return {
            "round_counter": self.round_counter,
            "epoch_index": self.epoch_index,
            "context_index": self.context_index,
            "initial_action": self.initial_action,
            "round_steps": self.round_steps,
            "active_worker_count": self.active_worker_count,
            "buffer_size": self.buffer_size,
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
    decay_epochs=None,
):
    effective_decay_epochs = num_epoch if decay_epochs is None else decay_epochs
    return EpochTrainingParams(
        epsilon=_linear_value(
            epsilon_init, epsilon_min, epoch_index, effective_decay_epochs
        ),
        ada=_linear_value(ada_init, ada_min, epoch_index, effective_decay_epochs),
        lr=_held_then_linear_value(lr_init, lr_min, epoch_index, num_epoch),
    )


def apply_epoch_training_params(trainer: Weighted_Contexts_DQN, epoch_index: int):
    """按 epoch 计算并写入 epsilon/ada/lr（学习率同步到 optimizer 参数组）。"""
    params = compute_epoch_training_params(
        epoch_index=epoch_index,
        num_epoch=trainer.num_epoch,
        epsilon_init=trainer.epsilon_init,
        epsilon_min=trainer.epsilon_min,
        ada_init=trainer.ada_init,
        ada_min=trainer.ada_min,
        lr_init=trainer.lr_init,
        lr_min=trainer.lr_min,
        decay_epochs=trainer.decay_epochs,
    )
    trainer.epsilon = params.epsilon
    trainer.ada = params.ada
    trainer.lr = params.lr
    for param_group in trainer.optimizer.param_groups:
        param_group["lr"] = trainer.lr
    logger.info("epoch %d: epsilon=%.6f, ada=%.6f, lr=%.6f", epoch_index, params.epsilon, params.ada, params.lr)


def make_cpu_state_dict(module: torch.nn.Module):
    """生成与模型存储完全独立的 CPU numpy state_dict，用于跨进程传输。

    torch tensor 经 torch.multiprocessing 队列传输时会为每个 tensor 分配
    共享内存 fd（外加 resource_sharer socket），一次性向多个 worker 派发
    整个 state_dict 会耗尽文件描述符（Errno 24 Too many open files）。
    numpy 数组按纯字节序列化，不占用任何 fd。
    """
    return {
        name: tensor.detach().cpu().clone().numpy()
        for name, tensor in module.state_dict().items()
    }


def load_worker_state_dict(model: torch.nn.Module, state_dict: dict[str, Any]):
    """将 numpy state_dict 转回 tensor 并载入 worker 侧模型。"""
    model.load_state_dict(
        {name: torch.from_numpy(value) for name, value in state_dict.items()}
    )


def sort_round_records(
    round_results: list[WorkerRoundResult],
) -> list[tuple[int, WorkerTransitionRecord]]:
    ordered = []
    for result in sorted(round_results, key=lambda item: item.df_index):
        for record in sorted(
            result.transitions,
            key=lambda transition: transition.step_index,
        ):
            ordered.append((result.df_index, record))
    return ordered


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


def _freeze_transition_value(value):
    if isinstance(value, np.ndarray):
        return ("ndarray", value.shape, value.dtype.str, value.tobytes())
    if isinstance(value, np.generic):
        return ("npscalar", value.dtype.str, value.item())
    if isinstance(value, (bool, int, float, str)):
        return ("scalar", type(value).__name__, value)
    if isinstance(value, (list, tuple)):
        return ("sequence", tuple(_freeze_transition_value(item) for item in value))
    if isinstance(value, dict):
        return (
            "mapping",
            tuple(
                (str(key), _freeze_transition_value(value[key]))
                for key in sorted(value, key=str)
            ),
        )
    return ("other", type(value).__name__, repr(value))


def build_semantic_transition_key(
    state: np.ndarray,
    action: int,
    info: dict[str, Any],
) -> tuple[int, int, int, int, int]:
    """提取经验的语义决策键，剔除浮点路径噪音、累计 reward 与 Teacher Q 值。

    键由以下离散正交状态组成：
    (state_hash, previous_action, action, pos_dir, dur_bucket)
    """
    state_hash = int.from_bytes(
        hashlib.blake2b(state.tobytes(), digest_size=8).digest(), "big"
    )

    previous_action = int(info["previous_action"])
    trading_info = info["trading_info"]
    pos_dir = int(np.round(float(trading_info[0])))
    raw_steps = int(float(trading_info[3]) * 180)
    if raw_steps <= 2:
        dur_bucket = 0
    elif raw_steps <= 12:
        dur_bucket = 1
    elif raw_steps <= 36:
        dur_bucket = 2
    else:
        dur_bucket = 3

    return (state_hash, previous_action, int(action), pos_dir, dur_bucket)


def build_transition_fingerprint(transition):
    """计算经验的內容指纹。

    state / info / action / reward / next_state / next_info / done 完全一致的
    transition 视为同一条经验。指纹为 64 位整数摘要（blake2b），在常规经验池
    规模（<1e8 条）下碰撞概率可忽略。
    """
    frozen = _freeze_transition_value(tuple(transition))
    digest = hashlib.blake2b(repr(frozen).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def write_round_transitions_to_buffer(
    buffer_diverse: RegimeStratifiedReplayBuffer,
    round_results: list[WorkerRoundResult],
    seen_fingerprints: Any = None,
) -> int:
    """按 (df_index, step_index) 顺序写入体制分层经验池，基于各格语义键与 TD-Error 择优保留。"""
    duplicate_count = 0
    records = sort_round_records(round_results)
    for df_index, record in records:
        grid_id = buffer_diverse.add_transition(
            record.transition,
            td_error=record.td_error,
        )
        if grid_id in (-1, -2):
            duplicate_count += 1
    return duplicate_count


def run_diverse_training_phase(
    trainer: Weighted_Contexts_DQN,
    buffer_diverse: RegimeStratifiedReplayBuffer,
    update_count: int,
    epoch_index: int,
):
    """完整训练阶段：根据轮次课程表激活对应体制经验池子集并执行分层平衡更新。"""
    if update_count <= 0:
        logger.info(
            "diverse training phase skipped | epoch_index=%d | update_count=%d",
            epoch_index,
            update_count,
        )
        return None
    sampler = buffer_diverse.create_sampler(
        epoch_index=epoch_index,
        block_epochs=trainer.curriculum_block_epochs,
    )
    buffer_size = len(buffer_diverse)

    last_losses = None
    for _window in range(UPDATE_WINDOWS_PER_EPOCH):
        logger.info(
            "diverse training phase window | epoch_index=%d | window=%d | update_count=%d | "
            "buffer_size=%d",
            epoch_index,
            _window,
            update_count,
            buffer_size,
        )
        for _ in range(update_count):
            (
                states,
                infos,
                actions,
                rewards,
                next_states,
                next_infos,
                dones,
            ) = sampler.sample()
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
            trainer.writer.add_scalar("total_loss", total_loss, trainer.update_counter)
            trainer.writer.add_scalar("KL_loss", KL_loss, trainer.update_counter)
            trainer.writer.add_scalar("td_loss", td_loss, trainer.update_counter)
        logger.info(
            "diverse training phase complete | epoch_index=%d | update_count=%d | "
            "total_loss=%.6f | KL_loss=%.6f | td_loss=%.6f",
            epoch_index,
            update_count,
            last_losses[0],
            last_losses[1],
            last_losses[2],
        )
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
):
    return ParallelRoundSummary(
        round_counter=int(round_counter),
        epoch_index=int(epoch_index),
        context_index=int(context_index),
        initial_action=int(initial_action),
        round_steps=int(sum(result.worker_steps for result in round_results)),
        active_worker_count=int(len(round_results)),
        buffer_size=int(buffer_size),
    )


def build_epoch_model_path(model_path: str, epoch_index: int) -> str:
    return os.path.join(model_path, "epoch_{}".format(epoch_index + 1))


def build_diverse_buffer_path(model_path: str) -> str:
    return os.path.join(model_path, "buffer_diverse.pkl")


def save_diverse_buffer(buffer_diverse: RegimeStratifiedReplayBuffer, model_path: str):
    """将 9 格经验池快照以分批张量化格式保存到文件。"""
    buffer_path = build_diverse_buffer_path(model_path)
    payload = buffer_diverse.extract_stacked_tensor_dicts()
    torch.save(payload, buffer_path)
    logger.info(
        "diverse buffer snapshot saved | path=%s | memory_size=%d",
        buffer_path,
        len(buffer_diverse),
    )
    return payload


@dataclass
class _WorkerEpisode:
    """单个 df 在一个探索任务内的回合状态（子进程侧）。"""

    env: object
    state: object
    info: dict
    done: bool
    reward_sum: float = 0.0
    transition_count: int = 0


class DfRolloutWorkerRunner:
    """探索子进程内的回合运行器：单个子进程可同时服务多个 df。

    与 pretrain 阶段完全相同的数据处理方式：主进程把若干 df 以
    round-robin 方式分配给每个子进程，ResetWorkerTask / ExploreWorkerRound
    消息携带 df_index，本类按 df_index 维护各自独立的 env 与回合状态。
    """

    def __init__(self, worker_config):
        self.df_indices = worker_config["df_indices"]
        self.train_df_by_df = worker_config["train_df_by_df"]
        self.env_kwargs = worker_config["env_kwargs"]
        self.device = worker_config["device"]
        self.leverage_choices = worker_config["leverage_choices"]
        self.position_list = worker_config["position_list"]
        self.initial_wallet_balance = worker_config["initial_wallet_balance"]
        self.initial_unrealized_pnL = worker_config["initial_unrealized_pnL"]
        self.gamma = float(worker_config.get("gamma", 0.99))
        self.n_step = int(worker_config.get("n_step", 12))
        self.model = create_parallel_worker_model(worker_config).to(self.device)
        # df_index -> 该 df 当前探索任务的回合状态
        self.episodes = {}

    def reset_task(self, message):
        """按 message.df_index 为对应数据文件重建 env 并重置回合状态。"""
        train_df = self.train_df_by_df[message.df_index]
        _, _, _, initial_state = build_initial_state(
            train_df,
            message.initial_action,
            self.leverage_choices,
            self.position_list,
            self.initial_wallet_balance,
            self.initial_unrealized_pnL,
        )
        env = create_demo_env(train_df, self.env_kwargs, initial_state)
        state, info = env.reset()
        self.episodes[message.df_index] = _WorkerEpisode(
            env=env,
            state=state,
            info=info,
            done=False,
        )

    def _act(self, state, info, context_index, epsilon):
        if np.random.uniform() <= epsilon:
            return int(np.random.choice(info["avaiable_action_list"])), 0.0
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
            context_q = q_values[:, context_index, :]
            action = int(torch.max(context_q, 1)[1].data.cpu().numpy()[0])
            chosen_q = float(context_q[0, action].item())
            return action, chosen_q

    def explore_round(self, message):
        """对 message.df_index 执行完整探索：单条消息一次性跑到回合结束。

        探索完整性由 worker 侧保证：循环 env.step 直到 episode.done（df 数据
        走完或爆仓）才返回，不依赖主进程的多轮重派发机制。
        """
        episode = self.episodes[message.df_index]
        load_worker_state_dict(self.model, message.state_dict)
        self.model.eval()
        transitions = []
        while not episode.done:
            action, chosen_q = self._act(
                episode.state,
                episode.info,
                message.context_index,
                message.epsilon,
            )
            next_state, reward, done, next_info = episode.env.step(action)
            if "q_value" in episode.info and len(episode.info["q_value"]) > action:
                teacher_q = float(episode.info["q_value"][action])
                td_error = abs(teacher_q - chosen_q)
            else:
                td_error = abs(float(reward) - chosen_q)

            transitions.append(
                WorkerTransitionRecord(
                    step_index=episode.transition_count,
                    transition=(
                        episode.state,
                        episode.info,
                        action,
                        reward,
                        next_state,
                        next_info,
                        done,
                    ),
                    td_error=float(td_error),
                )
            )
            episode.reward_sum += reward
            episode.transition_count += 1
            episode.state, episode.info, episode.done = next_state, next_info, done
        final_balance = episode.env.unrealized_pnl + episode.env.wallet_balance
        raw_tuples = [r.transition for r in transitions]
        acc_tuples = accumulate_trajectory_n_step(
            raw_tuples,
            gamma=self.gamma,
            n_step=self.n_step,
        )
        accumulated_transitions = [
            WorkerTransitionRecord(
                step_index=r.step_index,
                transition=acc_t,
                td_error=r.td_error,
            )
            for r, acc_t in zip(transitions, acc_tuples)
        ]
        return WorkerRoundResult(
            df_index=message.df_index,
            epoch_index=message.epoch_index,
            context_index=message.context_index,
            initial_action=message.initial_action,
            round_counter=message.round_counter,
            worker_steps=len(accumulated_transitions),
            transitions=accumulated_transitions,
            rollout_metrics=[
                RolloutMetrics(
                    epoch_index=message.epoch_index,
                    context_index=message.context_index,
                    initial_action=message.initial_action,
                    df_index=message.df_index,
                    transition_count=episode.transition_count,
                    reward_sum=float(episode.reward_sum),
                    final_balance=float(final_balance),
                    return_rate=float(
                        final_balance / (self.initial_wallet_balance + 1e-12) - 1
                    ),
                )
            ],
            done=episode.done,
        )


def start_parallel_workers(
    trainer: Weighted_Contexts_DQN,
    train_df_cache: dict[int, pd.DataFrame],
    env_kwargs: dict[str, Any],
):
    """为本轮探索创建全新的子进程（与 pretrain 阶段相同的数据处理方式）。

    所有 df 以 round-robin 方式分配给子进程：子进程数量 = min(df 数量,
    MAX_EXPLORATION_WORKERS)，每个子进程负责多个 df，消息携带 df_index 路由。
    每轮探索（每个 epoch）开始前调用，上一轮的子进程此时应已被彻底关闭。
    """
    trainer.worker_input_queues = {}
    trainer.worker_processes = []
    worker_context = create_worker_context()
    trainer.worker_result_queue = worker_context.Queue()
    effective_df_indices = build_effective_df_indices(trainer.total_df_index_length)
    num_workers = min(len(effective_df_indices), MAX_EXPLORATION_WORKERS)
    for worker_id in range(num_workers):
        assigned_df_indices = [
            df_index
            for i, df_index in enumerate(effective_df_indices)
            if i % num_workers == worker_id
        ]
        input_queue = worker_context.Queue()
        worker_config = {
            "worker_id": worker_id,
            "df_index": assigned_df_indices[0],
            "df_indices": assigned_df_indices,
            "train_df_by_df": {
                df_index: train_df_cache[df_index]
                for df_index in assigned_df_indices
            },
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
            "gamma": trainer.gamma,
            "n_step": trainer.n_step,
        }
        process = worker_context.Process(
            target=df_rollout_worker,
            args=(worker_config, input_queue, trainer.worker_result_queue),
        )
        process.start()
        for df_index in assigned_df_indices:
            trainer.worker_input_queues[df_index] = input_queue
        trainer.worker_processes.append(process)
    logger.info(
        "exploration workers started | df_count=%d | worker_count=%d | "
        "max_workers=%d",
        len(effective_df_indices),
        num_workers,
        MAX_EXPLORATION_WORKERS,
    )


def shutdown_exploration_workers(trainer: Weighted_Contexts_DQN):
    """彻底关闭全部探索子进程，并逐一确认退出后才返回。

    训练阶段启动前必须调用：先发送 ShutdownWorker 让各子进程正常退出并
    join 等待；超时未退出的进程以 terminate 兜底后再次 join；若仍有存活
    进程则视为资源泄漏，直接抛错阻止训练启动。
    """
    shutdown_workers(
        trainer.worker_input_queues.values(),
        trainer.worker_processes,
    )
    for process in trainer.worker_processes:
        process.join(timeout=WORKER_JOIN_TIMEOUT_SECONDS)
        if process.is_alive():
            process.terminate()
            process.join(timeout=WORKER_JOIN_TIMEOUT_SECONDS)
    alive_pids = [
        process.pid
        for process in trainer.worker_processes
        if process.is_alive()
    ]
    trainer.worker_input_queues = {}
    trainer.worker_processes = []
    trainer.worker_result_queue = None
    if alive_pids:
        raise RuntimeError(
            "exploration workers failed to terminate: pids={}".format(alive_pids)
        )
    logger.info("exploration workers shut down")


def reset_worker_task(
    trainer: Weighted_Contexts_DQN,
    epoch_index: int,
    context_index: int,
    initial_action: int,
    active_df_indices: set[int] | list[int],
):
    """为每个 df 派发回合重置消息（按 df_index 路由到对应回合状态）。"""
    for df_index in sorted(active_df_indices):
        trainer.worker_input_queues[df_index].put(
            ResetWorkerTask(
                df_index=df_index,
                epoch_index=epoch_index,
                context_index=context_index,
                initial_action=initial_action,
            )
        )


def send_worker_rounds(
    trainer: Weighted_Contexts_DQN,
    active_df_indices: set[int] | list[int],
    epoch_index: int,
    context_index: int,
    initial_action: int,
    round_counter: int,
    state_dict: dict[str, Any],
):
    """为全部 df 派发一轮探索消息（携带最新模型参数与 epsilon）。

    worker 侧的 explore_round 收到消息后一次性探索至回合结束（整个 df），
    因此每个任务只需派发一轮消息，无需多轮重派发。
    """
    for df_index in sorted(active_df_indices):
        trainer.worker_input_queues[df_index].put(
            ExploreWorkerRound(
                df_index=df_index,
                epoch_index=epoch_index,
                context_index=context_index,
                initial_action=initial_action,
                round_counter=round_counter,
                state_dict=state_dict,
                epsilon=trainer.epsilon,
            )
        )


def collect_worker_rounds(
    trainer: Weighted_Contexts_DQN,
    active_df_indices: set[int] | list[int],
    round_counter: int,
):
    """收集本轮全部活跃 df 的结果；任一子进程上报错误则原样返回交由上层抛错。"""
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
    trainer: Weighted_Contexts_DQN,
    epoch_index: int,
    context_index: int,
    initial_action: int,
    buffer_diverse: Multi_step_ReplayBuffer_multi_info,
    step_counter_diverse: int,
    round_counter: int,
    seen_fingerprints: set[Any],
):
    """单个 (epoch, context, initial_action) 任务的完整探索，不执行任何参数更新。

    探索完整性由 worker 侧的 explore_round 保证：单条消息一次性探索至回合
    结束（整个 df）。主进程只需一轮派发/收集；若任一 worker 上报未 done，
    则违反探索完整性约定，直接抛错。
    """
    active_df_indices = set(build_effective_df_indices(trainer.total_df_index_length))
    reset_worker_task(
        trainer,
        epoch_index,
        context_index,
        initial_action,
        active_df_indices,
    )
    task_metrics = []
    task_start_step_counter = step_counter_diverse
    task_record_count = 0
    task_duplicate_count = 0
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
    unfinished_df_indices = sorted(
        result.df_index for result in round_results if not result.done
    )
    if unfinished_df_indices:
        raise RuntimeError(
            "worker round finished without done | df_indices={}".format(
                unfinished_df_indices
            )
        )
    task_duplicate_count += write_round_transitions_to_buffer(
        buffer_diverse,
        round_results,
        seen_fingerprints,
    )
    task_record_count += sum(
        len(result.transitions) for result in round_results
    )
    task_metrics.extend(
        metrics for result in round_results for metrics in result.rollout_metrics
    )
    step_counter_diverse += sum(result.worker_steps for result in round_results)
    round_summary = summarize_parallel_round(
        round_counter=round_counter,
        epoch_index=epoch_index,
        context_index=context_index,
        initial_action=initial_action,
        round_results=round_results,
        buffer_size=len(buffer_diverse),
    )
    logger.info(
        "parallel rollout round complete | round_counter=%d | epoch_index=%d | "
        "context_index=%d | initial_action=%d | round_steps=%d | "
        "active_worker_count=%d | buffer_size=%d",
        round_summary.round_counter,
        round_summary.epoch_index,
        round_summary.context_index,
        round_summary.initial_action,
        round_summary.round_steps,
        round_summary.active_worker_count,
        round_summary.buffer_size,
    )
    logger.info(
        "diverse rollout task exploration complete | epoch_index=%d | "
        "context_index=%d | initial_action=%d | steps_collected=%d | "
        "records=%d | duplicates_skipped=%d | buffer_size=%d",
        epoch_index,
        context_index,
        initial_action,
        step_counter_diverse - task_start_step_counter,
        task_record_count,
        task_duplicate_count,
        len(buffer_diverse),
    )
    return round_counter + 1, step_counter_diverse, task_metrics


def save_parallel_epoch_model(trainer: Weighted_Contexts_DQN, epoch_index: int):
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


def write_context_rollout_scalars(
    trainer: Weighted_Contexts_DQN,
    context_index: int,
    context_metrics: list[RolloutMetrics],
    epoch_index: int,
):
    """写入单个 context 的训练期 rollout 标量。"""
    if not context_metrics:
        return
    summary = summarize_rollout_metrics(context_metrics)
    trainer.writer.add_scalar(
        tag="return_rate_train_{}".format(context_index),
        scalar_value=summary.mean_return_rate,
        global_step=epoch_index + 1,
        walltime=None,
    )
    trainer.writer.add_scalar(
        tag="reward_sum_train_{}".format(context_index),
        scalar_value=summary.mean_reward_sum,
        global_step=epoch_index + 1,
        walltime=None,
    )


def write_epoch_rollout_scalars(
    trainer: Weighted_Contexts_DQN,
    epoch_metrics: list[RolloutMetrics],
    epoch_index: int,
):
    """写入整个 epoch 的训练期 rollout 标量。"""
    if not epoch_metrics:
        return
    summary = summarize_rollout_metrics(epoch_metrics)
    trainer.writer.add_scalar(
        tag="epoch_return_rate_train",
        scalar_value=summary.mean_return_rate,
        global_step=epoch_index + 1,
        walltime=None,
    )
    trainer.writer.add_scalar(
        tag="epoch_final_balance_train",
        scalar_value=summary.mean_final_balance,
        global_step=epoch_index + 1,
        walltime=None,
    )
    trainer.writer.add_scalar(
        tag="epoch_reward_sum_train",
        scalar_value=summary.mean_reward_sum,
        global_step=epoch_index + 1,
        walltime=None,
    )


def is_buffer_full(
    buffer_diverse: RegimeStratifiedReplayBuffer,
    trainer: Weighted_Contexts_DQN,
) -> bool:
    """判断经验池是否已达到容量上限。"""
    return len(buffer_diverse) >= trainer.buffer_size


def run_epoch_exploration(
    trainer: Weighted_Contexts_DQN,
    epoch_index: int,
    train_df_cache: dict[int, pd.DataFrame],
    env_kwargs: dict[str, Any],
    buffer_diverse: Multi_step_ReplayBuffer_multi_info,
    step_counter_diverse: int,
    round_counter: int,
    seen_fingerprints: set[Any],
    diverse_rollout_latest_metrics_by_df: dict[int, Any],
):
    """一个 epoch 的完整探索阶段：创建全新子进程 -> 全任务探索 -> 彻底关闭。

    返回 (epoch_metrics, step_counter_diverse, round_counter)。无论探索正常
    结束还是中途异常，finally 都会彻底关闭本轮全部探索子进程并确认退出，
    保证训练阶段开始前不存在任何存活子进程。
    """
    epoch_metrics = []
    epoch_start_step_counter = step_counter_diverse
    explored_task_count = 0
    buffer_full = False
    try:
        start_parallel_workers(trainer, train_df_cache, env_kwargs)
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
                    buffer_diverse=buffer_diverse,
                    step_counter_diverse=step_counter_diverse,
                    round_counter=round_counter,
                    seen_fingerprints=seen_fingerprints,
                )
                explored_task_count += 1
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
                if is_buffer_full(buffer_diverse, trainer):
                    buffer_full = True
                    break
            if buffer_full:
                break
            write_context_rollout_scalars(
                trainer, context_index, context_metrics, epoch_index
            )
             
        # 探索完整性确认：全部任务结束或经验池已满
        logger.info(
            "epoch exploration complete | epoch_index=%d | explored_tasks=%d | "
            "steps_collected=%d | buffer_size=%d",
            epoch_index,
            explored_task_count,
            step_counter_diverse - epoch_start_step_counter,
            len(buffer_diverse),
        )
    finally:
        # 所有探索子进程彻底关闭并确认退出后，才允许进入训练阶段
        shutdown_exploration_workers(trainer)
    return epoch_metrics, step_counter_diverse, round_counter


def run_parallel_diverse_training(
    trainer: Weighted_Contexts_DQN,
    train_df_cache: dict[int, pd.DataFrame],
    env_kwargs: dict[str, Any],
    buffer_diverse: RegimeStratifiedReplayBuffer,
    step_counter_diverse: int,
    diverse_rollout_latest_metrics_by_df: dict[int, Any],
):
    """多样化训练主循环：每个 epoch 严格遵循「完整探索 -> 完整训练」。

    阶段切换条件：
    - 探索 -> 训练：本 epoch 内全部 (context_index, initial_action) 任务均完成
      对所有 df 的完整探索（worker 侧 explore_round 单条消息一次性探索至
      回合结束，主进程一轮派发/收集即完成），且全部探索子进程已被彻底
      关闭并确认退出；
    - 训练 -> 下一轮探索：本轮全部参数更新执行完毕并保存模型。
    经验按内容指纹去重，重复经验不会写入经验池。
    每次探索完成后将经验池快照保存到 model_path/buffer_diverse.pkl。
    连续 MAX_CONSECUTIVE_NO_NEW_EXPERIENCE_EPOCHS 个 epoch 探索均未新增任何
    经验后，后续 epoch 跳过探索阶段（不再创建子进程），仅执行训练。
    """
    if trainer.total_df_index_length <= 0:
        raise ValueError("parallel diverse training requires total_df_index_length > 0")
    round_counter = 0
    seen_fingerprints = {}
    consecutive_no_new_experience_epochs = 0
    skip_exploration = False
    tensor_snapshot = None
    best_loss = float("inf")
    best_model_file = None
    best_epoch_index = -1
    for epoch_index in range(trainer.num_epoch):
        apply_epoch_training_params(trainer, epoch_index)
        buffer_full = is_buffer_full(buffer_diverse, trainer)
        if skip_exploration or buffer_full:
            skip_exploration = True
            logger.info(
                "epoch exploration skipped | epoch_index=%d | buffer_full=%s | "
                "buffer_size=%d | consecutive_no_new_experience_epochs=%d",
                epoch_index,
                buffer_full,
                len(buffer_diverse),
                consecutive_no_new_experience_epochs,
            )
            epoch_metrics = []
        else:
            # 阶段一：完整探索 —— 本轮探索创建全新子进程（上限 20），结束即彻底关闭
            buffer_len_before_exploration = len(buffer_diverse)
            epoch_metrics, step_counter_diverse, round_counter = run_epoch_exploration(
                trainer,
                epoch_index,
                train_df_cache,
                env_kwargs,
                buffer_diverse,
                step_counter_diverse,
                round_counter,
                seen_fingerprints,
                diverse_rollout_latest_metrics_by_df,
            )
            # 阶段二：保存经验池 —— 探索完成且经验已全部写入后，落盘最新快照
            tensor_snapshot = save_diverse_buffer(buffer_diverse, trainer.model_path)
            new_experience_count = (
                len(buffer_diverse) - buffer_len_before_exploration
            )
            if new_experience_count == 0:
                consecutive_no_new_experience_epochs += 1
            else:
                consecutive_no_new_experience_epochs = 0
            if is_buffer_full(buffer_diverse, trainer):
                skip_exploration = True
                logger.info(
                    "buffer full reached | epoch_index=%d | buffer_size=%d | "
                    "subsequent epochs will skip exploration",
                    epoch_index,
                    len(buffer_diverse),
                )
            elif (
                consecutive_no_new_experience_epochs
                >= MAX_CONSECUTIVE_NO_NEW_EXPERIENCE_EPOCHS
            ):
                skip_exploration = True
                logger.info(
                    "exploration exhausted | epoch_index=%d | "
                    "consecutive_no_new_experience_epochs=%d | "
                    "subsequent epochs will skip exploration",
                    epoch_index,
                    consecutive_no_new_experience_epochs,
                )
            log_diverse_rollout_latest_metrics(
                epoch_index + 1,
                diverse_rollout_latest_metrics_by_df,
                logger,
            )

        # 阶段三：完整训练 —— 经验池已冻结，且无任何探索子进程存活
        training_source = (
            tensor_snapshot if tensor_snapshot is not None else buffer_diverse
        )
        last_losses = run_diverse_training_phase(
            trainer,
            training_source,
            trainer.update_times,
            epoch_index,
        )
        write_epoch_rollout_scalars(trainer, epoch_metrics, epoch_index)
        save_parallel_epoch_model(trainer, epoch_index)

        epoch_model_file = os.path.join(
            build_epoch_model_path(trainer.model_path, epoch_index),
            "trained_model.pkl",
        )
        if last_losses is not None:
            epoch_total_loss = float(last_losses[0])
            if epoch_total_loss < best_loss:
                best_loss = epoch_total_loss
                best_model_file = epoch_model_file
                best_epoch_index = epoch_index
                logger.info(
                    "更新最小 total_loss 模型 | epoch=%d | total_loss=%.6f | 模型路径=%s",
                    epoch_index + 1,
                    best_loss,
                    best_model_file,
                )
        elif best_model_file is None:
            best_model_file = epoch_model_file
            best_epoch_index = epoch_index

    if best_model_file is None:
        pretrain_model = os.path.join(trainer.model_path, "pretrain_model.pkl")
        if os.path.exists(pretrain_model):
            best_model_file = pretrain_model
        else:
            last_epoch_model = os.path.join(
                build_epoch_model_path(trainer.model_path, max(0, trainer.num_epoch - 1)),
                "trained_model.pkl",
            )
            if os.path.exists(last_epoch_model):
                best_model_file = last_epoch_model

    logger.info(
        "并行训练结束，准备使用 total loss 最小的模型进行最终评估 | "
        "最优 epoch=%d | 最小 total_loss=%.6f | 模型路径=%s",
        best_epoch_index + 1 if best_epoch_index >= 0 else 0,
        best_loss if best_loss != float("inf") else 0.0,
        best_model_file,
    )

    eval_metrics = evaluate_parallel_diverse_model(trainer, best_model_file)
    trainer.best_model_path = best_model_file
    trainer.best_loss = best_loss
    trainer.best_epoch_index = best_epoch_index
    trainer.diverse_eval_metrics = eval_metrics
    return step_counter_diverse


def evaluate_parallel_diverse_model(
    trainer: Weighted_Contexts_DQN,
    model_path: str,
) -> list[dict[str, Any]]:
    """使用指定的模型（通常为 total loss 最小的模型）执行一次评估。

    使用 evaluate_sub_agents.evaluates 多进程评估器在数据文件上运行评估，
    等待所有评估子进程结束并输出统计后返回。
    """
    if not model_path or not os.path.exists(model_path):
        logger.warning("跳过多样化训练评估 | 未找到模型文件=%s", model_path)
        return []

    if not os.path.exists(trainer.train_data_path):
        logger.warning("跳过多样化训练评估 | 数据目录不存在=%s", trainer.train_data_path)
        return []

    data_file_paths = sorted(
        os.path.join(trainer.train_data_path, file_name)
        for file_name in os.listdir(trainer.train_data_path)
        if file_name.startswith("df_") and file_name.endswith(".feather")
    )

    if not data_file_paths:
        logger.warning("跳过多样化训练评估 | 未找到评估数据文件")
        return []

    logg_file_path = os.path.join(trainer.model_path, "diverse_evaluation.log")

    logger.info(
        "开始执行多样化训练评估 | 模型=%s | 数据文件数=%d | 评估日志=%s",
        model_path,
        len(data_file_paths),
        logg_file_path,
    )
    eval_metrics = evaluates(
        logg_file_path=logg_file_path,
        data_file_paths=data_file_paths,
        model_path=model_path,
        tech_indicator_list_path=trainer.tech_indicator_list_path,
        maintenance_margin_ratio_dict_path=trainer.maintenance_margin_ratio_dict_path,
        transcation_cost=trainer.transcation_cost,
        max_holding_number=trainer.max_holding_number,
        position_choices=trainer.position_choices,
        N=trainer.N,
        time_info_dim=trainer.time_info_dim,
        hidden_nodes=trainer.hidden_nodes,
        leverage_choices=trainer.leverage_choices,
        initial_leverage=trainer.initial_leverage,
        initial_position=trainer.initial_position,
        initial_wallet_balance=trainer.initial_wallet_balance,
        order_book_depth=trainer.order_book_depth,
        early_stop=trainer.early_stop,
        enable_limit_reward=trainer.enable_limit_reward,
        limit_hold_bonus=trainer.limit_hold_bonus,
        limit_stay_bonus=trainer.limit_stay_bonus,
        limit_reverse_penalty=trainer.limit_reverse_penalty,
        near_limit_threshold=trainer.near_limit_threshold,
        allow_reverse_position=trainer.allow_reverse_position,
    )
    logger.info(
        "多样化训练评估完成 | 评估样本数=%d | 模型=%s",
        len(eval_metrics),
        model_path,
    )
    return eval_metrics


def update(
    trainer: Weighted_Contexts_DQN,
    states: torch.Tensor,
    info: dict[str, Any],
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    info_: dict[str, Any],
    dones: torch.Tensor,
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
