# Parallel pretraining collection + orchestration components extracted from
# parallel_weight_advantage_pretrain.py for easier review.
#
# This module holds the rule-based (DP-from-Q-table) experience collection
# runners and the exhaustive-warmup orchestration used by the pretrain stage.
# It does NOT import the orchestrator module at top level to keep the
# dependency graph acyclic; shared infrastructure is imported lazily inside
# the functions that need it.

import logging
import os
import sys
import torch
import numpy as np
from dataclasses import dataclass

import torch.nn.functional as F
from env.env_class.futures_util import get_dp_action_from_qtable
from env.env_class.policy_util import get_close_element
from RL.DiHFT.low_level.pretrain_qtable_diagnostics import (
    build_initial_state,
    create_demo_env,
)
from RL.util.update import (
    evaluate_quantile_at_action,
    recalculate_q_demonstration,
    update_params,
    soft_copy_params,
)
from RL.DiHFT.low_level.loss_nan_diagnostics import log_loss_nan_diagnostics
from RL.DiHFT.low_level.weight_advantage_pretrain import (
    calculate_paper_supervisor_kl_loss,
)

# Reuse the orchestrator's configured logger so all log output flows through
# the same file handler set up by configure_logger() in
# parallel_weight_advantage_pretrain.
logger = logging.getLogger(
    "RL.DiHFT.low_level.parallel_weight_advantage_pretrain"
)


@dataclass(frozen=True)
class CollectPretrainEpisode:
    initial_action: int
    rollout_index: int
    df_index: int = None


@dataclass(frozen=True)
class PretrainCollectResult:
    df_index: int
    initial_action: int
    rollout_index: int
    transitions: list
    reward_sum: float
    final_balance: float
    transition_count: int


def select_pretrain_action(
    info,
    optimal_step_counter,
    rollout_index,
    perfection_action_list,
    position_choices,
    leverage_choices,
):
    avaliable_action_list = info["avaiable_action_list"]
    if rollout_index == 0:
        action = perfection_action_list[optimal_step_counter]
        return get_close_element(action, avaliable_action_list)
    elif rollout_index == 1:
        action = (position_choices - 1) * len(leverage_choices)
        return get_close_element(action, avaliable_action_list)
    elif rollout_index == 2:
        action = len(leverage_choices) - 1
        return get_close_element(action, avaliable_action_list)
    elif rollout_index == 3:
        action = (position_choices // 2) * len(leverage_choices)
        return get_close_element(action, avaliable_action_list)
    raise ValueError("rollout_index must be in 0-3, got {}".format(rollout_index))


class PretrainCollectRunner:
    def __init__(self, worker_config):
        self.df_index = worker_config.get("df_index")
        self.df_indices = worker_config.get(
            "df_indices",
            [self.df_index] if self.df_index is not None else [],
        )
        if "train_df_by_df" in worker_config:
            self.train_df_by_df = worker_config["train_df_by_df"]
            self.q_table_by_df = worker_config["q_table_by_df"]
        else:
            self.train_df_by_df = {self.df_index: worker_config.get("train_df")}
            self.q_table_by_df = {self.df_index: worker_config.get("q_table")}
        self.env_kwargs = worker_config["env_kwargs"]
        self.leverage_choices = worker_config["leverage_choices"]
        self.position_list = worker_config["position_list"]
        self.position_choices = worker_config["position_choices"]
        self.initial_wallet_balance = worker_config["initial_wallet_balance"]
        self.initial_unrealized_pnL = worker_config["initial_unrealized_pnL"]
        self._env_cache = {}
        self._perfection_cache = {}

    def collect_episode(self, message):
        df_index = message.df_index if message.df_index is not None else self.df_index
        train_df = self.train_df_by_df[df_index]
        q_table = self.q_table_by_df[df_index]
        initial_action = message.initial_action
        cache_key = (df_index, initial_action)
        if cache_key not in self._env_cache:
            _, _, _, initial_state = build_initial_state(
                train_df,
                initial_action,
                self.leverage_choices,
                self.position_list,
                self.initial_wallet_balance,
                self.initial_unrealized_pnL,
            )
            self._env_cache[cache_key] = create_demo_env(
                train_df, self.env_kwargs, initial_state
            )
            self._perfection_cache[cache_key] = get_dp_action_from_qtable(
                q_table, initial_action
            )
        env = self._env_cache[cache_key]
        perfection_action_list = self._perfection_cache[cache_key]
        state, info = env.reset()
        optimal_step_counter = 0
        transitions = []
        reward_sum = 0.0
        while True:
            action = select_pretrain_action(
                info,
                optimal_step_counter,
                message.rollout_index,
                perfection_action_list,
                self.position_choices,
                self.leverage_choices,
            )
            optimal_step_counter += 1
            next_state, reward, done, next_info = env.step(action)
            transitions.append(
                (state, info, action, reward, next_state, next_info, done)
            )
            reward_sum += reward
            state, info = next_state, next_info
            if done:
                break
        final_balance = env.unrealized_pnl + env.wallet_balance
        return PretrainCollectResult(
            df_index=df_index,
            initial_action=initial_action,
            rollout_index=message.rollout_index,
            transitions=transitions,
            reward_sum=float(reward_sum),
            final_balance=float(final_balance),
            transition_count=len(transitions),
        )


def write_pretrain_loss_scalars(trainer, total_loss, KL_loss, td_loss):
    trainer.writer.add_scalar(
        tag="total_loss",
        scalar_value=total_loss,
        global_step=trainer.update_counter,
        walltime=None,
    )
    trainer.writer.add_scalar(
        tag="KL_loss",
        scalar_value=KL_loss,
        global_step=trainer.update_counter,
        walltime=None,
    )
    trainer.writer.add_scalar(
        tag="td_loss",
        scalar_value=td_loss,
        global_step=trainer.update_counter,
        walltime=None,
    )


def start_pretrain_collect_workers(trainer, train_df_cache, env_kwargs, q_table_cache):
    from RL.DiHFT.low_level.parallel_weight_advantage_pretrain import (
        build_effective_df_indices,
        create_worker_context,
        df_rollout_worker,
    )

    worker_context = create_worker_context()
    trainer.worker_result_queue = worker_context.Queue()
    trainer.worker_input_queues = {}
    trainer.worker_processes = []
    effective_df_indices = build_effective_df_indices(trainer.total_df_index_length)
    max_workers = trainer.pretrain_num_workers
    if max_workers <= 0:
        raise ValueError("pretrain_num_workers must be positive")
    num_workers = min(len(effective_df_indices), max_workers)

    for worker_id in range(num_workers):
        assigned_df_indices = [
            df_index
            for i, df_index in enumerate(effective_df_indices)
            if i % num_workers == worker_id
        ]
        input_queue = worker_context.Queue()
        worker_config = {
            "worker_id": worker_id,
            "df_indices": assigned_df_indices,
            "df_index": assigned_df_indices[0] if assigned_df_indices else None,
            "train_df": train_df_cache[assigned_df_indices[0]] if len(assigned_df_indices) == 1 else None,
            "train_df_by_df": {df: train_df_cache[df] for df in assigned_df_indices},
            "env_kwargs": env_kwargs,
            "leverage_choices": trainer.leverage_choices,
            "position_list": trainer.position_list,
            "position_choices": trainer.position_choices,
            "initial_wallet_balance": trainer.initial_wallet_balance,
            "initial_unrealized_pnL": trainer.initial_unrealized_pnL,
            "q_table": q_table_cache[assigned_df_indices[0]] if len(assigned_df_indices) == 1 else None,
            "q_table_by_df": {df: q_table_cache[df] for df in assigned_df_indices},
            "runner_factory": PretrainCollectRunner,
        }
        process = worker_context.Process(
            target=df_rollout_worker,
            args=(worker_config, input_queue, trainer.worker_result_queue),
        )
        process.start()
        for df_index in assigned_df_indices:
            trainer.worker_input_queues[df_index] = input_queue
        trainer.worker_processes.append(process)


def extract_buffer_transitions(buffer_pretrain):
    if hasattr(buffer_pretrain, "memory"):
        return [tuple(e) for e in buffer_pretrain.memory]
    if hasattr(buffer_pretrain, "items"):
        return list(buffer_pretrain.items)
    return list(buffer_pretrain)


def populate_buffer_transitions(buffer_pretrain, transitions):
    if hasattr(buffer_pretrain, "memory") and hasattr(buffer_pretrain, "experience"):
        for item in transitions:
            buffer_pretrain.memory.append(buffer_pretrain.experience(*item))
        # 直接填充 memory 绕过了 add() -> calc_multistep_return() 的惰性初始化，
        # 需要从加载的 transitions 恢复 info_key，否则 sample() 无法访问该属性
        if transitions and not hasattr(buffer_pretrain, "info_key"):
            buffer_pretrain.info_key = transitions[0][6].keys()
    elif hasattr(buffer_pretrain, "add"):
        for item in transitions:
            buffer_pretrain.add(*item)
    elif hasattr(buffer_pretrain, "items"):
        buffer_pretrain.items.extend(transitions)


def extract_stacked_tensor_dict(buffer, chunk_size=50000):
    """将经验池（ReplayBuffer 或快照字典）分批提取并堆叠为连续 Tensor 字典。

    使用预分配 Tensor + 分块（chunk_size）写入，避免数百万个小对象的列表推导式
    与全量 np.stack 造成数十 GB 临时内存爆炸。
    """
    if isinstance(buffer, dict) and "states" in buffer:
        return buffer

    from RL.util.replay_buffer_DQN import NETWORK_INFO_KEYS

    if isinstance(buffer, dict) and "memory" in buffer:
        memory = buffer["memory"]
        n_step_buffer = buffer.get("n_step_buffer", [])
    elif hasattr(buffer, "memory"):
        memory = list(buffer.memory)
        n_step_buffer = [
            [tuple(t) for t in deque_item]
            for deque_item in getattr(buffer, "n_step_buffer", [])
        ]
    else:
        memory = list(buffer)
        n_step_buffer = [
            [tuple(t) for t in deque_item]
            for deque_item in getattr(buffer, "n_step_buffer", [])
        ]

    n = len(memory)
    if n == 0:
        return {
            "states": torch.empty((0, 0), dtype=torch.float32),
            "actions": torch.empty((0, 1), dtype=torch.int64),
            "rewards": torch.empty((0, 1), dtype=torch.float32),
            "next_states": torch.empty((0, 0), dtype=torch.float32),
            "dones": torch.empty((0, 1), dtype=torch.float32),
            "infos": {},
            "next_infos": {},
            "info_keys": [],
            "n_step_buffer": n_step_buffer,
            "buffer_size": 0,
        }

    first = memory[0]
    first_state = first[0]
    first_info = first[1]
    first_next_info = first[6]

    state_shape = np.asarray(first_state).shape
    raw_info_keys = getattr(buffer, "info_key", None)
    if raw_info_keys is None:
        raw_info_keys = list(first_info.keys())
    target_info_keys = [k for k in raw_info_keys if k in NETWORK_INFO_KEYS]
    if not target_info_keys and raw_info_keys:
        target_info_keys = list(raw_info_keys)

    states = torch.empty((n, *state_shape), dtype=torch.float32)
    actions = torch.empty((n, 1), dtype=torch.int64)
    rewards = torch.empty((n, 1), dtype=torch.float32)
    next_states = torch.empty((n, *state_shape), dtype=torch.float32)
    dones = torch.empty((n, 1), dtype=torch.float32)

    infos = {}
    next_infos = {}
    for k in target_info_keys:
        val = np.asarray(first_info[k])
        val_dtype = torch.float32 if np.issubdtype(val.dtype, np.floating) else torch.int64
        infos[k] = torch.empty((n, *val.shape), dtype=val_dtype)
        val_ = np.asarray(first_next_info[k])
        val_dtype_ = torch.float32 if np.issubdtype(val_.dtype, np.floating) else torch.int64
        next_infos[k] = torch.empty((n, *val_.shape), dtype=val_dtype_)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = memory[start:end]
        c_states = [e[0] for e in chunk]
        c_infos = [e[1] for e in chunk]
        c_actions = [e[2] for e in chunk]
        c_rewards = [e[3] for e in chunk]
        c_next_states = [e[4] for e in chunk]
        c_dones = [e[5] for e in chunk]
        c_next_infos = [e[6] for e in chunk]

        states[start:end] = torch.from_numpy(np.stack(c_states)).float()
        actions[start:end] = torch.from_numpy(np.vstack(c_actions)).long()
        rewards[start:end] = torch.from_numpy(np.vstack(c_rewards)).float()
        next_states[start:end] = torch.from_numpy(np.stack(c_next_states)).float()
        dones[start:end] = torch.from_numpy(np.vstack(c_dones)).float()

        for k in target_info_keys:
            infos[k][start:end] = torch.as_tensor(np.stack([np.asarray(inf[k]) for inf in c_infos]))
            next_infos[k][start:end] = torch.as_tensor(np.stack([np.asarray(inf[k]) for inf in c_next_infos]))

    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "next_states": next_states,
        "dones": dones,
        "infos": infos,
        "next_infos": next_infos,
        "info_keys": target_info_keys,
        "n_step_buffer": n_step_buffer,
        "buffer_size": n,
    }


class StackedTransitionSampler:
    """经验池预堆叠采样器。

    支持两种输入：
    1. 张量化字典（{"states": tensor, ...}），无需重新堆叠，零额外内存开销；
    2. 传统 buffer 对象（含 memory deque），按分块（chunk_size）预堆叠为张量，严格控制峰值内存。
    采样语义与 Multi_step_ReplayBuffer_multi_info.sample() 一致：无放回均匀采样。
    """

    def __init__(self, buffer, batch_size, device):
        if isinstance(buffer, dict) and "states" in buffer:
            self.n = buffer["buffer_size"] if "buffer_size" in buffer else buffer["states"].shape[0]
            if self.n < batch_size:
                raise ValueError(
                    "buffer size ({}) is smaller than batch_size ({})".format(
                        self.n, batch_size
                    )
                )
            self.batch_size = batch_size
            self.device = device
            self.states = buffer["states"]
            self.actions = buffer["actions"]
            self.rewards = buffer["rewards"]
            self.next_states = buffer["next_states"]
            self.dones = buffer["dones"]
            self.info_keys = list(buffer.get("infos", {}).keys())
            self.infos = buffer.get("infos", {})
            self.next_infos = buffer.get("next_infos", {})
        else:
            payload = extract_stacked_tensor_dict(buffer)
            self.n = payload["buffer_size"]
            if self.n < batch_size:
                raise ValueError(
                    "buffer size ({}) is smaller than batch_size ({})".format(
                        self.n, batch_size
                    )
                )
            self.batch_size = batch_size
            self.device = device
            self.states = payload["states"]
            self.actions = payload["actions"]
            self.rewards = payload["rewards"]
            self.next_states = payload["next_states"]
            self.dones = payload["dones"]
            self.info_keys = payload["info_keys"]
            self.infos = payload["infos"]
            self.next_infos = payload["next_infos"]

    def sample(self):
        idx = np.random.choice(self.n, size=self.batch_size, replace=False)
        states = (
            torch.as_tensor(self.states[idx]).float().to(self.device)
        )
        infos = {
            k: torch.as_tensor(v[idx]).float().to(self.device)
            for k, v in self.infos.items()
        }
        actions = torch.as_tensor(self.actions[idx]).long().to(self.device)
        rewards = (
            torch.as_tensor(self.rewards[idx]).float().to(self.device)
        )
        next_states = (
            torch.as_tensor(self.next_states[idx]).float().to(self.device)
        )
        next_infos = {
            k: torch.as_tensor(v[idx]).float().to(self.device)
            for k, v in self.next_infos.items()
        }
        dones = (
            torch.as_tensor(self.dones[idx]).float().to(self.device)
        )
        return (states, infos, actions, rewards, next_states, next_infos, dones)


def save_pretrain_buffer_file(buffer_pretrain, buffer_path, step_counter):
    transitions = extract_buffer_transitions(buffer_pretrain)
    payload = {
        "transitions": transitions,
        "step_counter": step_counter,
        "buffer_size": len(buffer_pretrain),
    }
    dir_name = os.path.dirname(os.path.abspath(buffer_path))
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    torch.save(payload, buffer_path)


def load_pretrain_buffer_file(buffer_pretrain, buffer_path, current_step_counter=0):
    payload = torch.load(buffer_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "transitions" in payload:
        transitions = payload["transitions"]
        step_counter = payload.get(
            "step_counter", current_step_counter + len(transitions)
        )
    elif isinstance(payload, list):
        transitions = payload
        step_counter = current_step_counter + len(transitions)
    else:
        transitions = []
        step_counter = current_step_counter
    populate_buffer_transitions(buffer_pretrain, transitions)
    return step_counter


def resolve_pretrain_paths(trainer):
    model_path = trainer.model_path
    if not model_path:
        return None, None
    pretrain_buffer_path = os.path.join(model_path, "pretrain_buffer.pt")
    pretrain_model_path = os.path.join(model_path, "pretrain_model.pkl")
    return pretrain_buffer_path, pretrain_model_path


def run_exhaustive_warmup(
    trainer,
    q_table_cache,
    train_df_cache,
    env_kwargs,
    buffer_pretrain,
    step_counter_pretrain,
):
    from RL.DiHFT.low_level.parallel_weight_advantage_pretrain import (
        WorkerErrorMessage,
        raise_for_worker_error,
    )

    pretrain_buffer_path, pretrain_model_path = resolve_pretrain_paths(trainer)

    if trainer.load_pretrain_model and pretrain_model_path is not None and os.path.exists(pretrain_model_path):
        state_dict = torch.load(
            pretrain_model_path,
            map_location=trainer.device,
        )
        trainer.eval_net.load_state_dict(state_dict)
        if trainer.target_net is not None:
            trainer.target_net.load_state_dict(state_dict)
        logger.info(
            "已读取已训练的预先训练模型并跳过预先训练 | 模型路径=%s",
            pretrain_model_path,
        )
        
        return {
            "episodes": 0,
            "transitions": step_counter_pretrain,
            "update_count": 0,
        }, step_counter_pretrain
   
    if trainer.total_df_index_length <= 0:
        raise ValueError("exhaustive warmup requires total_df_index_length > 0")
    if trainer.pretrain_epoch < 0:
        raise ValueError("pretrain_epoch must be non-negative")
    if trainer.pretrain_epoch > 0 and trainer.update_times <= 0:
        raise ValueError("update_times must be positive when pretrain_epoch > 0")
    total_episodes = trainer.total_df_index_length * trainer.position_choices * 4

    # Check if pretrain buffer already exists
    if pretrain_buffer_path is not None and os.path.exists(pretrain_buffer_path):
        step_counter_pretrain = load_pretrain_buffer_file(
            buffer_pretrain, pretrain_buffer_path, step_counter_pretrain
        )
        logger.info(
            "预训练经验池已存在，直接加载并跳过探索 | 文件=%s | 经验池大小=%d",
            pretrain_buffer_path,
            len(buffer_pretrain),
        )
    else:
        logger.info(
            "exhaustive warmup collect start | df_count=%d | position_choices=%d | "
            "episodes=%d",
            trainer.total_df_index_length,
            trainer.position_choices,
            total_episodes,
        )
        start_pretrain_collect_workers(
            trainer, train_df_cache, env_kwargs, q_table_cache
        )
        for df_index in range(trainer.total_df_index_length):
            for initial_action in range(trainer.position_choices):
                for rollout_index in range(4):
                    trainer.worker_input_queues[df_index].put(
                        CollectPretrainEpisode(
                            initial_action=initial_action,
                            rollout_index=rollout_index,
                            df_index=df_index,
                        )
                    )
        collected = 0
        progress_log_every = max(1, total_episodes // 20)
        while collected < total_episodes:
            result = trainer.worker_result_queue.get()
            if isinstance(result, WorkerErrorMessage):
                trainer._shutdown_parallel_workers()
                raise_for_worker_error(result)
            for transition in result.transitions:
                buffer_pretrain.add(*transition)
                step_counter_pretrain += 1
            return_rate = result.final_balance / (
                trainer.initial_wallet_balance + 1e-12
            ) - 1
            trainer.writer.add_scalar(
                tag="pretrain_return_rate_train_{}".format(result.rollout_index),
                scalar_value=return_rate,
                global_step=collected,
                walltime=None,
            )
            trainer.writer.add_scalar(
                tag="pretrain_reward_sum_train_{}".format(result.rollout_index),
                scalar_value=result.reward_sum,
                global_step=collected,
                walltime=None,
            )
            collected += 1
            if collected % progress_log_every == 0 or collected == total_episodes:
                logger.info(
                    "exhaustive warmup collect progress | episodes=%d/%d | "
                    "transitions=%d | buffer=%d",
                    collected,
                    total_episodes,
                    step_counter_pretrain,
                    len(buffer_pretrain),
                )
        trainer._shutdown_parallel_workers()
        logger.info(
            "exhaustive warmup collect done | episodes=%d | transitions=%d | "
            "buffer=%d",
            total_episodes,
            step_counter_pretrain,
            len(buffer_pretrain),
        )
        if pretrain_buffer_path is not None:
            save_pretrain_buffer_file(
                buffer_pretrain, pretrain_buffer_path, step_counter_pretrain
            )
            logger.info(
                "探索完成，已保存经验池到文件 | 文件=%s | 经验池大小=%d",
                pretrain_buffer_path,
                len(buffer_pretrain),
            )

    update_count = 0
    eval_metrics = []
    eval_interval = 30
    if trainer.pretrain_epoch > 0:
        if len(buffer_pretrain) < trainer.batch_size:
            raise ValueError(
                "buffer_pretrain size ({}) is smaller than batch_size ({})".format(
                    len(buffer_pretrain), trainer.batch_size
                )
            )
        sampler = StackedTransitionSampler(
            buffer_pretrain, trainer.batch_size, trainer.device
        )
        logger.info(
            "exhaustive warmup train start | rounds=%d | updates_per_round=%d",
            trainer.pretrain_epoch,
            trainer.update_times,
        )
        for epoch in range(trainer.pretrain_epoch):
            last_losses = None
            for _ in range(trainer.update_times):
                (
                    states,
                    infos,
                    actions,
                    rewards,
                    next_states,
                    next_infos,
                    dones,
                ) = sampler.sample()
                last_losses = update_pretrain(
                    trainer,
                    states,
                    infos,
                    actions,
                    rewards,
                    next_states,
                    next_infos,
                    dones,
                )
                write_pretrain_loss_scalars(trainer, *last_losses)
                update_count += 1
            logger.info(
                "exhaustive warmup train epoch | epoch=%d/%d | total_loss=%.6f | "
                "KL_loss=%.6f | td_loss=%.6f | update_count=%d",
                epoch + 1,
                trainer.pretrain_epoch,
                last_losses[0],
                last_losses[1],
                last_losses[2],
                update_count,
            )
            

        
        dir_name = os.path.dirname(os.path.abspath(pretrain_model_path))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
       
        torch.save(trainer.eval_net.state_dict(), pretrain_model_path)
        logger.info(
            "exhaustive warmup 学习结束 | 模型已保存至=%s",
            pretrain_model_path,
        )
        
    else:
        logger.info("exhaustive warmup train skipped (pretrain_epoch=0)")
        

    return {
        "episodes": total_episodes,
        "transitions": step_counter_pretrain,
        "update_count": update_count,
        "eval_metrics": eval_metrics,
    }, step_counter_pretrain

def update_pretrain(
    trainer,
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
    current_sa_quantiles = current_sa_quantiles.squeeze(-1)
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
        target_sa_quantiles = target_sa_quantiles.permute(0, 2, 1)
        assert target_sa_quantiles.shape == (
            trainer.batch_size,
            trainer.N,
            1,
        )
    target_sa_quantiles = target_sa_quantiles.squeeze(-1)
    td_loss = trainer.loss_func_pretrain(current_sa_quantiles, target_sa_quantiles)
    td_loss = td_loss.sum(dim=1)
    td_loss = td_loss.mean()

    batch_weights = torch.ones(
        trainer.batch_size,
        trainer.N,
        device=trainer.device,
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
    loss = td_loss + KL_div * trainer.ada
    update_params(
        trainer.optimizer,
        loss,
        trainer.eval_net,
        retain_graph=False,
        grad_cliping=trainer.grad_clip,
    )
    soft_copy_params(trainer.eval_net, trainer.target_net, trainer.tau)
    trainer.update_counter += 1
    if torch.isnan(loss):
        log_loss_nan_diagnostics(
            logger=logger,
            numeric_values={
                "loss": loss,
                "KL_div": KL_div,
                "td_loss": td_loss,
                "states": states,
                "next_states": states_,
                "actions": actions,
                "rewards": rewards,
                "dones": dones,
                "time_input": time_input,
                "next_time_input": time_input_,
                "previous_action": previous_action,
                "next_previous_action": previous_action_,
                "avaliable_action": avaliable_action,
                "next_avaliable_action": avaliable_action_,
                "current_sa_quantiles": current_sa_quantiles,
                "target_sa_quantiles": target_sa_quantiles,
                "predict_action_distrbution": predict_action_distrbution,
                "q_value": q_value,
                "batch_weights": batch_weights,
            },
            info_values={"info": info, "info_": info_},
            trainer=trainer,
        )
        raise ValueError("loss is nan")
    return loss.item(), KL_div.item(), td_loss.item()
