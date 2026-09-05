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


from RL.DiHFT.low_level.evaluate_sub_agents import (
    DEFAULT_EVAL_NUM_WORKERS,
    DEFAULT_PRETRAIN_EVAL_NUM_WORKERS,
    SubAgentEvalMetric,
    SubAgentEvalTask,
    WarmupEvalMetric,
    WarmupEvalTask,
    act_test,
    evaluate_single_sub_agent_df,
    evaluate_sub_agents,
    evaluate_warmup_sub_agents,
    select_greedy_model_action,
)


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
        df_index = getattr(message, "df_index", None)
        if df_index is None:
            df_index = self.df_index
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
    max_workers = getattr(trainer, "pretrain_num_workers", 150)
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
    elif hasattr(buffer_pretrain, "add"):
        for item in transitions:
            buffer_pretrain.add(*item)
    elif hasattr(buffer_pretrain, "items"):
        buffer_pretrain.items.extend(transitions)


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
    payload = torch.load(buffer_path, map_location="cpu")
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
    model_path = getattr(trainer, "model_path", None)
    if not isinstance(model_path, str):
        model_path = None

    pretrain_buffer_path = None
    pretrain_model_path = None
    if model_path is not None:
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

    load_model = getattr(trainer, "load_pretrain_model", False)
    if isinstance(load_model, bool) and load_model:
        if pretrain_model_path is not None and os.path.exists(pretrain_model_path):
            state_dict = torch.load(
                pretrain_model_path,
                map_location=getattr(trainer, "device", "cpu"),
            )
            if hasattr(trainer, "eval_net") and hasattr(
                trainer.eval_net, "load_state_dict"
            ):
                trainer.eval_net.load_state_dict(state_dict)
            if hasattr(trainer, "target_net") and hasattr(
                trainer.target_net, "load_state_dict"
            ):
                trainer.target_net.load_state_dict(state_dict)
            logger.info(
                "已读取已训练的预先训练模型并跳过预先训练 | 模型路径=%s",
                pretrain_model_path,
            )
            eval_metrics = evaluate_warmup_sub_agents(
                trainer=trainer,
                train_df_cache=train_df_cache,
                env_kwargs=env_kwargs,
            )
            return {
                "episodes": 0,
                "transitions": step_counter_pretrain,
                "update_count": 0,
                "eval_metrics": eval_metrics,
            }, step_counter_pretrain
        else:
            raise FileNotFoundError(
                f"pretrain model file not found: {pretrain_model_path}"
            )

    if trainer.total_df_index_length <= 0:
        raise ValueError("exhaustive warmup requires total_df_index_length > 0")
    if getattr(trainer, "pretrain_epoch", 0) < 0:
        raise ValueError("pretrain_epoch must be non-negative")
    if trainer.pretrain_epoch > 0 and getattr(trainer, "update_times", 0) <= 0:
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
    raw_interval = getattr(trainer, "eval_every_rounds", None)
    eval_interval = (
        raw_interval if isinstance(raw_interval, int) and raw_interval > 0 else 30
    )
    if trainer.pretrain_epoch > 0:
        if len(buffer_pretrain) < trainer.batch_size:
            raise ValueError(
                "buffer_pretrain size ({}) is smaller than batch_size ({})".format(
                    len(buffer_pretrain), trainer.batch_size
                )
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
                ) = buffer_pretrain.sample()
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
        eval_metrics = evaluate_sub_agents(
            trainer=trainer,
            train_df_cache=train_df_cache,
            env_kwargs=env_kwargs,
        )
    else:
        logger.info("exhaustive warmup train skipped (pretrain_epoch=0)")
        eval_metrics = evaluate_sub_agents(
            trainer=trainer,
            train_df_cache=train_df_cache,
            env_kwargs=env_kwargs,
        )

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
