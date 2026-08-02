# Parallel pretraining collection + orchestration components extracted from
# parallel_weight_advantage_pretrain.py for easier review.
#
# This module holds the rule-based (DP-from-Q-table) experience collection
# runners and the exhaustive-warmup orchestration used by the pretrain stage.
# It does NOT import the orchestrator module at top level to keep the
# dependency graph acyclic; shared infrastructure is imported lazily inside
# the functions that need it.

import logging
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
        return perfection_action_list[optimal_step_counter]
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
        self.df_index = worker_config["df_index"]
        self.train_df = worker_config["train_df"]
        self.env_kwargs = worker_config["env_kwargs"]
        self.leverage_choices = worker_config["leverage_choices"]
        self.position_list = worker_config["position_list"]
        self.position_choices = worker_config["position_choices"]
        self.initial_wallet_balance = worker_config["initial_wallet_balance"]
        self.initial_unrealized_pnL = worker_config["initial_unrealized_pnL"]
        self.q_table = worker_config["q_table"]
        self._env_cache = {}
        self._perfection_cache = {}

    def collect_episode(self, message):
        initial_action = message.initial_action
        if initial_action not in self._env_cache:
            _, _, _, initial_state = build_initial_state(
                self.train_df,
                initial_action,
                self.leverage_choices,
                self.position_list,
                self.initial_wallet_balance,
                self.initial_unrealized_pnL,
            )
            self._env_cache[initial_action] = create_demo_env(
                self.train_df, self.env_kwargs, initial_state
            )
            self._perfection_cache[initial_action] = get_dp_action_from_qtable(
                self.q_table, initial_action
            )
        env = self._env_cache[initial_action]
        perfection_action_list = self._perfection_cache[initial_action]
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
            df_index=self.df_index,
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
    for df_index in build_effective_df_indices(trainer.total_df_index_length):
        input_queue = worker_context.Queue()
        worker_config = {
            "df_index": df_index,
            "train_df": train_df_cache[df_index],
            "env_kwargs": env_kwargs,
            "leverage_choices": trainer.leverage_choices,
            "position_list": trainer.position_list,
            "position_choices": trainer.position_choices,
            "initial_wallet_balance": trainer.initial_wallet_balance,
            "initial_unrealized_pnL": trainer.initial_unrealized_pnL,
            "q_table": q_table_cache[df_index],
            "runner_factory": PretrainCollectRunner,
        }
        process = worker_context.Process(
            target=df_rollout_worker,
            args=(worker_config, input_queue, trainer.worker_result_queue),
        )
        process.start()
        trainer.worker_input_queues[df_index] = input_queue
        trainer.worker_processes.append(process)


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

    if trainer.total_df_index_length <= 0:
        raise ValueError("exhaustive warmup requires total_df_index_length > 0")
    total_episodes = trainer.total_df_index_length * trainer.position_choices * 4
    logger.info(
        "exhaustive warmup collect start | df_count=%d | position_choices=%d | "
        "episodes=%d",
        trainer.total_df_index_length,
        trainer.position_choices,
        total_episodes,
    )
    start_pretrain_collect_workers(trainer, train_df_cache, env_kwargs, q_table_cache)
    for df_index in range(trainer.total_df_index_length):
        for initial_action in range(trainer.position_choices):
            for rollout_index in range(4):
                trainer.worker_input_queues[df_index].put(
                    CollectPretrainEpisode(initial_action, rollout_index)
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

    update_count = 0
    if trainer.pretrain_epoch > 0:
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
    else:
        logger.info("exhaustive warmup train skipped (pretrain_epoch=0)")
    return {
        "episodes": total_episodes,
        "transitions": step_counter_pretrain,
        "update_count": update_count,
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

    batch_weights = torch.ones(trainer.batch_size, trainer.N).to(trainer.device)
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

    weighted_action_distribution = torch.einsum(
        "ijk,ij->ik", predict_action_distrbution, batch_weights
    )
    q_value = recalculate_q_demonstration(info["q_value"], info["avaliable_action"])
    KL_div = F.kl_div(
        (weighted_action_distribution.softmax(dim=-1) + 1e-8).log(),
        (q_value.softmax(dim=-1) + 1e-8),
        reduction="batchmean",
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
                "weighted_action_distribution": weighted_action_distribution,
                "q_value": q_value,
                "batch_weights": batch_weights,
            },
            info_values={"info": info, "info_": info_},
            trainer=trainer,
        )
        raise ValueError("loss is nan")
    return loss.item(), KL_div.item(), td_loss.item()
