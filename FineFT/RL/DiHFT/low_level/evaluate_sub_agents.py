# Sub-agent evaluation across datasets for pretraining and diverse training.
#
# This module evaluates each (sub_agent, dataset) combination using an
# independent subprocess via a multiprocessing process pool.
# Action selection aligns with test_agent_index.py's act_test semantics.

import copy
import logging
import multiprocessing as mp
import os
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from env.env_class.policy_util import get_close_element
from RL.DiHFT.low_level.pretrain_qtable_diagnostics import (
    build_initial_state,
    create_demo_env,
)

logger = logging.getLogger(
    "RL.DiHFT.low_level.parallel_weight_advantage_pretrain"
)

DEFAULT_EVAL_NUM_WORKERS = 150
DEFAULT_PRETRAIN_EVAL_NUM_WORKERS = DEFAULT_EVAL_NUM_WORKERS


@dataclass(frozen=True)
class SubAgentEvalMetric:
    context_index: int
    df_index: int
    initial_action: int
    reward_sum: float
    final_balance: float
    return_rate: float


# Backward compatibility alias
WarmupEvalMetric = SubAgentEvalMetric


@dataclass(frozen=True)
class SubAgentEvalTask:
    context_index: int
    df_index: int
    train_df: Any
    env_kwargs: Dict[str, Any]
    initial_action: int
    leverage_choices: List[Any]
    position_list: List[Any]
    initial_wallet_balance: float
    initial_unrealized_pnL: float
    model: Any
    device: str


# Backward compatibility alias
WarmupEvalTask = SubAgentEvalTask


def select_greedy_model_action(
    model: Any,
    state: Any,
    info: Dict[str, Any],
    context_index: int,
    device: str = "cpu",
) -> int:
    """
    Greedy action selection with the exact logical semantics of test_agent_index.act_test.
    Evaluates model.qnet_list[context_index] when available, or slices context_index
    from model output, with fallback to get_close_element if avaiable_action_list is provided.
    """
    with torch.inference_mode():
        if isinstance(state, torch.Tensor):
            state_tensor = state.float().reshape(1, -1).to(device)
        else:
            state_tensor = torch.unsqueeze(torch.FloatTensor(state).reshape(-1), 0).to(
                device
            )

        previous_action_raw = info["previous_action"]
        if isinstance(previous_action_raw, torch.Tensor):
            previous_action = previous_action_raw.float().reshape(1, -1).to(device)
        else:
            previous_action = torch.tensor(
                [previous_action_raw], dtype=torch.float32
            ).reshape(1, -1).to(device)

        avaliable_action_raw = info["avaliable_action"]
        if isinstance(avaliable_action_raw, torch.Tensor):
            avaliable_action = avaliable_action_raw.to(device)
            if avaliable_action.ndim == 1:
                avaliable_action = avaliable_action.unsqueeze(0)
        else:
            avaliable_action = torch.tensor(
                avaliable_action_raw
            ).unsqueeze(0).to(device)

        hour_count_down_raw = info["funding_count_down_hour"]
        if isinstance(hour_count_down_raw, torch.Tensor):
            hour_count_down = hour_count_down_raw.float().reshape(1, -1).to(device)
        else:
            hour_count_down = torch.tensor(
                [hour_count_down_raw], dtype=torch.float32
            ).reshape(1, -1).to(device)

        minute_count_down_raw = info["funding_count_down_minute"]
        if isinstance(minute_count_down_raw, torch.Tensor):
            minute_count_down = minute_count_down_raw.float().reshape(1, -1).to(device)
        else:
            minute_count_down = torch.tensor(
                [minute_count_down_raw], dtype=torch.float32
            ).reshape(1, -1).to(device)

        time_input = torch.cat([hour_count_down, minute_count_down], dim=1).to(device)

        raw_trading_info = info["trading_info"]
        if isinstance(raw_trading_info, np.ndarray):
            trading_info = (
                torch.from_numpy(raw_trading_info).float().reshape(1, -1).to(device)
            )
        elif isinstance(raw_trading_info, torch.Tensor):
            trading_info = raw_trading_info.float().reshape(1, -1).to(device)
        else:
            trading_info = (
                torch.tensor(raw_trading_info, dtype=torch.float32)
                .reshape(1, -1)
                .to(device)
            )

        # Exact test_agent_index.act_test logic: call sub-network qnet_list[context_index] directly
        if hasattr(model, "qnet_list"):
            action_value_chosen_index = model.qnet_list[context_index](
                state=state_tensor,
                time=time_input,
                previous_action=previous_action,
                avaliable_action=avaliable_action,
                trading_info=trading_info,
            )
            raw_action = int(
                torch.max(action_value_chosen_index, 1)[1].data.cpu().numpy()[0]
            )
        else:
            q_values = model(
                state=state_tensor,
                time=time_input,
                previous_action=previous_action,
                avaliable_action=avaliable_action,
                trading_info=trading_info,
            )
            if isinstance(q_values, np.ndarray):
                raw_action = int(np.argmax(q_values[0, context_index, :]))
            else:
                raw_action = int(
                    torch.max(q_values[:, context_index, :], 1)[1].data.cpu().numpy()[0]
                )

        if "avaiable_action_list" in info:
            return get_close_element(raw_action, info["avaiable_action_list"])
        return raw_action


# Logical equivalence alias
act_test = select_greedy_model_action


def evaluate_single_sub_agent_df(task: SubAgentEvalTask) -> SubAgentEvalMetric:
    """
    Evaluates one (sub-agent, dataset) combination in an independent subprocess.
    """
    try:
        from RL.DiHFT.low_level import pretrain_qtable_diagnostics as pqd

        default_build_fn = pqd.build_initial_state
        default_env_fn = pqd.create_demo_env

        eval_mod = sys.modules.get("RL.DiHFT.low_level.evaluate_sub_agents")
        pp_mod = sys.modules.get("RL.DiHFT.low_level.parallel_pretrain")

        build_fn = getattr(eval_mod, "build_initial_state", default_build_fn)
        if build_fn is default_build_fn and pp_mod is not None:
            build_fn = getattr(pp_mod, "build_initial_state", default_build_fn)

        env_fn = getattr(eval_mod, "create_demo_env", default_env_fn)
        if env_fn is default_env_fn and pp_mod is not None:
            env_fn = getattr(pp_mod, "create_demo_env", default_env_fn)

        with torch.inference_mode():
            _, _, _, initial_state = build_fn(
                task.train_df,
                task.initial_action,
                task.leverage_choices,
                task.position_list,
                task.initial_wallet_balance,
                task.initial_unrealized_pnL,
            )
            env = env_fn(task.train_df, task.env_kwargs, initial_state)
            state, info = env.reset()
            reward_sum = 0.0
            while True:
                action = select_greedy_model_action(
                    model=task.model,
                    state=state,
                    info=info,
                    context_index=task.context_index,
                    device=task.device,
                )
                next_state, reward, done, next_info = env.step(action)
                reward_sum += reward
                state, info = next_state, next_info
                if done:
                    break
            final_balance = float(env.unrealized_pnl + env.wallet_balance)
            return_rate = float(
                final_balance / (task.initial_wallet_balance + 1e-12) - 1
            )
            return SubAgentEvalMetric(
                context_index=task.context_index,
                df_index=task.df_index,
                initial_action=task.initial_action,
                reward_sum=float(reward_sum),
                final_balance=final_balance,
                return_rate=return_rate,
            )
    except Exception as exc:
        logger.error(
            "Evaluation worker error | context_index=%d | df_index=%d: %s\n%s",
            task.context_index,
            task.df_index,
            exc,
            traceback.format_exc(),
        )
        raise


def get_evaluation_pool_context():
    """
    Returns the multiprocessing context for evaluation workers.
    Prefers 'fork' on platforms supporting it to allow copy-on-write memory
    sharing of dataset caches and models without IPC overhead.
    """
    if hasattr(mp, "get_context"):
        available = mp.get_all_start_methods()
        if "fork" in available:
            return mp.get_context("fork")
        return mp.get_context()
    return mp


def evaluate_sub_agents(
    trainer: Any,
    train_df_cache: Dict[int, Any],
    env_kwargs: Dict[str, Any],
    num_workers: Optional[int] = None,
    tag_prefix: Optional[str] = None,
    mp_context: Optional[Any] = None,
    pool_factory: Optional[Any] = None,
) -> List[SubAgentEvalMetric]:
    """
    Evaluates all sub-agents across all datasets in parallel using a process pool.
    Each (sub-agent, dataset) task runs in an independent subprocess.
    Usable for both pretrain warmup evaluation and subsequent diverse training evaluation.

    Args:
        trainer: The trainer holding eval_net, N (sub-agent count), total_df_index_length,
            hyperparameters, and optional tensorboard writer.
        train_df_cache: Mapping from df_index to DataFrame.
        env_kwargs: Kwargs for constructing the evaluation environment.
        num_workers: Total worker processes in the pool, defaults to 150
            (or trainer.pretrain_eval_num_workers / trainer.eval_num_workers if configured).
        tag_prefix: TensorBoard scalar tag prefix (defaults to 'pretrain_eval' or trainer.eval_tag_prefix).
        mp_context: Optional multiprocessing context.
        pool_factory: Optional custom callable to create a Pool (useful for mocking).

    Returns:
        List of SubAgentEvalMetric records sorted by (context_index, df_index).
    """
    eval_metrics: List[SubAgentEvalMetric] = []
    model = getattr(trainer, "eval_net", None)
    if model is None:
        return eval_metrics

    ensemble_n = getattr(trainer, "N", 0)
    if not isinstance(ensemble_n, int) or ensemble_n <= 0:
        return eval_metrics

    df_count = getattr(trainer, "total_df_index_length", 0)
    if not isinstance(df_count, int) or df_count <= 0:
        return eval_metrics

    for df_index in range(df_count):
        if train_df_cache.get(df_index) is None:
            return eval_metrics

    if num_workers is None:
        raw_workers = getattr(trainer, "pretrain_eval_num_workers", None)
        if not isinstance(raw_workers, int):
            raw_workers = getattr(trainer, "eval_num_workers", None)
        if isinstance(raw_workers, int):
            effective_num_workers = raw_workers
        else:
            effective_num_workers = DEFAULT_EVAL_NUM_WORKERS
    else:
        effective_num_workers = num_workers

    if not isinstance(effective_num_workers, int) or effective_num_workers <= 0:
        raise ValueError(
            f"sub-agent evaluation num_workers must be positive, got {effective_num_workers}"
        )

    if tag_prefix is None:
        tag_prefix = getattr(trainer, "eval_tag_prefix", "pretrain_eval")

    logger.info(
        "sub-agent evaluation start | sub_agents=%d | df_count=%d | "
        "initial_action=0 | pool_workers=%d",
        ensemble_n,
        df_count,
        effective_num_workers,
    )

    was_training = model.training if isinstance(model, torch.nn.Module) else False
    device = getattr(trainer, "device", "cpu")

    try:
        with torch.no_grad():
            if isinstance(model, torch.nn.Module):
                eval_model = copy.deepcopy(model).to("cpu")
                eval_model.eval()
                for p in eval_model.parameters():
                    p.requires_grad_(False)
                eval_device = "cpu"
            else:
                eval_model = model
                eval_device = device

            tasks: List[SubAgentEvalTask] = []
            for context_index in range(ensemble_n):
                for df_index in range(df_count):
                    tasks.append(
                        SubAgentEvalTask(
                            context_index=context_index,
                            df_index=df_index,
                            train_df=train_df_cache[df_index],
                            env_kwargs=env_kwargs,
                            initial_action=0,
                            leverage_choices=list(trainer.leverage_choices),
                            position_list=list(trainer.position_list),
                            initial_wallet_balance=float(trainer.initial_wallet_balance),
                            initial_unrealized_pnL=float(trainer.initial_unrealized_pnL),
                            model=eval_model,
                            device=eval_device,
                        )
                    )

            if not tasks:
                return eval_metrics

            # Check if task payload is picklable by multiprocessing
            can_pickle = True
            try:
                from multiprocessing.reduction import ForkingPickler
                import io

                buf = io.BytesIO()
                ForkingPickler(buf).dump(tasks[0])
            except Exception:
                can_pickle = False

            if can_pickle:
                pool_workers = min(len(tasks), effective_num_workers)
                ctx = mp_context if mp_context is not None else get_evaluation_pool_context()
                if pool_factory is not None:
                    pool_cm = pool_factory(processes=pool_workers)
                else:
                    pool_cm = ctx.Pool(processes=pool_workers)

                with pool_cm as pool:
                    raw_results = pool.map(evaluate_single_sub_agent_df, tasks)
            else:
                logger.warning(
                    "Evaluation task is not picklable by multiprocessing; "
                    "falling back to sequential evaluation."
                )
                raw_results = [evaluate_single_sub_agent_df(task) for task in tasks]
    finally:
        if isinstance(model, torch.nn.Module):
            model.train(was_training)

    sorted_results = sorted(
        raw_results, key=lambda m: (m.context_index, m.df_index)
    )

    for m in sorted_results:
        logger.info(
            "sub-agent eval | context_index=%d | df_index=%d | "
            "initial_action=0 | reward_sum=%.4f | final_balance=%.4f | return_rate=%.6f",
            m.context_index,
            m.df_index,
            m.reward_sum,
            m.final_balance,
            m.return_rate,
        )
        if getattr(trainer, "writer", None) is not None:
            trainer.writer.add_scalar(
                tag=f"{tag_prefix}_return_rate_context_{m.context_index}",
                scalar_value=m.return_rate,
                global_step=m.df_index,
                walltime=None,
            )
            trainer.writer.add_scalar(
                tag=f"{tag_prefix}_reward_sum_context_{m.context_index}",
                scalar_value=m.reward_sum,
                global_step=m.df_index,
                walltime=None,
            )
        eval_metrics.append(m)

    return eval_metrics


# Backward compatibility alias
evaluate_warmup_sub_agents = evaluate_sub_agents
