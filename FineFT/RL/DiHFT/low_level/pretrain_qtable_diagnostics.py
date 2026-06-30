import multiprocessing as mp
import os
import random

import numpy as np
import pandas as pd

from env.env_class.futures_util import (
    create_optimal_q_table_from_df,
    get_dp_action_from_qtable,
    map_action_to_position_leverage,
)
from env.env_initiate.demo_initiate import initiate_demo_env


def build_sample_plan(num_sample, total_df_index_length, position_choices):
    sample_plan = []
    for _ in range(num_sample):
        df_index = random.choices(range(total_df_index_length), k=1)[0]
        initial_action = random.choices(range(position_choices), k=1)[0]
        sample_plan.append((df_index, initial_action))
    return sample_plan


def _create_q_table_worker(args):
    df_index, train_data_path, qtable_kwargs = args
    df_path = os.path.join(train_data_path, "df_{}.feather".format(df_index))
    train_df = pd.read_feather(df_path)
    q_table = create_optimal_q_table_from_df(df=train_df, **qtable_kwargs)
    return df_index, train_df, q_table


def build_q_table_cache(sample_plan, train_data_path, qtable_kwargs, process_count=None):
    unique_df_indices = sorted({df_index for df_index, _ in sample_plan})
    if process_count is None:
        process_count = min(len(unique_df_indices), os.cpu_count() or 1)
    process_count = max(1, process_count)

    worker_args = [
        (df_index, train_data_path, qtable_kwargs) for df_index in unique_df_indices
    ]
    if process_count == 1:
        results = [_create_q_table_worker(args) for args in worker_args]
    else:
        with mp.Pool(processes=process_count) as pool:
            results = pool.map(_create_q_table_worker, worker_args)

    train_df_cache = {}
    q_table_cache = {}
    for df_index, train_df, q_table in results:
        train_df_cache[df_index] = train_df
        q_table_cache[df_index] = q_table
    return q_table_cache, train_df_cache


def build_initial_state(
    train_df,
    initial_action,
    leverage_choices,
    position_list,
    initial_wallet_balance,
    initial_unrealized_pnl,
):
    initial_position, initial_leverage = map_action_to_position_leverage(
        initial_action, leverage_choices, position_list
    )
    current_markprice = train_df["mark_price"].values[0]
    initial_margin = np.abs(initial_position * current_markprice / initial_leverage)
    initial_state = (
        initial_wallet_balance,
        initial_margin,
        initial_unrealized_pnl,
        initial_position,
        initial_leverage,
    )
    return initial_position, initial_leverage, initial_margin, initial_state


def create_demo_env(train_df, env_kwargs, initial_state):
    return initiate_demo_env(
        df=train_df,
        feature_list=env_kwargs["feature_list"],
        max_holding_number=env_kwargs["max_holding_number"],
        order_book_depth=env_kwargs["order_book_depth"],
        position_choices=env_kwargs["position_choices"],
        leverage_choice=env_kwargs["leverage_choices"],
        long_estimated_rate=env_kwargs["long_estimated_rate"],
        short_estimated_rate=env_kwargs["short_estimated_rate"],
        commission_rate=env_kwargs["commission_rate"],
        maintenance_margin_ratio_dict=env_kwargs["maintenance_margin_ratio_dict"],
        early_stop=env_kwargs["early_stop"],
        initial_state=initial_state,
        gamma=env_kwargs["gamma"],
        max_punishment=1e10,
    )


def _value_from_row(row, column):
    return row[column] if column in row.index else np.nan


def _diagnostic_row(
    train_df,
    env,
    sample_index,
    df_index,
    initial_action,
    step_index,
    action,
    previous_action,
    reward,
    cumulative_profit,
    previous_slippage_sum,
):
    source_row = train_df.iloc[min(step_index, len(train_df) - 1)]
    step_slippage = env.slippage_sum - previous_slippage_sum
    return {
        "sample_index": sample_index + 1,
        "df_index": df_index,
        "initial_action": initial_action,
        "step_index": step_index,
        "timestamp": _value_from_row(source_row, "timestamp"),
        "open": _value_from_row(source_row, "open"),
        "high": _value_from_row(source_row, "high"),
        "low": _value_from_row(source_row, "low"),
        "close": _value_from_row(source_row, "close"),
        "volume": _value_from_row(source_row, "volume"),
        "mark_price": _value_from_row(source_row, "mark_price"),
        "action": action,
        "previous_action": previous_action,
        "position": env.position,
        "leverage": env.leverage,
        "commission_rate": env.commission_rate,
        "step_slippage": step_slippage,
        "step_reward": reward,
        "cumulative_profit": cumulative_profit,
        "profitable": cumulative_profit > 0,
    }


def evaluate_and_export_sample(
    sample_index,
    df_index,
    initial_action,
    train_df,
    q_table,
    env_kwargs,
    output_dir,
):
    _, _, _, initial_state = build_initial_state(
        train_df,
        initial_action,
        env_kwargs["leverage_choices"],
        env_kwargs["position_list"],
        env_kwargs["initial_wallet_balance"],
        env_kwargs["initial_unrealized_pnl"],
    )
    env = create_demo_env(train_df, env_kwargs, initial_state)
    action_list = get_dp_action_from_qtable(q_table, initial_action)
    _, info = env.reset()
    cumulative_profit = 0
    previous_slippage_sum = env.slippage_sum
    rows = []

    for step_index, action in enumerate(action_list):
        previous_action = info["previous_action"]
        _, reward, done, info = env.step(action)
        cumulative_profit += reward
        rows.append(
            _diagnostic_row(
                train_df,
                env,
                sample_index,
                df_index,
                initial_action,
                step_index,
                action,
                previous_action,
                reward,
                cumulative_profit,
                previous_slippage_sum,
            )
        )
        previous_slippage_sum = env.slippage_sum
        if done:
            break

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(
        output_dir,
        "sample_{:04d}_df_{}_initial_action_{}.csv".format(
            sample_index + 1, df_index, initial_action
        ),
    )
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return {
        "sample_index": sample_index + 1,
        "df_index": df_index,
        "initial_action": initial_action,
        "episode_reward_sum": cumulative_profit,
        "profitable": cumulative_profit > 0,
        "csv_path": csv_path,
    }


def prepare_pretrain_qtable_diagnostics(
    num_sample,
    total_df_index_length,
    position_choices,
    train_data_path,
    qtable_kwargs,
    env_kwargs,
    output_dir,
    logger=None,
    process_count=None,
):
    sample_plan = build_sample_plan(
        num_sample, total_df_index_length, position_choices
    )
    q_table_cache, train_df_cache = build_q_table_cache(
        sample_plan,
        train_data_path,
        qtable_kwargs,
        process_count=process_count,
    )
    diagnostics = []
    for sample_index, (df_index, initial_action) in enumerate(sample_plan):
        diagnostic = evaluate_and_export_sample(
            sample_index,
            df_index,
            initial_action,
            train_df_cache[df_index],
            q_table_cache[df_index],
            env_kwargs,
            output_dir,
        )
        diagnostics.append(diagnostic)
        message = (
            "qtable诊断 | sample={sample_index} | df_index={df_index} | "
            "initial_action={initial_action} | episode_reward_sum={episode_reward_sum:.4f} | "
            "profitable={profitable} | csv_path={csv_path}"
        ).format(**diagnostic)
        if logger is not None:
            logger.info(message)
        print(message.replace(" | ", " "))
    return sample_plan, q_table_cache, train_df_cache, diagnostics
