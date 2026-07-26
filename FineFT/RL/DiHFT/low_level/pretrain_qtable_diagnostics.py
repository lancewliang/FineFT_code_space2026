import json
import multiprocessing as mp
import os
import random
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl

from env.env_class.futures_util import (
    create_optimal_q_table_from_df,
    get_dp_action_from_qtable,
    map_action_to_position_leverage,
)
from env.env_initiate.demo_initiate import initiate_demo_env


DIAGNOSTIC_CSV_PATTERN = re.compile(
    r"^df_(?P<df_index>\d+)_initial_action_(?P<initial_action>\d+)\.csv$"
)
DIAGNOSTIC_MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True, order=True)
class SamplePlanItem:
    df_index: int
    initial_action: int

    def to_tuple(self):
        return (self.df_index, self.initial_action)


@dataclass(frozen=True)
class QTableDiagnosticsManifest:
    diagnostic_count: int
    total_df_index_length: int
    position_choices: int
    qtable_kwargs: dict
    env_kwargs: dict

    def to_dict(self):
        return _normalize_manifest_value(
            {
                "diagnostic_count": self.diagnostic_count,
                "total_df_index_length": self.total_df_index_length,
                "position_choices": self.position_choices,
                "qtable_kwargs": self.qtable_kwargs,
                "env_kwargs": self.env_kwargs,
            }
        )


@dataclass(frozen=True)
class DiagnosticCsvRow:
    df_index: int
    initial_action: int
    step_index: int
    timestamp: object
    open: object
    high: object
    low: object
    close: object
    volume: object
    mark_price: object
    action: int
    previous_action: int
    position: float
    leverage: float
    commission_rate: float
    step_slippage: float
    step_reward: float
    cumulative_profit: float
    profitable: bool

    def to_dict(self):
        return {
            "df_index": self.df_index,
            "initial_action": self.initial_action,
            "step_index": self.step_index,
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "mark_price": self.mark_price,
            "action": self.action,
            "previous_action": self.previous_action,
            "position": self.position,
            "leverage": self.leverage,
            "commission_rate": self.commission_rate,
            "step_slippage": self.step_slippage,
            "step_reward": self.step_reward,
            "cumulative_profit": self.cumulative_profit,
            "profitable": self.profitable,
        }


@dataclass(frozen=True)
class SampleDiagnostic:
    df_index: int
    initial_action: int
    episode_reward_sum: float
    profitable: bool
    csv_path: str
    action_list: list[int]

    @property
    def sample_item(self):
        return SamplePlanItem(self.df_index, self.initial_action)

    def to_dict(self, include_action_list=True):
        payload = {
            "df_index": self.df_index,
            "initial_action": self.initial_action,
            "episode_reward_sum": self.episode_reward_sum,
            "profitable": self.profitable,
            "csv_path": self.csv_path,
        }
        if include_action_list:
            payload["action_list"] = list(self.action_list)
        return payload


@dataclass(frozen=True)
class QTableCacheBuildResult:
    q_table_cache: dict
    train_df_cache: dict
    diagnostics: list[SampleDiagnostic]


@dataclass(frozen=True)
class PretrainQTableDiagnosticsResult:
    sample_plan: list[SamplePlanItem]
    q_table_cache: dict
    train_df_cache: dict
    diagnostics: list[SampleDiagnostic]
    sample_action_cache: dict[SamplePlanItem, list[int]]


def build_sample_plan(total_df_index_length, position_choices):
    return [
        SamplePlanItem(df_index, initial_action)
        for df_index in range(total_df_index_length)
        for initial_action in range(position_choices)
    ]


def select_sample_from_plan(sample_plan):
    return random.choice(sample_plan)


def build_balanced_training_schedule(sample_plan, num_sample):
    if num_sample < 0:
        raise ValueError("num_sample must be non-negative")
    if num_sample == 0:
        return []

    sample_plan = [_coerce_sample_item(item) for item in sample_plan]
    if not sample_plan:
        raise ValueError("sample_plan must not be empty when num_sample > 0")

    sample_items_by_df = {}
    for item in sorted(
        sample_plan,
        key=lambda item: (item.df_index, item.initial_action),
    ):
        sample_items_by_df.setdefault(item.df_index, []).append(item)

    df_indices = sorted(sample_items_by_df)
    remaining_items_by_df = {df_index: [] for df_index in df_indices}
    schedule = []
    while len(schedule) < num_sample:
        df_cycle = list(df_indices)
        random.shuffle(df_cycle)
        for df_index in df_cycle:
            if len(schedule) >= num_sample:
                break
            remaining_items = remaining_items_by_df[df_index]
            if not remaining_items:
                remaining_items.extend(sample_items_by_df[df_index])
                random.shuffle(remaining_items)
            schedule.append(remaining_items.pop(0))
    return schedule


def _coerce_sample_item(sample_item):
    if isinstance(sample_item, SamplePlanItem):
        return sample_item
    df_index, initial_action = sample_item
    return SamplePlanItem(int(df_index), int(initial_action))


def get_sample_action_from_cache(
    sample_action_cache_by_plan,
    sample_item_or_df_index,
    initial_action=None,
):
    if isinstance(sample_item_or_df_index, SamplePlanItem):
        sample_key = sample_item_or_df_index
    else:
        sample_key = SamplePlanItem(int(sample_item_or_df_index), int(initial_action))
    if sample_key not in sample_action_cache_by_plan:
        legacy_sample_key = sample_key.to_tuple()
        if legacy_sample_key in sample_action_cache_by_plan:
            return sample_action_cache_by_plan[legacy_sample_key]
        raise KeyError(
            "sample_action_cache missing for df_index={} initial_action={}".format(
                sample_key.df_index,
                sample_key.initial_action,
            )
        )
    return sample_action_cache_by_plan[sample_key]


def _create_q_table_worker(args):
    (
        df_index,
        train_data_path,
        qtable_kwargs,
        sample_tasks,
        env_kwargs,
        output_dir,
    ) = args
    df_path = os.path.join(train_data_path, "df_{}.feather".format(df_index))
    train_df = pd.read_feather(df_path)
    q_table = create_optimal_q_table_from_df(df=train_df, **qtable_kwargs)
    diagnostics = []
    if sample_tasks and env_kwargs is not None and output_dir is not None:
        for initial_action in sample_tasks:
            diagnostics.append(
                evaluate_and_export_sample(
                    df_index,
                    initial_action,
                    train_df,
                    q_table,
                    env_kwargs,
                    output_dir,
                )
            )
    return QTableCacheBuildResult(
        q_table_cache={df_index: q_table},
        train_df_cache={df_index: train_df},
        diagnostics=diagnostics,
    )


def build_q_table_cache(
    sample_plan,
    train_data_path,
    qtable_kwargs,
    env_kwargs=None,
    output_dir=None,
    process_count=None,
):
    sample_plan = [_coerce_sample_item(item) for item in sample_plan]
    unique_df_indices = sorted({item.df_index for item in sample_plan})
    if process_count is None:
        process_count = min(len(unique_df_indices), os.cpu_count() or 1)
    process_count = max(1, process_count)

    sample_tasks_by_df = {}
    if env_kwargs is not None and output_dir is not None:
        for item in sample_plan:
            sample_tasks_by_df.setdefault(item.df_index, []).append(
                item.initial_action
            )

    worker_args = [
        (
            df_index,
            train_data_path,
            qtable_kwargs,
            sample_tasks_by_df.get(df_index, []),
            env_kwargs,
            output_dir,
        )
        for df_index in unique_df_indices
    ]
    if process_count == 1:
        results = [_create_q_table_worker(args) for args in worker_args]
    else:
        with mp.Pool(processes=process_count) as pool:
            results = pool.map(_create_q_table_worker, worker_args)

    train_df_cache = {}
    q_table_cache = {}
    diagnostics = []
    for result in results:
        q_table_cache.update(result.q_table_cache)
        train_df_cache.update(result.train_df_cache)
        diagnostics.extend(result.diagnostics)
    return QTableCacheBuildResult(q_table_cache, train_df_cache, diagnostics)


def extend_q_table_cache(
    df_indices,
    train_data_path,
    qtable_kwargs,
    q_table_cache=None,
    train_df_cache=None,
    process_count=None,
):
    q_table_cache = dict(q_table_cache or {})
    train_df_cache = dict(train_df_cache or {})
    missing_df_indices = [
        df_index
        for df_index in sorted(set(df_indices))
        if df_index not in q_table_cache or df_index not in train_df_cache
    ]
    if not missing_df_indices:
        return q_table_cache, train_df_cache

    missing_plan = [SamplePlanItem(df_index, 0) for df_index in missing_df_indices]
    cache_result = build_q_table_cache(
        missing_plan,
        train_data_path,
        qtable_kwargs,
        process_count=process_count,
    )
    q_table_cache.update(cache_result.q_table_cache)
    train_df_cache.update(cache_result.train_df_cache)
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
        allow_reverse_position=env_kwargs.get("allow_reverse_position", False),
    )


def _normalize_manifest_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            str(key): _normalize_manifest_value(value[key])
            for key in sorted(value, key=str)
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_manifest_value(item) for item in value]
    return value


def _build_diagnostics_manifest(
    diagnostic_count,
    total_df_index_length,
    position_choices,
    qtable_kwargs,
    env_kwargs,
):
    return QTableDiagnosticsManifest(
        diagnostic_count=diagnostic_count,
        total_df_index_length=total_df_index_length,
        position_choices=position_choices,
        qtable_kwargs=qtable_kwargs,
        env_kwargs=env_kwargs,
    )


def _manifest_payload(manifest):
    return manifest.to_dict() if hasattr(manifest, "to_dict") else manifest


def _manifest_matches(output_dir, expected_manifest):
    manifest_path = os.path.join(output_dir, DIAGNOSTIC_MANIFEST_NAME)
    if not os.path.isfile(manifest_path):
        return False
    try:
        with open(manifest_path, "r", encoding="utf-8") as manifest_file:
            existing_manifest = json.load(manifest_file)
    except (OSError, json.JSONDecodeError):
        return False
    return existing_manifest == _manifest_payload(expected_manifest)


def _write_diagnostics_manifest(output_dir, manifest):
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, DIAGNOSTIC_MANIFEST_NAME)
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(_manifest_payload(manifest), manifest_file, sort_keys=True, indent=2)
        manifest_file.write("\n")


def _value_from_row(row, column):
    return row[column] if column in row.index else np.nan


def _diagnostic_row(
    train_df,
    env,
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
    return DiagnosticCsvRow(
        df_index=df_index,
        initial_action=initial_action,
        step_index=step_index,
        timestamp=_value_from_row(source_row, "timestamp"),
        open=_value_from_row(source_row, "open"),
        high=_value_from_row(source_row, "high"),
        low=_value_from_row(source_row, "low"),
        close=_value_from_row(source_row, "close"),
        volume=_value_from_row(source_row, "volume"),
        mark_price=_value_from_row(source_row, "mark_price"),
        action=action,
        previous_action=previous_action,
        position=env.position,
        leverage=env.leverage,
        commission_rate=env.commission_rate,
        step_slippage=step_slippage,
        step_reward=reward,
        cumulative_profit=cumulative_profit,
        profitable=cumulative_profit > 0,
    )


def evaluate_and_export_sample(
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
        "df_{}_initial_action_{}.csv".format(df_index, initial_action),
    )
    pl.DataFrame([row.to_dict() for row in rows]).write_csv(csv_path)
    return SampleDiagnostic(
        df_index=df_index,
        initial_action=initial_action,
        episode_reward_sum=cumulative_profit,
        profitable=cumulative_profit > 0,
        csv_path=csv_path,
        action_list=[row.action for row in rows],
    )


def _load_existing_diagnostics(
    train_data_path,
    output_dir,
    manifest,
    total_df_index_length,
    position_choices,
):
    if not os.path.isdir(output_dir):
        return None
    if not _manifest_matches(output_dir, manifest):
        return None

    csv_by_plan = {}
    for file_name in os.listdir(output_dir):
        match = DIAGNOSTIC_CSV_PATTERN.match(file_name)
        if match is None:
            continue
        df_index = int(match.group("df_index"))
        initial_action = int(match.group("initial_action"))
        csv_by_plan[SamplePlanItem(df_index, initial_action)] = os.path.join(
            output_dir, file_name
        )

    sample_plan = [
        SamplePlanItem(df_index, initial_action)
        for df_index in range(total_df_index_length)
        for initial_action in range(position_choices)
    ]
    if any(plan_item not in csv_by_plan for plan_item in sample_plan):
        return None

    diagnostics = []
    sample_action_cache = {}
    train_df_cache = {}
    for sample_item in sample_plan:
        csv_path = csv_by_plan[sample_item]
        diagnostic_df = pl.read_csv(csv_path)
        if "action" not in diagnostic_df.columns:
            return None
        if "cumulative_profit" in diagnostic_df.columns and len(diagnostic_df) > 0:
            episode_reward_sum = diagnostic_df["cumulative_profit"][-1]
        elif "step_reward" in diagnostic_df.columns:
            episode_reward_sum = diagnostic_df["step_reward"].sum()
        else:
            return None

        action_list = diagnostic_df["action"].cast(pl.Int64).to_list()
        sample_action_cache[sample_item] = action_list
        episode_reward_sum = float(episode_reward_sum)
        diagnostics.append(
            SampleDiagnostic(
                df_index=sample_item.df_index,
                initial_action=sample_item.initial_action,
                episode_reward_sum=episode_reward_sum,
                profitable=episode_reward_sum > 0,
                csv_path=csv_path,
                action_list=action_list,
            )
        )
        if sample_item.df_index not in train_df_cache:
            df_path = os.path.join(
                train_data_path, "df_{}.feather".format(sample_item.df_index)
            )
            train_df_cache[sample_item.df_index] = pd.read_feather(df_path)

    return PretrainQTableDiagnosticsResult(
        sample_plan=sample_plan,
        q_table_cache={},
        train_df_cache=train_df_cache,
        diagnostics=diagnostics,
        sample_action_cache=sample_action_cache,
    )


def prepare_pretrain_qtable_diagnostics(
    total_df_index_length,
    position_choices,
    train_data_path,
    qtable_kwargs,
    env_kwargs,
    output_dir,
    logger=None,
    process_count=None,
    num_sample=None,
):
    sample_count = total_df_index_length * position_choices
    manifest = _build_diagnostics_manifest(
        sample_count,
        total_df_index_length,
        position_choices,
        qtable_kwargs,
        env_kwargs,
    )
    existing = _load_existing_diagnostics(
        train_data_path,
        output_dir,
        manifest,
        total_df_index_length,
        position_choices,
    )
    if existing is not None:
        for diagnostic in existing.diagnostics:
            message = (
                "qtable诊断 | df_index={df_index} | "
                "initial_action={initial_action} | episode_reward_sum={episode_reward_sum:.4f} | "
                "profitable={profitable} | csv_path={csv_path} | source=csv"
            ).format(**diagnostic.to_dict(include_action_list=False))
            if logger is not None:
                logger.info(message)
        return existing

    sample_plan = build_sample_plan(total_df_index_length, position_choices)
    cache_result = build_q_table_cache(
        sample_plan,
        train_data_path,
        qtable_kwargs,
        env_kwargs=env_kwargs,
        output_dir=output_dir,
        process_count=process_count,
    )
    sample_action_cache = {}
    diagnostics = sorted(
        cache_result.diagnostics, key=lambda item: (item.df_index, item.initial_action)
    )
    for diagnostic in diagnostics:
        sample_action_cache[diagnostic.sample_item] = diagnostic.action_list
        message = (
            "qtable诊断 | df_index={df_index} | "
            "initial_action={initial_action} | episode_reward_sum={episode_reward_sum:.4f} | "
            "profitable={profitable} | csv_path={csv_path}"
        ).format(**diagnostic.to_dict(include_action_list=False))
        if logger is not None:
            logger.info(message)
    _write_diagnostics_manifest(output_dir, manifest)
    return PretrainQTableDiagnosticsResult(
        sample_plan=sample_plan,
        q_table_cache=cache_result.q_table_cache,
        train_df_cache=cache_result.train_df_cache,
        diagnostics=diagnostics,
        sample_action_cache=sample_action_cache,
    )
