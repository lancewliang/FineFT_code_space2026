"""Select a 4x4 low-level agent grid from slope and volatility results.

This script is intentionally independent from
``FineFT_single_agent_with_different_position.py``.  It reads the CSV artifacts
already produced by ``test_agent_index.py``, writes selection reports, and
assembles ``model.pth`` without changing the high-level agent.

For every candidate ``(epoch_path, bin_index)`` and label pair, performance is
measured in three ways:

* the complete volatility-label slices from ``analysis_result.csv``;
* the complete slope-label slices from ``analysis_result.csv``;
* the timestamp intersection of both labels from each label type's trading
  detail CSV.

The two trading-detail evaluations deliberately remain separate because a
slope slice and a volatility slice can reset the initial position at different
timestamps.  A candidate must be profitable under both reset contexts.  If no
candidate passes all gates, the logical slot is recorded as a null/flat slot.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

FINEFT_ROOT = Path(__file__).resolve().parents[2]
if str(FINEFT_ROOT) not in sys.path:
    sys.path.insert(0, str(FINEFT_ROOT))

from model.low_level import ensemble_Qnet


LABEL_TYPES = ("volatility", "slope")
LABEL_PATTERN = re.compile(r"^label_(\d+)$")
EPOCH_PATTERN = re.compile(r"^epoch_(\d+)$")

ANALYSIS_COLUMNS = {
    "标签": "label",
    "初始动作": "initial_action",
    "分箱索引": "bin_index",
    "合约": "contract",
    "数据文件": "df_path",
    "奖励总和": "reward_sum",
    "数据长度": "df_length",
    "换手率": "turnover",
}
DETAIL_COLUMNS = {
    "标签": "label",
    "数据文件": "df_path",
    "初始动作": "initial_action",
    "分箱索引": "bin_index",
    "时间戳": "timestamp",
    "单步奖励": "reward",
}

CANDIDATE_KEYS = [
    "candidate_id",
    "epoch_number",
    "epoch_path",
    "model_path",
    "bin_index",
]


@dataclass(frozen=True)
class SelectionConfig:
    """Risk and coverage gates used by the selector."""

    num_labels: int = 4
    lcb_z: float = 1.0
    min_marginal_contracts: int = 2
    min_joint_contracts: int = 2
    min_positive_contract_ratio: float = 0.60
    min_mean_return: float = 0.0
    min_lcb: float = 0.0
    min_worst_initial_position_return: float = 0.0
    csv_chunk_size: int = 250_000
    missing_joint_policy: str = "empty_model"


@dataclass
class SelectionArtifacts:
    """In-memory result returned through the module's selection interface."""

    marginal_metrics: pd.DataFrame
    joint_metrics: pd.DataFrame
    candidate_rankings: pd.DataFrame
    selected_slots: pd.DataFrame
    manifest: dict[str, Any]

    def write(self, output_dir: Path) -> dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "marginal_metrics": output_dir / "two_dimensional_marginal_metrics.csv",
            "joint_metrics": output_dir / "two_dimensional_joint_metrics.csv",
            "candidate_rankings": output_dir
            / "two_dimensional_candidate_rankings.csv",
            "selected_slots": output_dir / "two_dimensional_selection.csv",
            "manifest": output_dir / "two_dimensional_selection_manifest.json",
        }
        self.marginal_metrics.to_csv(paths["marginal_metrics"], index=False)
        self.joint_metrics.to_csv(paths["joint_metrics"], index=False)
        self.candidate_rankings.to_csv(paths["candidate_rankings"], index=False)
        self.selected_slots.to_csv(paths["selected_slots"], index=False)
        with paths["manifest"].open("w", encoding="utf-8") as file:
            json.dump(_json_safe(self.manifest), file, ensure_ascii=False, indent=2)
        return paths


class TwoDimensionalAgentSelector:
    """Build a robust logical agent grid behind one selection interface."""

    def __init__(self, config: SelectionConfig) -> None:
        self.config = config
        self.labels = [f"label_{index}" for index in range(config.num_labels)]

    def select(
        self,
        candidate_root: Path,
        valid_root: Path,
        *,
        min_epoch: int | None = None,
        max_epoch: int | None = None,
    ) -> SelectionArtifacts:
        """Select all logical slots and return reports without writing files."""

        result_files = self._discover_common_result_files(
            candidate_root, min_epoch=min_epoch, max_epoch=max_epoch
        )
        marginal_rows = self._load_marginal_rows(result_files)
        marginal_rows, coverage = self._keep_complete_candidates(marginal_rows)
        marginal_metrics = self._calculate_metrics(
            marginal_rows,
            group_keys=CANDIDATE_KEYS + ["label_type", "label"],
        )

        label_lookup, joint_support = self._load_joint_label_lookup(valid_root)
        joint_rows = self._load_joint_detail_rows(
            result_files,
            label_lookup,
            allowed_candidates=set(marginal_rows["candidate_id"]),
        )
        self._validate_joint_totals(marginal_rows, joint_rows)
        joint_metrics = self._calculate_metrics(
            joint_rows,
            group_keys=CANDIDATE_KEYS
            + ["route_context", "volatility_label", "slope_label"],
        )
        rankings, slots = self._select_grid(
            marginal_metrics, joint_metrics, joint_support
        )
        manifest = self._build_manifest(
            candidate_root,
            valid_root,
            result_files,
            coverage,
            slots,
        )
        return SelectionArtifacts(
            marginal_metrics=marginal_metrics,
            joint_metrics=joint_metrics,
            candidate_rankings=rankings,
            selected_slots=slots,
            manifest=manifest,
        )

    def _discover_common_result_files(
        self,
        candidate_root: Path,
        *,
        min_epoch: int | None,
        max_epoch: int | None,
    ) -> dict[int, dict[str, Path]]:
        if not candidate_root.is_dir():
            raise FileNotFoundError(f"candidate root does not exist: {candidate_root}")

        by_epoch: dict[int, dict[str, Path]] = {}
        for epoch_path in candidate_root.glob("epoch_*"):
            match = EPOCH_PATTERN.fullmatch(epoch_path.name)
            if not match or not epoch_path.is_dir():
                continue
            epoch_number = int(match.group(1))
            if min_epoch is not None and epoch_number < min_epoch:
                continue
            if max_epoch is not None and epoch_number > max_epoch:
                continue
            files = {
                label_type: epoch_path / label_type / "analysis_result.csv"
                for label_type in LABEL_TYPES
            }
            if all(path.is_file() for path in files.values()):
                for label_type in LABEL_TYPES:
                    detail_path = (
                        epoch_path
                        / label_type
                        / f"trading_action_detail_epoch_{epoch_number}.csv"
                    )
                    if not detail_path.is_file():
                        raise FileNotFoundError(
                            f"missing trading detail for common epoch: {detail_path}"
                        )
                    files[f"{label_type}_detail"] = detail_path
                model_path = epoch_path / "trained_model.pkl"
                if not model_path.is_file():
                    raise FileNotFoundError(
                        f"missing checkpoint for common epoch: {model_path}"
                    )
                by_epoch[epoch_number] = files

        if not by_epoch:
            raise ValueError(
                "no epoch has complete slope and volatility CSV results"
            )
        return dict(sorted(by_epoch.items()))

    def _load_marginal_rows(
        self, result_files: dict[int, dict[str, Path]]
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for epoch_number, files in result_files.items():
            epoch_path = files["slope"].parent.parent
            model_path = epoch_path / "trained_model.pkl"
            for label_type in LABEL_TYPES:
                frame = pd.read_csv(files[label_type])
                missing = set(ANALYSIS_COLUMNS) - set(frame.columns)
                if missing:
                    raise ValueError(
                        f"{files[label_type]} missing columns: {sorted(missing)}"
                    )
                frame = frame.rename(columns=ANALYSIS_COLUMNS)
                for record in frame[list(ANALYSIS_COLUMNS.values())].to_dict("records"):
                    label = str(record["label"])
                    self._validate_label(label, files[label_type])
                    arrays = {
                        field: _parse_json_array(
                            record[field], files[label_type], field
                        )
                        for field in (
                            "contract",
                            "df_path",
                            "reward_sum",
                            "df_length",
                            "turnover",
                        )
                    }
                    lengths = {field: len(values) for field, values in arrays.items()}
                    if len(set(lengths.values())) != 1 or not arrays["reward_sum"]:
                        raise ValueError(
                            "unaligned or empty arrays in "
                            f"{files[label_type]}: {lengths}"
                        )
                    bin_index = int(record["bin_index"])
                    candidate = _candidate_fields(
                        epoch_number, epoch_path, model_path, bin_index
                    )
                    for index in range(lengths["reward_sum"]):
                        reward_sum = float(arrays["reward_sum"][index])
                        df_length = int(arrays["df_length"][index])
                        turnover = float(arrays["turnover"][index])
                        if df_length <= 0 or not math.isfinite(reward_sum):
                            raise ValueError(
                                f"invalid reward or length in {files[label_type]}"
                            )
                        transition_count = df_length - 1
                        if transition_count == 0:
                            if not math.isclose(reward_sum, 0.0, abs_tol=1e-12):
                                raise ValueError(
                                    "single-row slice has non-zero reward in "
                                    f"{files[label_type]}"
                                )
                            continue
                        rows.append(
                            {
                                **candidate,
                                "label_type": label_type,
                                "label": label,
                                "initial_action": int(record["initial_action"]),
                                "contract": str(arrays["contract"][index]),
                                "df_path": str(arrays["df_path"][index]),
                                "reward_sum": reward_sum,
                                "step_count": transition_count,
                                "turnover_sum": turnover,
                            }
                        )
        return pd.DataFrame(rows)

    def _keep_complete_candidates(
        self, rows: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        initial_actions = sorted(
            int(value) for value in rows["initial_action"].unique()
        )
        expected = {
            (label_type, label, initial_action)
            for label_type in LABEL_TYPES
            for label in self.labels
            for initial_action in initial_actions
        }
        complete: list[str] = []
        for candidate_id, candidate_rows in rows.groupby("candidate_id", sort=False):
            actual = set(
                candidate_rows[["label_type", "label", "initial_action"]]
                .drop_duplicates()
                .itertuples(index=False, name=None)
            )
            if actual == expected:
                complete.append(str(candidate_id))

        if not complete:
            raise ValueError("no candidate has complete two-dimensional label coverage")
        filtered = rows[rows["candidate_id"].isin(complete)].copy()
        coverage = {
            "discovered_candidate_count": int(rows["candidate_id"].nunique()),
            "complete_candidate_count": len(complete),
            "excluded_incomplete_candidate_count": int(
                rows["candidate_id"].nunique() - len(complete)
            ),
            "initial_actions": initial_actions,
        }
        return filtered, coverage

    def _load_joint_label_lookup(
        self, valid_root: Path
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        frames: dict[str, pd.DataFrame] = {}
        for label_type in LABEL_TYPES:
            label_root = valid_root / label_type
            paths = sorted(label_root.glob("*/label_*/*.feather"))
            if not paths:
                raise FileNotFoundError(
                    f"no validation label slices found under {label_root}"
                )
            parts: list[pd.DataFrame] = []
            for path in paths:
                label = path.parent.name
                self._validate_label(label, path)
                contract = path.parent.parent.name
                data = pd.read_feather(path, columns=["timestamp"])
                parts.append(
                    pd.DataFrame(
                        {
                            "contract": contract,
                            "timestamp": data["timestamp"].astype(str),
                            f"{label_type}_label": label,
                        }
                    )
                )
            frame = pd.concat(parts, ignore_index=True)
            if frame.duplicated(["contract", "timestamp"]).any():
                raise ValueError(
                    f"{label_type} validation slices contain duplicate timestamps"
                )
            frames[label_type] = frame

        lookup = frames["volatility"].merge(
            frames["slope"],
            on=["contract", "timestamp"],
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
        unmatched = lookup["_merge"] != "both"
        if unmatched.any():
            raise ValueError(
                "slope and volatility validation timestamps do not cover the "
                "same rows; "
                f"unmatched={int(unmatched.sum())}"
            )
        lookup = lookup.drop(columns="_merge")

        support = (
            lookup.groupby(["volatility_label", "slope_label"], as_index=False)
            .agg(
                joint_row_count=("timestamp", "size"),
                joint_contract_count=("contract", "nunique"),
            )
        )
        grid = pd.MultiIndex.from_product(
            [self.labels, self.labels],
            names=["volatility_label", "slope_label"],
        ).to_frame(index=False)
        support = grid.merge(
            support,
            on=["volatility_label", "slope_label"],
            how="left",
        ).fillna({"joint_row_count": 0, "joint_contract_count": 0})
        support[["joint_row_count", "joint_contract_count"]] = support[
            ["joint_row_count", "joint_contract_count"]
        ].astype(int)
        return lookup, support

    def _load_joint_detail_rows(
        self,
        result_files: dict[int, dict[str, Path]],
        label_lookup: pd.DataFrame,
        *,
        allowed_candidates: set[str],
    ) -> pd.DataFrame:
        summaries: list[pd.DataFrame] = []
        lookup = label_lookup[
            ["contract", "timestamp", "volatility_label", "slope_label"]
        ]
        use_columns = list(DETAIL_COLUMNS)

        for epoch_number, files in result_files.items():
            epoch_path = files["slope"].parent.parent
            model_path = epoch_path / "trained_model.pkl"
            for route_context in LABEL_TYPES:
                detail_path = files[f"{route_context}_detail"]
                for chunk in pd.read_csv(
                    detail_path,
                    usecols=use_columns,
                    chunksize=self.config.csv_chunk_size,
                ):
                    chunk = chunk.rename(columns=DETAIL_COLUMNS)
                    chunk["contract"] = chunk["df_path"].str.split("/", n=1).str[0]
                    chunk["timestamp"] = chunk["timestamp"].astype(str)
                    chunk["bin_index"] = chunk["bin_index"].astype(int)
                    chunk["candidate_id"] = chunk["bin_index"].map(
                        lambda bin_index: _candidate_id(epoch_number, int(bin_index))
                    )
                    chunk = chunk[chunk["candidate_id"].isin(allowed_candidates)]
                    if chunk.empty:
                        continue
                    chunk = chunk.merge(
                        lookup,
                        on=["contract", "timestamp"],
                        how="left",
                        validate="many_to_one",
                    )
                    if chunk[["volatility_label", "slope_label"]].isna().any().any():
                        raise ValueError(
                            f"{detail_path} contains timestamps absent from "
                            "validation labels"
                        )
                    own_label = f"{route_context}_label"
                    mismatch = chunk["label"] != chunk[own_label]
                    if mismatch.any():
                        raise ValueError(
                            f"{detail_path} label disagrees with timestamp label lookup"
                        )
                    chunk["reward"] = pd.to_numeric(chunk["reward"], errors="raise")
                    grouped = (
                        chunk.groupby(
                            [
                                "candidate_id",
                                "bin_index",
                                "initial_action",
                                "contract",
                                "volatility_label",
                                "slope_label",
                            ],
                            as_index=False,
                        )
                        .agg(
                            reward_sum=("reward", "sum"),
                            step_count=("reward", "size"),
                        )
                    )
                    grouped["turnover_sum"] = 0.0
                    grouped["epoch_number"] = epoch_number
                    grouped["epoch_path"] = str(epoch_path)
                    grouped["model_path"] = str(model_path)
                    grouped["route_context"] = route_context
                    summaries.append(grouped)

        if not summaries:
            raise ValueError("trading detail files produced no joint evaluation rows")
        joint_rows = pd.concat(summaries, ignore_index=True)
        group_keys = CANDIDATE_KEYS + [
            "route_context",
            "volatility_label",
            "slope_label",
            "initial_action",
            "contract",
        ]
        return (
            joint_rows.groupby(group_keys, as_index=False)
            .agg(
                reward_sum=("reward_sum", "sum"),
                step_count=("step_count", "sum"),
                turnover_sum=("turnover_sum", "sum"),
            )
        )

    def _validate_joint_totals(
        self, marginal_rows: pd.DataFrame, joint_rows: pd.DataFrame
    ) -> None:
        """Ensure timestamp partitioning preserves every source reward and step."""

        comparison_keys = CANDIDATE_KEYS + [
            "label_type",
            "label",
            "initial_action",
            "contract",
        ]
        expected = (
            marginal_rows.groupby(comparison_keys, as_index=False)
            .agg(
                expected_reward_sum=("reward_sum", "sum"),
                expected_step_count=("step_count", "sum"),
            )
        )
        actual = joint_rows.copy()
        actual["label_type"] = actual["route_context"]
        actual["label"] = np.where(
            actual["route_context"] == "volatility",
            actual["volatility_label"],
            actual["slope_label"],
        )
        actual = (
            actual.groupby(comparison_keys, as_index=False)
            .agg(
                actual_reward_sum=("reward_sum", "sum"),
                actual_step_count=("step_count", "sum"),
            )
        )
        comparison = expected.merge(
            actual,
            on=comparison_keys,
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
        reward_matches = np.isclose(
            comparison["expected_reward_sum"],
            comparison["actual_reward_sum"],
            rtol=1e-9,
            atol=1e-6,
            equal_nan=False,
        )
        step_matches = (
            comparison["expected_step_count"]
            == comparison["actual_step_count"]
        )
        invalid = (comparison["_merge"] != "both") | ~reward_matches | ~step_matches
        if invalid.any():
            sample = comparison.loc[invalid].head(5).to_dict("records")
            raise ValueError(
                "joint timestamp partition does not preserve analysis_result.csv "
                f"totals; invalid_groups={int(invalid.sum())}, sample={sample}"
            )

    def _calculate_metrics(
        self, rows: pd.DataFrame, *, group_keys: list[str]
    ) -> pd.DataFrame:
        base_keys = group_keys + ["contract", "initial_action"]
        contract_position = (
            rows.groupby(base_keys, as_index=False)
            .agg(
                reward_sum=("reward_sum", "sum"),
                step_count=("step_count", "sum"),
                turnover_sum=("turnover_sum", "sum"),
            )
        )
        contract_position["return_per_step"] = (
            contract_position["reward_sum"] / contract_position["step_count"]
        )

        metric_rows: list[dict[str, Any]] = []
        grouper: str | list[str] = group_keys[0] if len(group_keys) == 1 else group_keys
        for key, group in contract_position.groupby(grouper, sort=False):
            key_values = (key,) if not isinstance(key, tuple) else key
            identity = dict(zip(group_keys, key_values))
            contract_returns = group.groupby("contract")["return_per_step"].mean()
            position_returns = group.groupby("initial_action")["return_per_step"].mean()
            contract_count = int(contract_returns.size)
            mean_return = float(contract_returns.mean())
            contract_std = (
                float(contract_returns.std(ddof=1)) if contract_count > 1 else math.nan
            )
            standard_error = (
                contract_std / math.sqrt(contract_count)
                if contract_count > 1
                else math.nan
            )
            if self.config.lcb_z == 0.0:
                lcb = mean_return
            elif math.isfinite(standard_error):
                lcb = mean_return - self.config.lcb_z * standard_error
            else:
                lcb = math.nan
            metric_rows.append(
                {
                    **identity,
                    "mean_return_per_step": mean_return,
                    "contract_std": contract_std,
                    "standard_error": standard_error,
                    "lcb_return_per_step": lcb,
                    "worst_initial_position_return": float(position_returns.min()),
                    "positive_contract_ratio": float((contract_returns > 0).mean()),
                    "contract_count": contract_count,
                    "initial_position_count": int(position_returns.size),
                    "step_count": int(group["step_count"].sum()),
                    "turnover_sum": float(group["turnover_sum"].sum()),
                }
            )
        return pd.DataFrame(metric_rows)

    def _select_grid(
        self,
        marginal_metrics: pd.DataFrame,
        joint_metrics: pd.DataFrame,
        joint_support: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        rankings: list[pd.DataFrame] = []
        slots: list[dict[str, Any]] = []

        for volatility_index, volatility_label in enumerate(self.labels):
            for slope_index, slope_label in enumerate(self.labels):
                slot_id = volatility_index * self.config.num_labels + slope_index
                pair = self._pair_candidates(
                    marginal_metrics,
                    joint_metrics,
                    volatility_label,
                    slope_label,
                )
                support_row = joint_support[
                    (joint_support["volatility_label"] == volatility_label)
                    & (joint_support["slope_label"] == slope_label)
                ].iloc[0]
                pair["slot_id"] = slot_id
                pair["joint_row_count"] = int(support_row["joint_row_count"])
                pair["joint_contract_count"] = int(
                    support_row["joint_contract_count"]
                )
                pair = self._apply_selection_gates(pair)
                pair = pair.sort_values(
                    [
                        "eligible",
                        "pair_score",
                        "mean_lcb",
                        "epoch_number",
                        "bin_index",
                    ],
                    ascending=[False, False, False, True, True],
                    na_position="last",
                ).reset_index(drop=True)
                pair["rank_in_slot"] = np.arange(1, len(pair) + 1)
                rankings.append(pair)
                eligible = pair[pair["eligible"]]
                if eligible.empty:
                    fallback_candidate = None
                    if (
                        int(support_row["joint_row_count"]) == 0
                        and self.config.missing_joint_policy == "slope_marginal_best"
                    ):
                        marginal_gates = (
                            pair["gate_marginal_contracts"]
                            & (
                                pair["mean_return_per_step_volatility"]
                                > self.config.min_mean_return
                            )
                            & (
                                pair["mean_return_per_step_slope"]
                                > self.config.min_mean_return
                            )
                            & (
                                pair["lcb_return_per_step_volatility"]
                                > self.config.min_lcb
                            )
                            & (
                                pair["lcb_return_per_step_slope"]
                                > self.config.min_lcb
                            )
                            & (
                                pair["worst_initial_position_return_volatility"]
                                >= self.config.min_worst_initial_position_return
                            )
                            & (
                                pair["worst_initial_position_return_slope"]
                                >= self.config.min_worst_initial_position_return
                            )
                            & (
                                pair["positive_contract_ratio_volatility"]
                                >= self.config.min_positive_contract_ratio
                            )
                            & (
                                pair["positive_contract_ratio_slope"]
                                >= self.config.min_positive_contract_ratio
                            )
                        )
                        marginal_eligible = pair[marginal_gates].copy()
                        if not marginal_eligible.empty:
                            marginal_eligible["marginal_score"] = marginal_eligible[
                                ["lcb_return_per_step_volatility", "lcb_return_per_step_slope"]
                            ].min(axis=1)
                            fallback_candidate = marginal_eligible.sort_values(
                                ["marginal_score", "mean_lcb", "epoch_number", "bin_index"],
                                ascending=[False, False, True, True],
                            ).iloc[0].copy()
                            fallback_candidate["pair_score"] = fallback_candidate["marginal_score"]
                    if fallback_candidate is not None:
                        slots.append(
                            self._model_slot(
                                slot_id,
                                volatility_label,
                                slope_label,
                                fallback_candidate,
                                support_row,
                                reason="fallback_from_slope_marginal",
                            )
                        )
                    else:
                        slots.append(
                            self._null_slot(
                                slot_id,
                                volatility_label,
                                slope_label,
                                pair,
                                support_row,
                            )
                        )
                else:
                    slots.append(
                        self._model_slot(
                            slot_id,
                            volatility_label,
                            slope_label,
                            eligible.iloc[0],
                            support_row,
                        )
                    )

        return pd.concat(rankings, ignore_index=True), pd.DataFrame(slots)

    def _pair_candidates(
        self,
        marginal_metrics: pd.DataFrame,
        joint_metrics: pd.DataFrame,
        volatility_label: str,
        slope_label: str,
    ) -> pd.DataFrame:
        volatility = marginal_metrics[
            (marginal_metrics["label_type"] == "volatility")
            & (marginal_metrics["label"] == volatility_label)
        ].drop(columns=["label_type", "label"])
        slope = marginal_metrics[
            (marginal_metrics["label_type"] == "slope")
            & (marginal_metrics["label"] == slope_label)
        ].drop(columns=["label_type", "label"])
        pair = volatility.merge(
            slope,
            on=CANDIDATE_KEYS,
            how="inner",
            validate="one_to_one",
            suffixes=("_volatility", "_slope"),
        )

        joint = joint_metrics[
            (joint_metrics["volatility_label"] == volatility_label)
            & (joint_metrics["slope_label"] == slope_label)
        ].drop(columns=["volatility_label", "slope_label"])
        context_frames: list[pd.DataFrame] = []
        metric_columns = [
            column for column in joint.columns if column not in CANDIDATE_KEYS
        ]
        for route_context in LABEL_TYPES:
            context = joint[joint["route_context"] == route_context].drop(
                columns="route_context"
            )
            context = context.rename(
                columns={
                    column: f"{column}_{route_context}_run"
                    for column in metric_columns
                    if column != "route_context"
                }
            )
            context_frames.append(context)
        for context in context_frames:
            pair = pair.merge(
                context,
                on=CANDIDATE_KEYS,
                how="left",
                validate="one_to_one",
            )
        pair["volatility_label"] = volatility_label
        pair["slope_label"] = slope_label
        return pair

    def _apply_selection_gates(self, pair: pd.DataFrame) -> pd.DataFrame:
        pair = pair.copy()
        lcb_columns = [
            "lcb_return_per_step_volatility",
            "lcb_return_per_step_slope",
            "lcb_return_per_step_volatility_run",
            "lcb_return_per_step_slope_run",
        ]
        pair["pair_score"] = pair[lcb_columns].min(axis=1, skipna=False)
        pair["mean_lcb"] = pair[lcb_columns].mean(axis=1, skipna=False)

        gates = {
            "joint_support": pair["joint_contract_count"]
            >= self.config.min_joint_contracts,
            "marginal_contracts": (
                pair["contract_count_volatility"]
                >= self.config.min_marginal_contracts
            )
            & (pair["contract_count_slope"] >= self.config.min_marginal_contracts),
            "mean_return": (
                pair["mean_return_per_step_volatility"]
                > self.config.min_mean_return
            )
            & (
                pair["mean_return_per_step_slope"] > self.config.min_mean_return
            )
            & (
                pair["mean_return_per_step_volatility_run"]
                > self.config.min_mean_return
            )
            & (
                pair["mean_return_per_step_slope_run"]
                > self.config.min_mean_return
            ),
            "lcb": (pair[lcb_columns] > self.config.min_lcb).all(axis=1),
            "worst_initial_position": (
                pair["worst_initial_position_return_volatility"]
                >= self.config.min_worst_initial_position_return
            )
            & (
                pair["worst_initial_position_return_slope"]
                >= self.config.min_worst_initial_position_return
            )
            & (
                pair["worst_initial_position_return_volatility_run"]
                >= self.config.min_worst_initial_position_return
            )
            & (
                pair["worst_initial_position_return_slope_run"]
                >= self.config.min_worst_initial_position_return
            ),
            "positive_contract_ratio": (
                pair["positive_contract_ratio_volatility"]
                >= self.config.min_positive_contract_ratio
            )
            & (
                pair["positive_contract_ratio_slope"]
                >= self.config.min_positive_contract_ratio
            )
            & (
                pair["positive_contract_ratio_volatility_run"]
                >= self.config.min_positive_contract_ratio
            )
            & (
                pair["positive_contract_ratio_slope_run"]
                >= self.config.min_positive_contract_ratio
            ),
        }
        for name, values in gates.items():
            pair[f"gate_{name}"] = values.fillna(False)
        gate_columns = [f"gate_{name}" for name in gates]
        pair["eligible"] = pair[gate_columns].all(axis=1)
        pair["rejection_reasons"] = pair.apply(
            lambda row: ",".join(
                name for name in gates if not bool(row[f"gate_{name}"])
            ),
            axis=1,
        )
        return pair

    def _model_slot(
        self,
        slot_id: int,
        volatility_label: str,
        slope_label: str,
        selected: pd.Series,
        support: pd.Series,
        reason: str = "passed all marginal and joint-context gates",
    ) -> dict[str, Any]:
        return {
            "slot_id": slot_id,
            "volatility_label": volatility_label,
            "slope_label": slope_label,
            "kind": "model",
            "candidate_id": selected["candidate_id"],
            "epoch_number": int(selected["epoch_number"]),
            "epoch_path": selected["epoch_path"],
            "model_path": selected["model_path"],
            "bin_index": int(selected["bin_index"]),
            "pair_score": float(selected["pair_score"]),
            "joint_row_count": int(support["joint_row_count"]),
            "joint_contract_count": int(support["joint_contract_count"]),
            "selection_reason": reason,
            **_selected_metric_columns(selected),
        }

    def _null_slot(
        self,
        slot_id: int,
        volatility_label: str,
        slope_label: str,
        pair: pd.DataFrame,
        support: pd.Series,
    ) -> dict[str, Any]:
        if int(support["joint_row_count"]) == 0:
            reason = "no_joint_validation_rows"
        elif int(support["joint_contract_count"]) < self.config.min_joint_contracts:
            reason = "insufficient_joint_contract_coverage"
        else:
            reason = "no_candidate_passed_all_profitability_and_stability_gates"
        best = pair.iloc[0] if not pair.empty else None
        return {
            "slot_id": slot_id,
            "volatility_label": volatility_label,
            "slope_label": slope_label,
            "kind": "empty_model",
            "candidate_id": None,
            "epoch_number": None,
            "epoch_path": None,
            "model_path": None,
            "bin_index": None,
            "pair_score": 0.0,
            "joint_row_count": int(support["joint_row_count"]),
            "joint_contract_count": int(support["joint_contract_count"]),
            "selection_reason": reason,
            "best_rejected_candidate_id": (
                None if best is None else best["candidate_id"]
            ),
            "best_rejected_pair_score": (
                None
                if best is None or pd.isna(best["pair_score"])
                else float(best["pair_score"])
            ),
            "best_rejected_reasons": (
                None if best is None else best["rejection_reasons"]
            ),
        }

    def _build_manifest(
        self,
        candidate_root: Path,
        valid_root: Path,
        result_files: dict[int, dict[str, Path]],
        coverage: dict[str, Any],
        slots: pd.DataFrame,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "selection_method": "two_dimensional_marginal_and_dual_context_lcb",
            "candidate_root": str(candidate_root),
            "valid_root": str(valid_root),
            "axes": {
                "volatility": self.labels,
                "slope": self.labels,
            },
            "slot_count": self.config.num_labels**2,
            "slot_index_formula": "volatility_index * num_labels + slope_index",
            "null_policy": {
                "logical_kind": "empty_model",
                "intended_runtime_behavior": "flat_position",
                "model_assembly_status": "not_built_by_this_script",
            },
            "candidate_scope": {
                "common_epochs": sorted(result_files),
                **coverage,
            },
            "selection_config": asdict(self.config),
            "metric_definition": {
                "return": (
                    "sum(reward) / transition_count within contract and "
                    "initial position; transition_count = df_length - 1"
                ),
                "aggregation": (
                    "initial-position mean within each contract, then "
                    "contract-equal mean"
                ),
                "lcb": "mean_return - lcb_z * contract_standard_error",
                "pair_score": (
                    "minimum LCB across volatility marginal, slope marginal, "
                    "volatility-run joint subset, and slope-run joint subset"
                ),
                "joint_context_note": (
                    "Joint rewards are filtered by timestamp from both existing "
                    "trading trajectories. They are kept separate because slice "
                    "reset points can differ."
                ),
            },
            "artifacts": {
                "model_assembly": "not_performed",
                "high_level_model_change": "not_performed",
            },
            "slots": slots.to_dict("records"),
        }

    def _validate_label(self, label: str, source: Path) -> None:
        match = LABEL_PATTERN.fullmatch(label)
        if not match or int(match.group(1)) >= self.config.num_labels:
            raise ValueError(
                f"invalid or unexpected label {label!r} in {source}; "
                f"expected label_0..label_{self.config.num_labels - 1}"
            )


def _candidate_id(epoch_number: int, bin_index: int) -> str:
    return f"epoch_{epoch_number}:bin_{bin_index}"


def _candidate_fields(
    epoch_number: int, epoch_path: Path, model_path: Path, bin_index: int
) -> dict[str, Any]:
    return {
        "candidate_id": _candidate_id(epoch_number, bin_index),
        "epoch_number": epoch_number,
        "epoch_path": str(epoch_path),
        "model_path": str(model_path),
        "bin_index": bin_index,
    }


def _parse_json_array(value: Any, source: Path, field: str) -> list[Any]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON array {field!r} in {source}") from error
    if not isinstance(parsed, list):
        raise ValueError(f"field {field!r} in {source} is not a JSON array")
    return parsed


def _selected_metric_columns(row: pd.Series) -> dict[str, Any]:
    names = [
        "mean_return_per_step_volatility",
        "lcb_return_per_step_volatility",
        "mean_return_per_step_slope",
        "lcb_return_per_step_slope",
        "mean_return_per_step_volatility_run",
        "lcb_return_per_step_volatility_run",
        "mean_return_per_step_slope_run",
        "lcb_return_per_step_slope_run",
        "worst_initial_position_return_volatility",
        "worst_initial_position_return_slope",
        "worst_initial_position_return_volatility_run",
        "worst_initial_position_return_slope_run",
    ]
    return {
        name: None if pd.isna(row.get(name)) else float(row[name]) for name in names
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def assemble_and_save_ensemble(
    slots: pd.DataFrame,
    output_path: Path,
    *,
    n_states: int,
    n_actions: int,
    hidden_nodes: int,
    time_info_dim: int,
    trading_info_dim: int = 4,
) -> Path:
    """Assemble selected agents and flat placeholders in logical slot order."""

    ordered_slots = slots.sort_values("slot_id").reset_index(drop=True)
    expected_slot_ids = list(range(len(ordered_slots)))
    actual_slot_ids = ordered_slots["slot_id"].astype(int).tolist()
    if actual_slot_ids != expected_slot_ids:
        raise ValueError(
            "slot_id values must be contiguous and start at zero; "
            f"got {actual_slot_ids}"
        )
    if n_actions <= 0 or n_actions % 2 == 0:
        raise ValueError(
            f"n_actions must be a positive odd number, got {n_actions}"
        )

    ensemble = ensemble_Qnet(
        n_states,
        n_actions,
        hidden_nodes,
        time_info_dim,
        ensemble_number=len(ordered_slots),
        TRADING_INFO_DIM=trading_info_dim,
    )
    zero_position_action = n_actions // 2
    for row in ordered_slots.to_dict("records"):
        slot_id = int(row["slot_id"])
        kind = row["kind"]
        qnet = ensemble.qnet_list[slot_id]
        if kind == "model":
            model_path = Path(str(row["model_path"]))
            bin_index = int(row["bin_index"])
            state_dict = torch.load(
                model_path,
                map_location="cpu",
                weights_only=True,
            )
            prefix = f"qnet_list.{bin_index}."
            qnet_state_dict = {
                key.removeprefix(prefix): value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
            if not qnet_state_dict:
                raise ValueError(
                    f"{model_path} has no parameters for qnet_list.{bin_index}"
                )
            qnet.load_state_dict(qnet_state_dict)
        elif kind == "empty_model":
            with torch.no_grad():
                for parameter in qnet.parameters():
                    parameter.zero_()
                qnet.out.bias[zero_position_action] = 1.0
        else:
            raise ValueError(f"unknown slot kind {kind!r} for slot {slot_id}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ensemble.state_dict(), output_path)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select a 4x4 slope/volatility low-level agent grid."
    )
    parser.add_argument("--dataset_name", default="fu")
    parser.add_argument("--experiment_name", default="30min_multi")
    parser.add_argument("--result_root", default="result/DiHFT/low_level")
    parser.add_argument("--base_path", default="dataset/30min")
    parser.add_argument(
        "--save_path", default="analysis_result/DiHFT/low_level"
    )
    parser.add_argument("--num_labels", type=int, default=4)
    parser.add_argument("--min_epoch", type=int)
    parser.add_argument("--max_epoch", type=int)
    parser.add_argument("--lcb_z", type=float, default=1.0)
    parser.add_argument("--min_marginal_contracts", type=int, default=2)
    parser.add_argument("--min_joint_contracts", type=int, default=2)
    parser.add_argument("--min_positive_contract_ratio", type=float, default=0.60)
    parser.add_argument("--csv_chunk_size", type=int, default=250_000)
    parser.add_argument("--position_choices", type=int, default=5)
    parser.add_argument("--hidden_nodes", type=int, default=128)
    parser.add_argument("--time_info_dim", type=int, default=2)
    parser.add_argument("--trading_info_dim", type=int, default=4)
    parser.add_argument(
        "--missing_joint_policy",
        choices=["empty_model", "slope_marginal_best"],
        default="empty_model",
        help="Fallback policy when a slot has zero joint validation rows.",
    )
    parser.add_argument(
        "--output_dir",
        help="Override the default two_dimensional_selection output directory.",
    )
    parser.add_argument(
        "--model_output_path",
        help="Override the default <output_dir>/model.pth output path.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    candidate_root = (
        Path(args.result_root)
        / args.dataset_name
        / args.experiment_name
        / "weights_advantage_pretrain"
    )
    valid_root = Path(args.base_path) / args.dataset_name / "valid"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(args.save_path)
        / args.dataset_name
        / args.experiment_name
        / "two_dimensional_selection"
    )
    config = SelectionConfig(
        num_labels=args.num_labels,
        lcb_z=args.lcb_z,
        min_marginal_contracts=args.min_marginal_contracts,
        min_joint_contracts=args.min_joint_contracts,
        min_positive_contract_ratio=args.min_positive_contract_ratio,
        csv_chunk_size=args.csv_chunk_size,
        missing_joint_policy=args.missing_joint_policy,
    )
    artifacts = TwoDimensionalAgentSelector(config).select(
        candidate_root,
        valid_root,
        min_epoch=args.min_epoch,
        max_epoch=args.max_epoch,
    )
    model_output_path = (
        Path(args.model_output_path)
        if args.model_output_path
        else output_dir / "model.pth"
    )
    state_features_path = valid_root.parent / "state_features.npy"
    n_states = len(np.load(state_features_path))
    assemble_and_save_ensemble(
        artifacts.selected_slots,
        model_output_path,
        n_states=n_states,
        n_actions=args.position_choices,
        hidden_nodes=args.hidden_nodes,
        time_info_dim=args.time_info_dim,
        trading_info_dim=args.trading_info_dim,
    )
    artifacts.manifest["null_policy"]["model_assembly_status"] = (
        "built_as_flat_qnet"
    )
    artifacts.manifest["artifacts"]["model_assembly"] = str(model_output_path)
    paths = artifacts.write(output_dir)
    paths["model"] = model_output_path
    model_slots = int((artifacts.selected_slots["kind"] == "model").sum())
    null_slots = int((artifacts.selected_slots["kind"] == "empty_model").sum())
    complete_count = artifacts.manifest["candidate_scope"][
        "complete_candidate_count"
    ]
    print(f"common candidates: {complete_count}")
    print(f"selected model slots: {model_slots}")
    print(f"selected null slots: {null_slots}")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
    main()
