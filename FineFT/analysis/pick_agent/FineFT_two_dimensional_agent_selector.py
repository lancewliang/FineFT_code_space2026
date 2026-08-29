"""Select a 4x4 low-level agent grid from slope and volatility results.

This script is intentionally independent from
``FineFT_single_agent_with_different_position.py``.  It reads the CSV artifacts
already produced by ``test_agent_index.py``, writes selection reports, and
assembles ``model.pth`` without changing the high-level agent.

For every candidate ``(epoch_path, bin_index)`` and label pair:
* if the two dimensions have intersecting/joint data, the candidate is selected
  based on the profitability and stability of the joint data (evaluated under
  both volatility-run and slope-run reset contexts);
* if there is no intersecting/joint data, the candidate is selected based on the
  slope dimension's profitability and stability (slope marginal metrics).

If no candidate passes all gates for a slot, the logical slot is recorded as a
null/flat slot.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
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
    lcb_z: float = 0.0
    min_marginal_contracts: int = 1
    min_joint_contracts: int = 1
    min_positive_contract_ratio: float = 0.40
    min_mean_return: float = 0.0
    min_lcb: float = 0.0
    min_worst_initial_position_return: float = -1.0
    min_worst_initial_position_return_v2: float | None = -3.0
    min_worst_initial_position_return_v3: float | None = -5.0
    csv_chunk_size: int = 250_000
    missing_joint_policy: str = "slope_marginal_best"
    contract_weighting: str = "step_weighted"
    min_slice_steps: int = 0


@dataclass
class SelectionArtifacts:
    """In-memory result returned through the module's selection interface."""

    marginal_metrics: pl.DataFrame
    joint_metrics: pl.DataFrame
    candidate_rankings: pl.DataFrame
    selected_slots: pl.DataFrame
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
        self.marginal_metrics.write_csv(paths["marginal_metrics"])
        self.joint_metrics.write_csv(paths["joint_metrics"])
        self.candidate_rankings.write_csv(paths["candidate_rankings"])
        self.selected_slots.write_csv(paths["selected_slots"])
        with paths["manifest"].open("w", encoding="utf-8") as file:
            json.dump(_json_safe(self.manifest), file, ensure_ascii=False, indent=2)
        return paths


class TwoDimensionalAgentSelector:
    """Build a robust logical agent grid behind one selection interface."""

    def __init__(self, config: SelectionConfig) -> None:
        self.config = config
        if self.config.contract_weighting not in ("step_weighted", "contract_equal"):
            raise ValueError(
                f"unsupported contract_weighting {self.config.contract_weighting!r}; "
                "expected 'step_weighted' or 'contract_equal'"
            )
        if self.config.min_slice_steps < 0:
            raise ValueError(
                f"min_slice_steps must be non-negative, got {self.config.min_slice_steps}"
            )
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

        t0 = time.time()
        print("Starting 2D agent selection pipeline...", flush=True)

        result_files = self._discover_common_result_files(
            candidate_root, min_epoch=min_epoch, max_epoch=max_epoch
        )
        marginal_rows, zero_transition_slices = self._load_marginal_rows(result_files)
        marginal_rows, coverage = self._keep_complete_candidates(marginal_rows)

        print(
            "[3/6] Calculating marginal metrics for complete candidates...",
            flush=True,
        )
        marginal_metrics = self._calculate_metrics(
            marginal_rows,
            group_keys=CANDIDATE_KEYS + ["label_type", "label"],
        )
        print(
            f"      Marginal metrics calculated: {marginal_metrics.height} candidate-label groups.",
            flush=True,
        )

        label_lookup, joint_support = self._load_joint_label_lookup(valid_root)
        joint_rows = self._load_joint_detail_rows(
            result_files,
            label_lookup,
            allowed_candidates=set(marginal_rows["candidate_id"].to_list()),
            zero_transition_slices=zero_transition_slices,
        )
        self._validate_joint_totals(marginal_rows, joint_rows)

        print("[5/6] Calculating joint metrics...", flush=True)
        joint_metrics = self._calculate_metrics(
            joint_rows,
            group_keys=CANDIDATE_KEYS
            + ["route_context", "volatility_label", "slope_label"],
        )
        print(
            f"      Joint metrics calculated: {joint_metrics.height} candidate-joint groups.",
            flush=True,
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
        print(
            f"2D agent selection completed in {time.time() - t0:.1f}s.",
            flush=True,
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

        print(
            f"[1/6] Discovering candidate epochs in {candidate_root} "
            f"(min_epoch={min_epoch}, max_epoch={max_epoch})...",
            flush=True,
        )
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
        sorted_epochs = sorted(by_epoch.keys())
        print(
            f"      Found {len(sorted_epochs)} common epochs "
            f"[{sorted_epochs[0]}..{sorted_epochs[-1]}] with complete results.",
            flush=True,
        )
        return dict(sorted(by_epoch.items()))

    def _load_marginal_rows(
        self, result_files: dict[int, dict[str, Path]]
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        print(
            f"[2/6] Loading marginal analysis rows from {len(result_files)} epochs...",
            flush=True,
        )
        t_start = time.time()
        rows: list[dict[str, Any]] = []
        zero_transition_rows: list[dict[str, Any]] = []
        for epoch_number, files in result_files.items():
            epoch_path = files["slope"].parent.parent
            model_path = epoch_path / "trained_model.pkl"
            for label_type in LABEL_TYPES:
                frame = pl.read_csv(files[label_type])
                missing = set(ANALYSIS_COLUMNS) - set(frame.columns)
                if missing:
                    raise ValueError(
                        f"{files[label_type]} missing columns: {sorted(missing)}"
                    )
                frame = frame.rename(ANALYSIS_COLUMNS)
                for record in frame.select(list(ANALYSIS_COLUMNS.values())).iter_rows(named=True):
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
                        if transition_count < max(1, self.config.min_slice_steps):
                            if transition_count == 0 and not math.isclose(reward_sum, 0.0, abs_tol=1e-12):
                                raise ValueError(
                                    "single-row slice has non-zero reward in "
                                    f"{files[label_type]}"
                                )
                            zero_transition_rows.append(
                                {
                                    "candidate_id": candidate["candidate_id"],
                                    "label_type": label_type,
                                    "label": label,
                                    "initial_action": int(record["initial_action"]),
                                    "df_path": str(arrays["df_path"][index]),
                                }
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
        print(
            f"      Loaded {len(rows)} marginal records ({time.time() - t_start:.1f}s).",
            flush=True,
        )
        marginal_schema = {
            "candidate_id": pl.String,
            "epoch_number": pl.Int64,
            "epoch_path": pl.String,
            "model_path": pl.String,
            "bin_index": pl.Int64,
            "label_type": pl.String,
            "label": pl.String,
            "initial_action": pl.Int64,
            "contract": pl.String,
            "df_path": pl.String,
            "reward_sum": pl.Float64,
            "step_count": pl.Int64,
            "turnover_sum": pl.Float64,
        }
        zero_schema = {
            "candidate_id": pl.String,
            "label_type": pl.String,
            "label": pl.String,
            "initial_action": pl.Int64,
            "df_path": pl.String,
        }
        zero_transition_slices = pl.DataFrame(
            zero_transition_rows,
            schema=zero_schema,
        ).unique()
        return pl.DataFrame(rows, schema=marginal_schema), zero_transition_slices

    def _keep_complete_candidates(
        self, rows: pl.DataFrame
    ) -> tuple[pl.DataFrame, dict[str, Any]]:
        initial_actions = sorted(
            int(value) for value in rows["initial_action"].unique().to_list()
        )
        expected_count = len(LABEL_TYPES) * len(self.labels) * len(initial_actions)
        cand_combos = (
            rows.select(["candidate_id", "label_type", "label", "initial_action"])
            .unique()
            .filter(
                pl.col("label_type").is_in(LABEL_TYPES)
                & pl.col("label").is_in(self.labels)
                & pl.col("initial_action").is_in(initial_actions)
            )
            .group_by("candidate_id")
            .agg(pl.len().alias("combo_count"))
            .filter(pl.col("combo_count") == expected_count)
        )
        complete_set = set(cand_combos["candidate_id"].to_list())

        seen: set[str] = set()
        ordered_complete: list[str] = []
        for candidate_id in rows["candidate_id"].to_list():
            if candidate_id in complete_set and candidate_id not in seen:
                seen.add(candidate_id)
                ordered_complete.append(candidate_id)

        if not ordered_complete:
            raise ValueError("no candidate has complete two-dimensional label coverage")
        filtered = rows.filter(pl.col("candidate_id").is_in(complete_set))
        discovered_count = rows["candidate_id"].n_unique()
        coverage = {
            "discovered_candidate_count": discovered_count,
            "complete_candidate_count": len(ordered_complete),
            "excluded_incomplete_candidate_count": discovered_count - len(ordered_complete),
            "initial_actions": initial_actions,
        }
        print(
            f"      Candidate coverage: {coverage['complete_candidate_count']} complete candidates kept "
            f"({coverage['discovered_candidate_count']} discovered, "
            f"{coverage['excluded_incomplete_candidate_count']} incomplete excluded).",
            flush=True,
        )
        return filtered, coverage

    def _load_joint_label_lookup(
        self, valid_root: Path
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        print(
            f"[4/6] Loading joint validation label lookup from {valid_root}...",
            flush=True,
        )
        t_start = time.time()
        frames: dict[str, pl.DataFrame] = {}
        for label_type in LABEL_TYPES:
            label_root = valid_root / label_type
            paths = sorted(label_root.glob("*/label_*/*.feather"))
            if not paths:
                raise FileNotFoundError(
                    f"no validation label slices found under {label_root}"
                )
            parts: list[pl.DataFrame] = []
            for path in paths:
                label = path.parent.name
                self._validate_label(label, path)
                contract = path.parent.parent.name
                data = pl.read_ipc(path, columns=["timestamp"], memory_map=False)
                if data.schema["timestamp"].is_temporal():
                    ts_expr = pl.col("timestamp").dt.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    ts_expr = pl.col("timestamp").cast(pl.String)
                parts.append(
                    data.select([
                        pl.lit(contract).alias("contract"),
                        ts_expr.alias("timestamp"),
                        pl.lit(label).alias(f"{label_type}_label"),
                    ])
                )
            frame = pl.concat(parts)
            if frame.select(["contract", "timestamp"]).is_duplicated().any():
                raise ValueError(
                    f"{label_type} validation slices contain duplicate timestamps"
                )
            frames[label_type] = frame

        lookup = frames["volatility"].join(
            frames["slope"],
            on=["contract", "timestamp"],
            how="full",
            coalesce=True,
        )
        unmatched_count = lookup.filter(
            pl.col("volatility_label").is_null() | pl.col("slope_label").is_null()
        ).height
        if unmatched_count > 0:
            raise ValueError(
                "slope and volatility validation timestamps do not cover the "
                "same rows; "
                f"unmatched={unmatched_count}"
            )

        support_actual = (
            lookup.group_by(["volatility_label", "slope_label"])
            .agg(
                pl.len().alias("joint_row_count"),
                pl.col("contract").n_unique().alias("joint_contract_count"),
            )
        )
        grid_data = [
            {"volatility_label": vl, "slope_label": sl}
            for vl in self.labels
            for sl in self.labels
        ]
        grid = pl.DataFrame(
            grid_data,
            schema={"volatility_label": pl.String, "slope_label": pl.String},
        )
        support = (
            grid.join(support_actual, on=["volatility_label", "slope_label"], how="left")
            .with_columns(
                pl.col("joint_row_count").fill_null(0).cast(pl.Int64),
                pl.col("joint_contract_count").fill_null(0).cast(pl.Int64),
            )
        )
        print(
            f"      Validation lookup built: {lookup.height} timestamps across "
            f"{lookup['contract'].n_unique()} contracts ({time.time() - t_start:.1f}s).",
            flush=True,
        )
        return lookup, support

    def _load_joint_detail_rows(
        self,
        result_files: dict[int, dict[str, Path]],
        label_lookup: pl.DataFrame,
        *,
        allowed_candidates: set[str],
        zero_transition_slices: pl.DataFrame,
    ) -> pl.DataFrame:
        total_epochs = len(result_files)
        print(
            f"      Loading joint detail CSVs across {total_epochs} epochs...",
            flush=True,
        )
        t_start = time.time()
        summaries: list[pl.DataFrame] = []
        lookup = label_lookup.select(
            ["contract", "timestamp", "volatility_label", "slope_label"]
        )

        for idx, (epoch_number, files) in enumerate(result_files.items(), 1):
            epoch_t0 = time.time()
            epoch_path = files["slope"].parent.parent
            model_path = epoch_path / "trained_model.pkl"
            print(
                f"        [{idx}/{total_epochs}] Processing epoch_{epoch_number}...",
                end="",
                flush=True,
            )
            for route_context in LABEL_TYPES:
                detail_path = files[f"{route_context}_detail"]
                excluded_slices = (
                    zero_transition_slices.filter(
                        pl.col("label_type") == route_context
                    ).select(["candidate_id", "label", "initial_action", "df_path"])
                )
                detail_df = pl.read_csv(
                    detail_path,
                    columns=list(DETAIL_COLUMNS.keys()),
                    schema_overrides={
                        "标签": pl.String,
                        "数据文件": pl.String,
                        "初始动作": pl.Int64,
                        "分箱索引": pl.Int64,
                        "时间戳": pl.String,
                        "单步奖励": pl.Float64,
                    },
                ).rename(DETAIL_COLUMNS)

                detail_df = detail_df.with_columns(
                    pl.col("df_path").str.split("/").list.get(0).alias("contract"),
                    pl.col("timestamp").cast(pl.String),
                    pl.col("bin_index").cast(pl.Int64),
                    pl.concat_str([
                        pl.lit(f"epoch_{epoch_number}:bin_"),
                        pl.col("bin_index").cast(pl.String),
                    ]).alias("candidate_id"),
                )
                detail_df = detail_df.filter(
                    pl.col("candidate_id").is_in(allowed_candidates)
                )
                if excluded_slices.height > 0:
                    detail_df = detail_df.join(
                        excluded_slices,
                        on=["candidate_id", "label", "initial_action", "df_path"],
                        how="anti",
                    )
                if detail_df.height == 0:
                    continue

                detail_df = detail_df.join(
                    lookup,
                    on=["contract", "timestamp"],
                    how="left",
                )
                if detail_df.select(
                    pl.col("volatility_label").is_null().any()
                    | pl.col("slope_label").is_null().any()
                ).item():
                    raise ValueError(
                        f"{detail_path} contains timestamps absent from "
                        "validation labels"
                    )

                own_label = f"{route_context}_label"
                if (detail_df["label"] != detail_df[own_label]).any():
                    raise ValueError(
                        f"{detail_path} label disagrees with timestamp label lookup"
                    )

                grouped = (
                    detail_df.group_by([
                        "candidate_id",
                        "bin_index",
                        "initial_action",
                        "contract",
                        "volatility_label",
                        "slope_label",
                    ])
                    .agg(
                        pl.col("reward").cast(pl.Float64).sum().alias("reward_sum"),
                        pl.len().alias("step_count"),
                    )
                    .with_columns(
                        pl.lit(0.0).alias("turnover_sum"),
                        pl.lit(epoch_number).cast(pl.Int64).alias("epoch_number"),
                        pl.lit(str(epoch_path)).alias("epoch_path"),
                        pl.lit(str(model_path)).alias("model_path"),
                        pl.lit(route_context).alias("route_context"),
                    )
                )
                summaries.append(grouped)
            epoch_elapsed = time.time() - epoch_t0
            print(f" done ({epoch_elapsed:.1f}s)", flush=True)

        if not summaries:
            raise ValueError("trading detail files produced no joint evaluation rows")
        print("      Aggregating joint detail summaries...", flush=True)
        joint_rows = pl.concat(summaries)
        group_keys = CANDIDATE_KEYS + [
            "route_context",
            "volatility_label",
            "slope_label",
            "initial_action",
            "contract",
        ]
        aggregated = (
            joint_rows.group_by(group_keys)
            .agg(
                pl.col("reward_sum").sum().alias("reward_sum"),
                pl.col("step_count").sum().alias("step_count"),
                pl.col("turnover_sum").sum().alias("turnover_sum"),
            )
        )
        print(
            f"      Joint detail rows loaded and aggregated: {aggregated.height} rows "
            f"({time.time() - t_start:.1f}s total).",
            flush=True,
        )
        return aggregated

    def _validate_joint_totals(
        self, marginal_rows: pl.DataFrame, joint_rows: pl.DataFrame
    ) -> None:
        """Ensure timestamp partitioning preserves every source reward and step."""

        print(
            "      Validating joint totals against marginal summaries...",
            flush=True,
        )
        comparison_keys = CANDIDATE_KEYS + [
            "label_type",
            "label",
            "initial_action",
            "contract",
        ]
        expected = (
            marginal_rows.group_by(comparison_keys)
            .agg(
                pl.col("reward_sum").sum().alias("expected_reward_sum"),
                pl.col("step_count").sum().alias("expected_step_count"),
            )
        )
        actual = (
            joint_rows.with_columns(
                pl.col("route_context").alias("label_type"),
                pl.when(pl.col("route_context") == "volatility")
                .then(pl.col("volatility_label"))
                .otherwise(pl.col("slope_label"))
                .alias("label"),
            )
            .group_by(comparison_keys)
            .agg(
                pl.col("reward_sum").sum().alias("actual_reward_sum"),
                pl.col("step_count").sum().alias("actual_step_count"),
            )
        )
        comparison = expected.join(
            actual,
            on=comparison_keys,
            how="full",
            coalesce=True,
        )
        diff = (pl.col("expected_reward_sum") - pl.col("actual_reward_sum")).abs()
        tol = 1e-6 + 1e-9 * pl.col("actual_reward_sum").abs()
        reward_matches = diff <= tol
        step_matches = pl.col("expected_step_count") == pl.col("actual_step_count")
        both_present = (
            pl.col("expected_reward_sum").is_not_null()
            & pl.col("actual_reward_sum").is_not_null()
        )
        valid = both_present & reward_matches & step_matches
        invalid_df = comparison.filter(~valid)
        if invalid_df.height > 0:
            sample = invalid_df.head(5).to_dicts()
            raise ValueError(
                "joint timestamp partition does not preserve analysis_result.csv "
                f"totals; invalid_groups={invalid_df.height}, sample={sample}"
            )
        print("      Joint totals consistency check passed.", flush=True)

    def _calculate_metrics(
        self, rows: pl.DataFrame, *, group_keys: list[str]
    ) -> pl.DataFrame:
        base_keys = group_keys + ["contract", "initial_action"]
        contract_position = (
            rows.group_by(base_keys)
            .agg(
                pl.col("reward_sum").sum().alias("reward_sum"),
                pl.col("step_count").sum().alias("step_count"),
                pl.col("turnover_sum").sum().alias("turnover_sum"),
            )
            .with_columns(
                (pl.col("reward_sum") / pl.col("step_count")).alias("return_per_step")
            )
        )

        contract_summary = (
            contract_position.group_by(group_keys + ["contract"])
            .agg(
                pl.col("return_per_step").mean().alias("contract_return"),
                pl.col("step_count").sum().alias("step_count"),
            )
        )

        if self.config.contract_weighting == "step_weighted":
            mean_return_expr = (
                pl.when(pl.col("step_count").sum() > 0)
                .then(
                    (pl.col("contract_return") * pl.col("step_count")).sum()
                    / pl.col("step_count").sum()
                )
                .otherwise(0.0)
                .alias("mean_return_per_step")
            )
            position_return_expr = (
                pl.when(pl.col("step_count").sum() > 0)
                .then(
                    (pl.col("return_per_step") * pl.col("step_count")).sum()
                    / pl.col("step_count").sum()
                )
                .otherwise(0.0)
                .alias("position_return")
            )
        elif self.config.contract_weighting == "contract_equal":
            mean_return_expr = (
                pl.col("contract_return").mean().alias("mean_return_per_step")
            )
            position_return_expr = (
                pl.col("return_per_step").mean().alias("position_return")
            )
        else:
            raise ValueError(
                f"unsupported contract_weighting {self.config.contract_weighting!r}; "
                "expected 'step_weighted' or 'contract_equal'"
            )

        contract_metrics = (
            contract_summary.group_by(group_keys)
            .agg(
                pl.len().alias("contract_count"),
                mean_return_expr,
                pl.col("contract_return").std(ddof=1).alias("contract_std"),
                (pl.col("contract_return") > 0.0)
                .cast(pl.Float64)
                .mean()
                .alias("positive_contract_ratio"),
            )
            .with_columns(
                pl.when(pl.col("contract_count") > 1)
                .then(
                    pl.col("contract_std")
                    / (pl.col("contract_count").cast(pl.Float64).sqrt())
                )
                .otherwise(None)
                .alias("standard_error")
            )
            .with_columns(
                (
                    pl.col("mean_return_per_step")
                    if self.config.lcb_z == 0.0
                    else pl.when(
                        pl.col("standard_error").is_not_null()
                        & pl.col("standard_error").is_finite()
                    )
                    .then(
                        pl.col("mean_return_per_step")
                        - self.config.lcb_z * pl.col("standard_error")
                    )
                    .otherwise(None)
                ).alias("lcb_return_per_step")
            )
        )

        position_summary = (
            contract_position.group_by(group_keys + ["initial_action"])
            .agg(
                position_return_expr
            )
        )

        position_metrics = (
            position_summary.group_by(group_keys)
            .agg(
                pl.len().alias("initial_position_count"),
                pl.col("position_return").min().alias("worst_initial_position_return"),
            )
        )

        totals = (
            contract_position.group_by(group_keys)
            .agg(
                pl.col("step_count").sum().alias("step_count"),
                pl.col("turnover_sum").sum().alias("turnover_sum"),
            )
        )

        cols_order = group_keys + [
            "mean_return_per_step",
            "contract_std",
            "standard_error",
            "lcb_return_per_step",
            "worst_initial_position_return",
            "positive_contract_ratio",
            "contract_count",
            "initial_position_count",
            "step_count",
            "turnover_sum",
        ]

        return (
            contract_metrics.join(position_metrics, on=group_keys, how="inner")
            .join(totals, on=group_keys, how="inner")
            .select(cols_order)
        )

    def _worst_position_threshold_for_volatility(
        self, volatility_label: str | None
    ) -> float:
        if volatility_label is None:
            return self.config.min_worst_initial_position_return
        match = LABEL_PATTERN.fullmatch(volatility_label)
        if not match:
            return self.config.min_worst_initial_position_return
        v_idx = int(match.group(1))
        if v_idx >= 3 and self.config.min_worst_initial_position_return_v3 is not None:
            return self.config.min_worst_initial_position_return_v3
        if v_idx == 2 and self.config.min_worst_initial_position_return_v2 is not None:
            return self.config.min_worst_initial_position_return_v2
        return self.config.min_worst_initial_position_return

    def _select_grid(
        self,
        marginal_metrics: pl.DataFrame,
        joint_metrics: pl.DataFrame,
        joint_support: pl.DataFrame,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        total_slots = self.config.num_labels * self.config.num_labels
        print(
            f"[6/6] Selecting 2D agent grid ({self.config.num_labels}x{self.config.num_labels} = {total_slots} slots)...",
            flush=True,
        )
        rankings: list[pl.DataFrame] = []
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
                support_row = joint_support.filter(
                    (pl.col("volatility_label") == volatility_label)
                    & (pl.col("slope_label") == slope_label)
                ).row(0, named=True)
                joint_row_count = int(support_row["joint_row_count"])
                joint_contract_count = int(support_row["joint_contract_count"])
                pair = pair.with_columns(
                    pl.lit(slot_id).alias("slot_id"),
                    pl.lit(joint_row_count).alias("joint_row_count"),
                    pl.lit(joint_contract_count).alias(
                        "joint_contract_count"
                    ),
                )

                has_joint_data = (
                    joint_row_count > 0
                    and joint_contract_count >= self.config.min_joint_contracts
                )

                if has_joint_data:
                    pair = self._apply_selection_gates(
                        pair,
                        volatility_label=volatility_label,
                        slope_label=slope_label,
                        mode="joint",
                    )
                    pair = pair.sort(
                        by=["eligible", "pair_score", "mean_lcb", "epoch_number", "bin_index"],
                        descending=[True, True, True, False, False],
                        nulls_last=True,
                    ).with_columns(
                        (pl.int_range(1, pl.len() + 1)).alias("rank_in_slot")
                    )
                    rankings.append(pair)
                    eligible = pair.filter(pl.col("eligible"))
                    if eligible.height > 0:
                        chosen = eligible.row(0, named=True)
                        slot_dict = self._model_slot(
                            slot_id,
                            volatility_label,
                            slope_label,
                            chosen,
                            support_row,
                            reason="passed all joint-context gates",
                        )
                        slots.append(slot_dict)
                        print(
                            f"      Slot {slot_id:02d} [{volatility_label}, {slope_label}]: "
                            f"model -> {chosen['candidate_id']} (pair_score={chosen['pair_score']:.6f})",
                            flush=True,
                        )
                    else:
                        slot_dict = self._null_slot(
                            slot_id,
                            volatility_label,
                            slope_label,
                            pair,
                            support_row,
                        )
                        slots.append(slot_dict)
                        print(
                            f"      Slot {slot_id:02d} [{volatility_label}, {slope_label}]: "
                            f"empty_model (reason={slot_dict['selection_reason']})",
                            flush=True,
                        )
                else:
                    if self.config.missing_joint_policy == "slope_marginal_best":
                        pair = self._apply_selection_gates(
                            pair,
                            volatility_label=volatility_label,
                            slope_label=slope_label,
                            mode="slope_marginal",
                        )
                        pair = pair.sort(
                            by=["eligible", "pair_score", "mean_lcb", "epoch_number", "bin_index"],
                            descending=[True, True, True, False, False],
                            nulls_last=True,
                        ).with_columns(
                            (pl.int_range(1, pl.len() + 1)).alias("rank_in_slot")
                        )
                        rankings.append(pair)
                        eligible = pair.filter(pl.col("eligible"))
                        if eligible.height > 0:
                            chosen = eligible.row(0, named=True)
                            slot_dict = self._model_slot(
                                slot_id,
                                volatility_label,
                                slope_label,
                                chosen,
                                support_row,
                                reason="fallback_from_slope_marginal",
                            )
                            slots.append(slot_dict)
                            print(
                                f"      Slot {slot_id:02d} [{volatility_label}, {slope_label}]: "
                                f"fallback model -> {chosen['candidate_id']} (score={chosen['pair_score']:.6f})",
                                flush=True,
                            )
                        else:
                            slot_dict = self._null_slot(
                                slot_id,
                                volatility_label,
                                slope_label,
                                pair,
                                support_row,
                            )
                            slots.append(slot_dict)
                            print(
                                f"      Slot {slot_id:02d} [{volatility_label}, {slope_label}]: "
                                f"empty_model (reason={slot_dict['selection_reason']})",
                                flush=True,
                            )
                    else:
                        pair = self._apply_selection_gates(
                            pair,
                            volatility_label=volatility_label,
                            slope_label=slope_label,
                            mode="joint",
                        )
                        pair = pair.sort(
                            by=["eligible", "pair_score", "mean_lcb", "epoch_number", "bin_index"],
                            descending=[True, True, True, False, False],
                            nulls_last=True,
                        ).with_columns(
                            (pl.int_range(1, pl.len() + 1)).alias("rank_in_slot")
                        )
                        rankings.append(pair)
                        slot_dict = self._null_slot(
                            slot_id,
                            volatility_label,
                            slope_label,
                            pair,
                            support_row,
                        )
                        slots.append(slot_dict)
                        print(
                            f"      Slot {slot_id:02d} [{volatility_label}, {slope_label}]: "
                            f"empty_model (reason={slot_dict['selection_reason']})",
                            flush=True,
                        )

        return pl.concat(rankings), pl.DataFrame(slots)

    def _pair_candidates(
        self,
        marginal_metrics: pl.DataFrame,
        joint_metrics: pl.DataFrame,
        volatility_label: str,
        slope_label: str,
    ) -> pl.DataFrame:
        volatility = marginal_metrics.filter(
            (pl.col("label_type") == "volatility")
            & (pl.col("label") == volatility_label)
        ).drop(["label_type", "label"])
        volatility = volatility.rename({
            col: f"{col}_volatility"
            for col in volatility.columns
            if col not in CANDIDATE_KEYS
        })

        slope = marginal_metrics.filter(
            (pl.col("label_type") == "slope")
            & (pl.col("label") == slope_label)
        ).drop(["label_type", "label"])
        slope = slope.rename({
            col: f"{col}_slope"
            for col in slope.columns
            if col not in CANDIDATE_KEYS
        })

        pair = volatility.join(slope, on=CANDIDATE_KEYS, how="inner")

        joint = joint_metrics.filter(
            (pl.col("volatility_label") == volatility_label)
            & (pl.col("slope_label") == slope_label)
        ).drop(["volatility_label", "slope_label"])

        for route_context in LABEL_TYPES:
            context = joint.filter(
                pl.col("route_context") == route_context
            ).drop("route_context")
            context = context.rename({
                col: f"{col}_{route_context}_run"
                for col in context.columns
                if col not in CANDIDATE_KEYS
            })
            pair = pair.join(context, on=CANDIDATE_KEYS, how="left")

        pair = pair.with_columns(
            pl.lit(volatility_label).alias("volatility_label"),
            pl.lit(slope_label).alias("slope_label"),
        )
        return pair

    def _apply_selection_gates(
        self,
        pair: pl.DataFrame,
        *,
        volatility_label: str | None = None,
        slope_label: str | None = None,
        mode: str = "joint",
    ) -> pl.DataFrame:
        if mode == "joint":
            lcb_columns = [
                "lcb_return_per_step_volatility_run",
                "lcb_return_per_step_slope_run",
            ]
            any_lcb_missing = pl.any_horizontal(
                pl.col(lcb_columns).is_null() | pl.col(lcb_columns).is_nan()
            )
            pair_score_expr = (
                pl.when(any_lcb_missing)
                .then(None)
                .otherwise(pl.min_horizontal(lcb_columns))
                .alias("pair_score")
            )
            mean_lcb_expr = (
                pl.when(any_lcb_missing)
                .then(None)
                .otherwise(pl.mean_horizontal(lcb_columns))
                .alias("mean_lcb")
            )

            vol_thresh = self._worst_position_threshold_for_volatility(volatility_label)

            gate_exprs = {
                "joint_support": (
                    pl.col("joint_contract_count").fill_null(0)
                    >= self.config.min_joint_contracts
                ),
                "marginal_contracts": pl.lit(True),
                "mean_return": (
                    (
                        pl.col("mean_return_per_step_volatility_run")
                        > self.config.min_mean_return
                    )
                    & (
                        pl.col("mean_return_per_step_slope_run")
                        > self.config.min_mean_return
                    )
                ).fill_null(False),
                "lcb": (
                    pl.all_horizontal([pl.col(c) > self.config.min_lcb for c in lcb_columns])
                ).fill_null(False),
                "worst_initial_position": (
                    (
                        pl.col("worst_initial_position_return_volatility_run")
                        >= vol_thresh
                    )
                    & (
                        pl.col("worst_initial_position_return_slope_run")
                        >= vol_thresh
                    )
                ).fill_null(False),
                "positive_contract_ratio": (
                    (
                        pl.col("positive_contract_ratio_volatility_run")
                        >= self.config.min_positive_contract_ratio
                    )
                    & (
                        pl.col("positive_contract_ratio_slope_run")
                        >= self.config.min_positive_contract_ratio
                    )
                ).fill_null(False),
            }
        elif mode == "slope_marginal":
            lcb_columns = [
                "lcb_return_per_step_slope",
            ]
            any_lcb_missing = pl.any_horizontal(
                pl.col(lcb_columns).is_null() | pl.col(lcb_columns).is_nan()
            )
            pair_score_expr = (
                pl.when(any_lcb_missing)
                .then(None)
                .otherwise(pl.col("lcb_return_per_step_slope"))
                .alias("pair_score")
            )
            mean_lcb_expr = (
                pl.when(any_lcb_missing)
                .then(None)
                .otherwise(pl.col("lcb_return_per_step_slope"))
                .alias("mean_lcb")
            )

            base_thresh = self.config.min_worst_initial_position_return

            gate_exprs = {
                "joint_support": pl.lit(True),
                "marginal_contracts": (
                    pl.col("contract_count_slope").fill_null(0)
                    >= self.config.min_marginal_contracts
                ),
                "mean_return": (
                    pl.col("mean_return_per_step_slope")
                    > self.config.min_mean_return
                ).fill_null(False),
                "lcb": (
                    pl.col("lcb_return_per_step_slope")
                    > self.config.min_lcb
                ).fill_null(False),
                "worst_initial_position": (
                    pl.col("worst_initial_position_return_slope")
                    >= base_thresh
                ).fill_null(False),
                "positive_contract_ratio": (
                    pl.col("positive_contract_ratio_slope")
                    >= self.config.min_positive_contract_ratio
                ).fill_null(False),
            }
        else:
            raise ValueError(f"unsupported selection gate mode: {mode!r}")

        gate_cols = [f"gate_{name}" for name in gate_exprs]
        failed_exprs = [
            pl.when(~pl.col(f"gate_{name}")).then(pl.lit(name)).otherwise(None)
            for name in gate_exprs
        ]

        pair = pair.with_columns(
            pair_score_expr,
            mean_lcb_expr,
            *[expr.alias(f"gate_{name}") for name, expr in gate_exprs.items()],
        )
        return pair.with_columns(
            pl.all_horizontal(gate_cols).alias("eligible"),
            pl.concat_list(failed_exprs)
            .list.drop_nulls()
            .list.join(",")
            .alias("rejection_reasons"),
        )

    def _model_slot(
        self,
        slot_id: int,
        volatility_label: str,
        slope_label: str,
        selected: dict[str, Any],
        support: dict[str, Any],
        reason: str = "passed all marginal and joint-context gates",
    ) -> dict[str, Any]:
        return {
            "slot_id": slot_id,
            "volatility_label": volatility_label,
            "slope_label": slope_label,
            "kind": "model",
            "candidate_id": selected["candidate_id"],
            "epoch_number": int(selected["epoch_number"]),
            "epoch_path": str(selected["epoch_path"]),
            "model_path": str(selected["model_path"]),
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
        pair: pl.DataFrame,
        support: dict[str, Any],
    ) -> dict[str, Any]:
        if int(support["joint_row_count"]) == 0:
            reason = "no_joint_validation_rows"
        elif int(support["joint_contract_count"]) < self.config.min_joint_contracts:
            reason = "insufficient_joint_contract_coverage"
        else:
            reason = "no_candidate_passed_all_profitability_and_stability_gates"
        best = pair.row(0, named=True) if pair.height > 0 else None
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
                if best is None
                or best.get("pair_score") is None
                or not math.isfinite(best["pair_score"])
                else float(best["pair_score"])
            ),
            "best_rejected_reasons": (
                None if best is None else best.get("rejection_reasons")
            ),
        }

    def _build_manifest(
        self,
        candidate_root: Path,
        valid_root: Path,
        result_files: dict[int, dict[str, Path]],
        coverage: dict[str, Any],
        slots: pl.DataFrame,
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
                    f"{self.config.contract_weighting} mean across contracts"
                ),
                "lcb": "mean_return - lcb_z * contract_standard_error",
                "pair_score": (
                    "minimum LCB across volatility-run and slope-run joint subsets "
                    "when joint data exists; slope marginal LCB when no joint data exists"
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
            "slots": slots.to_dicts(),
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


def _selected_metric_columns(row: dict[str, Any]) -> dict[str, Any]:
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
        name: (
            None
            if row.get(name) is None
            or not math.isfinite(float(row[name]))
            else float(row[name])
        )
        for name in names
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
    slots: pl.DataFrame,
    output_path: Path,
    *,
    n_states: int,
    n_actions: int,
    hidden_nodes: int,
    time_info_dim: int,
    trading_info_dim: int = 4,
) -> Path:
    """Assemble selected agents and flat placeholders in logical slot order."""

    ordered_slots = slots.sort("slot_id")
    expected_slot_ids = list(range(ordered_slots.height))
    actual_slot_ids = ordered_slots["slot_id"].cast(pl.Int64).to_list()
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
        ensemble_number=ordered_slots.height,
        TRADING_INFO_DIM=trading_info_dim,
    )
    zero_position_action = n_actions // 2
    for row in ordered_slots.to_dicts():
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
    parser.add_argument(
        "--lcb_z",
        type=float,
        default=0.0,
        help="LCB standard error multiplier (default: 0.0 for pure mean return; >0 for risk penalty).",
    )
    parser.add_argument(
        "--min_marginal_contracts",
        type=int,
        default=1,
        help="Minimum contracts required for marginal slices (default: 1).",
    )
    parser.add_argument(
        "--min_joint_contracts",
        type=int,
        default=1,
        help="Minimum contracts required for joint slices (default: 1).",
    )
    parser.add_argument(
        "--min_positive_contract_ratio",
        type=float,
        default=0.60,
        help="Minimum ratio of positive contracts (default: 0.40).",
    )
    parser.add_argument(
        "--min_mean_return",
        type=float,
        default=0.0,
        help="Minimum mean return per step across slices (default: 0.0).",
    )
    parser.add_argument(
        "--min_lcb",
        type=float,
        default=0.0,
        help="Minimum LCB return per step (default: 0.0).",
    )
    parser.add_argument(
        "--min_worst_initial_position_return",
        type=float,
        default=-1.0,
        help="Minimum return across worst initial position (default: -1.0 for low vol).",
    )
    parser.add_argument(
        "--min_worst_initial_position_return_v2",
        type=float,
        default=-3.0,
        help="Minimum return across worst initial position for volatility label_2 (default: -3.0).",
    )
    parser.add_argument(
        "--min_worst_initial_position_return_v3",
        type=float,
        default=-5.0,
        help="Minimum return across worst initial position for volatility label_3 (default: -5.0).",
    )
    parser.add_argument("--csv_chunk_size", type=int, default=250_000)
    parser.add_argument("--position_choices", type=int, default=5)
    parser.add_argument("--hidden_nodes", type=int, default=128)
    parser.add_argument("--time_info_dim", type=int, default=2)
    parser.add_argument("--trading_info_dim", type=int, default=4)
    parser.add_argument(
        "--missing_joint_policy",
        choices=["empty_model", "slope_marginal_best"],
        default="slope_marginal_best",
        help="Fallback policy when a slot has zero or insufficient joint validation rows (default: slope_marginal_best).",
    )
    parser.add_argument(
        "--contract_weighting",
        choices=["step_weighted", "contract_equal"],
        default="step_weighted",
        help="Contract weighting method for mean_return_per_step aggregation (default: step_weighted).",
    )
    parser.add_argument(
        "--min_slice_steps",
        type=int,
        default=30,
        help="Minimum step count for an individual slice to be included (default: 30).",
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
    start_time = time.time()
    print("=" * 60, flush=True)
    print("FineFT 2D Agent Selector", flush=True)
    print(f"Dataset: {args.dataset_name}, Experiment: {args.experiment_name}", flush=True)
    print(f"Num labels: {args.num_labels} ({args.num_labels}x{args.num_labels} = {args.num_labels**2} slots)", flush=True)
    print(f"Position choices: {args.position_choices}, LCB z: {args.lcb_z}", flush=True)
    print(f"Contract weighting: {args.contract_weighting}, Min slice steps: {args.min_slice_steps}", flush=True)
    print(f"Missing joint policy: {args.missing_joint_policy}", flush=True)
    print("=" * 60, flush=True)

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
        min_mean_return=args.min_mean_return,
        min_lcb=args.min_lcb,
        min_worst_initial_position_return=args.min_worst_initial_position_return,
        min_worst_initial_position_return_v2=args.min_worst_initial_position_return_v2,
        min_worst_initial_position_return_v3=args.min_worst_initial_position_return_v3,
        csv_chunk_size=args.csv_chunk_size,
        missing_joint_policy=args.missing_joint_policy,
        contract_weighting=args.contract_weighting,
        min_slice_steps=args.min_slice_steps,
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
    print(
        f"Assembling ensemble model (n_states={n_states}, n_actions={args.position_choices}) -> {model_output_path}...",
        flush=True,
    )
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
    print(f"Writing selection artifacts -> {output_dir}...", flush=True)
    paths = artifacts.write(output_dir)
    paths["model"] = model_output_path
    model_slots = int((artifacts.selected_slots["kind"] == "model").sum())
    null_slots = int((artifacts.selected_slots["kind"] == "empty_model").sum())
    complete_count = artifacts.manifest["candidate_scope"][
        "complete_candidate_count"
    ]
    elapsed = time.time() - start_time
    print("=" * 60, flush=True)
    print(f"Selection completed in {elapsed:.1f}s ({elapsed/60:.2f} min)", flush=True)
    print(f"Common candidates: {complete_count}", flush=True)
    print(f"Selected model slots: {model_slots} / {artifacts.selected_slots.height}", flush=True)
    print(f"Selected null slots: {null_slots} / {artifacts.selected_slots.height}", flush=True)
    for name, path in paths.items():
        print(f"  {name}: {path}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
    main()
