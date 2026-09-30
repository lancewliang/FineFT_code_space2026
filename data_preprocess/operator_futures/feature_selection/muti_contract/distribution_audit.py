from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import polars as pl
from scipy.stats import ks_2samp

from operator_futures.feature_selection.muti_contract.types import (
    DistributionAuditConfig,
    PipelineStepResult,
)

logger = logging.getLogger(__name__)

PSI_EPSILON: float = 1e-6
DEFAULT_MAX_MEAN_PSI: float = 0.10
DEFAULT_MAX_PAIR_PSI: float = 0.25
RELAXED_MAX_MEAN_PSI: float = 0.15
DEFAULT_MIN_DRIFT_SURVIVORS: int = 20
DEFAULT_DISTRIBUTION_NUM_BINS: int = 10


@dataclass
class DistributionAuditResult:
    metrics_df: pl.DataFrame
    surviving_features: list[str]
    dropped_features: list[str]
    fallback_triggered: bool
    mean_psi_by_feature: dict[str, float]
    forward_psi_by_feature: dict[str, float] = field(default_factory=dict)

    def to_step_result(self) -> PipelineStepResult:
        return PipelineStepResult(
            step_name="distribution_audit",
            surviving_features=self.surviving_features,
            dropped_features=self.dropped_features,
            audit_metrics_df=self.metrics_df,
            diagnostics={
                "fallback_triggered": self.fallback_triggered,
            },
        )


def _compute_bin_probabilities(
    valid_arrays: list[np.ndarray],
    num_bins: int,
) -> tuple[list[np.ndarray], int]:
    pooled = np.concatenate(valid_arrays)
    if pooled.size == 0 or np.std(pooled) == 0.0:
        return [np.array([1.0]) for _ in valid_arrays], 1

    unique_vals = np.unique(pooled)
    if len(unique_vals) <= 5:
        contract_probs: list[np.ndarray] = []
        for arr in valid_arrays:
            probs = np.array(
                [float(np.sum(arr == val)) / float(len(arr)) for val in unique_vals]
            )
            contract_probs.append(probs)
        return contract_probs, len(unique_vals)

    # Zero-isolated adaptive binning
    zeros_count = np.sum(pooled == 0.0)
    is_zero_inflated = (float(zeros_count) / float(len(pooled))) >= 0.05
    if is_zero_inflated:
        nonzero_pooled = pooled[pooled != 0.0]
        unique_nonzero = np.unique(nonzero_pooled)
        if len(unique_nonzero) < 2:
            contract_probs = []
            for arr in valid_arrays:
                p0 = float(np.sum(arr == 0.0)) / float(len(arr))
                p1 = float(np.sum(arr != 0.0)) / float(len(arr))
                contract_probs.append(np.array([p0, p1]))
            return contract_probs, 2

        k_nonzero = max(num_bins - 1, 1)
        q = np.linspace(0.0, 1.0, k_nonzero + 1)
        raw_edges = np.quantile(nonzero_pooled, q)
        nonzero_edges = np.unique(raw_edges).astype(float)
        if len(nonzero_edges) < 2:
            contract_probs = []
            for arr in valid_arrays:
                p0 = float(np.sum(arr == 0.0)) / float(len(arr))
                p1 = float(np.sum(arr != 0.0)) / float(len(arr))
                contract_probs.append(np.array([p0, p1]))
            return contract_probs, 2

        nonzero_edges[0] = -np.inf
        nonzero_edges[-1] = np.inf
        contract_probs = []
        for arr in valid_arrays:
            p0 = float(np.sum(arr == 0.0)) / float(len(arr))
            arr_nz = arr[arr != 0.0]
            if len(arr_nz) > 0:
                counts, _ = np.histogram(arr_nz, bins=nonzero_edges)
                p_nz = (counts / float(len(arr))).tolist()
            else:
                p_nz = [0.0] * (len(nonzero_edges) - 1)
            contract_probs.append(np.array([p0, *p_nz]))
        total_bins = 1 + (len(nonzero_edges) - 1)
        return contract_probs, total_bins

    # Standard continuous quantile binning
    q = np.linspace(0.0, 1.0, num_bins + 1)
    raw_edges = np.quantile(pooled, q)
    edges = np.unique(raw_edges).astype(float)
    if len(edges) < 2:
        return [np.array([1.0]) for _ in valid_arrays], 1
    edges[0] = -np.inf
    edges[-1] = np.inf
    contract_probs = []
    for arr in valid_arrays:
        counts, _ = np.histogram(arr, bins=edges)
        contract_probs.append(counts / float(len(arr)))
    return contract_probs, len(edges) - 1


def _compute_pairwise_psi_and_ks(
    arrays: list[np.ndarray],
    num_bins: int,
) -> tuple[float, float, float, float]:
    valid_arrays = []
    for arr in arrays:
        valid = arr[np.isfinite(arr)]
        if len(valid) > 0:
            valid_arrays.append(valid)

    if len(valid_arrays) < 2:
        return 0.0, 0.0, 0.0, 1.0

    contract_probs, _ = _compute_bin_probabilities(valid_arrays, num_bins)

    psi_values: list[float] = []
    ks_d_values: list[float] = []
    ks_p_values: list[float] = []

    num_contracts = len(valid_arrays)
    for i in range(num_contracts):
        for j in range(i + 1, num_contracts):
            prob_i = contract_probs[i]
            prob_j = contract_probs[j]

            term = (prob_i - prob_j) * np.log(
                (prob_i + PSI_EPSILON) / (prob_j + PSI_EPSILON)
            )
            psi_ij = float(np.sum(term))
            psi_values.append(max(0.0, psi_ij))

            ks_res = ks_2samp(valid_arrays[i], valid_arrays[j])
            ks_d_values.append(float(ks_res.statistic))
            ks_p_values.append(float(ks_res.pvalue))

    if not psi_values:
        return 0.0, 0.0, 0.0, 1.0

    mean_psi = float(np.mean(psi_values))
    max_pair_psi = float(np.max(psi_values))
    max_ks_d = float(np.max(ks_d_values))
    min_ks_p = float(np.min(ks_p_values))
    return mean_psi, max_pair_psi, max_ks_d, min_ks_p


def audit_distribution_drift(
    frames: dict[str, pl.DataFrame],
    feature_universe: list[str],
    num_bins: int = DEFAULT_DISTRIBUTION_NUM_BINS,
    max_mean_psi: float = DEFAULT_MAX_MEAN_PSI,
    max_pair_psi: float = DEFAULT_MAX_PAIR_PSI,
    min_drift_survivors: int = DEFAULT_MIN_DRIFT_SURVIVORS,
    forward_outpost_frame: pl.DataFrame | None = None,
    forward_outpost_max_psi: float = 0.15,
    config: DistributionAuditConfig | None = None,
) -> DistributionAuditResult:
    if config is not None:
        num_bins = config.num_bins
        max_mean_psi = config.max_mean_psi
        max_pair_psi = config.max_pair_psi
        min_drift_survivors = config.min_drift_survivors
        forward_outpost_max_psi = config.forward_outpost_max_psi

    if not feature_universe:
        schema = {
            "feature": pl.Utf8,
            "mean_psi": pl.Float64,
            "max_pair_psi": pl.Float64,
            "max_ks_d": pl.Float64,
            "min_ks_p": pl.Float64,
            "drift_passed": pl.Boolean,
        }
        if forward_outpost_frame is not None:
            schema["forward_psi"] = pl.Float64
        empty_df = pl.DataFrame(schema=schema)
        return DistributionAuditResult(
            metrics_df=empty_df,
            surviving_features=[],
            dropped_features=[],
            fallback_triggered=False,
            mean_psi_by_feature={},
            forward_psi_by_feature={},
        )

    contract_columns: dict[str, dict[str, np.ndarray]] = {}
    for contract, frame in frames.items():
        contract_cols = {}
        for feat in feature_universe:
            if feat in frame.columns:
                contract_cols[feat] = frame[feat].to_numpy()
        contract_columns[contract] = contract_cols

    mean_psi_dict: dict[str, float] = {}
    max_pair_psi_dict: dict[str, float] = {}
    max_ks_d_dict: dict[str, float] = {}
    min_ks_p_dict: dict[str, float] = {}
    forward_psi_dict: dict[str, float] = {}

    for feat in feature_universe:
        arrays = [
            contract_columns[contract][feat]
            for contract in frames
            if feat in contract_columns[contract]
        ]
        mean_psi, max_pair_psi, max_ks_d, min_ks_p = _compute_pairwise_psi_and_ks(
            arrays, num_bins
        )
        mean_psi_dict[feat] = mean_psi
        max_pair_psi_dict[feat] = max_pair_psi
        max_ks_d_dict[feat] = max_ks_d
        min_ks_p_dict[feat] = min_ks_p

        if forward_outpost_frame is not None and feat in forward_outpost_frame.columns:
            val_col = forward_outpost_frame[feat].to_numpy()
            val_clean = val_col[np.isfinite(val_col)]
            train_clean = [
                arr[np.isfinite(arr)]
                for arr in arrays
                if len(arr[np.isfinite(arr)]) > 0
            ]
            if len(train_clean) > 0 and len(val_clean) > 0:
                train_pooled = np.concatenate(train_clean)
                pair_probs, _ = _compute_bin_probabilities(
                    [train_pooled, val_clean], num_bins
                )
                p_train = pair_probs[0]
                p_val = pair_probs[1]
                term = (p_train - p_val) * np.log(
                    (p_train + PSI_EPSILON) / (p_val + PSI_EPSILON)
                )
                forward_psi_dict[feat] = float(max(0.0, np.sum(term)))
            else:
                forward_psi_dict[feat] = 0.0

    passing = [
        feat
        for feat in feature_universe
        if mean_psi_dict[feat] <= max_mean_psi
        and max_pair_psi_dict[feat] <= max_pair_psi
        and (
            forward_outpost_frame is None
            or forward_psi_dict[feat] <= forward_outpost_max_psi
        )
    ]

    target_survivors = min(len(feature_universe), min_drift_survivors)
    fallback_triggered = False

    if len(passing) < target_survivors:
        fallback_triggered = True
        logger.warning(
            "Distribution drift gate: only %d / %d features passed initial threshold "
            "(mean_psi<=%.2f, max_pair_psi<=%.2f). Relaxing threshold to mean_psi<=%.2f.",
            len(passing),
            len(feature_universe),
            max_mean_psi,
            max_pair_psi,
            RELAXED_MAX_MEAN_PSI,
        )
        passing_relaxed = [
            feat
            for feat in feature_universe
            if mean_psi_dict[feat] <= RELAXED_MAX_MEAN_PSI
            and max_pair_psi_dict[feat] <= max_pair_psi
            and (
                forward_outpost_frame is None
                or forward_psi_dict[feat] <= forward_outpost_max_psi
            )
        ]
        if len(passing_relaxed) >= target_survivors:
            passing = passing_relaxed
        else:
            logger.warning(
                "Distribution drift gate: only %d features passed relaxed threshold. "
                "Retaining top %d features by lowest mean_psi.",
                len(passing_relaxed),
                target_survivors,
            )
            sorted_by_psi = sorted(feature_universe, key=lambda f: mean_psi_dict[f])
            passing = sorted_by_psi[:target_survivors]

    surviving_set = set(passing)
    records: list[dict[str, float | str | bool]] = []
    for feat in feature_universe:
        row: dict[str, float | str | bool] = {
            "feature": feat,
            "mean_psi": mean_psi_dict[feat],
            "max_pair_psi": max_pair_psi_dict[feat],
            "max_ks_d": max_ks_d_dict[feat],
            "min_ks_p": min_ks_p_dict[feat],
        }
        if forward_outpost_frame is not None:
            row["forward_psi"] = forward_psi_dict[feat]
        row["drift_passed"] = feat in surviving_set
        records.append(row)

    metrics_df = pl.DataFrame(records)
    dropped = [feat for feat in feature_universe if feat not in surviving_set]

    return DistributionAuditResult(
        metrics_df=metrics_df,
        surviving_features=passing,
        dropped_features=dropped,
        fallback_triggered=fallback_triggered,
        mean_psi_by_feature=mean_psi_dict,
        forward_psi_by_feature=forward_psi_dict,
    )
