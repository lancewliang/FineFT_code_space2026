from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import polars as pl
from scipy.stats import ks_2samp

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

    pooled = np.concatenate(valid_arrays)
    if len(pooled) == 0:
        return 0.0, 0.0, 0.0, 1.0

    q = np.linspace(0.0, 1.0, num_bins + 1)
    raw_edges = np.quantile(pooled, q)
    edges = np.unique(raw_edges)
    if len(edges) < 2:
        return 0.0, 0.0, 0.0, 1.0

    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf

    contract_probs: list[np.ndarray] = []
    for arr in valid_arrays:
        counts, _ = np.histogram(arr, bins=edges)
        contract_probs.append(counts / float(len(arr)))

    psi_values: list[float] = []
    ks_d_values: list[float] = []
    ks_p_values: list[float] = []

    num_contracts = len(valid_arrays)
    for i in range(num_contracts):
        for j in range(i + 1, num_contracts):
            prob_i = contract_probs[i]
            prob_j = contract_probs[j]

            # PSI(C_i, C_j) = sum_k (P_ik - Q_jk) * ln((P_ik + eps) / (Q_jk + eps))
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
) -> DistributionAuditResult:
    if not feature_universe:
        empty_df = pl.DataFrame(
            schema={
                "feature": pl.Utf8,
                "mean_psi": pl.Float64,
                "max_pair_psi": pl.Float64,
                "max_ks_d": pl.Float64,
                "min_ks_p": pl.Float64,
                "drift_passed": pl.Boolean,
            }
        )
        return DistributionAuditResult(
            metrics_df=empty_df,
            surviving_features=[],
            dropped_features=[],
            fallback_triggered=False,
            mean_psi_by_feature={},
        )

    # Pre-extract numpy columns per contract
    contract_columns: dict[str, dict[str, np.ndarray]] = {}
    for contract, frame in frames.items():
        contract_cols = {}
        for feat in feature_universe:
            if feat in frame.columns:
                contract_cols[feat] = frame[feat].to_numpy()
        contract_columns[contract] = contract_cols

    records: list[dict[str, float | str | bool]] = []
    mean_psi_dict: dict[str, float] = {}
    max_pair_psi_dict: dict[str, float] = {}
    max_ks_d_dict: dict[str, float] = {}
    min_ks_p_dict: dict[str, float] = {}

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

    passing = [
        feat
        for feat in feature_universe
        if mean_psi_dict[feat] <= max_mean_psi
        and max_pair_psi_dict[feat] <= max_pair_psi
    ]

    target_survivors = min(len(feature_universe), min_drift_survivors)
    fallback_triggered = False

    if len(passing) < target_survivors:
        fallback_triggered = True
        logger.warning(
            "Distribution drift gate: only %d / %d features passed initial threshold (mean_psi<=%.2f, max_pair_psi<=%.2f). "
            "Relaxing threshold to mean_psi<=%.2f.",
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
    for feat in feature_universe:
        records.append(
            {
                "feature": feat,
                "mean_psi": mean_psi_dict[feat],
                "max_pair_psi": max_pair_psi_dict[feat],
                "max_ks_d": max_ks_d_dict[feat],
                "min_ks_p": min_ks_p_dict[feat],
                "drift_passed": feat in surviving_set,
            }
        )

    metrics_df = pl.DataFrame(records)
    dropped = [feat for feat in feature_universe if feat not in surviving_set]

    return DistributionAuditResult(
        metrics_df=metrics_df,
        surviving_features=passing,
        dropped_features=dropped,
        fallback_triggered=fallback_triggered,
        mean_psi_by_feature=mean_psi_dict,
    )
