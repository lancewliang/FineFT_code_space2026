from __future__ import annotations

import logging
import math
import re
from typing import Sequence

import numpy as np
import polars as pl
from statsmodels.tsa.stattools import adfuller

from operator_futures.feature_selection.manifests import PersistenceDiagnostic
from operator_futures.feature_selection.muti_contract.types import (
    PipelineStepResult,
    StationarityAuditConfig,
)

logger = logging.getLogger(__name__)


def lag1_autocorrelation(values: Sequence[float] | np.ndarray) -> float | None:
    values = np.asarray(values, dtype=float)
    if values.size < 3:
        return None
    left = values[:-1]
    right = values[1:]
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]
    if left.size < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    val = float(np.corrcoef(left, right)[0, 1])
    return val if np.isfinite(val) else None


def directional_half_life_bars(autocorrelation: float | None) -> float | None:
    if autocorrelation is None:
        return None
    if autocorrelation <= 0.0:
        return 0.0
    if autocorrelation >= 1.0:
        return None
    return float(math.log(0.5) / math.log(autocorrelation))


def sign_alternation_rate(values: Sequence[float] | np.ndarray) -> float | None:
    values = np.asarray(values, dtype=float)
    valid = values[np.isfinite(values)]
    if len(valid) < 2:
        return None
    signs = np.sign(valid)
    diff = signs[1:] != signs[:-1]
    return float(np.mean(diff))


def median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=float)))


def execute_stationarity_audit(
    frames: dict[str, pl.DataFrame],
    features: list[str],
    config: StationarityAuditConfig,
) -> PipelineStepResult:
    if not features:
        return PipelineStepResult(
            step_name="stationarity_audit",
            surviving_features=[],
            dropped_features=[],
            audit_metrics_df=None,
            diagnostics={
                "fallback_triggered": False,
                "adf_dropped": [],
                "persistence_dropped": [],
                "sar_dropped": [],
                "persistence_diagnostics": [],
            },
        )

    active_regex = (
        re.compile(config.active_feature_pattern)
        if config.active_feature_pattern is not None
        else None
    )

    feature_pvalues: dict[str, list[float]] = {}
    feature_passing_ratios: dict[str, float] = {}
    feature_mean_pvalues: dict[str, float] = {}
    feature_half_lives: dict[str, float | None] = {}
    feature_autocorrs: dict[str, float | None] = {}
    feature_sars: dict[str, float | None] = {}
    persistence_diagnostics: list[PersistenceDiagnostic] = []
    active_filter_flags: dict[str, bool] = {}

    for feat in features:
        p_vals: list[float] = []
        half_lives: list[float] = []
        autocorrs: list[float] = []
        sars: list[float] = []

        is_active = (
            bool(active_regex.search(feat)) if active_regex is not None else True
        )
        active_filter_flags[feat] = is_active

        for frame in frames.values():
            if feat not in frame.columns:
                continue
            arr = frame[feat].to_numpy()
            valid = arr[np.isfinite(arr)]

            # ADF Stationarity
            if len(valid) < 8 or np.std(valid) == 0.0:
                p_vals.append(1.0)
            else:
                try:
                    res = adfuller(valid, autolag="AIC")
                    p_vals.append(float(res[1]))
                except Exception:
                    p_vals.append(1.0)

            # Autocorrelation and half-life
            rho = lag1_autocorrelation(valid)
            if rho is not None:
                autocorrs.append(rho)
            hl = directional_half_life_bars(rho)
            if hl is not None:
                half_lives.append(hl)

            # SAR
            sar = sign_alternation_rate(valid)
            if sar is not None:
                sars.append(sar)

        feature_pvalues[feat] = p_vals
        passing_count = sum(1 for p in p_vals if p < config.adf_significance_level)
        total_eval = max(len(p_vals), 1)
        passing_ratio = float(passing_count) / float(total_eval)
        feature_passing_ratios[feat] = passing_ratio
        feature_mean_pvalues[feat] = float(np.mean(p_vals)) if p_vals else 1.0

        med_autocorr = median_or_none(autocorrs)
        med_half_life = median_or_none(half_lives)
        med_sar = median_or_none(sars)
        feature_autocorrs[feat] = med_autocorr
        feature_half_lives[feat] = med_half_life
        feature_sars[feat] = med_sar

        persistence_diagnostics.append(
            {
                "feature": feat,
                "lag1_autocorrelation_median": med_autocorr,
                "half_life_bars_median": med_half_life,
                "active_filter": is_active,
            }
        )

    # 1. ADF Gating
    surviving_adf = [
        feat
        for feat in features
        if feature_passing_ratios[feat] >= config.min_passing_contract_ratio
    ]

    target_floor = min(len(features), config.min_survivors_floor)
    fallback_triggered = False

    if len(surviving_adf) < target_floor:
        fallback_triggered = True
        logger.warning(
            "Stationarity audit: only %d / %d features passed ADF threshold (p<%.2f, ratio>=%.2f). "
            "Relaxing threshold to p<%.2f.",
            len(surviving_adf),
            len(features),
            config.adf_significance_level,
            config.min_passing_contract_ratio,
            config.fallback_significance_level,
        )
        relaxed_passing = [
            feat
            for feat in features
            if (
                sum(
                    1
                    for p in feature_pvalues[feat]
                    if p < config.fallback_significance_level
                )
                / max(len(feature_pvalues[feat]), 1)
            )
            >= config.min_passing_contract_ratio
        ]
        if len(relaxed_passing) >= target_floor:
            surviving_adf = relaxed_passing
        else:
            logger.warning(
                "Stationarity audit: only %d features passed relaxed ADF threshold. "
                "Retaining top %d features by lowest mean ADF p-value.",
                len(relaxed_passing),
                target_floor,
            )
            sorted_by_p = sorted(features, key=lambda f: feature_mean_pvalues[f])
            surviving_adf = sorted_by_p[:target_floor]

    adf_dropped = [f for f in features if f not in set(surviving_adf)]

    # 2. Persistence Half-Life Gating
    persistence_dropped: list[str] = []
    after_persistence: list[str] = []
    for feat in surviving_adf:
        hl = feature_half_lives[feat]
        is_active = active_filter_flags[feat]
        if (
            config.min_half_life_bars > 0.0
            and is_active
            and hl is not None
            and hl < config.min_half_life_bars
        ):
            persistence_dropped.append(feat)
        else:
            after_persistence.append(feat)

    # 3. Sign Alternation Rate (SAR) Gating
    sar_dropped: list[str] = []
    final_surviving: list[str] = []
    for feat in after_persistence:
        sar = feature_sars[feat]
        if (
            config.max_sign_alternation_rate < 1.0
            and sar is not None
            and sar > config.max_sign_alternation_rate
        ):
            sar_dropped.append(feat)
        else:
            final_surviving.append(feat)

    all_dropped = [f for f in features if f not in set(final_surviving)]

    audit_rows = []
    for feat in features:
        audit_rows.append(
            {
                "feature": feat,
                "mean_adf_pvalue": feature_mean_pvalues[feat],
                "passing_contract_ratio": feature_passing_ratios[feat],
                "half_life_median": feature_half_lives[feat],
                "sar_median": feature_sars[feat],
                "stationarity_passed": feat in set(final_surviving),
            }
        )
    audit_metrics_df = pl.DataFrame(audit_rows)

    return PipelineStepResult(
        step_name="stationarity_audit",
        surviving_features=final_surviving,
        dropped_features=all_dropped,
        audit_metrics_df=audit_metrics_df,
        diagnostics={
            "fallback_triggered": fallback_triggered,
            "adf_dropped": adf_dropped,
            "persistence_dropped": persistence_dropped,
            "sar_dropped": sar_dropped,
            "persistence_diagnostics": persistence_diagnostics,
        },
    )
