from __future__ import annotations

import logging
import re
from typing import Sequence

import numpy as np
import polars as pl

from operator_futures.feature_selection.muti_contract.types import (
    DataHygieneConfig,
    PipelineStepResult,
)

logger = logging.getLogger(__name__)


def apply_feature_blacklist(
    features: Sequence[str], blacklist: Sequence[str]
) -> tuple[list[str], list[str]]:
    if not blacklist:
        return list(features), []
    bl_set = set(blacklist)
    kept = [f for f in features if f not in bl_set]
    dropped = [f for f in features if f in bl_set]
    return kept, dropped


def apply_feature_ablation_patterns(
    features: Sequence[str], patterns: Sequence[str]
) -> tuple[list[str], list[str]]:
    if not patterns:
        return list(features), []
    compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
    kept = [f for f in features if not any(p.search(f) for p in compiled)]
    dropped = [f for f in features if f not in set(kept)]
    return kept, dropped


def execute_data_hygiene(
    frames: dict[str, pl.DataFrame],
    candidate_features: list[str],
    config: DataHygieneConfig,
) -> tuple[dict[str, pl.DataFrame], PipelineStepResult]:
    surviving = list(candidate_features)

    # 1. Front-loaded Blacklist
    surviving, blacklist_dropped = apply_feature_blacklist(
        surviving, config.feature_blacklist
    )

    # 2. Front-loaded Regex Ablation
    surviving, ablation_dropped = apply_feature_ablation_patterns(
        surviving, config.feature_ablation_patterns
    )

    # 3. Near-Zero Variance and Mode Frequency
    zero_variance_dropped: list[str] = []
    quasi_constant_dropped: list[str] = []
    audit_rows: list[dict[str, float | str | bool]] = []
    clean_surviving: list[str] = []

    for feature in surviving:
        all_vals: list[np.ndarray] = []
        for frame in frames.values():
            if feature in frame.columns:
                arr = frame[feature].to_numpy()
                valid = arr[np.isfinite(arr)]
                if len(valid) > 0:
                    all_vals.append(valid)

        if not all_vals:
            zero_variance_dropped.append(feature)
            audit_rows.append(
                {
                    "feature": feature,
                    "variance": 0.0,
                    "mode_frequency": 1.0,
                    "hygiene_passed": False,
                }
            )
            continue

        pooled = np.concatenate(all_vals)
        variance = float(np.var(pooled))
        _, counts = np.unique(pooled, return_counts=True)
        mode_frequency = float(counts.max() / len(pooled))

        is_zero_var = variance <= config.min_variance
        is_quasi_const = mode_frequency >= config.max_mode_frequency

        passed = not is_zero_var and not is_quasi_const
        audit_rows.append(
            {
                "feature": feature,
                "variance": variance,
                "mode_frequency": mode_frequency,
                "hygiene_passed": passed,
            }
        )

        if is_zero_var:
            zero_variance_dropped.append(feature)
        elif is_quasi_const:
            quasi_constant_dropped.append(feature)
        else:
            clean_surviving.append(feature)

    # 4. In-Memory 5*IQR Winsorization on clean surviving features
    cleaned_frames = {k: v.clone() for k, v in frames.items()}
    if config.enable_winsorization and clean_surviving:
        for feature in clean_surviving:
            all_vals = []
            for frame in frames.values():
                if feature in frame.columns:
                    arr = frame[feature].to_numpy()
                    valid = arr[np.isfinite(arr)]
                    if len(valid) > 0:
                        all_vals.append(valid)
            if not all_vals:
                continue
            pooled = np.concatenate(all_vals)
            median = float(np.median(pooled))
            q25 = float(np.percentile(pooled, 25))
            q75 = float(np.percentile(pooled, 75))
            iqr = q75 - q25
            if iqr > 0.0:
                lower = median - config.winsorize_iqr_multiplier * iqr
                upper = median + config.winsorize_iqr_multiplier * iqr
            else:
                std = float(np.std(pooled))
                lower = median - config.winsorize_iqr_multiplier * std
                upper = median + config.winsorize_iqr_multiplier * std

            for contract, frame in cleaned_frames.items():
                if feature in frame.columns:
                    cleaned_frames[contract] = frame.with_columns(
                        pl.col(feature).clip(lower, upper)
                    )

    all_dropped = (
        blacklist_dropped
        + ablation_dropped
        + zero_variance_dropped
        + quasi_constant_dropped
    )
    diagnostics: dict[str, float | str | bool | list[str]] = {
        "blacklist_dropped": blacklist_dropped,
        "ablation_dropped": ablation_dropped,
        "zero_variance_dropped": zero_variance_dropped,
        "quasi_constant_dropped": quasi_constant_dropped,
    }
    audit_metrics_df = pl.DataFrame(audit_rows) if audit_rows else None

    result = PipelineStepResult(
        step_name="data_hygiene",
        surviving_features=clean_surviving,
        dropped_features=all_dropped,
        audit_metrics_df=audit_metrics_df,
        diagnostics=diagnostics,
    )
    return cleaned_frames, result
