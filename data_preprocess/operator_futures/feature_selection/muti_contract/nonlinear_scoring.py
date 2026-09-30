from __future__ import annotations

import logging
import math
from typing import Sequence

import numpy as np
import polars as pl

from operator_futures.feature_selection.muti_contract.metrics import (
    _catboost_importance,
    calculate_future_return,
)
from operator_futures.feature_selection.muti_contract.types import (
    NonlinearScoringConfig,
    PipelineStepResult,
)

logger = logging.getLogger(__name__)


def execute_nonlinear_scoring(
    frames: dict[str, pl.DataFrame],
    features: list[str],
    aggregate_metrics_df: pl.DataFrame,
    mean_psi_by_feature: dict[str, float],
    config: NonlinearScoringConfig,
    metric_frames: list[pl.DataFrame] | None = None,
) -> tuple[list[str], pl.DataFrame, PipelineStepResult]:
    if not features:
        return [], aggregate_metrics_df, PipelineStepResult(
            step_name="nonlinear_scoring",
            surviving_features=[],
            dropped_features=[],
            audit_metrics_df=aggregate_metrics_df,
            diagnostics={
                "target_decision_window": config.decision_window,
                "composite_drop_ratio": config.composite_drop_ratio,
                "dropped_features": [],
            },
        )

    w_dec = config.decision_window
    per_contract_catboost: dict[str, dict[str, float]] = {}

    # 1. Fit CatBoost only on target decision window for each contract
    for contract, frame in frames.items():
        future_return = calculate_future_return(frame, w_dec)
        if future_return.size == 0:
            per_contract_catboost[contract] = {f: 0.0 for f in features}
            continue
        imp_dict = _catboost_importance(
            frame,
            features,
            future_return,
            window_length=w_dec,
            iterations=config.catboost_iterations,
            early_stopping_rounds=config.early_stopping_rounds,
            depth=config.catboost_depth,
            random_seed=config.random_seed,
        )
        per_contract_catboost[contract] = imp_dict

    # 2. Compute contract-averaged CatBoost importance
    catboost_mean: dict[str, float] = {}
    for feat in features:
        vals = [
            per_contract_catboost[c].get(feat, 0.0)
            for c in frames
            if c in per_contract_catboost
        ]
        catboost_mean[feat] = float(np.mean(vals)) if vals else 0.0

    # 3. Update metric_frames in place if provided (decision window gets importance, other windows remain 0.0)
    if metric_frames is not None:
        for idx, (contract, mf) in enumerate(zip(frames.keys(), metric_frames)):
            contract_imp = per_contract_catboost.get(contract, {})
            # Update rows where window == w_dec
            if "window" in mf.columns and "CatBoost Importance" in mf.columns:
                metric_frames[idx] = mf.with_columns(
                    pl.when(pl.col("window") == w_dec)
                    .then(
                        pl.col("feature").map_elements(
                            lambda f: contract_imp.get(f, 0.0), return_dtype=pl.Float64
                        )
                    )
                    .otherwise(pl.col("CatBoost Importance"))
                    .alias("CatBoost Importance")
                )

    # 4. Update aggregate_metrics_df
    scored_input = aggregate_metrics_df.filter(pl.col("feature").is_in(features))
    feat_list = scored_input["feature"].to_list()
    cb_series = pl.Series(
        "CatBoost Importance_Mean", [catboost_mean.get(f, 0.0) for f in feat_list]
    )
    if "CatBoost Importance_Mean" in scored_input.columns:
        scored_input = scored_input.with_columns(cb_series)
    else:
        scored_input = scored_input.with_columns(cb_series)

    # 5. Composite Priority Scoring
    height = float(max(scored_input.height, 1))
    rank_ic_score = pl.col("RankIC_Mean").abs().fill_null(0.0)
    catboost_score = pl.col("CatBoost Importance_Mean").fill_null(0.0)
    inv_psi_values = [
        1.0 / (float(mean_psi_by_feature.get(f, 0.0)) + 1e-4) for f in feat_list
    ]

    scored = (
        scored_input.with_columns(
            [
                rank_ic_score.alias("Composite RankIC Score"),
                catboost_score.alias("Composite Importance Score"),
                pl.Series("inv_mean_psi", inv_psi_values),
            ]
        )
        .with_columns(
            (
                0.40 * (pl.col("inv_mean_psi").rank() / height)
                + 0.35 * (pl.col("Composite RankIC Score").rank() / height)
                + 0.25 * (pl.col("Composite Importance Score").rank() / height)
            ).alias("Priority")
        )
        .with_columns(pl.col("Priority").alias("Composite Score"))
        .sort(
            ["Priority", "Composite RankIC Score", "Composite Importance Score"],
            descending=[True, True, True],
        )
    )

    # 6. Bottom Truncation
    drop_count = min(
        math.ceil(scored.height * config.composite_drop_ratio),
        max(scored.height - 1, 0),
    )
    kept_df = scored.head(scored.height - drop_count) if drop_count else scored
    dropped_df = scored.tail(drop_count) if drop_count else scored.head(0)

    surviving = kept_df["feature"].to_list()
    dropped = dropped_df["feature"].to_list()
    if not surviving:
        raise ValueError(
            "feature selection produced an empty list after Composite Score"
        )

    step_result = PipelineStepResult(
        step_name="nonlinear_scoring",
        surviving_features=surviving,
        dropped_features=dropped,
        audit_metrics_df=scored,
        diagnostics={
            "decision_window": w_dec,
            "catboost_mean_importance": catboost_mean,
            "dropped_count": drop_count,
            "composite_drop_ratio": config.composite_drop_ratio,
            "per_contract_catboost": per_contract_catboost,
        },
    )
    return surviving, scored, step_result
