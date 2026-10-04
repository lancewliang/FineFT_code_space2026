from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import polars as pl
from scipy import stats

from operator_futures.feature_selection.muti_contract.metrics import (
    _permutation_importance,
    aggregate_metric_frames,
    calculate_future_return,
    calculate_ic,
    calculate_rank_ic,
    calculate_sharpe,
)
from operator_futures.feature_selection.muti_contract.types import (
    PipelineStepResult,
    PredictiveAuditConfig,
)

logger = logging.getLogger(__name__)


def execute_predictive_audit(
    frames: dict[str, pl.DataFrame],
    features: list[str],
    config: PredictiveAuditConfig,
) -> tuple[pl.DataFrame, PipelineStepResult]:
    if not features:
        empty_agg = pl.DataFrame(
            schema={
                "feature": pl.Utf8,
                "RankIC_Mean": pl.Float64,
                "RankIC_Std": pl.Float64,
                "IC_Mean": pl.Float64,
                "IC_Std": pl.Float64,
                "VolRankIC_Mean": pl.Float64,
                "VolRankIC_Std": pl.Float64,
                "VolIC_Mean": pl.Float64,
                "VolIC_Std": pl.Float64,
            }
        )
        step_result = PipelineStepResult(
            step_name="predictive_audit",
            surviving_features=[],
            dropped_features=[],
            audit_metrics_df=empty_agg,
            diagnostics={
                "anti_causal_dropped": [],
                "hard_filter_dropped": [],
                "sign_consistency_dropped": [],
                "stability_dropped": [],
                "fdr_dropped": [],
                "metric_frames": [],
                "sign_consistency_map": {},
                "vol_sign_consistency_map": {},
            },
        )
        return empty_agg, step_result

    # 1. Vectorized multi-window metrics calculation across contracts
    contract_metric_frames: list[pl.DataFrame] = []
    for contract, frame in frames.items():
        rows = []
        for window_length in config.windows_list:
            future_return = calculate_future_return(frame, window_length)
            if future_return.size == 0:
                continue
            future_vol = np.abs(future_return)
            metric_df = frame.slice(0, future_return.size)
            for feature in features:
                values = metric_df[feature].cast(pl.Float64, strict=False).to_numpy()
                rows.append(
                    {
                        "feature": feature,
                        "window": window_length,
                        "Permutation Importance": _permutation_importance(
                            values, future_return
                        ),
                        "CatBoost Importance": 0.0,
                        "IC": calculate_ic(values, future_return),
                        "RankIC": calculate_rank_ic(values, future_return),
                        "VolIC": calculate_ic(values, future_vol),
                        "VolRankIC": calculate_rank_ic(values, future_vol),
                        "Sharpe": calculate_sharpe(values, future_return),
                    }
                )
        if rows:
            contract_metric_frames.append(pl.DataFrame(rows))

    if not contract_metric_frames:
        raise ValueError("future return is empty; cannot calculate feature metrics")

    aggregate_df = aggregate_metric_frames(contract_metric_frames)

    # 2. Extract SignConsistency at target decision window
    combined_mf = pl.concat(contract_metric_frames, how="vertical")
    dec_window = config.target_decision_window
    w_frames = combined_mf.filter(pl.col("window") == dec_window)
    n_contracts = float(len(contract_metric_frames))

    if w_frames.height > 0:
        sc_df = (
            w_frames.group_by("feature")
            .agg(
                [
                    (pl.col("RankIC") > 0.0).sum().alias("pos_cnt"),
                    (pl.col("RankIC") < 0.0).sum().alias("neg_cnt"),
                ]
            )
            .with_columns(
                (pl.max_horizontal("pos_cnt", "neg_cnt") / n_contracts).alias(
                    "SignConsistency"
                )
            )
        )
        sc_map = dict(
            zip(sc_df["feature"].to_list(), sc_df["SignConsistency"].to_list())
        )
        if "VolRankIC" in w_frames.columns:
            vol_sc_df = (
                w_frames.group_by("feature")
                .agg(
                    [
                        (pl.col("VolRankIC") > 0.0).sum().alias("vol_pos_cnt"),
                        (pl.col("VolRankIC") < 0.0).sum().alias("vol_neg_cnt"),
                    ]
                )
                .with_columns(
                    (pl.max_horizontal("vol_pos_cnt", "vol_neg_cnt") / n_contracts).alias(
                        "VolSignConsistency"
                    )
                )
            )
            vol_sc_map = dict(
                zip(vol_sc_df["feature"].to_list(), vol_sc_df["VolSignConsistency"].to_list())
            )
        else:
            vol_sc_map = {feat: 1.0 for feat in features}
    elif "SignConsistency_Mean" in aggregate_df.columns:
        sc_map = dict(
            zip(
                aggregate_df["feature"].to_list(),
                aggregate_df["SignConsistency_Mean"].to_list(),
            )
        )
        vol_sc_map = (
            dict(
                zip(
                    aggregate_df["feature"].to_list(),
                    aggregate_df["VolSignConsistency_Mean"].to_list(),
                )
            )
            if "VolSignConsistency_Mean" in aggregate_df.columns
            else {feat: 1.0 for feat in features}
        )
    else:
        sc_map = {feat: 1.0 for feat in features}
        vol_sc_map = {feat: 1.0 for feat in features}

    for feat in features:
        if feat not in sc_map:
            sc_map[feat] = 0.0
        if feat not in vol_sc_map:
            vol_sc_map[feat] = 0.0

    rank_ic_mean_map = dict(
        zip(aggregate_df["feature"].to_list(), aggregate_df["RankIC_Mean"].to_list())
    )
    rank_ic_std_map = dict(
        zip(aggregate_df["feature"].to_list(), aggregate_df["RankIC_Std"].to_list())
    )
    ic_mean_map = dict(
        zip(aggregate_df["feature"].to_list(), aggregate_df["IC_Mean"].to_list())
    )
    ic_std_map = dict(
        zip(aggregate_df["feature"].to_list(), aggregate_df["IC_Std"].to_list())
    )

    vol_rank_ic_mean_map = (
        dict(zip(aggregate_df["feature"].to_list(), aggregate_df["VolRankIC_Mean"].to_list()))
        if "VolRankIC_Mean" in aggregate_df.columns
        else {feat: 0.0 for feat in features}
    )
    vol_rank_ic_std_map = (
        dict(zip(aggregate_df["feature"].to_list(), aggregate_df["VolRankIC_Std"].to_list()))
        if "VolRankIC_Std" in aggregate_df.columns
        else {feat: 0.0 for feat in features}
    )
    vol_ic_mean_map = (
        dict(zip(aggregate_df["feature"].to_list(), aggregate_df["VolIC_Mean"].to_list()))
        if "VolIC_Mean" in aggregate_df.columns
        else {feat: 0.0 for feat in features}
    )
    vol_ic_std_map = (
        dict(zip(aggregate_df["feature"].to_list(), aggregate_df["VolIC_Std"].to_list()))
        if "VolIC_Std" in aggregate_df.columns
        else {feat: 0.0 for feat in features}
    )

    # 3. Gate 1: Anti-causality Anomaly Screening
    anti_causal_dropped: list[str] = []
    surviving_after_anti_causal: list[str] = []
    for feat in features:
        mean_rank_ic = rank_ic_mean_map[feat]
        mean_ic = ic_mean_map[feat]
        vol_mean_rank_ic = vol_rank_ic_mean_map[feat]
        vol_mean_ic = vol_ic_mean_map[feat]
        if (
            (abs(mean_rank_ic) >= config.ic_anomaly_ceiling and abs(vol_mean_rank_ic) >= config.ic_anomaly_ceiling)
            or (abs(mean_ic) >= config.ic_anomaly_ceiling and abs(vol_mean_ic) >= config.ic_anomaly_ceiling)
        ):
            anti_causal_dropped.append(feat)
            logger.warning(
                "Feature %s flagged and dropped by anti-causality screen (|RankIC|=%.4f, |IC|=%.4f >= %.2f)",
                feat,
                abs(mean_rank_ic),
                abs(mean_ic),
                config.ic_anomaly_ceiling,
            )
        else:
            surviving_after_anti_causal.append(feat)

    # 4. Gate 2: Hard RankIC Filter (Survives if directional OR volatility passes)
    hard_filter_dropped: list[str] = []
    surviving_after_hard: list[str] = []
    for feat in surviving_after_anti_causal:
        mean_rank_ic = rank_ic_mean_map[feat]
        vol_mean_rank_ic = vol_rank_ic_mean_map[feat]
        if config.rank_ic_mode == "signed":
            passed_dir = mean_rank_ic >= config.min_abs_ic
        else:
            passed_dir = abs(mean_rank_ic) >= config.min_abs_ic
        passed_vol = abs(vol_mean_rank_ic) >= config.min_abs_ic

        if passed_dir or passed_vol:
            surviving_after_hard.append(feat)
        else:
            hard_filter_dropped.append(feat)

    # 5. Gate 3: Sign Consistency Filter (Survives if directional OR volatility passes)
    sign_consistency_dropped: list[str] = []
    surviving_after_sign: list[str] = []
    for feat in surviving_after_hard:
        passed_dir = sc_map[feat] >= config.min_sign_consistency
        passed_vol = vol_sc_map[feat] >= config.min_sign_consistency
        if passed_dir or passed_vol:
            surviving_after_sign.append(feat)
        else:
            sign_consistency_dropped.append(feat)

    # 6. Gate 4: Stability IR Filter (Survives if directional OR volatility passes)
    stability_dropped: list[str] = []
    surviving_after_stability: list[str] = []
    for feat in surviving_after_sign:
        dir_ir = abs(rank_ic_mean_map[feat]) / (rank_ic_std_map[feat] + 1e-6)
        vol_ir = abs(vol_rank_ic_mean_map[feat]) / (vol_rank_ic_std_map[feat] + 1e-6)
        passed_dir = (dir_ir >= config.min_rank_ic_ir and ic_std_map[feat] <= config.max_metric_std)
        passed_vol = (vol_ir >= config.min_rank_ic_ir and vol_ic_std_map[feat] <= config.max_metric_std)
        if passed_dir or passed_vol:
            surviving_after_stability.append(feat)
        else:
            stability_dropped.append(feat)

    # 7. Gate 5: Benjamini-Hochberg FDR Control
    fdr_dropped: list[str] = []
    total_eff_samples = sum(frame.height for frame in frames.values())
    if (
        config.fdr_threshold < 1.0
        and config.min_abs_ic > 0.0
        and total_eff_samples >= 30
        and surviving_after_stability
    ):
        df_deg = max(total_eff_samples - 2, 1)
        p_values: list[tuple[str, float]] = []
        for feat in surviving_after_stability:
            r_dir = rank_ic_mean_map[feat]
            t_stat_dir = (r_dir * np.sqrt(df_deg)) / np.sqrt(max(1.0 - min(r_dir**2, 0.9999), 1e-6))
            p_val_dir = float(2.0 * stats.t.sf(abs(t_stat_dir), df=df_deg))

            r_vol = vol_rank_ic_mean_map[feat]
            t_stat_vol = (r_vol * np.sqrt(df_deg)) / np.sqrt(max(1.0 - min(r_vol**2, 0.9999), 1e-6))
            p_val_vol = float(2.0 * stats.t.sf(abs(t_stat_vol), df=df_deg))

            p_val = min(p_val_dir, p_val_vol)
            p_values.append((feat, p_val))

        p_values.sort(key=lambda item: item[1])
        m_features = len(p_values)
        k_max = -1
        for idx, (_, p_val) in enumerate(p_values):
            k = idx + 1
            threshold = (float(k) / float(m_features)) * config.fdr_threshold
            if p_val <= threshold:
                k_max = k

        if k_max > 0:
            final_surviving = [feat for feat, _ in p_values[:k_max]]
            fdr_dropped = [feat for feat, _ in p_values[k_max:]]
        else:
            final_surviving = []
            fdr_dropped = [feat for feat, _ in p_values]
    else:
        final_surviving = surviving_after_stability

    all_dropped = [feat for feat in features if feat not in set(final_surviving)]

    step_result = PipelineStepResult(
        step_name="predictive_audit",
        surviving_features=final_surviving,
        dropped_features=all_dropped,
        audit_metrics_df=aggregate_df,
        diagnostics={
            "anti_causal_dropped": anti_causal_dropped,
            "hard_filter_dropped": hard_filter_dropped,
            "sign_consistency_dropped": sign_consistency_dropped,
            "stability_dropped": stability_dropped,
            "fdr_dropped": fdr_dropped,
            "metric_frames": contract_metric_frames,
            "sign_consistency_map": sc_map,
            "vol_sign_consistency_map": vol_sc_map,
        },
    )
    return aggregate_df, step_result
