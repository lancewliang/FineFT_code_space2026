from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import polars as pl
from scipy import stats

from operator_futures.feature_selection.muti_contract.metrics import (
    aggregate_metric_frames,
    calculate_metric_frame,
)
from operator_futures.feature_selection.muti_contract.types import (
    PipelineStepResult,
    PredictiveAuditConfig,
    SCALE_TIER_CONFIGS,
    classify_feature_scale,
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
        try:
            metric_df = calculate_metric_frame(
                frame,
                features,
                windows_list=list(config.windows_list),
                compute_catboost=False,
            )
            if metric_df.height > 0:
                contract_metric_frames.append(metric_df)
        except ValueError:
            continue

    if not contract_metric_frames:
        raise ValueError("future return is empty; cannot calculate feature metrics")

    aggregate_df = aggregate_metric_frames(contract_metric_frames)

    # 2. ADR-0054: Three-Tier Multi-Horizon Native Predictive Audit
    combined_mf = pl.concat(contract_metric_frames, how="vertical")
    n_contracts = float(len(contract_metric_frames))

    # Initialize per-tier diagnostic tracking
    tier_diagnostics: dict[str, dict[str, Any]] = {
        tier_name: {
            "forward_horizons": list(t_cfg.forward_horizons),
            "decision_horizon": t_cfg.decision_horizon,
            "thresholds": {
                "min_abs_ic": t_cfg.min_abs_ic,
                "sign_consistency": t_cfg.min_sign_consistency,
                "stability_ir": t_cfg.min_rank_ic_ir,
            },
            "candidates": 0,
            "anti_causal_dropped": 0,
            "rank_ic_dropped": 0,
            "sign_consistency_dropped": 0,
            "stability_dropped": 0,
            "survivors": 0,
        }
        for tier_name, t_cfg in SCALE_TIER_CONFIGS.items()
    }

    rank_ic_mean_map: dict[str, float] = {}
    rank_ic_std_map: dict[str, float] = {}
    ic_mean_map: dict[str, float] = {}
    ic_std_map: dict[str, float] = {}
    vol_rank_ic_mean_map: dict[str, float] = {}
    vol_rank_ic_std_map: dict[str, float] = {}
    vol_ic_mean_map: dict[str, float] = {}
    vol_ic_std_map: dict[str, float] = {}
    sc_map: dict[str, float] = {}
    vol_sc_map: dict[str, float] = {}

    has_vol = "VolRankIC" in combined_mf.columns

    for feat in features:
        feat_tier = classify_feature_scale(feat)
        tier_cfg = SCALE_TIER_CONFIGS[feat_tier]
        tier_diagnostics[feat_tier]["candidates"] += 1

        f_sub = combined_mf.filter(
            (pl.col("feature") == feat) & (pl.col("window").is_in(list(tier_cfg.forward_horizons)))
        )
        if f_sub.height == 0:
            f_sub = combined_mf.filter(pl.col("feature") == feat)

        if f_sub.height > 0:
            rank_ic_mean_map[feat] = float(f_sub["RankIC"].mean())
            rank_ic_std_map[feat] = float(f_sub["RankIC"].std()) if f_sub.height > 1 else 0.0
            ic_mean_map[feat] = float(f_sub["IC"].mean())
            ic_std_map[feat] = float(f_sub["IC"].std()) if f_sub.height > 1 else 0.0
            if has_vol:
                vol_rank_ic_mean_map[feat] = float(f_sub["VolRankIC"].mean())
                vol_rank_ic_std_map[feat] = float(f_sub["VolRankIC"].std()) if f_sub.height > 1 else 0.0
                vol_ic_mean_map[feat] = float(f_sub["VolIC"].mean())
                vol_ic_std_map[feat] = float(f_sub["VolIC"].std()) if f_sub.height > 1 else 0.0
            else:
                vol_rank_ic_mean_map[feat] = 0.0
                vol_rank_ic_std_map[feat] = 0.0
                vol_ic_mean_map[feat] = 0.0
                vol_ic_std_map[feat] = 0.0
        else:
            rank_ic_mean_map[feat] = 0.0
            rank_ic_std_map[feat] = 0.0
            ic_mean_map[feat] = 0.0
            ic_std_map[feat] = 0.0
            vol_rank_ic_mean_map[feat] = 0.0
            vol_rank_ic_std_map[feat] = 0.0
            vol_ic_mean_map[feat] = 0.0
            vol_ic_std_map[feat] = 0.0

        f_dec = combined_mf.filter(
            (pl.col("feature") == feat) & (pl.col("window") == tier_cfg.decision_horizon)
        )
        if f_dec.height == 0:
            avail = combined_mf.filter(pl.col("feature") == feat)
            if avail.height > 0:
                windows_avail = sorted(avail["window"].unique().to_list())
                closest_w = min(windows_avail, key=lambda w: abs(w - tier_cfg.decision_horizon))
                f_dec = combined_mf.filter((pl.col("feature") == feat) & (pl.col("window") == closest_w))

        if f_dec.height > 0:
            pos_cnt = float((f_dec["RankIC"] > 0.0).sum())
            neg_cnt = float((f_dec["RankIC"] < 0.0).sum())
            sc_map[feat] = max(pos_cnt, neg_cnt) / float(f_dec.height)
            if has_vol:
                vol_pos = float((f_dec["VolRankIC"] > 0.0).sum())
                vol_neg = float((f_dec["VolRankIC"] < 0.0).sum())
                vol_sc_map[feat] = max(vol_pos, vol_neg) / float(f_dec.height)
            else:
                vol_sc_map[feat] = 1.0
        else:
            sc_map[feat] = 1.0
            vol_sc_map[feat] = 1.0

    # 3. Gate 1: Anti-causality Anomaly Screening
    anti_causal_dropped: list[str] = []
    surviving_after_anti_causal: list[str] = []
    for feat in features:
        mean_rank_ic = rank_ic_mean_map[feat]
        mean_ic = ic_mean_map[feat]
        vol_mean_rank_ic = vol_rank_ic_mean_map[feat]
        vol_mean_ic = vol_ic_mean_map[feat]
        feat_tier = classify_feature_scale(feat)
        if (
            (abs(mean_rank_ic) >= config.ic_anomaly_ceiling and abs(vol_mean_rank_ic) >= config.ic_anomaly_ceiling)
            or (abs(mean_ic) >= config.ic_anomaly_ceiling and abs(vol_mean_ic) >= config.ic_anomaly_ceiling)
        ):
            anti_causal_dropped.append(feat)
            tier_diagnostics[feat_tier]["anti_causal_dropped"] += 1
            logger.warning(
                "Feature %s flagged and dropped by anti-causality screen (|RankIC|=%.4f, |IC|=%.4f >= %.2f)",
                feat,
                abs(mean_rank_ic),
                abs(mean_ic),
                config.ic_anomaly_ceiling,
            )
        else:
            surviving_after_anti_causal.append(feat)

    # 4. Gate 2: Hard RankIC Filter (Evaluated against tier-specific min_abs_ic)
    hard_filter_dropped: list[str] = []
    surviving_after_hard: list[str] = []
    for feat in surviving_after_anti_causal:
        mean_rank_ic = rank_ic_mean_map[feat]
        vol_mean_rank_ic = vol_rank_ic_mean_map[feat]
        feat_tier = classify_feature_scale(feat)
        tier_cfg = SCALE_TIER_CONFIGS[feat_tier]

        if config.rank_ic_mode == "signed":
            passed_dir = mean_rank_ic >= tier_cfg.min_abs_ic
        else:
            passed_dir = abs(mean_rank_ic) >= tier_cfg.min_abs_ic
        passed_vol = abs(vol_mean_rank_ic) >= tier_cfg.min_abs_ic

        if passed_dir or passed_vol:
            surviving_after_hard.append(feat)
        else:
            hard_filter_dropped.append(feat)
            tier_diagnostics[feat_tier]["rank_ic_dropped"] += 1

    # 5. Gate 3: Sign Consistency Filter (Evaluated against tier-specific min_sign_consistency)
    sign_consistency_dropped: list[str] = []
    surviving_after_sign: list[str] = []
    for feat in surviving_after_hard:
        feat_tier = classify_feature_scale(feat)
        tier_cfg = SCALE_TIER_CONFIGS[feat_tier]

        passed_dir = sc_map[feat] >= tier_cfg.min_sign_consistency
        passed_vol = vol_sc_map[feat] >= tier_cfg.min_sign_consistency
        if passed_dir or passed_vol:
            surviving_after_sign.append(feat)
        else:
            sign_consistency_dropped.append(feat)
            tier_diagnostics[feat_tier]["sign_consistency_dropped"] += 1

    # 6. Gate 4: Stability IR Filter (Evaluated against tier-specific min_rank_ic_ir)
    stability_dropped: list[str] = []
    surviving_after_stability: list[str] = []
    for feat in surviving_after_sign:
        feat_tier = classify_feature_scale(feat)
        tier_cfg = SCALE_TIER_CONFIGS[feat_tier]

        dir_ir = abs(rank_ic_mean_map[feat]) / (rank_ic_std_map[feat] + 1e-6)
        vol_ir = abs(vol_rank_ic_mean_map[feat]) / (vol_rank_ic_std_map[feat] + 1e-6)
        passed_dir = (dir_ir >= tier_cfg.min_rank_ic_ir and ic_std_map[feat] <= config.max_metric_std)
        passed_vol = (vol_ir >= tier_cfg.min_rank_ic_ir and vol_ic_std_map[feat] <= config.max_metric_std)
        if passed_dir or passed_vol:
            surviving_after_stability.append(feat)
        else:
            stability_dropped.append(feat)
            tier_diagnostics[feat_tier]["stability_dropped"] += 1

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

    for feat in final_surviving:
        feat_tier = classify_feature_scale(feat)
        tier_diagnostics[feat_tier]["survivors"] += 1

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
            "tier_breakdown": tier_diagnostics,
        },
    )
    return aggregate_df, step_result
