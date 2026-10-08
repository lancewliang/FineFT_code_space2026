from __future__ import annotations

import argparse
from dataclasses import replace
import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

from operator_futures.feature_selection.manifests import (
    FeatureSelectionContractRecord,
    FeatureSelectionManifest,
    FeatureSelectionResult,
    PersistenceDiagnostic,
    StreamAuditRecord,
)
from operator_futures.feature_selection.muti_contract.data_hygiene import (
    execute_data_hygiene,
)
from operator_futures.feature_selection.muti_contract.distribution_audit import (
    DEFAULT_DISTRIBUTION_NUM_BINS,
    DEFAULT_MAX_MEAN_PSI,
    DEFAULT_MAX_PAIR_PSI,
    DEFAULT_MIN_DRIFT_SURVIVORS,
    audit_distribution_drift,
)
from operator_futures.feature_selection.muti_contract.io_manager import (
    PipelineIOManager,
    extract_state_features,
)
from operator_futures.feature_selection.muti_contract.metrics import (
    DEFAULT_WINDOWS_LIST,
    aggregate_metric_frames,
    calculate_metric_frame,
)
from operator_futures.feature_selection.muti_contract.nonlinear_scoring import (
    execute_nonlinear_scoring,
)
from operator_futures.feature_selection.muti_contract.orthogonal_dedup import (
    execute_orthogonal_deduplication,
)
from operator_futures.feature_selection.muti_contract.predictive_audit import (
    execute_predictive_audit,
)
from operator_futures.feature_selection.muti_contract.regime_audit import (
    audit_regimes,
    compute_regime_quantiles,
    default_target_regime_bins,
)
from operator_futures.feature_selection.muti_contract.stationarity_audit import (
    execute_stationarity_audit,
)
from operator_futures.feature_selection.muti_contract.types import (
    DEFAULT_RL_PROFILE,
    DEFAULT_VAE_SLOPE_PROFILE,
    DEFAULT_VAE_VOLATILITY_PROFILE,
    SCALE_TIER_CONFIGS,
    STREAM_TIER_QUOTAS,
    classify_feature_scale,
    DataHygieneConfig,
    DistributionAuditConfig,
    FeatureSelectionPipelineConfig,
    NonlinearScoringConfig,
    OrthogonalDedupConfig,
    PredictiveAuditConfig,
    RegimeAuditConfig,
    StationarityAuditConfig,
    StreamFilterProfile,
)

logger = logging.getLogger(__name__)

DEFAULT_PERSISTENCE_FILTER_PATTERN = r"_log_return_(1|2)$"


def _state_features(df: pl.DataFrame, *, orderbook_depth: int) -> list[str]:
    return extract_state_features(df, orderbook_depth=orderbook_depth)




def _build_config_from_legacy_kwargs(**kwargs) -> FeatureSelectionPipelineConfig:
    root_path = Path(kwargs["root_path"])
    symbol = kwargs["symbol"]
    target_freq = kwargs["target_freq"]
    stage = kwargs["stage"]
    windows_list = tuple(
        DEFAULT_WINDOWS_LIST
        if kwargs.get("windows_list") is None
        else kwargs["windows_list"]
    )
    target_decision_window = kwargs.get("target_decision_window")
    dec_window = (
        target_decision_window
        if target_decision_window is not None
        else (6 if 6 in windows_list else windows_list[0])
    )
    min_half_life_bars = float(kwargs.get("min_half_life_bars", 0.0))
    persistence_pattern = kwargs.get(
        "persistence_filter_pattern", DEFAULT_PERSISTENCE_FILTER_PATTERN
    )

    def _create_stream_profile(
        base_profile: StreamFilterProfile, profile_key: str, blacklist_key: str
    ) -> StreamFilterProfile:
        if kwargs.get(profile_key) is not None:
            prof = kwargs[profile_key]
        else:
            prof = base_profile
            overrides = {}
            if kwargs.get("min_clusters") is not None:
                overrides["min_clusters"] = int(kwargs["min_clusters"])
            if kwargs.get("max_clusters") is not None:
                overrides["max_clusters"] = int(kwargs["max_clusters"])
            if kwargs.get("filter_micro_persistence") is not None:
                overrides["filter_micro_persistence"] = bool(kwargs["filter_micro_persistence"])
            if overrides:
                prof = replace(prof, **overrides)
        if kwargs.get(blacklist_key) is not None:
            prof = replace(prof, feature_blacklist=tuple(kwargs[blacklist_key]))
        return prof

    vae_slope_profile = _create_stream_profile(DEFAULT_VAE_SLOPE_PROFILE, "vae_slope_profile", "vae_slope_feature_blacklist")
    vae_volatility_profile = _create_stream_profile(DEFAULT_VAE_VOLATILITY_PROFILE, "vae_volatility_profile", "vae_volatility_feature_blacklist")
    rl_profile = _create_stream_profile(DEFAULT_RL_PROFILE, "rl_profile", "rl_feature_blacklist")

    raw_blacklist = list(kwargs.get("feature_blacklist") or ())
    vae_slope_bl = (
        kwargs.get("vae_slope_feature_blacklist")
        if kwargs.get("vae_slope_feature_blacklist") is not None
        else (vae_slope_profile.feature_blacklist if vae_slope_profile.feature_blacklist else None)
    )
    vae_vol_bl = (
        kwargs.get("vae_volatility_feature_blacklist")
        if kwargs.get("vae_volatility_feature_blacklist") is not None
        else (vae_volatility_profile.feature_blacklist if vae_volatility_profile.feature_blacklist else None)
    )
    rl_bl = (
        kwargs.get("rl_feature_blacklist")
        if kwargs.get("rl_feature_blacklist") is not None
        else (rl_profile.feature_blacklist if rl_profile.feature_blacklist else None)
    )

    bl_sets = [set(bl) for bl in (vae_slope_bl, vae_vol_bl, rl_bl) if bl is not None]
    if len(bl_sets) == 3:
        effective_hygiene_blacklist = tuple(
            sorted(set(raw_blacklist).union(bl_sets[0].intersection(bl_sets[1]).intersection(bl_sets[2])))
        )
    elif len(bl_sets) > 0:
        common = bl_sets[0]
        for s in bl_sets[1:]:
            common = common.intersection(s)
        effective_hygiene_blacklist = tuple(sorted(set(raw_blacklist).union(common)))
    else:
        effective_hygiene_blacklist = tuple(raw_blacklist)

    return FeatureSelectionPipelineConfig(
        root_path=root_path,
        symbol=symbol,
        target_freq=target_freq,
        stage=stage,
        split_path=kwargs.get(
            "split_path",
            "PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST",
        ),
        save_path=kwargs.get(
            "save_path", "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION"
        ),
        orderbook_depth=int(kwargs.get("orderbook_depth", 5)),
        mandatory_state_features=tuple(kwargs.get("mandatory_state_features") or ()),
        persistence_filter_pattern=persistence_pattern,
        hygiene=DataHygieneConfig(
            feature_blacklist=effective_hygiene_blacklist,
            feature_ablation_patterns=tuple(
                kwargs.get("feature_ablation_patterns") or ()
            ),
        ),
        drift=DistributionAuditConfig(
            num_bins=int(
                kwargs.get("distribution_num_bins", DEFAULT_DISTRIBUTION_NUM_BINS)
            ),
            max_mean_psi=float(kwargs.get("max_mean_psi", DEFAULT_MAX_MEAN_PSI)),
            max_pair_psi=float(kwargs.get("max_pair_psi", DEFAULT_MAX_PAIR_PSI)),
            min_drift_survivors=int(
                kwargs.get("min_drift_survivors", DEFAULT_MIN_DRIFT_SURVIVORS)
            ),
        ),
        stationarity=StationarityAuditConfig(
            min_half_life_bars=min_half_life_bars,
            active_feature_pattern=(
                persistence_pattern if min_half_life_bars > 0.0 else None
            ),
            max_sign_alternation_rate=1.0 if min_half_life_bars == 0.0 else 0.40,
        ),
        predictive=PredictiveAuditConfig(
            min_abs_ic=float(kwargs.get("min_abs_ic", 0.02)),
            min_sign_consistency=float(kwargs.get("min_sign_consistency", 0.75)),
            min_rank_ic_ir=float(kwargs.get("min_rank_ic_ir", 0.40)),
            target_decision_window=dec_window,
            windows_list=windows_list,
            fdr_threshold=float(kwargs.get("fdr_threshold", 0.05)),
            ic_anomaly_ceiling=float(kwargs.get("ic_anomaly_ceiling", 1.0)),
            rank_ic_mode=str(kwargs.get("rank_ic_mode", "absolute")),
            max_metric_std=float(kwargs.get("max_metric_std", 1.0)),
        ),
        scoring=NonlinearScoringConfig(
            decision_window=dec_window,
            composite_drop_ratio=float(kwargs.get("composite_drop_ratio", 0.1)),
        ),
        dedup=OrthogonalDedupConfig(
            max_correlation=float(kwargs.get("max_correlation", 0.7)),
            dedup_method=str(kwargs.get("dedup_method", "greedy")),
            max_vif=float(kwargs.get("max_vif", 10.0)),
        ),
        regime=RegimeAuditConfig(
            enable_conditional_anchors=bool(
                kwargs.get("enable_conditional_anchors", True)
            ),
            regime_bins=int(kwargs.get("regime_bins", 4)),
            target_regime_bins=(
                tuple(kwargs["target_regime_bins"])
                if kwargs.get("target_regime_bins") is not None
                else None
            ),
        ),
        vae_slope_profile=vae_slope_profile,
        vae_volatility_profile=vae_volatility_profile,
        rl_profile=rl_profile,
    )



def _build_process_documentation() -> str:
    return """# Multi-Horizon Cross-Frequency Triple-Stream Feature Selection Process

## 1. Multi-Scale Hierarchy & Regular Expression Contracting
Candidate features are classified into three temporal scale tiers based on causal naming contracts:
- **Micro Tier (k in [1..12] bars, 10m ~ 2h)**: High-frequency orderbook microstructure, depth imbalances, and short returns. Default fallback for technical indicators.
- **Meso Tier (k in [16..96] bars, 2.6h ~ 16h)**: Intraday wave momentum, session indicators (`session_`, `trading_minute_`), multiday bars (`prev_day_`, `prev_2_day_`), and calendar features (`base_time_`, `contract_month_`, `contract_life_`).
- **Macro Tier (k in [192..720] bars, 32h ~ 120h)**: Continuous causal macro operators (`_(720|1440|2160)`, `prev_(5|10|15|20|30)_day`, `prev_(1|2|4|6)_week`, `cm_.*_(720|1440)`), including mark price EMA deviations, ROC, trend betas, trend-to-noise ratios, and cross-month rolling Z-scores.

## 2. Three-Tier Native Predictive Funnel
To prevent long-horizon macro features from being erroneously discarded by short-term IC tests, candidate features are evaluated strictly against their tier-matched forward return windows:
- **Micro Gate**: forward horizons [1, 2, 6, 12], decision horizon 6. Thresholds: |RankIC| >= 0.010, SignConsistency >= 0.55, RankIC-IR >= 0.18.
- **Meso Gate**: forward horizons [16, 24, 48, 96], decision horizon 24. Thresholds: |RankIC| >= 0.015, SignConsistency >= 0.58, RankIC-IR >= 0.15.
- **Macro Gate**: forward horizons [192, 384, 720], decision horizon 192. Thresholds: |RankIC| >= 0.020, SignConsistency >= 0.60, RankIC-IR >= 0.12.

## 3. Asymmetric Triple-Stream Specialization & Allocation
- **Slope VAE Stream (Strategic Direction Router)**: Operates strictly on Meso (8~10 features) and Macro (4~6 features). Micro features are strictly blacklisted (0% micro) to eliminate latent space jitter and false OOD halts.
- **Volatility VAE Stream (Regime Dispersion Router)**: Operates on Meso (6~8 features) and Macro (4~5 features), with at most 0~1 micro feature.
- **RL Decision Stream (Execution Policy)**: Flat multi-horizon state vector with 85~100 micro, 35~45 meso, and 10~15 macro features (total 135~160 dimensions).

## 4. Stratified Correlation Clustering Quotas & OOD Prevention
- Agglomerative correlation clustering (Ward linkage) is executed within each scale tier independently to enforce tier capacity quotas and prevent macro factors from being crowded out by micro indicators.
- Macro operators are mathematically scale-invariant (EMA deviation normalized by ATR, ROC normalized by base price, trend-to-noise bounded in [0, 1]) and incorporate expanding-window fallbacks (min_periods=48), ensuring zero-NaN generation and eliminating out-of-distribution (OOD) feature drift across market price level regimes.
"""

def _evaluate_stream_branch(
    profile: StreamFilterProfile,
    candidate_features: list[str],
    aggregate_metrics_df: pl.DataFrame,
    dist_metrics_df: pl.DataFrame,
    frames: dict[str, pl.DataFrame],
    mandatory_features: list[str],
    persistence_diagnostics: list[PersistenceDiagnostic] | None,
    min_half_life_bars: float,
    active_persistence_pattern: str | None,
    raw_universe_size: int,
    catboost_mean_importance: dict[str, float],
    sign_consistency_map: dict[str, float],
    retained_anchors: list[str],
    *,
    vol_sign_consistency_map: dict[str, float] | None = None,
    pred_tier_diagnostics: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[str], dict[str, list[str]], StreamAuditRecord]:
    import re
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    from operator_futures.feature_selection.muti_contract.orthogonal_dedup import (
        compute_contract_normalized_spearman_correlation_matrix,
        prune_by_vif,
    )

    # 1. Mandatory features partition
    if profile.mandatory_feature_pattern is not None:
        mandatory_regex = re.compile(profile.mandatory_feature_pattern)
        stream_mandatory = [f for f in mandatory_features if mandatory_regex.search(f)]
    else:
        stream_mandatory = list(mandatory_features)

    # 2. Lookup metrics
    mean_psi_map = dict(
        zip(dist_metrics_df["feature"].to_list(), dist_metrics_df["mean_psi"].to_list())
    )
    max_pair_psi_map = dict(
        zip(dist_metrics_df["feature"].to_list(), dist_metrics_df["max_pair_psi"].to_list())
    )

    is_vol = (profile.target_metric == "VolRankIC")
    target_ic_col = "VolRankIC_Mean" if is_vol and "VolRankIC_Mean" in aggregate_metrics_df.columns else "RankIC_Mean"
    target_ic_std_col = "VolRankIC_Std" if is_vol and "VolRankIC_Std" in aggregate_metrics_df.columns else "RankIC_Std"
    effective_sc_map = vol_sign_consistency_map if is_vol and vol_sign_consistency_map is not None else sign_consistency_map

    target_ic_mean_map = dict(
        zip(aggregate_metrics_df["feature"].to_list(), aggregate_metrics_df[target_ic_col].to_list())
    )
    target_ic_std_map = dict(
        zip(aggregate_metrics_df["feature"].to_list(), aggregate_metrics_df[target_ic_std_col].to_list())
    )

    pool = list(candidate_features)
    filter_drops: dict[str, list[str]] = {}

    if profile.feature_blacklist:
        stream_blacklist_set = set(profile.feature_blacklist)
        stream_mandatory = [f for f in stream_mandatory if f not in stream_blacklist_set]
        stream_dropped = [f for f in pool if f in stream_blacklist_set]
        pool = [f for f in pool if f not in stream_blacklist_set]
        if stream_dropped:
            filter_drops["Stream Blacklist Dropped"] = stream_dropped

    has_macro = any(classify_feature_scale(f) == "macro" for f in pool)

    if has_macro and profile.name in STREAM_TIER_QUOTAS and STREAM_TIER_QUOTAS[profile.name]["micro"].max_quota == 0:
        micro_drops = [f for f in pool if classify_feature_scale(f) == "micro"]
        if micro_drops:
            pool = [f for f in pool if classify_feature_scale(f) != "micro"]
            stream_mandatory = [f for f in stream_mandatory if classify_feature_scale(f) != "micro"]
            if "Stream Blacklist Dropped" in filter_drops:
                filter_drops["Stream Blacklist Dropped"].extend(micro_drops)
            else:
                filter_drops["Stream Blacklist Dropped"] = micro_drops

    # (a) Distribution Drift Gating
    psi_dropped: list[str] = []
    psi_surviving: list[str] = []
    for f in pool:
        if has_macro:
            f_tier = classify_feature_scale(f)
            tier_cfg = SCALE_TIER_CONFIGS.get(f_tier)
            eff_max_mean_psi = max(profile.max_mean_psi, tier_cfg.max_mean_psi if tier_cfg else 0.0)
            eff_max_pair_psi = max(profile.max_pair_psi, tier_cfg.max_pair_psi if tier_cfg else 0.0)
        else:
            eff_max_mean_psi = profile.max_mean_psi
            eff_max_pair_psi = profile.max_pair_psi
        mean_psi = mean_psi_map.get(f, 0.0)
        max_pair_psi = max_pair_psi_map.get(f, 0.0)
        if mean_psi <= eff_max_mean_psi and max_pair_psi <= eff_max_pair_psi:
            psi_surviving.append(f)
        else:
            psi_dropped.append(f)
    pool = psi_surviving
    if psi_dropped:
        filter_drops["Distribution Drift Dropped"] = psi_dropped

    # (b) Persistence Noise Gating
    if profile.filter_micro_persistence and min_half_life_bars > 0.0 and persistence_diagnostics is not None and active_persistence_pattern is not None:
        hl_diag_map = {row["feature"]: row for row in persistence_diagnostics}
        active_pat = re.compile(active_persistence_pattern)
        pers_dropped: list[str] = []
        pers_surviving: list[str] = []
        for f in pool:
            diag = hl_diag_map.get(f)
            is_active = (
                diag["active_filter"]
                if (diag is not None and "active_filter" in diag)
                else bool(active_pat.search(f))
            )
            hl = diag["half_life_bars_median"] if diag is not None else None
            if is_active and (hl is None or hl < min_half_life_bars):
                pers_dropped.append(f)
            else:
                pers_surviving.append(f)
        pool = pers_surviving
        if pers_dropped:
            filter_drops["Persistence Filter Dropped"] = pers_dropped

    # (c) Predictive Gating (Hard RankIC, Sign Consistency, Stability IR)
    hard_dropped: list[str] = []
    hard_surviving: list[str] = []
    for f in pool:
        if has_macro:
            f_tier = classify_feature_scale(f)
            tier_cfg = SCALE_TIER_CONFIGS.get(f_tier)
            eff_min_abs_ic = tier_cfg.min_abs_ic if (tier_cfg and profile.min_abs_ic > 0.0) else profile.min_abs_ic
        else:
            eff_min_abs_ic = profile.min_abs_ic
        if abs(target_ic_mean_map.get(f, 0.0)) >= eff_min_abs_ic:
            hard_surviving.append(f)
        else:
            hard_dropped.append(f)
    pool = hard_surviving
    if hard_dropped:
        filter_drops["Hard Filter Dropped"] = hard_dropped

    sc_dropped: list[str] = []
    sc_surviving: list[str] = []
    for f in pool:
        if has_macro:
            f_tier = classify_feature_scale(f)
            tier_cfg = SCALE_TIER_CONFIGS.get(f_tier)
            eff_min_sc = tier_cfg.min_sign_consistency if (tier_cfg and profile.min_sign_consistency > 0.0) else profile.min_sign_consistency
        else:
            eff_min_sc = profile.min_sign_consistency
        if effective_sc_map.get(f, 1.0) >= eff_min_sc:
            sc_surviving.append(f)
        else:
            sc_dropped.append(f)
    pool = sc_surviving
    if sc_dropped:
        filter_drops["Sign Consistency Filter Dropped"] = sc_dropped

    stab_dropped: list[str] = []
    stab_surviving: list[str] = []
    for f in pool:
        if has_macro:
            f_tier = classify_feature_scale(f)
            tier_cfg = SCALE_TIER_CONFIGS.get(f_tier)
            eff_min_ir = tier_cfg.min_rank_ic_ir if (tier_cfg and profile.min_rank_ic_ir > 0.0) else profile.min_rank_ic_ir
        else:
            eff_min_ir = profile.min_rank_ic_ir
        mean_r = abs(target_ic_mean_map.get(f, 0.0))
        std_r = target_ic_std_map.get(f, 0.0)
        ir = mean_r / (std_r + 1e-6)
        if ir >= eff_min_ir:
            stab_surviving.append(f)
        else:
            stab_dropped.append(f)
    pool = stab_surviving
    if stab_dropped:
        filter_drops["Stability Filter Dropped"] = stab_dropped

    # (c.1) Regime Differentiation: Monotonicity & ANOVA F-test
    if (profile.require_monotonic or profile.min_anova_f > 0.0) and pool:
        from operator_futures.feature_selection.muti_contract.regime_audit import extract_slope_and_volatility
        from scipy import stats

        frame_metrics: dict[str, np.ndarray] = {}
        all_metrics: list[np.ndarray] = []
        is_vol_regime = (profile.name == "vae_volatility" or profile.target_metric == "VolRankIC")
        for cname, frame in frames.items():
            if frame.height < 48:
                continue
            slope_arr, vol_arr = extract_slope_and_volatility(frame)
            arr = vol_arr if is_vol_regime else slope_arr
            frame_metrics[cname] = arr
            all_metrics.append(arr[47:])

        if all_metrics:
            cat_metrics = np.concatenate(all_metrics)
            if len(cat_metrics) >= 6:
                q1, q2 = np.quantile(cat_metrics, [1.0 / 3.0, 2.0 / 3.0])
                if q1 < q2:
                    mono_dropped: list[str] = []
                    anova_dropped: list[str] = []
                    for f in pool:
                        b0_vals: list[np.ndarray] = []
                        b1_vals: list[np.ndarray] = []
                        b2_vals: list[np.ndarray] = []
                        for cname, frame in frames.items():
                            if f not in frame.columns:
                                continue
                            m_arr = frame_metrics[cname]
                            f_arr = frame[f].to_numpy().astype(float)
                            start_idx = 47 if len(m_arr) >= 48 else 0
                            sub_m = m_arr[start_idx:]
                            sub_f = f_arr[start_idx:]
                            valid_m = ~(np.isnan(sub_m) | np.isnan(sub_f))
                            sub_m = sub_m[valid_m]
                            sub_f = sub_f[valid_m]
                            if len(sub_m) == 0:
                                continue
                            b0_vals.append(sub_f[sub_m < q1])
                            b1_vals.append(sub_f[(sub_m >= q1) & (sub_m < q2)])
                            b2_vals.append(sub_f[sub_m >= q2])

                        v0 = np.concatenate(b0_vals) if b0_vals else np.array([])
                        v1 = np.concatenate(b1_vals) if b1_vals else np.array([])
                        v2 = np.concatenate(b2_vals) if b2_vals else np.array([])

                        if len(v0) >= 2 and len(v1) >= 2 and len(v2) >= 2:
                            m0, m1, m2 = float(np.mean(v0)), float(np.mean(v1)), float(np.mean(v2))
                            if profile.require_monotonic:
                                if is_vol_regime:
                                    is_mono = (m0 < m1 < m2)
                                else:
                                    is_mono = (m0 < m1 < m2) or (m0 > m1 > m2)
                                if not is_mono:
                                    mono_dropped.append(f)
                                    continue

                            if profile.min_anova_f > 0.0:
                                f_stat, p_val = stats.f_oneway(v0, v1, v2)
                                if np.isnan(f_stat) or f_stat < profile.min_anova_f:
                                    anova_dropped.append(f)
                                    continue

                    if mono_dropped:
                        pool = [f for f in pool if f not in mono_dropped]
                        filter_drops["Monotonicity Filter Dropped"] = mono_dropped
                    if anova_dropped:
                        pool = [f for f in pool if f not in anova_dropped]
                        filter_drops["ANOVA F Filter Dropped"] = anova_dropped

    # (d) Composite Priority Scoring
    if pool:
        height = float(len(pool))
        inv_psi_vals = np.array([1.0 / (float(mean_psi_map.get(f, 0.0)) + 1e-4) for f in pool])
        rank_ic_vals = np.array([abs(float(target_ic_mean_map.get(f, 0.0))) for f in pool])
        cb_vals = np.array([float(catboost_mean_importance.get(f, 0.0)) for f in pool])

        inv_psi_ranks = (np.argsort(np.argsort(inv_psi_vals)) + 1) / height
        rank_ic_ranks = (np.argsort(np.argsort(rank_ic_vals)) + 1) / height
        cb_ranks = (np.argsort(np.argsort(cb_vals)) + 1) / height

        priority_scores = (
            profile.psi_weight * inv_psi_ranks
            + profile.rank_ic_weight * rank_ic_ranks
            + profile.catboost_weight * cb_ranks
        )
        sorted_indices = np.argsort(-priority_scores)
        pool = [pool[i] for i in sorted_indices]
        filter_drops["Composite Score"] = pool

    # (e) Orthogonal Deduplication & Clustering (ADR-0054 Scale-Stratified Quotas)
    corre_df = (
        compute_contract_normalized_spearman_correlation_matrix(frames, pool)
        if len(pool) > 1
        else None
    )
    corr_np = (
        np.array(corre_df.select(pool).to_numpy(), copy=True)
        if corre_df is not None
        else None
    )
    if corr_np is not None:
        np.fill_diagonal(corr_np, 1.0)

    if profile.name in STREAM_TIER_QUOTAS and has_macro:
        tier_quotas_dict = STREAM_TIER_QUOTAS[profile.name]
        canonical_max = (
            DEFAULT_RL_PROFILE.max_clusters
            if profile.name == "rl_decision"
            else (
                DEFAULT_VAE_SLOPE_PROFILE.max_clusters
                if profile.name == "vae_slope"
                else DEFAULT_VAE_VOLATILITY_PROFILE.max_clusters
            )
        )
        scale_factor = (
            profile.max_clusters / float(canonical_max)
            if profile.max_clusters != canonical_max
            else 1.0
        )

        selected_candidates: list[str] = []
        all_cluster_dropped: list[str] = []
        tier_target_max_map: dict[str, int] = {}
        selected_by_tier: dict[str, list[str]] = {t: [] for t in ("micro", "meso", "macro")}

        for tier_name in ("micro", "meso", "macro"):
            t_quota = tier_quotas_dict[tier_name]
            t_mandatory = [f for f in stream_mandatory if classify_feature_scale(f) == tier_name]
            t_pool = [f for f in pool if classify_feature_scale(f) == tier_name]

            if scale_factor != 1.0:
                eff_max = max(0, int(round(t_quota.max_quota * scale_factor)))
                eff_min = max(0, int(round(t_quota.min_quota * scale_factor)))
                if t_quota.max_quota > 0 and eff_max == 0:
                    eff_max = 1
            else:
                eff_max = t_quota.max_quota
                eff_min = t_quota.min_quota

            target_max_t = max(eff_max - len(t_mandatory), 0)
            target_min_t = max(eff_min - len(t_mandatory), 0)
            tier_target_max_map[tier_name] = target_max_t

            if target_max_t == 0:
                all_cluster_dropped.extend(t_pool)
                continue
            if not t_pool:
                continue
            if len(t_pool) <= target_max_t:
                t_selected = list(t_pool)
                t_dropped = []
            else:
                if corre_df is not None:
                    t_indices = [pool.index(f) for f in t_pool]
                    t_corr = corr_np[np.ix_(t_indices, t_indices)]
                    np.fill_diagonal(t_corr, 1.0)
                    dist_matrix = np.sqrt(np.clip((1.0 - t_corr) / 2.0, 0.0, 1.0))
                    np.fill_diagonal(dist_matrix, 0.0)
                    condensed_dist = squareform(dist_matrix, checks=False)
                    z = linkage(condensed_dist, method="ward")

                    dist_threshold = np.sqrt(max((1.0 - profile.max_correlation) / 2.0, 0.0))
                    cluster_ids = fcluster(z, t=dist_threshold, criterion="distance")
                    num_clusters = len(np.unique(cluster_ids))

                    if num_clusters > target_max_t and len(t_pool) > target_max_t:
                        cluster_ids = fcluster(z, t=target_max_t, criterion="maxclust")
                        num_clusters = len(np.unique(cluster_ids))

                    cluster_members = []
                    for cid in sorted(np.unique(cluster_ids)):
                        members = [t_pool[idx] for idx, c in enumerate(cluster_ids) if c == cid]
                        members_sorted = sorted(members, key=lambda f: t_pool.index(f))
                        cluster_members.append(members_sorted)

                    cluster_selected = [m[0] for m in cluster_members]
                    if len(cluster_selected) < target_min_t and len(t_pool) > len(cluster_selected):
                        remaining_candidates = []
                        for members in cluster_members:
                            remaining_candidates.extend(members[1:])
                        remaining_candidates.sort(key=lambda f: t_pool.index(f))
                        deficit = min(
                            target_max_t - len(cluster_selected),
                            len(remaining_candidates),
                        )
                        cluster_selected.extend(remaining_candidates[:deficit])

                    t_dropped = [f for f in t_pool if f not in set(cluster_selected)]
                    t_selected = cluster_selected
                else:
                    t_selected = t_pool[:target_max_t]
                    t_dropped = t_pool[target_max_t:]

            if corre_df is not None and len(t_selected) > 1:
                t_selected, vif_dropped = prune_by_vif(
                    t_selected, corre_df, max_vif=profile.max_vif
                )
                t_dropped.extend(vif_dropped)

            selected_candidates.extend(t_selected)
            selected_by_tier[tier_name].extend(t_selected)
            all_cluster_dropped.extend(t_dropped)

        target_max_candidates = max(profile.max_clusters - len(stream_mandatory), 1)
        target_min_candidates = max(profile.min_clusters - len(stream_mandatory), 1)
        if len(selected_candidates) < target_min_candidates and len(pool) > len(selected_candidates):
            for f in pool:
                if len(selected_candidates) >= min(target_min_candidates, target_max_candidates):
                    break
                if f in set(selected_candidates):
                    continue
                f_tier = classify_feature_scale(f)
                max_t = tier_target_max_map.get(f_tier, 0)
                if len(selected_by_tier[f_tier]) < max_t:
                    selected_candidates.append(f)
                    selected_by_tier[f_tier].append(f)

        all_cluster_dropped = [f for f in pool if f not in set(selected_candidates)]
        if all_cluster_dropped:
            filter_drops["Correlation Filter Dropped"] = all_cluster_dropped
    else:
        target_max_candidates = max(profile.max_clusters - len(stream_mandatory), 1)
        target_min_candidates = max(profile.min_clusters - len(stream_mandatory), 1)

        if not pool:
            selected_candidates = []
            cluster_dropped = []
        elif len(pool) == 1:
            selected_candidates = list(pool)
            cluster_dropped = []
        else:
            assert corre_df is not None
            corr_np = np.array(corre_df.select(pool).to_numpy(), copy=True)
            np.fill_diagonal(corr_np, 1.0)
            n_feat = len(pool)

            dist_matrix = np.sqrt(np.clip((1.0 - corr_np) / 2.0, 0.0, 1.0))
            np.fill_diagonal(dist_matrix, 0.0)
            condensed_dist = squareform(dist_matrix, checks=False)
            z = linkage(condensed_dist, method="ward")

            dist_threshold = np.sqrt(max((1.0 - profile.max_correlation) / 2.0, 0.0))
            cluster_ids = fcluster(z, t=dist_threshold, criterion="distance")
            num_clusters = len(np.unique(cluster_ids))

            if num_clusters > target_max_candidates and n_feat > target_max_candidates:
                cluster_ids = fcluster(z, t=target_max_candidates, criterion="maxclust")
                num_clusters = len(np.unique(cluster_ids))

            cluster_members = []
            for cid in sorted(np.unique(cluster_ids)):
                members = [pool[idx] for idx, c in enumerate(cluster_ids) if c == cid]
                members_sorted = sorted(members, key=lambda f: pool.index(f))
                cluster_members.append(members_sorted)

            cluster_selected = [m[0] for m in cluster_members]
            if len(cluster_selected) < target_min_candidates and len(pool) > len(cluster_selected):
                remaining_candidates = []
                for members in cluster_members:
                    remaining_candidates.extend(members[1:])
                remaining_candidates.sort(key=lambda f: pool.index(f))
                deficit = min(
                    target_max_candidates - len(cluster_selected),
                    len(remaining_candidates),
                )
                cluster_selected.extend(remaining_candidates[:deficit])

            cluster_dropped = [f for f in pool if f not in set(cluster_selected)]
            selected_candidates = cluster_selected

            selected_candidates, vif_dropped = prune_by_vif(
                selected_candidates, corre_df, max_vif=profile.max_vif
            )
            all_dedup_dropped = [f for f in pool if f not in set(selected_candidates)]
            if all_dedup_dropped:
                filter_drops["Correlation Filter Dropped"] = all_dedup_dropped

    if retained_anchors:
        for a in retained_anchors:
            if (
                a not in selected_candidates
                and a not in stream_mandatory
                and (not profile.feature_blacklist or a not in set(profile.feature_blacklist))
            ):
                selected_candidates.append(a)

    final_stream_features = selected_candidates + stream_mandatory

    # Fail-Fast Minimum Cluster Count Check
    total_count = len(final_stream_features)

    min_required = min(raw_universe_size + len(stream_mandatory), profile.min_clusters)
    if total_count < min_required:
        raise ValueError(
            f"Stream {profile.name} yielded {total_count} features, which is below "
            f"the configured minimum cluster count {profile.min_clusters}"
        )

    dropped_counts = {
        k: len(v) for k, v in filter_drops.items() if k.endswith("Dropped")
    }

    tier_breakdown: dict[str, dict[str, Any]] = {}
    for t_name in ("micro", "meso", "macro"):
        t_sel = [f for f in final_stream_features if classify_feature_scale(f) == t_name]
        t_cfg = SCALE_TIER_CONFIGS[t_name]
        t_diag = (pred_tier_diagnostics or {}).get(t_name, {})
        t_q = STREAM_TIER_QUOTAS.get(profile.name, {}).get(t_name)
        q_min = t_q.min_quota if t_q else 0
        q_max = t_q.max_quota if t_q else len(t_sel)
        tier_breakdown[t_name] = {
            "forward_horizons": t_diag.get("forward_horizons", list(t_cfg.forward_horizons)),
            "decision_horizon": t_diag.get("decision_horizon", t_cfg.decision_horizon),
            "thresholds": t_diag.get("thresholds", {
                "min_abs_ic": t_cfg.min_abs_ic,
                "sign_consistency": t_cfg.min_sign_consistency,
                "stability_ir": t_cfg.min_rank_ic_ir,
            }),
            "candidates": t_diag.get("candidates", 0),
            "anti_causal_dropped": t_diag.get("anti_causal_dropped", 0),
            "rank_ic_dropped": t_diag.get("rank_ic_dropped", 0),
            "sign_consistency_dropped": t_diag.get("sign_consistency_dropped", 0),
            "stability_dropped": t_diag.get("stability_dropped", 0),
            "survivors": t_diag.get("survivors", 0),
            "selected_quota": {"min": q_min, "max": q_max},
            "selected": len(t_sel),
            "features": t_sel,
        }

    audit_record = StreamAuditRecord(
        profile_name=profile.name,
        selected_features=final_stream_features,
        selected_feature_count=len(final_stream_features),
        filter_results=filter_drops,
        candidate_count=len(candidate_features),
        dropped_counts=dropped_counts,
        tier_breakdown=tier_breakdown,
    )
    return final_stream_features, filter_drops, audit_record


def _run_triple_stream_train_stage(
    io: PipelineIOManager,
    frames: dict[str, pl.DataFrame],
    raw_universe: list[str],
    config: FeatureSelectionPipelineConfig,
) -> FeatureSelectionResult:
    mandatory_features = list(config.mandatory_state_features)
    blacklist_set = set(config.hygiene.feature_blacklist)
    blacklisted_mandatory: list[str] = []
    if blacklist_set and mandatory_features:
        blacklisted_mandatory = sorted(blacklist_set.intersection(mandatory_features))
        if blacklisted_mandatory:
            mandatory_features = [
                f for f in mandatory_features if f not in blacklist_set
            ]

    for contract, frame in frames.items():
        missing = [feature for feature in raw_universe if feature not in frame.columns]
        if missing:
            raise ValueError(
                f"contract {contract} is missing required feature columns: {missing}"
            )
        io.validate_contract_frame(
            frame, contract=contract, feature_universe=raw_universe
        )

    candidate_universe = [f for f in raw_universe if f not in mandatory_features]

    # 1.1 Shared Data Hygiene
    cleaned_frames, hygiene_res = execute_data_hygiene(
        frames, candidate_universe, config.hygiene
    )
    if (
        config.hygiene.feature_blacklist
        and not hygiene_res.surviving_features
        and not mandatory_features
    ):
        raise ValueError(
            "feature selection produced an empty list after Feature Blacklist"
        )
    if not hygiene_res.surviving_features and not mandatory_features:
        raise ValueError(f"{config.stage} feature universe is empty")
    candidate_universe = hygiene_res.surviving_features

    # 1.2 Shared Distribution Drift Audit (Relaxed envelope across VAE Slope, VAE Vol, and RL)
    outpost_frame = io.load_validation_outpost_frame(candidate_universe)
    shared_drift_config = DistributionAuditConfig(
        num_bins=config.drift.num_bins,
        max_mean_psi=max(
            config.drift.max_mean_psi,
            config.rl_profile.max_mean_psi,
            config.vae_slope_profile.max_mean_psi,
            config.vae_volatility_profile.max_mean_psi,
        ),
        max_pair_psi=max(
            config.drift.max_pair_psi,
            config.rl_profile.max_pair_psi,
            config.vae_slope_profile.max_pair_psi,
            config.vae_volatility_profile.max_pair_psi,
        ),
        min_drift_survivors=config.drift.min_drift_survivors,
        forward_outpost_max_psi=config.drift.forward_outpost_max_psi,
    )
    dist_res = audit_distribution_drift(
        cleaned_frames,
        candidate_universe,
        config=shared_drift_config,
        forward_outpost_frame=outpost_frame,
    )
    dist_path = io.output_dir / "distribution_audit_metrics.csv"
    dist_res.metrics_df.write_csv(dist_path)

    # 1.3 Stationarity Diagnostics (Shared half-life calculation)
    persistence_diagnostics: list[PersistenceDiagnostic] | None = None
    if config.stationarity.min_half_life_bars > 0.0:
        stat_res = execute_stationarity_audit(
            cleaned_frames,
            dist_res.surviving_features,
            config=config.stationarity,
        )
        persistence_diagnostics = stat_res.diagnostics["persistence_diagnostics"]

    # 1.4 Shared Vectorized Predictive Audit (Relaxed envelope)
    shared_predictive_config = PredictiveAuditConfig(
        min_abs_ic=min(
            config.predictive.min_abs_ic,
            config.vae_slope_profile.min_abs_ic,
            config.vae_volatility_profile.min_abs_ic,
            config.rl_profile.min_abs_ic,
        ),
        min_sign_consistency=min(
            config.predictive.min_sign_consistency,
            config.vae_slope_profile.min_sign_consistency,
            config.vae_volatility_profile.min_sign_consistency,
            config.rl_profile.min_sign_consistency,
        ),
        min_rank_ic_ir=min(
            config.predictive.min_rank_ic_ir,
            config.vae_slope_profile.min_rank_ic_ir,
            config.vae_volatility_profile.min_rank_ic_ir,
            config.rl_profile.min_rank_ic_ir,
        ),
        target_decision_window=config.predictive.target_decision_window,
        windows_list=config.predictive.windows_list,
        fdr_threshold=config.predictive.fdr_threshold,
        ic_anomaly_ceiling=config.predictive.ic_anomaly_ceiling,
        rank_ic_mode=config.predictive.rank_ic_mode,
        max_metric_std=config.predictive.max_metric_std,
    )
    pred_df, pred_res = execute_predictive_audit(
        cleaned_frames,
        dist_res.surviving_features,
        config=shared_predictive_config,
    )
    aggregate_path = io.output_dir / "aggregate_metrics.csv"
    pred_df.write_csv(aggregate_path)

    contract_metric_frames = pred_res.diagnostics["metric_frames"]
    per_contract_dir = io.output_dir / "per_contract"
    per_contract_dir.mkdir(parents=True, exist_ok=True)
    per_contract_records: list[FeatureSelectionContractRecord] = []
    for contract, metric_frame in zip(frames.keys(), contract_metric_frames):
        metric_path = per_contract_dir / f"{contract}_metrics.csv"
        metric_frame.write_csv(metric_path)
        per_contract_records.append(
            FeatureSelectionContractRecord(
                contract=contract,
                input_path=str(io.input_dir / f"{contract}.feather"),
                metric_path=str(metric_path),
            )
        )

    # 1.5 Shared Nonlinear Importance Scoring
    mean_psi_by_feature = dict(
        zip(dist_res.metrics_df["feature"].to_list(), dist_res.metrics_df["mean_psi"].to_list())
    )
    scoring_candidates = pred_res.surviving_features
    if scoring_candidates:
        _, scored_df, scoring_res = execute_nonlinear_scoring(
            cleaned_frames,
            scoring_candidates,
            pred_df,
            mean_psi_by_feature,
            config=config.scoring,
        )
        catboost_mean_importance = scoring_res.diagnostics["catboost_mean_importance"]
    else:
        scored_df = pred_df
        catboost_mean_importance = {}

    sign_consistency_map = pred_res.diagnostics.get("sign_consistency_map", {})
    vol_sign_consistency_map = pred_res.diagnostics.get("vol_sign_consistency_map", {})

    # 1.6 Shared Multi-Regime Audit
    regime_bins = config.regime.regime_bins
    regime_quantiles = compute_regime_quantiles(cleaned_frames, num_bins=regime_bins)
    if config.regime.target_regime_bins is not None:
        effective_target_bins = list(config.regime.target_regime_bins)
    else:
        num_s = len(regime_quantiles["slope"]) + 1
        num_v = len(regime_quantiles["volatility"]) + 1
        effective_target_bins = default_target_regime_bins(num_s, num_v)

    regime_audit_df, retained_anchors, retention_details = audit_regimes(
        cleaned_frames,
        candidate_universe,
        regime_quantiles,
        list(config.predictive.windows_list),
        target_regime_bins=effective_target_bins,
        min_abs_ic=config.predictive.min_abs_ic,
        enable_conditional_anchors=config.regime.enable_conditional_anchors,
    )
    regime_audit_path = io.output_dir / "regime_audit_metrics.csv"
    regime_audit_df.write_csv(regime_audit_path)

    rl_anchors: list[str] = []
    if config.regime.enable_conditional_anchors and retained_anchors:
        rl_blacklist_set = set(config.rl_profile.feature_blacklist)
        rl_anchors = [
            a
            for a in retained_anchors
            if a in raw_universe and a not in blacklist_set and a not in rl_blacklist_set
        ]

    pred_tier_diagnostics = pred_res.diagnostics.get("tier_breakdown")

    # Branch A: Slope VAE Regime Stream Evaluation
    vae_slope_selected, vae_slope_filter_drops, vae_slope_audit = _evaluate_stream_branch(
        profile=config.vae_slope_profile,
        candidate_features=pred_res.surviving_features,
        aggregate_metrics_df=scored_df,
        dist_metrics_df=dist_res.metrics_df,
        frames=cleaned_frames,
        mandatory_features=mandatory_features,
        persistence_diagnostics=persistence_diagnostics,
        min_half_life_bars=float(config.stationarity.min_half_life_bars or 1.0),
        active_persistence_pattern=config.persistence_filter_pattern,
        raw_universe_size=len(raw_universe),
        catboost_mean_importance=catboost_mean_importance,
        sign_consistency_map=sign_consistency_map,
        retained_anchors=[],
        vol_sign_consistency_map=vol_sign_consistency_map,
        pred_tier_diagnostics=pred_tier_diagnostics,
    )

    # Branch B: Volatility VAE Regime Stream Evaluation
    vae_vol_selected, vae_vol_filter_drops, vae_vol_audit = _evaluate_stream_branch(
        profile=config.vae_volatility_profile,
        candidate_features=pred_res.surviving_features,
        aggregate_metrics_df=scored_df,
        dist_metrics_df=dist_res.metrics_df,
        frames=cleaned_frames,
        mandatory_features=mandatory_features,
        persistence_diagnostics=persistence_diagnostics,
        min_half_life_bars=float(config.stationarity.min_half_life_bars),
        active_persistence_pattern=config.persistence_filter_pattern,
        raw_universe_size=len(raw_universe),
        catboost_mean_importance=catboost_mean_importance,
        sign_consistency_map=sign_consistency_map,
        retained_anchors=[],
        vol_sign_consistency_map=vol_sign_consistency_map,
        pred_tier_diagnostics=pred_tier_diagnostics,
    )

    # Branch C: RL Decision Stream Evaluation
    rl_selected, rl_filter_drops, rl_audit = _evaluate_stream_branch(
        profile=config.rl_profile,
        candidate_features=pred_res.surviving_features,
        aggregate_metrics_df=scored_df,
        dist_metrics_df=dist_res.metrics_df,
        frames=cleaned_frames,
        mandatory_features=mandatory_features,
        persistence_diagnostics=persistence_diagnostics,
        min_half_life_bars=float(config.stationarity.min_half_life_bars),
        active_persistence_pattern=config.persistence_filter_pattern,
        raw_universe_size=len(raw_universe),
        catboost_mean_importance=catboost_mean_importance,
        sign_consistency_map=sign_consistency_map,
        retained_anchors=rl_anchors,
        vol_sign_consistency_map=vol_sign_consistency_map,
        pred_tier_diagnostics=pred_tier_diagnostics,
    )

    # Mathematical Union: S_union = S_vae_slope U S_vae_vol U S_rl
    union_set = set(vae_slope_selected).union(vae_vol_selected).union(rl_selected)
    union_candidates = [
        f for f in rl_selected if f not in mandatory_features
    ] + [
        f
        for f in vae_slope_selected
        if f not in mandatory_features and f not in set(rl_selected)
    ] + [
        f
        for f in vae_vol_selected
        if f not in mandatory_features and f not in set(rl_selected) and f not in set(vae_slope_selected)
    ]
    union_mandatory = [f for f in mandatory_features if f in union_set]
    final_selected = union_candidates + union_mandatory
    assert set(final_selected) == union_set

    # Persist triple-stream artifacts and filtered contract datasets
    vae_slope_file, vae_vol_file, rl_file = io.save_triple_stream_features(
        vae_slope_features=vae_slope_selected,
        vae_volatility_features=vae_vol_selected,
        rl_features=rl_selected,
    )
    filtered_outputs = io.write_filtered_outputs(frames, final_selected)

    # Shared filter results summary
    shared_filter_results: dict[str, list[str]] = {}
    all_blacklisted = sorted(
        set(hygiene_res.diagnostics["blacklist_dropped"]).union(blacklisted_mandatory)
    )
    if all_blacklisted:
        shared_filter_results["Feature Blacklist Dropped"] = all_blacklisted
    if hygiene_res.diagnostics["ablation_dropped"]:
        shared_filter_results["Feature Ablation Dropped"] = hygiene_res.diagnostics[
            "ablation_dropped"
        ]
    if dist_res.dropped_features:
        shared_filter_results["Distribution Drift Dropped"] = dist_res.dropped_features
    shared_filter_results["Hard Filter Survivors"] = pred_res.surviving_features

    manifest = FeatureSelectionManifest(
        symbol=config.symbol,
        target_freq=config.target_freq,
        stage=config.stage,
        split_input_dir=str(io.input_dir),
        rl_feature_file=str(rl_file),
        vae_slope_feature_file=str(vae_slope_file),
        vae_volatility_feature_file=str(vae_vol_file),
        union_selected_feature_count=len(final_selected),
        union_selected_features=final_selected,
        stream_mode="triple",
        vae_slope_stream=vae_slope_audit,
        vae_volatility_stream=vae_vol_audit,
        rl_stream=rl_audit,
        windows_list=list(config.predictive.windows_list),
        composite_drop_ratio=config.scoring.composite_drop_ratio,
        global_feature_blacklist=(
            list(config.hygiene.feature_blacklist)
            if config.hygiene.feature_blacklist
            else None
        ),
        feature_ablation_patterns=list(config.hygiene.feature_ablation_patterns),
        rank_ic_mode=config.predictive.rank_ic_mode,
        mandatory_state_features=(
            mandatory_features if mandatory_features else None
        ),
        persistence_filter=(
            {
                "min_half_life_bars": float(config.stationarity.min_half_life_bars),
                "active_feature_pattern": config.persistence_filter_pattern,
            }
            if config.stationarity.min_half_life_bars > 0.0
            else None
        ),
        persistence_diagnostics=persistence_diagnostics,
        aggregate_metrics_path=str(aggregate_path),
        shared_filter_results=shared_filter_results,
        contracts=per_contract_records,
        filtered_outputs=filtered_outputs,
        regime_bins=config.regime.regime_bins,
        target_regime_bins=effective_target_bins,
        regime_quantiles=regime_quantiles,
        regime_audit_path=str(regime_audit_path),
        distribution_audit_path=str(dist_path),
        global_max_mean_psi=config.drift.max_mean_psi,
        global_max_pair_psi=config.drift.max_pair_psi,
        min_drift_survivors=config.drift.min_drift_survivors,
        global_min_sign_consistency=config.predictive.min_sign_consistency,
        conditional_anchors_retained=retention_details if retention_details else None,
        process_documentation=_build_process_documentation(),
    )
    io.save_manifest(manifest)
    return FeatureSelectionResult(output_dir=io.output_dir, manifest=manifest)


def run_feature_selection(
    config: FeatureSelectionPipelineConfig | None = None,
    **kwargs,
) -> FeatureSelectionResult:
    if config is None:
        config = _build_config_from_legacy_kwargs(**kwargs)

    io = PipelineIOManager(config)
    frames = io.load_stage_frames()
    raw_universe = io.resolve_initial_feature_universe(frames)

    # 1. Validation Stage Short-Circuit
    if config.stage == "valid":
        return _run_validation_stage(io, frames, raw_universe, config)

    return _run_triple_stream_train_stage(io, frames, raw_universe, config)


def _run_validation_stage(
    io: PipelineIOManager,
    frames: dict[str, pl.DataFrame],
    raw_universe: list[str],
    config: FeatureSelectionPipelineConfig,
) -> FeatureSelectionResult:
    mandatory_features = list(config.mandatory_state_features)
    feature_universe = raw_universe + mandatory_features

    for contract, frame in frames.items():
        missing = [feature for feature in feature_universe if feature not in frame.columns]
        if missing:
            raise ValueError(
                f"contract {contract} is missing required feature columns: {missing}"
            )
        io.validate_contract_frame(
            frame, contract=contract, feature_universe=feature_universe
        )

    dist_res = audit_distribution_drift(
        frames, feature_universe, config=config.drift
    )
    dist_path = io.output_dir / "distribution_audit_metrics.csv"
    dist_res.metrics_df.write_csv(dist_path)

    per_contract_dir = io.output_dir / "per_contract"
    per_contract_dir.mkdir(parents=True, exist_ok=True)
    metric_frames: list[pl.DataFrame] = []
    per_contract_records: list[FeatureSelectionContractRecord] = []

    for contract, frame in frames.items():
        metrics = calculate_metric_frame(
            frame,
            raw_universe,
            windows_list=list(config.predictive.windows_list),
            compute_catboost=False,
        )
        metric_path = per_contract_dir / f"{contract}_metrics.csv"
        metrics.write_csv(metric_path)
        metric_frames.append(metrics)
        per_contract_records.append(
            FeatureSelectionContractRecord(
                contract=contract,
                input_path=str(io.input_dir / f"{contract}.feather"),
                metric_path=str(metric_path),
            )
        )

    aggregate = aggregate_metric_frames(metric_frames)
    aggregate_path = io.output_dir / "aggregate_metrics.csv"
    aggregate.write_csv(aggregate_path)

    train_manifest_path = (
        Path(config.root_path)
        / config.save_path
        / config.target_freq
        / config.symbol
        / "train"
        / "feature_selection_manifest.json"
    )
    regime_quantiles = None
    regime_bins = config.regime.regime_bins
    effective_target_bins = (
        list(config.regime.target_regime_bins)
        if config.regime.target_regime_bins is not None
        else None
    )

    if train_manifest_path.exists():
        try:
            train_manifest_data = json.loads(
                train_manifest_path.read_text(encoding="utf-8")
            )
            regime_quantiles = train_manifest_data.get("regime_quantiles")
            if train_manifest_data.get("regime_bins") is not None:
                regime_bins = int(train_manifest_data["regime_bins"])
            if (
                effective_target_bins is None
                and train_manifest_data.get("target_regime_bins") is not None
            ):
                effective_target_bins = [
                    (int(item[0]), int(item[1]))
                    for item in train_manifest_data["target_regime_bins"]
                ]
        except Exception:
            pass

    if regime_quantiles is None:
        regime_quantiles = compute_regime_quantiles(frames, num_bins=regime_bins)
    if effective_target_bins is None:
        num_s = len(regime_quantiles["slope"]) + 1
        num_v = len(regime_quantiles["volatility"]) + 1
        effective_target_bins = default_target_regime_bins(num_s, num_v)

    regime_audit_df, _, retention_details = audit_regimes(
        frames,
        raw_universe,
        regime_quantiles,
        list(config.predictive.windows_list),
        target_regime_bins=effective_target_bins,
        min_abs_ic=config.predictive.min_abs_ic,
        enable_conditional_anchors=config.regime.enable_conditional_anchors,
    )
    regime_audit_path = io.output_dir / "regime_audit_metrics.csv"
    regime_audit_df.write_csv(regime_audit_path)

    manifest = FeatureSelectionManifest(
        symbol=config.symbol,
        target_freq=config.target_freq,
        stage=config.stage,
        split_input_dir=str(io.input_dir),
        evaluated_feature_file=str(
            Path(config.root_path)
            / config.save_path
            / config.target_freq
            / config.symbol
            / "train"
            / "rl_state_features.npy"
        ),
        evaluated_feature_count=len(feature_universe),
        evaluated_features=feature_universe,
        windows_list=list(config.predictive.windows_list),
        aggregate_metrics_path=str(aggregate_path),
        contracts=per_contract_records,
        feature_ablation_patterns=list(config.hygiene.feature_ablation_patterns),
        rank_ic_mode=config.predictive.rank_ic_mode,
        report_only=True,
        regime_bins=regime_bins,
        target_regime_bins=effective_target_bins,
        regime_quantiles=regime_quantiles,
        regime_audit_path=str(regime_audit_path),
        distribution_audit_path=str(dist_path),
        global_max_mean_psi=config.drift.max_mean_psi,
        global_max_pair_psi=config.drift.max_pair_psi,
        min_drift_survivors=config.drift.min_drift_survivors,
        global_min_sign_consistency=config.predictive.min_sign_consistency,
        conditional_anchors_retained=retention_details if retention_details else None,
    )
    io.save_manifest(manifest)
    return FeatureSelectionResult(output_dir=io.output_dir, manifest=manifest)


def _parse_windows_list(value: list[int | str] | str | None) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = [value]
    result: list[int] = []
    for item in value:
        if isinstance(item, int):
            result.append(item)
        elif isinstance(item, str):
            for part in item.split(","):
                part = part.strip()
                if part:
                    result.append(int(part))
    return result if result else None


def _parse_target_regime_bins(
    value: list[tuple[int, int]] | list[str] | str | None,
) -> list[tuple[int, int]] | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = [value]
    result: list[tuple[int, int]] = []
    for item in value:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            result.append((int(item[0]), int(item[1])))
        elif isinstance(item, str):
            cleaned = item.strip("()[] ")
            for part in cleaned.split():
                part = part.strip("()[] ,")
                if not part:
                    continue
                if "," in part:
                    s, v = part.split(",", 1)
                    result.append((int(s.strip()), int(v.strip())))
    return result if result else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=Path, default=Path("."))
    parser.add_argument(
        "--split_path",
        type=str,
        default="PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION",
    )
    parser.add_argument("--symbol", "--symbols", dest="symbol", type=str, required=True)
    parser.add_argument("--target_freq", type=str, required=True)
    parser.add_argument("--stage", choices=["train", "valid"], required=True)
    parser.add_argument("--orderbook_depth", type=int, default=5)
    parser.add_argument("--min_abs_ic", type=float, default=0.02)
    parser.add_argument("--max_metric_std", type=float, default=1.0)
    parser.add_argument("--max_correlation", type=float, default=0.7)
    parser.add_argument("--min_rank_ic_ir", type=float, default=0.40)
    parser.add_argument("--min_sign_consistency", type=float, default=0.75)
    parser.add_argument("--target_decision_window", type=int, default=None)
    parser.add_argument("--composite_drop_ratio", type=float, default=0.1)
    parser.add_argument("--feature_blacklist", nargs="*", default=None)
    parser.add_argument(
        "--feature_ablation_patterns",
        nargs="*",
        default=[],
    )
    parser.add_argument(
        "--mandatory_state_features",
        nargs="*",
        default=None,
    )
    parser.add_argument("--min_half_life_bars", type=float, default=0.0)
    parser.add_argument(
        "--persistence_filter_pattern",
        type=str,
        default=DEFAULT_PERSISTENCE_FILTER_PATTERN,
    )
    parser.add_argument(
        "--rank_ic_mode",
        choices=["absolute", "signed"],
        default="absolute",
    )
    parser.add_argument(
        "--enable_conditional_anchors",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--disable_conditional_anchors",
        action="store_false",
        dest="enable_conditional_anchors",
    )
    parser.add_argument(
        "--windows_list",
        "--windows",
        dest="windows_list",
        nargs="*",
        default=None,
    )
    parser.add_argument(
        "--regime_bins",
        "--num_regime_bins",
        dest="regime_bins",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--target_regime_bins",
        nargs="*",
        default=None,
    )

    parser.add_argument(
        "--max_mean_psi",
        type=float,
        default=DEFAULT_MAX_MEAN_PSI,
    )
    parser.add_argument(
        "--max_pair_psi",
        type=float,
        default=DEFAULT_MAX_PAIR_PSI,
    )
    parser.add_argument(
        "--min_drift_survivors",
        type=int,
        default=DEFAULT_MIN_DRIFT_SURVIVORS,
    )
    parser.add_argument(
        "--distribution_num_bins",
        type=int,
        default=DEFAULT_DISTRIBUTION_NUM_BINS,
    )
    parser.add_argument(
        "--dedup_method",
        choices=["cluster", "greedy"],
        default="cluster",
    )
    parser.add_argument(
        "--max_vif",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--fdr_threshold",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--vae_slope_feature_blacklist",
        nargs="*",
        default=None,
        help="Stream-specific feature blacklist for Slope VAE regime stream.",
    )
    parser.add_argument(
        "--vae_volatility_feature_blacklist",
        nargs="*",
        default=None,
        help="Stream-specific feature blacklist for Volatility VAE regime stream.",
    )
    parser.add_argument(
        "--rl_feature_blacklist",
        nargs="*",
        default=None,
        help="Stream-specific feature blacklist for RL decision stream.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_feature_selection(
        root_path=args.root_path,
        split_path=args.split_path,
        save_path=args.save_path,
        symbol=args.symbol,
        target_freq=args.target_freq,
        stage=args.stage,
        orderbook_depth=args.orderbook_depth,
        min_abs_ic=args.min_abs_ic,
        max_metric_std=args.max_metric_std,
        max_correlation=args.max_correlation,
        min_rank_ic_ir=args.min_rank_ic_ir,
        min_sign_consistency=args.min_sign_consistency,
        target_decision_window=args.target_decision_window,
        windows_list=_parse_windows_list(args.windows_list),
        composite_drop_ratio=args.composite_drop_ratio,
        feature_blacklist=args.feature_blacklist,
        feature_ablation_patterns=args.feature_ablation_patterns,
        mandatory_state_features=args.mandatory_state_features,
        min_half_life_bars=args.min_half_life_bars,
        persistence_filter_pattern=args.persistence_filter_pattern,
        rank_ic_mode=args.rank_ic_mode,
        enable_conditional_anchors=args.enable_conditional_anchors,
        regime_bins=args.regime_bins,
        target_regime_bins=_parse_target_regime_bins(args.target_regime_bins),
        max_mean_psi=args.max_mean_psi,
        max_pair_psi=args.max_pair_psi,
        min_drift_survivors=args.min_drift_survivors,
        distribution_num_bins=args.distribution_num_bins,
        dedup_method=args.dedup_method,
        max_vif=args.max_vif,
        fdr_threshold=args.fdr_threshold,
        vae_slope_feature_blacklist=args.vae_slope_feature_blacklist,
        vae_volatility_feature_blacklist=args.vae_volatility_feature_blacklist,
        rl_feature_blacklist=args.rl_feature_blacklist,
    )
