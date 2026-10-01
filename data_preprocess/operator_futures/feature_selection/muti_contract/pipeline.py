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
    DEFAULT_VAE_PROFILE,
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

    vae_profile = (
        replace(
            kwargs.get("vae_profile", DEFAULT_VAE_PROFILE),
            feature_blacklist=tuple(kwargs["vae_feature_blacklist"]),
        )
        if kwargs.get("vae_feature_blacklist") is not None
        else kwargs.get("vae_profile", DEFAULT_VAE_PROFILE)
    )
    rl_profile = (
        replace(
            kwargs.get("rl_profile", DEFAULT_RL_PROFILE),
            feature_blacklist=tuple(kwargs["rl_feature_blacklist"]),
        )
        if kwargs.get("rl_feature_blacklist") is not None
        else kwargs.get("rl_profile", DEFAULT_RL_PROFILE)
    )

    raw_blacklist = list(kwargs.get("feature_blacklist") or ())
    vae_bl = (
        kwargs.get("vae_feature_blacklist")
        if kwargs.get("vae_feature_blacklist") is not None
        else (vae_profile.feature_blacklist if vae_profile.feature_blacklist else None)
    )
    rl_bl = (
        kwargs.get("rl_feature_blacklist")
        if kwargs.get("rl_feature_blacklist") is not None
        else (rl_profile.feature_blacklist if rl_profile.feature_blacklist else None)
    )

    if vae_bl is not None and rl_bl is not None:
        effective_hygiene_blacklist = tuple(
            sorted(set(raw_blacklist).union(set(vae_bl).intersection(set(rl_bl))))
        )
    elif vae_bl is not None:
        effective_hygiene_blacklist = tuple(
            sorted(set(raw_blacklist).union(set(vae_bl)))
        )
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
        dual_stream=bool(kwargs.get("dual_stream", False)),
        vae_profile=vae_profile,
        rl_profile=rl_profile,
    )



def _evaluate_stream_branch(
    profile: StreamFilterProfile,
    candidate_features: list[str],
    aggregate_metrics_df: pl.DataFrame,
    dist_metrics_df: pl.DataFrame,
    frames: dict[str, pl.DataFrame],
    mandatory_features: list[str],
    persistence_diagnostics: list[PersistenceDiagnostic],
    min_half_life_bars: float,
    active_persistence_pattern: str,
    raw_universe_size: int,
    catboost_mean_importance: dict[str, float],
    sign_consistency_map: dict[str, float],
    retained_anchors: list[str],
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
    rank_ic_mean_map = dict(
        zip(aggregate_metrics_df["feature"].to_list(), aggregate_metrics_df["RankIC_Mean"].to_list())
    )
    rank_ic_std_map = dict(
        zip(aggregate_metrics_df["feature"].to_list(), aggregate_metrics_df["RankIC_Std"].to_list())
    )
    ic_std_map = dict(
        zip(aggregate_metrics_df["feature"].to_list(), aggregate_metrics_df["IC_Std"].to_list())
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

    # (a) Distribution Drift Gating
    psi_dropped: list[str] = []
    psi_surviving: list[str] = []
    for f in pool:
        mean_psi = mean_psi_map.get(f, 0.0)
        max_pair_psi = max_pair_psi_map.get(f, 0.0)
        if mean_psi <= profile.max_mean_psi and max_pair_psi <= profile.max_pair_psi:
            psi_surviving.append(f)
        else:
            psi_dropped.append(f)
    pool = psi_surviving
    if psi_dropped:
        filter_drops["Distribution Drift Dropped"] = psi_dropped

    # (b) Persistence Noise Gating
    if profile.filter_micro_persistence and min_half_life_bars > 0.0:
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
        if abs(rank_ic_mean_map.get(f, 0.0)) >= profile.min_abs_ic:
            hard_surviving.append(f)
        else:
            hard_dropped.append(f)
    pool = hard_surviving
    if hard_dropped:
        filter_drops["Hard Filter Dropped"] = hard_dropped

    sc_dropped: list[str] = []
    sc_surviving: list[str] = []
    for f in pool:
        if sign_consistency_map.get(f, 1.0) >= profile.min_sign_consistency:
            sc_surviving.append(f)
        else:
            sc_dropped.append(f)
    pool = sc_surviving
    if sc_dropped:
        filter_drops["Sign Consistency Filter Dropped"] = sc_dropped

    stab_dropped: list[str] = []
    stab_surviving: list[str] = []
    for f in pool:
        mean_r = abs(rank_ic_mean_map.get(f, 0.0))
        std_r = rank_ic_std_map.get(f, 0.0)
        ir = mean_r / (std_r + 1e-6)
        if ir >= profile.min_rank_ic_ir:
            stab_surviving.append(f)
        else:
            stab_dropped.append(f)
    pool = stab_surviving
    if stab_dropped:
        filter_drops["Stability Filter Dropped"] = stab_dropped

    # (d) Composite Priority Scoring
    if pool:
        height = float(len(pool))
        inv_psi_vals = np.array([1.0 / (float(mean_psi_map.get(f, 0.0)) + 1e-4) for f in pool])
        rank_ic_vals = np.array([abs(float(rank_ic_mean_map.get(f, 0.0))) for f in pool])
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

    # (e) Orthogonal Deduplication & Clustering
    target_max_candidates = max(profile.max_clusters - len(stream_mandatory), 1)
    target_min_candidates = max(profile.min_clusters - len(stream_mandatory), 1)

    if not pool:
        selected_candidates: list[str] = []
        cluster_dropped: list[str] = []
    else:
        corre_df = compute_contract_normalized_spearman_correlation_matrix(frames, pool)
        corr_np = corre_df.select(pool).to_numpy()
        np.fill_diagonal(corr_np, 1.0)
        n_feat = len(pool)

        if n_feat == 1:
            selected_candidates = list(pool)
            cluster_dropped = []
        else:
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
            selected_candidates, corre_df, max_vif=10.0
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

    # Fail-Fast Minimum Cluster Count Check (User Story 18)
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
    audit_record = StreamAuditRecord(
        profile_name=profile.name,
        selected_features=final_stream_features,
        selected_feature_count=len(final_stream_features),
        filter_results=filter_drops,
        candidate_count=len(candidate_features),
        dropped_counts=dropped_counts,
    )
    return final_stream_features, filter_drops, audit_record


def _run_dual_stream_train_stage(
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

    # 1.2 Shared Distribution Drift Audit (Relaxed envelope across VAE and RL)
    outpost_frame = io.load_validation_outpost_frame(candidate_universe)
    shared_drift_config = DistributionAuditConfig(
        num_bins=config.drift.num_bins,
        max_mean_psi=max(
            config.drift.max_mean_psi,
            config.rl_profile.max_mean_psi,
            config.vae_profile.max_mean_psi,
        ),
        max_pair_psi=max(
            config.drift.max_pair_psi,
            config.rl_profile.max_pair_psi,
            config.vae_profile.max_pair_psi,
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
    candidate_universe = dist_res.surviving_features

    # 1.3 Shared Stationarity Audit (ADF and SAR evaluated; persistence diagnostics gathered)
    shared_stationarity_config = StationarityAuditConfig(
        adf_significance_level=config.stationarity.adf_significance_level,
        min_passing_contract_ratio=config.stationarity.min_passing_contract_ratio,
        fallback_significance_level=config.stationarity.fallback_significance_level,
        min_survivors_floor=config.stationarity.min_survivors_floor,
        min_half_life_bars=0.0,
        max_sign_alternation_rate=1.0,
        active_feature_pattern=config.persistence_filter_pattern,
    )
    stat_res = execute_stationarity_audit(
        cleaned_frames, candidate_universe, shared_stationarity_config
    )
    if not stat_res.surviving_features and not mandatory_features:
        raise ValueError(
            "feature selection produced an empty list after Stationarity Audit"
        )
    candidate_universe = stat_res.surviving_features
    persistence_diagnostics = stat_res.diagnostics["persistence_diagnostics"]

    # 1.4 Shared Vectorized Predictive Audit (Relaxed envelope)
    shared_predictive_config = PredictiveAuditConfig(
        min_abs_ic=min(
            config.predictive.min_abs_ic,
            config.vae_profile.min_abs_ic,
            config.rl_profile.min_abs_ic,
        ),
        min_sign_consistency=min(
            config.predictive.min_sign_consistency,
            config.vae_profile.min_sign_consistency,
            config.rl_profile.min_sign_consistency,
        ),
        min_rank_ic_ir=min(
            config.predictive.min_rank_ic_ir,
            config.vae_profile.min_rank_ic_ir,
            config.rl_profile.min_rank_ic_ir,
        ),
        target_decision_window=config.predictive.target_decision_window,
        windows_list=config.predictive.windows_list,
        fdr_threshold=config.predictive.fdr_threshold,
        ic_anomaly_ceiling=config.predictive.ic_anomaly_ceiling,
        rank_ic_mode=config.predictive.rank_ic_mode,
        max_metric_std=config.predictive.max_metric_std,
    )
    aggregate_df, pred_res = execute_predictive_audit(
        cleaned_frames, candidate_universe, shared_predictive_config
    )
    if not pred_res.surviving_features and not mandatory_features:
        raise ValueError("feature selection produced an empty list after Hard Filter")

    per_contract_dir = io.output_dir / "per_contract"
    per_contract_dir.mkdir(parents=True, exist_ok=True)
    metric_frames: list[pl.DataFrame] = pred_res.diagnostics["metric_frames"]
    per_contract_records: list[FeatureSelectionContractRecord] = []
    for (contract, frame), mf in zip(cleaned_frames.items(), metric_frames):
        metric_path = per_contract_dir / f"{contract}_metrics.csv"
        mf.write_csv(metric_path)
        per_contract_records.append(
            FeatureSelectionContractRecord(
                contract=contract,
                input_path=str(io.input_dir / f"{contract}.feather"),
                metric_path=str(metric_path),
            )
        )
    aggregate_path = io.output_dir / "aggregate_metrics.csv"
    aggregate_df.write_csv(aggregate_path)

    # Stage 2: Single-pass CatBoost fitting on target decision window w=6
    scored_features, scored_df, score_res = execute_nonlinear_scoring(
        cleaned_frames,
        pred_res.surviving_features,
        aggregate_df,
        dist_res.mean_psi_by_feature,
        config.scoring,
        metric_frames=metric_frames,
    )
    for (contract, _), mf in zip(cleaned_frames.items(), metric_frames):
        metric_path = per_contract_dir / f"{contract}_metrics.csv"
        mf.write_csv(metric_path)
    scored_df.write_csv(aggregate_path)
    catboost_mean_importance = score_res.diagnostics["catboost_mean_importance"]
    sign_consistency_map = pred_res.diagnostics["sign_consistency_map"]

    # Regime Audit
    regime_quantiles = compute_regime_quantiles(
        cleaned_frames, num_bins=config.regime.regime_bins
    )
    effective_target_bins = (
        list(config.regime.target_regime_bins)
        if config.regime.target_regime_bins is not None
        else default_target_regime_bins(
            config.regime.regime_bins, config.regime.regime_bins
        )
    )
    regime_audit_df, retained_anchors, retention_details = audit_regimes(
        cleaned_frames,
        raw_universe,
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

    # Branch A: VAE Regime Stream Evaluation
    vae_selected, vae_filter_drops, vae_audit = _evaluate_stream_branch(
        profile=config.vae_profile,
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
    )

    # Branch B: RL Decision Stream Evaluation
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
    )

    # Mathematical Union: S_union = S_vae U S_rl
    union_set = set(vae_selected).union(rl_selected)
    union_candidates = [
        f for f in rl_selected if f not in mandatory_features
    ] + [
        f
        for f in vae_selected
        if f not in mandatory_features and f not in set(rl_selected)
    ]
    union_mandatory = [f for f in mandatory_features if f in union_set]
    final_selected = union_candidates + union_mandatory
    assert set(final_selected) == union_set

    # Persist dual-stream artifacts and filtered contract datasets
    vae_file, rl_file = io.save_dual_stream_features(
        vae_features=vae_selected,
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
    shared_filter_results["Hard Filter"] = pred_res.surviving_features

    manifest = FeatureSelectionManifest(
        symbol=config.symbol,
        target_freq=config.target_freq,
        stage=config.stage,
        split_input_dir=str(io.input_dir),
        rl_feature_file=str(rl_file),
        vae_feature_file=str(vae_file),
        selected_feature_count=len(final_selected),
        selected_features=final_selected,
        stream_mode="dual",
        vae_stream=vae_audit,
        rl_stream=rl_audit,
        windows_list=list(config.predictive.windows_list),
        composite_drop_ratio=config.scoring.composite_drop_ratio,
        feature_blacklist=(
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
        filter_results=shared_filter_results,
        contracts=per_contract_records,
        filtered_outputs=filtered_outputs,
        regime_bins=config.regime.regime_bins,
        target_regime_bins=effective_target_bins,
        regime_quantiles=regime_quantiles,
        regime_audit_path=str(regime_audit_path),
        distribution_audit_path=str(dist_path),
        max_mean_psi=config.drift.max_mean_psi,
        max_pair_psi=config.drift.max_pair_psi,
        min_drift_survivors=config.drift.min_drift_survivors,
        min_sign_consistency=config.predictive.min_sign_consistency,
        conditional_anchors_retained=retention_details if retention_details else None,
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

    if config.dual_stream:
        return _run_dual_stream_train_stage(io, frames, raw_universe, config)
    return _run_single_stream_train_stage(io, frames, raw_universe, config)


def _run_single_stream_train_stage(
    io: PipelineIOManager,
    frames: dict[str, pl.DataFrame],
    raw_universe: list[str],
    config: FeatureSelectionPipelineConfig,
) -> FeatureSelectionResult:
    # 2. Stage 1: Fast Vectorized Statistical Gates

    mandatory_features = list(config.mandatory_state_features)
    blacklist_set = set(config.hygiene.feature_blacklist)
    blacklisted_mandatory: list[str] = []
    if blacklist_set and mandatory_features:
        blacklisted_mandatory = sorted(blacklist_set.intersection(mandatory_features))
        if blacklisted_mandatory:
            mandatory_features = [
                f for f in mandatory_features if f not in blacklist_set
            ]

    # Validate input frames and check missing columns
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

    # 1.1 Data Hygiene (NZV, Mode Frequency, Front-loaded Blacklist & Ablation, Winsorization)
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

    # 1.2 Distribution Drift Audit (Multi-contract PSI & Forward Boundary Outpost)
    outpost_frame = io.load_validation_outpost_frame(candidate_universe)
    dist_res = audit_distribution_drift(
        cleaned_frames,
        candidate_universe,
        config=config.drift,
        forward_outpost_frame=outpost_frame,
    )
    dist_path = io.output_dir / "distribution_audit_metrics.csv"
    dist_res.metrics_df.write_csv(dist_path)
    candidate_universe = dist_res.surviving_features

    # 1.3 Stationarity Audit (Within-contract ADF, Fallback Safeguard, Half-Life, SAR)
    stat_res = execute_stationarity_audit(
        cleaned_frames, candidate_universe, config.stationarity
    )
    if not stat_res.surviving_features and not mandatory_features:
        raise ValueError(
            "feature selection produced an empty list after Stationarity Audit"
        )
    candidate_universe = stat_res.surviving_features

    # 1.4 Vectorized Predictive Audit (Multi-window IC/RankIC, Sign Consistency, Stability IR, FDR)
    aggregate_df, pred_res = execute_predictive_audit(
        cleaned_frames, candidate_universe, config.predictive
    )
    if not pred_res.surviving_features and not mandatory_features:
        raise ValueError("feature selection produced an empty list after Hard Filter")

    # Persist initial per-contract metric frames and aggregate metrics
    per_contract_dir = io.output_dir / "per_contract"
    per_contract_dir.mkdir(parents=True, exist_ok=True)
    metric_frames: list[pl.DataFrame] = pred_res.diagnostics["metric_frames"]
    per_contract_records: list[FeatureSelectionContractRecord] = []
    for (contract, frame), mf in zip(cleaned_frames.items(), metric_frames):
        metric_path = per_contract_dir / f"{contract}_metrics.csv"
        mf.write_csv(metric_path)
        per_contract_records.append(
            FeatureSelectionContractRecord(
                contract=contract,
                input_path=str(io.input_dir / f"{contract}.feather"),
                metric_path=str(metric_path),
            )
        )
    aggregate_path = io.output_dir / "aggregate_metrics.csv"
    aggregate_df.write_csv(aggregate_path)

    # 3. Stage 2: Target-Horizon Nonlinear Scoring (Purged CatBoost on w_dec, Priority Score)
    scored_features, scored_df, score_res = execute_nonlinear_scoring(
        cleaned_frames,
        pred_res.surviving_features,
        aggregate_df,
        dist_res.mean_psi_by_feature,
        config.scoring,
        metric_frames=metric_frames,
    )
    # Update per-contract metrics with target-horizon CatBoost importance
    for (contract, _), mf in zip(cleaned_frames.items(), metric_frames):
        metric_path = per_contract_dir / f"{contract}_metrics.csv"
        mf.write_csv(metric_path)
    scored_df.write_csv(aggregate_path)
    candidate_universe = scored_features

    # 4. Stage 3: Orthogonal Representation & Regime Audit
    ortho_res = execute_orthogonal_deduplication(
        cleaned_frames, candidate_universe, scored_features, config.dedup
    )
    candidate_universe = ortho_res.surviving_features

    # Regime Audit
    regime_quantiles = compute_regime_quantiles(
        cleaned_frames, num_bins=config.regime.regime_bins
    )
    effective_target_bins = (
        list(config.regime.target_regime_bins)
        if config.regime.target_regime_bins is not None
        else default_target_regime_bins(
            config.regime.regime_bins, config.regime.regime_bins
        )
    )
    regime_audit_df, retained_anchors, retention_details = audit_regimes(
        cleaned_frames,
        raw_universe,
        regime_quantiles,
        list(config.predictive.windows_list),
        target_regime_bins=effective_target_bins,
        min_abs_ic=config.predictive.min_abs_ic,
        enable_conditional_anchors=config.regime.enable_conditional_anchors,
    )
    regime_audit_path = io.output_dir / "regime_audit_metrics.csv"
    regime_audit_df.write_csv(regime_audit_path)

    normal_selected = [f for f in candidate_universe if f not in mandatory_features]
    newly_retained: list[str] = []
    if config.regime.enable_conditional_anchors and retained_anchors:
        newly_retained = [
            a
            for a in retained_anchors
            if a in raw_universe
            and a not in normal_selected
            and a not in blacklist_set
        ]
        if newly_retained:
            normal_selected.extend(newly_retained)

    final_selected = normal_selected + mandatory_features

    # 5. Persist Output Datasets and Manifest
    selected_file = io.save_selected_features(final_selected)
    filtered_outputs = io.write_filtered_outputs(frames, final_selected)

    filter_results: dict[str, list[str]] = {}
    all_blacklisted = sorted(
        set(hygiene_res.diagnostics["blacklist_dropped"]).union(blacklisted_mandatory)
    )
    if all_blacklisted:
        filter_results["Feature Blacklist Dropped"] = all_blacklisted
    if hygiene_res.diagnostics["ablation_dropped"]:
        filter_results["Feature Ablation Dropped"] = hygiene_res.diagnostics[
            "ablation_dropped"
        ]
    if dist_res.dropped_features:
        filter_results["Distribution Drift Dropped"] = dist_res.dropped_features

    filter_results["Hard Filter"] = pred_res.surviving_features
    if pred_res.diagnostics["sign_consistency_dropped"]:
        filter_results["Sign Consistency Filter Dropped"] = pred_res.diagnostics[
            "sign_consistency_dropped"
        ]
    if config.stationarity.min_half_life_bars > 0.0:
        filter_results["Persistence Filter"] = stat_res.surviving_features
        if stat_res.diagnostics["persistence_dropped"]:
            filter_results["Persistence Filter Dropped"] = stat_res.diagnostics[
                "persistence_dropped"
            ]
    filter_results["Stability Filter"] = [
        f
        for f in pred_res.surviving_features
        if f not in pred_res.diagnostics["stability_dropped"]
    ]
    filter_results["Composite Score"] = scored_features
    if score_res.dropped_features:
        filter_results["Composite Score Dropped"] = score_res.dropped_features
    filter_results["Correlation Filter"] = ortho_res.surviving_features
    if newly_retained:
        filter_results["Conditional Anchor Retention"] = newly_retained

    persistence_diagnostics = (
        stat_res.diagnostics["persistence_diagnostics"]
        if config.stationarity.min_half_life_bars > 0.0
        else None
    )
    persistence_filter = (
        {
            "min_half_life_bars": float(config.stationarity.min_half_life_bars),
            "active_feature_pattern": config.persistence_filter_pattern,
        }
        if config.stationarity.min_half_life_bars > 0.0
        else None
    )

    manifest = FeatureSelectionManifest(
        symbol=config.symbol,
        target_freq=config.target_freq,
        stage=config.stage,
        split_input_dir=str(io.input_dir),
        rl_feature_file=str(selected_file),
        selected_feature_count=len(final_selected),
        selected_features=final_selected,
        windows_list=list(config.predictive.windows_list),
        composite_drop_ratio=config.scoring.composite_drop_ratio,
        feature_blacklist=(
            list(config.hygiene.feature_blacklist)
            if config.hygiene.feature_blacklist
            else None
        ),
        feature_ablation_patterns=list(config.hygiene.feature_ablation_patterns),
        rank_ic_mode=config.predictive.rank_ic_mode,
        mandatory_state_features=(
            mandatory_features if mandatory_features else None
        ),
        persistence_filter=persistence_filter,
        persistence_diagnostics=persistence_diagnostics,
        aggregate_metrics_path=str(aggregate_path),
        filter_results=filter_results,
        contracts=per_contract_records,
        filtered_outputs=filtered_outputs,
        regime_bins=config.regime.regime_bins,
        target_regime_bins=effective_target_bins,
        regime_quantiles=regime_quantiles,
        regime_audit_path=str(regime_audit_path),
        distribution_audit_path=str(dist_path),
        max_mean_psi=config.drift.max_mean_psi,
        max_pair_psi=config.drift.max_pair_psi,
        min_drift_survivors=config.drift.min_drift_survivors,
        min_sign_consistency=config.predictive.min_sign_consistency,
        conditional_anchors_retained=retention_details if retention_details else None,
    )
    io.save_manifest(manifest)
    return FeatureSelectionResult(output_dir=io.output_dir, manifest=manifest)


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
            frame, raw_universe, windows_list=list(config.predictive.windows_list)
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
        max_mean_psi=config.drift.max_mean_psi,
        max_pair_psi=config.drift.max_pair_psi,
        min_drift_survivors=config.drift.min_drift_survivors,
        min_sign_consistency=config.predictive.min_sign_consistency,
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
        "--dual_stream",
        action="store_true",
        default=True,
        help="Enable dual-stream feature selection (default: True).",
    )
    parser.add_argument(
        "--no_dual_stream",
        action="store_false",
        dest="dual_stream",
        help="Disable dual-stream feature selection and run single-stream mode.",
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
        "--vae_feature_blacklist",
        nargs="*",
        default=None,
        help="Stream-specific feature blacklist for VAE regime stream.",
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
        dual_stream=args.dual_stream,
        max_mean_psi=args.max_mean_psi,
        max_pair_psi=args.max_pair_psi,
        min_drift_survivors=args.min_drift_survivors,
        distribution_num_bins=args.distribution_num_bins,
        dedup_method=args.dedup_method,
        max_vif=args.max_vif,
        fdr_threshold=args.fdr_threshold,
        vae_feature_blacklist=args.vae_feature_blacklist,
        rl_feature_blacklist=args.rl_feature_blacklist,
    )
