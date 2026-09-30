from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl

from operator_futures.feature_selection.manifests import (
    FeatureSelectionContractRecord,
    FeatureSelectionManifest,
    FeatureSelectionResult,
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
    DataHygieneConfig,
    DistributionAuditConfig,
    FeatureSelectionPipelineConfig,
    NonlinearScoringConfig,
    OrthogonalDedupConfig,
    PredictiveAuditConfig,
    RegimeAuditConfig,
    StationarityAuditConfig,
)

logger = logging.getLogger(__name__)

DEFAULT_PERSISTENCE_FILTER_PATTERN = r"_log_return_(1|2)$"


def _state_features(df: pl.DataFrame, *, orderbook_depth: int) -> list[str]:
    return extract_state_features(df, orderbook_depth=orderbook_depth)


def _ordered_filter_features(
    frames: dict[str, pl.DataFrame],
    aggregate: pl.DataFrame,
    feature_universe: list[str],
    *,
    min_abs_ic: float = 0.02,
    max_metric_std: float = 1.0,
    max_correlation: float = 0.7,
    min_rank_ic_ir: float = 0.40,
    min_sign_consistency: float = 0.75,
    target_decision_window: int | None = None,
    windows_list: list[int] | None = None,
    metric_frames: list[pl.DataFrame] | None = None,
    composite_drop_ratio: float = 0.1,
    min_half_life_bars: float = 0.0,
    persistence_diagnostics: list[Any] | None = None,
    rank_ic_mode: str = "absolute",
    mean_psi_by_feature: dict[str, float] | None = None,
) -> tuple[list[str], dict[str, list[str]]]:
    import math
    from operator_futures.feature_selection.cor_util import (
        compute_contract_normalized_correlation_matrix,
        select_feature,
    )

    if composite_drop_ratio < 0 or composite_drop_ratio >= 1:
        raise ValueError("composite_drop_ratio must be in [0, 1)")
    if rank_ic_mode not in {"absolute", "signed"}:
        raise ValueError("rank_ic_mode must be 'absolute' or 'signed'")

    selected = aggregate.filter(pl.col("feature").is_in(feature_universe))
    rank_ic_filter = (
        pl.col("RankIC_Mean") >= min_abs_ic
        if rank_ic_mode == "signed"
        else pl.col("RankIC_Mean").abs() >= min_abs_ic
    )
    hard = selected.filter(rank_ic_filter)["feature"].to_list()
    if not hard:
        raise ValueError("feature selection produced an empty list after Hard Filter")

    sign_consistent = hard
    sign_consistency_dropped: list[str] = []
    if metric_frames is not None and len(metric_frames) > 0:
        eff_windows = list(windows_list or [6, 12, 24, 48])
        dec_window = (
            target_decision_window
            if target_decision_window is not None
            else (6 if 6 in eff_windows else eff_windows[0])
        )
        combined_mf = pl.concat(metric_frames, how="vertical")
        if "RankIC" in combined_mf.columns and "window" in combined_mf.columns:
            w_frames = combined_mf.filter(pl.col("window") == dec_window)
            if w_frames.height > 0:
                n_contracts = float(len(metric_frames))
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
                sign_consistent = [
                    f for f in hard if sc_map.get(f, 0.0) >= min_sign_consistency
                ]
                sign_consistency_dropped = [
                    f for f in hard if f not in set(sign_consistent)
                ]
                if not sign_consistent:
                    raise ValueError(
                        "feature selection produced an empty list after Sign Consistency Filter"
                    )
    elif "SignConsistency" in selected.columns:
        sc_map = dict(
            zip(selected["feature"].to_list(), selected["SignConsistency"].to_list())
        )
        sign_consistent = [
            f for f in hard if sc_map.get(f, 0.0) >= min_sign_consistency
        ]
        sign_consistency_dropped = [
            f for f in hard if f not in set(sign_consistent)
        ]
        if not sign_consistent:
            raise ValueError(
                "feature selection produced an empty list after Sign Consistency Filter"
            )

    persistence = sign_consistent
    persistence_dropped: list[str] = []
    if min_half_life_bars > 0.0:
        diag_map = {
            str(row["feature"]): row for row in (persistence_diagnostics or [])
        }
        for f in sign_consistent:
            d = diag_map.get(f)
            hl = d.get("half_life_bars_median") if d is not None else None
            if (
                d is not None
                and d.get("active_filter") is True
                and hl is not None
                and float(hl) < min_half_life_bars
            ):
                persistence_dropped.append(f)
            else:
                pass
        persistence = [f for f in sign_consistent if f not in set(persistence_dropped)]
        if not persistence:
            raise ValueError(
                "feature selection produced an empty list after Persistence Filter"
            )

    stability_cond = pl.col("IC_Std") <= max_metric_std
    if "RankIC_Std" in selected.columns:
        rank_ic_ir = pl.col("RankIC_Mean").abs() / (pl.col("RankIC_Std") + 1e-6)
        stability_cond = stability_cond & (rank_ic_ir >= min_rank_ic_ir)

    stability = (
        selected.filter(pl.col("feature").is_in(persistence))
        .filter(stability_cond)["feature"]
        .to_list()
    )
    if not stability:
        raise ValueError(
            "feature selection produced an empty list after Stability Filter"
        )

    scored_input = selected.filter(pl.col("feature").is_in(stability))
    height = float(max(scored_input.height, 1))

    rank_ic_score = (
        pl.col("RankIC_Mean").abs()
        if rank_ic_mode == "absolute"
        else pl.col("RankIC_Mean")
    ).fill_null(0.0)

    catboost_score = (
        pl.col("CatBoost Importance_Mean").fill_null(0.0)
        if "CatBoost Importance_Mean" in scored_input.columns
        else pl.lit(0.0)
    )

    if mean_psi_by_feature is not None:
        mean_psi_values = [
            float(mean_psi_by_feature.get(f, 0.0))
            for f in scored_input["feature"].to_list()
        ]
    elif "mean_psi" in scored_input.columns:
        mean_psi_values = scored_input["mean_psi"].fill_null(0.0).to_list()
    else:
        mean_psi_values = [0.0] * scored_input.height

    inv_psi_values = [1.0 / (psi + 1e-4) for psi in mean_psi_values]

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
    drop_count = min(
        math.ceil(scored.height * composite_drop_ratio),
        max(scored.height - 1, 0),
    )
    kept = scored.head(scored.height - drop_count) if drop_count else scored
    dropped = scored.tail(drop_count)["feature"].to_list() if drop_count else []
    composite = kept["feature"].to_list()
    if not composite:
        raise ValueError(
            "feature selection produced an empty list after Composite Score"
        )

    corre_df = compute_contract_normalized_correlation_matrix(
        frames=frames, features=composite
    )
    correlation = select_feature(
        features=composite, corre_df=corre_df, theshold=max_correlation
    )
    if not correlation:
        raise ValueError(
            "feature selection produced an empty list after Correlation Filter"
        )
    filter_results = {"Hard Filter": hard}
    if sign_consistency_dropped:
        filter_results["Sign Consistency Filter Dropped"] = sign_consistency_dropped
    if min_half_life_bars > 0.0:
        filter_results.update(
            {
                "Persistence Filter": persistence,
                "Persistence Filter Dropped": persistence_dropped,
            }
        )
    filter_results.update(
        {
            "Stability Filter": stability,
            "Composite Score": composite,
            "Composite Score Dropped": dropped,
            "Correlation Filter": correlation,
        }
    )
    return correlation, filter_results


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
            feature_blacklist=tuple(kwargs.get("feature_blacklist") or ()),
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
    )


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
        selected_feature_file=str(selected_file),
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
            / "state_features.npy"
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
    )
