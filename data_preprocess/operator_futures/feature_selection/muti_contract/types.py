from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl


@dataclass(frozen=True)
class PipelineStepResult:
    step_name: str
    surviving_features: list[str]
    dropped_features: list[str]
    audit_metrics_df: pl.DataFrame | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DataHygieneConfig:
    min_variance: float = 1e-6
    max_mode_frequency: float = 0.98
    enable_winsorization: bool = True
    winsorize_iqr_multiplier: float = 5.0
    feature_blacklist: tuple[str, ...] = field(default_factory=tuple)
    feature_ablation_patterns: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class DistributionAuditConfig:
    num_bins: int = 10
    max_mean_psi: float = 0.10
    max_pair_psi: float = 0.25
    min_drift_survivors: int = 20
    forward_outpost_max_psi: float = 0.15


@dataclass(frozen=True)
class StationarityAuditConfig:
    adf_significance_level: float = 0.05
    min_passing_contract_ratio: float = 0.70
    fallback_significance_level: float = 0.10
    min_survivors_floor: int = 25
    min_half_life_bars: float = 2.0
    max_sign_alternation_rate: float = 0.40
    active_feature_pattern: str | None = None


@dataclass(frozen=True)
class PredictiveAuditConfig:
    min_abs_ic: float = 0.02
    min_sign_consistency: float = 0.75
    min_rank_ic_ir: float = 0.40
    target_decision_window: int = 6
    windows_list: tuple[int, ...] = (1, 2, 6, 12, 24, 48)
    fdr_threshold: float = 0.05
    ic_anomaly_ceiling: float = 0.30
    rank_ic_mode: str = "absolute"
    max_metric_std: float = 1.0


@dataclass(frozen=True)
class NonlinearScoringConfig:
    decision_window: int = 6
    composite_drop_ratio: float = 0.10
    early_stopping_rounds: int = 30
    catboost_depth: int = 6
    catboost_iterations: int = 1000
    random_seed: int = 42


@dataclass(frozen=True)
class OrthogonalDedupConfig:
    max_correlation: float = 0.70
    correlation_method: str = "spearman"
    dedup_method: str = "cluster"
    enable_hierarchical_clustering: bool = True
    cluster_distance_threshold: float = 0.50
    min_clusters: int = 50
    max_clusters: int = 70
    max_vif: float = 10.0


@dataclass(frozen=True)
class RegimeAuditConfig:
    enable_conditional_anchors: bool = True
    regime_bins: int = 4
    target_regime_bins: tuple[tuple[int, int], ...] | None = None
    max_variance_ratio: float = 3.0


@dataclass(frozen=True)
class StreamFilterProfile:
    name: str
    max_mean_psi: float
    max_pair_psi: float
    min_abs_ic: float
    min_sign_consistency: float
    min_rank_ic_ir: float
    max_correlation: float
    min_clusters: int
    max_clusters: int
    psi_weight: float
    rank_ic_weight: float
    catboost_weight: float
    filter_micro_persistence: bool
    mandatory_feature_pattern: str | None = None


DEFAULT_VAE_PROFILE = StreamFilterProfile(
    name="vae_regime",
    max_mean_psi=0.10,
    max_pair_psi=0.20,
    min_abs_ic=0.015,
    min_sign_consistency=0.70,
    min_rank_ic_ir=0.35,
    max_correlation=0.65,
    min_clusters=12,
    max_clusters=18,
    psi_weight=0.50,
    rank_ic_weight=0.30,
    catboost_weight=0.20,
    filter_micro_persistence=True,
    mandatory_feature_pattern=r"^(base_time_|time_|trading_minute_)",
)

DEFAULT_RL_PROFILE = StreamFilterProfile(
    name="rl_decision",
    max_mean_psi=0.25,
    max_pair_psi=0.35,
    min_abs_ic=0.020,
    min_sign_consistency=0.65,
    min_rank_ic_ir=0.30,
    max_correlation=0.80,
    min_clusters=50,
    max_clusters=65,
    psi_weight=0.15,
    rank_ic_weight=0.50,
    catboost_weight=0.35,
    filter_micro_persistence=False,
    mandatory_feature_pattern=None,
)


@dataclass(frozen=True)
class FeatureSelectionPipelineConfig:
    root_path: Path
    symbol: str
    target_freq: str
    stage: str
    split_path: str = "PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST"
    save_path: str = "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION"
    orderbook_depth: int = 5
    mandatory_state_features: tuple[str, ...] = field(default_factory=tuple)
    persistence_filter_pattern: str = r"_log_return_(1|2)$"
    dual_stream: bool = True
    vae_profile: StreamFilterProfile = DEFAULT_VAE_PROFILE
    rl_profile: StreamFilterProfile = DEFAULT_RL_PROFILE
    hygiene: DataHygieneConfig = field(default_factory=DataHygieneConfig)
    drift: DistributionAuditConfig = field(default_factory=DistributionAuditConfig)
    stationarity: StationarityAuditConfig = field(default_factory=StationarityAuditConfig)
    predictive: PredictiveAuditConfig = field(default_factory=PredictiveAuditConfig)
    scoring: NonlinearScoringConfig = field(default_factory=NonlinearScoringConfig)
    dedup: OrthogonalDedupConfig = field(default_factory=OrthogonalDedupConfig)
    regime: RegimeAuditConfig = field(default_factory=RegimeAuditConfig)
