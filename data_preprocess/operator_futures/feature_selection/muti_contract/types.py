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
    max_mean_psi: float = 0.45
    max_pair_psi: float = 1.30
    min_drift_survivors: int = 20
    forward_outpost_max_psi: float = 0.80


@dataclass(frozen=True)
class StationarityAuditConfig:
    adf_significance_level: float = 0.05
    min_passing_contract_ratio: float = 0.70
    fallback_significance_level: float = 0.10
    min_survivors_floor: int = 25
    min_half_life_bars: float = 2.0
    max_sign_alternation_rate: float = 0.40
    active_feature_pattern: str | None = None


import re

MACRO_SCALE_PATTERN = re.compile(
    r"(_(720|1440|2160)(_|$)|prev_(5|10|15|20|30)_day|prev_(1|2|4|6)_week|cm_.*_(720|1440))"
)
MESO_SCALE_PATTERN = re.compile(
    r"(_(48|96|192)(_|$)|prev_day_|prev_2_day_|.*session.*|trading_minute_|base_time_|contract_month_|contract_life_|is_opening_|is_closing_)"
)


def classify_feature_scale(feature_name: str) -> str:
    if MACRO_SCALE_PATTERN.search(feature_name):
        return "macro"
    if MESO_SCALE_PATTERN.search(feature_name):
        return "meso"
    return "micro"


@dataclass(frozen=True)
class ScaleTierConfig:
    name: str
    forward_horizons: tuple[int, ...]
    decision_horizon: int
    min_abs_ic: float
    min_sign_consistency: float
    min_rank_ic_ir: float


SCALE_TIER_CONFIGS: dict[str, ScaleTierConfig] = {
    "micro": ScaleTierConfig(
        name="micro",
        forward_horizons=(1, 2, 6, 12),
        decision_horizon=6,
        min_abs_ic=0.010,
        min_sign_consistency=0.55,
        min_rank_ic_ir=0.18,
    ),
    "meso": ScaleTierConfig(
        name="meso",
        forward_horizons=(16, 24, 48, 96),
        decision_horizon=24,
        min_abs_ic=0.015,
        min_sign_consistency=0.58,
        min_rank_ic_ir=0.15,
    ),
    "macro": ScaleTierConfig(
        name="macro",
        forward_horizons=(192, 384, 720),
        decision_horizon=192,
        min_abs_ic=0.020,
        min_sign_consistency=0.60,
        min_rank_ic_ir=0.12,
    ),
}

DEFAULT_MULTI_HORIZON_WINDOWS: tuple[int, ...] = (
    1, 2, 6, 12, 16, 24, 48, 96, 192, 384, 720
)


@dataclass(frozen=True)
class StreamTierQuota:
    min_quota: int
    max_quota: int


STREAM_TIER_QUOTAS: dict[str, dict[str, StreamTierQuota]] = {
    "rl_decision": {
        "micro": StreamTierQuota(min_quota=85, max_quota=100),
        "meso": StreamTierQuota(min_quota=35, max_quota=45),
        "macro": StreamTierQuota(min_quota=10, max_quota=15),
    },
    "vae_slope": {
        "micro": StreamTierQuota(min_quota=0, max_quota=0),
        "meso": StreamTierQuota(min_quota=8, max_quota=10),
        "macro": StreamTierQuota(min_quota=4, max_quota=6),
    },
    "vae_volatility": {
        "micro": StreamTierQuota(min_quota=0, max_quota=1),
        "meso": StreamTierQuota(min_quota=6, max_quota=8),
        "macro": StreamTierQuota(min_quota=4, max_quota=5),
    },
}


@dataclass(frozen=True)
class PredictiveAuditConfig:
    min_abs_ic: float = 0.010
    min_sign_consistency: float = 0.55
    min_rank_ic_ir: float = 0.18
    target_decision_window: int = 6
    windows_list: tuple[int, ...] = DEFAULT_MULTI_HORIZON_WINDOWS
    fdr_threshold: float = 0.05
    ic_anomaly_ceiling: float = 0.45
    rank_ic_mode: str = "absolute"
    max_metric_std: float = 1.0


@dataclass(frozen=True)
class NonlinearScoringConfig:
    decision_window: int = 6
    composite_drop_ratio: float = 0.05
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
    feature_blacklist: tuple[str, ...] = field(default_factory=tuple)
    min_anova_f: float = 0.0
    require_monotonic: bool = False
    target_metric: str = "RankIC"
    max_vif: float = 10.0


DEFAULT_VAE_SLOPE_PROFILE = StreamFilterProfile(
    name="vae_slope",
    max_mean_psi=0.10,
    max_pair_psi=0.20,
    min_abs_ic=0.020,
    min_sign_consistency=0.75,
    min_rank_ic_ir=0.35,
    max_correlation=0.65,
    min_clusters=12,
    max_clusters=16,
    psi_weight=0.45,
    rank_ic_weight=0.35,
    catboost_weight=0.20,
    filter_micro_persistence=True,
    mandatory_feature_pattern=r"^(base_time_|time_|trading_minute_)",
    feature_blacklist=(),
    min_anova_f=4.0,
    require_monotonic=True,
    target_metric="RankIC",
    max_vif=10.0,
)

DEFAULT_VAE_VOLATILITY_PROFILE = StreamFilterProfile(
    name="vae_volatility",
    max_mean_psi=0.12,
    max_pair_psi=0.25,
    min_abs_ic=0.030,
    min_sign_consistency=0.75,
    min_rank_ic_ir=0.40,
    max_correlation=0.60,
    min_clusters=10,
    max_clusters=14,
    psi_weight=0.45,
    rank_ic_weight=0.35,
    catboost_weight=0.20,
    filter_micro_persistence=True,
    mandatory_feature_pattern=r"^(trading_minute_progress)",
    feature_blacklist=(),
    min_anova_f=6.0,
    require_monotonic=True,
    target_metric="VolRankIC",
    max_vif=8.0,
)

DEFAULT_RL_PROFILE = StreamFilterProfile(
    name="rl_decision",
    max_mean_psi=0.45,
    max_pair_psi=1.30,
    min_abs_ic=0.010,
    min_sign_consistency=0.55,
    min_rank_ic_ir=0.18,
    max_correlation=0.80,
    min_clusters=135,
    max_clusters=160,
    psi_weight=0.15,
    rank_ic_weight=0.70,
    catboost_weight=0.15,
    filter_micro_persistence=False,
    mandatory_feature_pattern=None,
    feature_blacklist=(),
    min_anova_f=0.0,
    require_monotonic=False,
    target_metric="RankIC",
    max_vif=10.0,
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
    vae_slope_profile: StreamFilterProfile = DEFAULT_VAE_SLOPE_PROFILE
    vae_volatility_profile: StreamFilterProfile = DEFAULT_VAE_VOLATILITY_PROFILE
    rl_profile: StreamFilterProfile = DEFAULT_RL_PROFILE
    hygiene: DataHygieneConfig = field(default_factory=DataHygieneConfig)
    drift: DistributionAuditConfig = field(default_factory=DistributionAuditConfig)
    stationarity: StationarityAuditConfig = field(default_factory=StationarityAuditConfig)
    predictive: PredictiveAuditConfig = field(default_factory=PredictiveAuditConfig)
    scoring: NonlinearScoringConfig = field(default_factory=NonlinearScoringConfig)
    dedup: OrthogonalDedupConfig = field(default_factory=OrthogonalDedupConfig)
    regime: RegimeAuditConfig = field(default_factory=RegimeAuditConfig)
