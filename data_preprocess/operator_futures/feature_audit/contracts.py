"""Domain contracts, types, and reason codes for Multi-Horizon Feature Audit."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any


class HorizonType(str, Enum):
    H1 = "1h"
    H2 = "2h"
    H4 = "4h"
    TODAY = "today"
    NEXT_TD = "next_td"
    NEXT_5TD = "next_5td"
    NEXT_15TD = "next_15td"


class TargetKind(str, Enum):
    RETURN = "return"
    ABS_RETURN = "abs_return"


class InvalidReason(str, Enum):
    NONE = "none"
    ZERO_LENGTH_HORIZON = "ZERO_LENGTH_HORIZON"
    STALE_PRICE = "STALE_PRICE"
    MISSING_TARGET_PRICE = "MISSING_TARGET_PRICE"
    CONTRACT_EXPIRED = "CONTRACT_EXPIRED"
    UNVERIFIED_TIMING = "UNVERIFIED_TIMING"
    NON_FINITE_PRICE = "NON_FINITE_PRICE"
    INSUFFICIENT_TAIL = "INSUFFICIENT_TAIL"


@dataclass(frozen=True)
class TradingSessionSpec:
    start_time: str  # "YYYY-MM-DD HH:MM:SS"
    end_time: str    # "YYYY-MM-DD HH:MM:SS"
    name: str = "session"


@dataclass(frozen=True)
class TradingDaySchedule:
    trading_day: str  # "YYYYMMDD"
    sessions: List[TradingSessionSpec]


@dataclass
class CandidateFeature:
    name: str
    dtype: str
    is_qualified: bool
    exclusion_reason: Optional[str] = None
    stream_membership: Optional[str] = None  # slope_vae, vol_vae, rl_decision, rejected


@dataclass
class FeatureMetricResult:
    feature: str
    horizon: str
    target_kind: str
    spearman_corr: Optional[float]
    valid_pairs: int
    total_pairs: int
    null_reason: Optional[str] = None
