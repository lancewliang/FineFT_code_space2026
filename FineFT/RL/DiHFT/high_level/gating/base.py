from __future__ import annotations

import abc
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence
import numpy as np


@dataclass(frozen=True, slots=True)
class GatingDecision:
    is_defensive: bool
    volatility_index: int
    slope_index: int
    reject_reason: str
    metrics: Mapping[str, object] = field(default_factory=dict)
    decision_id: str = ""
    selected_agent_id: str = ""
    gate_version: str = ""
    regime: str = ""
    dominant_slot: tuple[int, int] | None = None
    active_slot: tuple[int, int] | None = None
    candidate_slot: tuple[int, int] | None = None
    candidate_count: int = 0
    queried_slot: tuple[int, int] | None = None
    proposed_action: int | None = None
    completed_candidate_slot: tuple[int, int] | None = None
    completed_candidate_count: int = 0
    incumbent_invalidated: bool = False
    transition_reason: str = ""

    def __post_init__(self) -> None:
        """Copy metrics into an immutable, slotted decision value."""
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))


class BaseGatingStrategy(abc.ABC):
    """Abstract base interface for dual-axis VAE gating strategies."""

    @property
    @abc.abstractmethod
    def strategy_name(self) -> str:
        """Canonical identifier for this gating strategy."""
        ...

    @abc.abstractmethod
    def decide(
        self,
        volatility_weights: Sequence[float] | np.ndarray,
        slope_weights: Sequence[float] | np.ndarray,
        current_position: float = 0.0,
    ) -> GatingDecision:
        """Evaluate dual-axis weights and return a GatingDecision."""
        ...
