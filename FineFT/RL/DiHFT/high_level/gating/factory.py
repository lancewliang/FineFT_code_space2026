from __future__ import annotations

from RL.DiHFT.high_level.gating.base import BaseGatingStrategy
from RL.DiHFT.high_level.gating.expert_hysteresis import ExpertHysteresisGating


def create_gating_strategy(
    strategy_type: str,
    *,
    slope_threshold: float = 0.2,
    volatility_threshold: float = 0.2,
    hysteresis_exit_ratio: float = 0.65,
    expert_switch_lead: float = 0.0,
    expert_confirm_steps: int = 1,
) -> BaseGatingStrategy:
    """Build the sole supported high-level routing strategy."""
    normalized_type = strategy_type.strip().lower()

    if normalized_type != "expert_hysteresis":
        raise ValueError(
            f"Unsupported gating strategy {strategy_type!r}; "
            "the upgraded routing contract requires 'expert_hysteresis'"
        )
    return ExpertHysteresisGating(
        slope_threshold=slope_threshold,
        volatility_threshold=volatility_threshold,
        hysteresis_exit_ratio=hysteresis_exit_ratio,
        expert_switch_lead=expert_switch_lead,
        expert_confirm_steps=expert_confirm_steps,
    )
