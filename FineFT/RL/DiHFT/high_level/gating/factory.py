from __future__ import annotations

from typing import Any
from RL.DiHFT.high_level.gating.base import BaseGatingStrategy
from RL.DiHFT.high_level.gating.absolute_gating import AbsoluteThresholdGating
from RL.DiHFT.high_level.gating.hierarchical_gating import HierarchicalDualGating


def create_gating_strategy(
    strategy_type: str,
    *,
    slope_threshold: float = 0.2,
    volatility_threshold: float = 0.2,
    hysteresis_exit_ratio: float = 0.65,
    ood_threshold: float = 0.005,
    slope_margin_threshold: float = 0.12,
    volatility_margin_threshold: float = 0.12,
    **kwargs: Any,
) -> BaseGatingStrategy:
    """Factory function to instantiate a gating strategy by canonical name."""
    normalized_type = strategy_type.strip().lower()

    if normalized_type == "absolute":
        st = kwargs["slope_rule_base_threshold"] if "slope_rule_base_threshold" in kwargs else slope_threshold
        vt = kwargs["volatility_rule_base_threshold"] if "volatility_rule_base_threshold" in kwargs else volatility_threshold
        her = kwargs["hysteresis_exit_ratio"] if "hysteresis_exit_ratio" in kwargs else hysteresis_exit_ratio
        return AbsoluteThresholdGating(
            slope_threshold=float(st),
            volatility_threshold=float(vt),
            hysteresis_exit_ratio=float(her),
        )

    if normalized_type == "hierarchical":
        return HierarchicalDualGating(
            ood_threshold=float(ood_threshold),
            slope_margin_threshold=float(slope_margin_threshold),
            volatility_margin_threshold=float(volatility_margin_threshold),
        )

    raise ValueError(
        f"Unknown gating strategy type: {strategy_type!r}. Supported strategies: ['absolute', 'hierarchical']"
    )
