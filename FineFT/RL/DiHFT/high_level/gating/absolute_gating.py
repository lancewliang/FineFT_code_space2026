from __future__ import annotations

from typing import Sequence
import numpy as np

from RL.DiHFT.high_level.gating.base import BaseGatingStrategy, GatingDecision


class AbsoluteThresholdGating(BaseGatingStrategy):
    """Legacy gating strategy using absolute dual-axis VAE likelihood thresholds."""

    def __init__(self, slope_threshold: float, volatility_threshold: float) -> None:
        self.slope_threshold = float(slope_threshold)
        self.volatility_threshold = float(volatility_threshold)

    @property
    def strategy_name(self) -> str:
        return "absolute"

    def decide(
        self,
        volatility_weights: Sequence[float] | np.ndarray,
        slope_weights: Sequence[float] | np.ndarray,
    ) -> GatingDecision:
        vol_arr = np.asarray(volatility_weights, dtype=float)
        slope_arr = np.asarray(slope_weights, dtype=float)

        max_vol = float(np.max(vol_arr))
        max_slope = float(np.max(slope_arr))

        metrics = {
            "max_volatility_weight": max_vol,
            "max_slope_weight": max_slope,
        }

        if max_vol < self.volatility_threshold or max_slope < self.slope_threshold:
            return GatingDecision(
                is_defensive=True,
                volatility_index=-1,
                slope_index=-1,
                reject_reason="absolute_threshold",
                metrics=metrics,
            )

        vol_idx = int(np.argmax(vol_arr))
        slope_idx = int(np.argmax(slope_arr))

        return GatingDecision(
            is_defensive=False,
            volatility_index=vol_idx,
            slope_index=slope_idx,
            reject_reason="none",
            metrics=metrics,
        )
