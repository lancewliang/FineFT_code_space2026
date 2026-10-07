from __future__ import annotations

from typing import Sequence
import numpy as np

from RL.DiHFT.high_level.gating.base import BaseGatingStrategy, GatingDecision


class AbsoluteThresholdGating(BaseGatingStrategy):
    """Gating strategy using absolute dual-axis VAE likelihood thresholds with hysteresis."""

    def __init__(
        self,
        slope_threshold: float,
        volatility_threshold: float,
        hysteresis_exit_ratio: float = 0.65,
    ) -> None:
        self.slope_threshold = float(slope_threshold)
        self.volatility_threshold = float(volatility_threshold)
        self.hysteresis_exit_ratio = float(hysteresis_exit_ratio)
        if self.hysteresis_exit_ratio <= 0.0 or self.hysteresis_exit_ratio > 1.0:
            raise ValueError(
                f"hysteresis_exit_ratio must be in (0.0, 1.0], got {self.hysteresis_exit_ratio}"
            )

    @property
    def strategy_name(self) -> str:
        return "absolute"

    def decide(
        self,
        volatility_weights: Sequence[float] | np.ndarray,
        slope_weights: Sequence[float] | np.ndarray,
        current_position: float = 0.0,
    ) -> GatingDecision:
        vol_arr = np.asarray(volatility_weights, dtype=float)
        slope_arr = np.asarray(slope_weights, dtype=float)

        max_vol = float(np.max(vol_arr))
        max_slope = float(np.max(slope_arr))

        is_holding = abs(float(current_position)) > 1e-6
        effective_vol_thresh = (
            self.volatility_threshold * self.hysteresis_exit_ratio
            if is_holding
            else self.volatility_threshold
        )
        effective_slope_thresh = (
            self.slope_threshold * self.hysteresis_exit_ratio
            if is_holding
            else self.slope_threshold
        )

        metrics = {
            "max_volatility_weight": max_vol,
            "max_slope_weight": max_slope,
            "effective_volatility_threshold": effective_vol_thresh,
            "effective_slope_threshold": effective_slope_thresh,
        }

        if max_vol < effective_vol_thresh or max_slope < effective_slope_thresh:
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
