from __future__ import annotations

from typing import Sequence
import numpy as np

from RL.DiHFT.high_level.gating.base import BaseGatingStrategy, GatingDecision


class HierarchicalDualGating(BaseGatingStrategy):
    """Hierarchical dual-layer gating strategy separating tail-risk OOD circuit breaking from mode ambiguity filtering."""

    def __init__(
        self,
        ood_threshold: float = 0.005,
        slope_margin_threshold: float = 0.12,
        volatility_margin_threshold: float = 0.12,
    ) -> None:
        self.ood_threshold = float(ood_threshold)
        self.slope_margin_threshold = float(slope_margin_threshold)
        self.volatility_margin_threshold = float(volatility_margin_threshold)

    @property
    def strategy_name(self) -> str:
        return "hierarchical"

    @staticmethod
    def _compute_relative_margin(arr: np.ndarray) -> tuple[np.ndarray, float]:
        total = float(np.sum(arr))
        probs = arr / (total + 1e-12)
        if len(probs) < 2:
            return probs, 1.0
        sorted_probs = np.sort(probs)[::-1]
        margin = float(sorted_probs[0] - sorted_probs[1])
        return probs, margin

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

        metrics: dict[str, object] = {
            "max_volatility_weight": max_vol,
            "max_slope_weight": max_slope,
        }

        # Gate 1: Absolute Tail-Risk Circuit Breaker (Catastrophic OOD)
        if max_vol < self.ood_threshold or max_slope < self.ood_threshold:
            return GatingDecision(
                is_defensive=True,
                volatility_index=-1,
                slope_index=-1,
                reject_reason="ood_circuit_breaker",
                metrics=metrics,
            )

        # Compute relative probability distributions and margins
        vol_probs, vol_margin = self._compute_relative_margin(vol_arr)
        slope_probs, slope_margin = self._compute_relative_margin(slope_arr)

        metrics["volatility_margin"] = vol_margin
        metrics["slope_margin"] = slope_margin
        metrics["volatility_probs"] = vol_probs.tolist()
        metrics["slope_probs"] = slope_probs.tolist()

        # Gate 2: Relative Mode Clarity (Choppy Noise / Ambiguity Filter)
        if (
            vol_margin < self.volatility_margin_threshold
            or slope_margin < self.slope_margin_threshold
        ):
            return GatingDecision(
                is_defensive=True,
                volatility_index=-1,
                slope_index=-1,
                reject_reason="margin_ambiguity",
                metrics=metrics,
            )

        # Both gates pass cleanly: select dominant sub-agent
        vol_idx = int(np.argmax(vol_arr))
        slope_idx = int(np.argmax(slope_arr))

        return GatingDecision(
            is_defensive=False,
            volatility_index=vol_idx,
            slope_index=slope_idx,
            reject_reason="none",
            metrics=metrics,
        )
