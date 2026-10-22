from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class InputValidity:
    is_valid: bool
    reason: str = "none"

    @classmethod
    def valid(cls) -> "InputValidity":
        return cls(True, "none")

    @classmethod
    def invalid(cls, reason: str) -> "InputValidity":
        normalized = str(reason).strip()
        if not normalized or normalized == "none":
            raise ValueError("invalid input validity requires a concrete reason")
        return cls(False, normalized)


@dataclass(frozen=True)
class ExtremeOODDecision:
    degraded: bool
    lifecycle: str
    reason: str
    recovery_streak: int
    max_volatility_quantile: float
    max_slope_quantile: float


class ExtremeOODGate:
    """Stateful deterministic OOD and input-validity circuit breaker."""

    def __init__(
        self,
        enter_threshold: float = 0.005,
        recover_threshold: float = 0.01,
        recovery_steps: int = 3,
    ) -> None:
        self.enter_threshold = float(enter_threshold)
        self.recover_threshold = float(recover_threshold)
        self.recovery_steps = int(recovery_steps)
        if not 0.0 <= self.enter_threshold < self.recover_threshold <= 1.0:
            raise ValueError(
                "OOD thresholds must satisfy 0 <= enter < recover <= 1"
            )
        if self.recovery_steps <= 0:
            raise ValueError("recovery_steps must be positive")
        self._degraded = False
        self._recovery_streak = 0
        self._active_reason = "none"

    def reset(self) -> None:
        """Reset lifecycle state at a contract/run boundary."""
        self._degraded = False
        self._recovery_streak = 0
        self._active_reason = "none"

    @staticmethod
    def _best_weight(values: Sequence[float] | np.ndarray, axis: str) -> float:
        array = np.asarray(values, dtype=float).reshape(-1)
        if array.size == 0 or not np.all(np.isfinite(array)):
            raise ValueError(f"{axis} likelihood weights must be finite and non-empty")
        return float(np.max(array))

    def evaluate(
        self,
        volatility_weights: Sequence[float] | np.ndarray,
        slope_weights: Sequence[float] | np.ndarray,
        input_validity: InputValidity,
    ) -> ExtremeOODDecision:
        try:
            max_vol = self._best_weight(volatility_weights, "volatility")
            max_slope = self._best_weight(slope_weights, "slope")
        except ValueError as error:
            input_validity = InputValidity.invalid(str(error))
            max_vol = float("nan")
            max_slope = float("nan")

        failure_reason = "none"
        if not input_validity.is_valid:
            failure_reason = f"data_invalid:{input_validity.reason}"
        elif max_vol < self.enter_threshold:
            failure_reason = "vae_ood:volatility"
        elif max_slope < self.enter_threshold:
            failure_reason = "vae_ood:slope"

        if failure_reason != "none":
            lifecycle = "stay" if self._degraded else "enter"
            self._degraded = True
            self._recovery_streak = 0
            self._active_reason = failure_reason
            return self._decision(lifecycle, failure_reason, max_vol, max_slope)

        if self._degraded:
            recoverable = (
                input_validity.is_valid
                and max_vol >= self.recover_threshold
                and max_slope >= self.recover_threshold
            )
            if not recoverable:
                self._recovery_streak = 0
                return self._decision("stay", self._active_reason, max_vol, max_slope)
            self._recovery_streak += 1
            if self._recovery_streak < self.recovery_steps:
                return self._decision("stay", "recovery_pending", max_vol, max_slope)
            self._degraded = False
            self._recovery_streak = 0
            self._active_reason = "none"
            return self._decision("recover", "none", max_vol, max_slope)

        return self._decision("healthy", "none", max_vol, max_slope)

    def _decision(
        self,
        lifecycle: str,
        reason: str,
        max_vol: float,
        max_slope: float,
    ) -> ExtremeOODDecision:
        return ExtremeOODDecision(
            degraded=self._degraded,
            lifecycle=lifecycle,
            reason=reason,
            recovery_streak=self._recovery_streak,
            max_volatility_quantile=max_vol,
            max_slope_quantile=max_slope,
        )
