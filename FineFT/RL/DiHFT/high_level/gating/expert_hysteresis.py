from __future__ import annotations

import json
from numbers import Integral
from typing import TYPE_CHECKING, Sequence

import numpy as np

from RL.DiHFT.high_level.gating.base import BaseGatingStrategy, GatingDecision

if TYPE_CHECKING:
    from analysis.pick_agent.FineFT_two_dimensional_agent_selector import (
        TwoDimensionalSelectionManifest,
    )


class ExpertHysteresisGating(BaseGatingStrategy):
    """Keep expert ownership independently of positions and action persistence.

    State belongs to one contract rollout. Manifest binding validates identities
    without moving the confirmation clock; only ``decide`` advances evidence.
    """

    strategy_version = "expert_hysteresis.v1"

    def __init__(
        self,
        slope_threshold: float,
        volatility_threshold: float,
        hysteresis_exit_ratio: float = 0.65,
        expert_switch_lead: float = 0.0,
        expert_confirm_steps: int = 1,
    ) -> None:
        self.slope_threshold = float(slope_threshold)
        self.volatility_threshold = float(volatility_threshold)
        self.hysteresis_exit_ratio = float(hysteresis_exit_ratio)
        self.expert_switch_lead = float(expert_switch_lead)
        for name, value in (
            ("slope_threshold", self.slope_threshold),
            ("volatility_threshold", self.volatility_threshold),
        ):
            if not np.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be finite and in (0, 1]")
        if (
            not np.isfinite(self.hysteresis_exit_ratio)
            or not 0.0 < self.hysteresis_exit_ratio < 1.0
        ):
            raise ValueError("hysteresis_exit_ratio must be finite and in (0, 1)")
        for threshold in (self.slope_threshold, self.volatility_threshold):
            if not 0.0 <= threshold * self.hysteresis_exit_ratio < threshold:
                raise ValueError("derived holding threshold must satisfy 0 <= H < E")
        if not np.isfinite(self.expert_switch_lead) or self.expert_switch_lead < 0.0:
            raise ValueError("expert_switch_lead must be finite and non-negative")
        if (
            isinstance(expert_confirm_steps, bool)
            or not isinstance(expert_confirm_steps, Integral)
            or expert_confirm_steps <= 0
        ):
            raise ValueError("expert_confirm_steps must be a positive integer")
        self.expert_confirm_steps = int(expert_confirm_steps)
        self.reset()

    @property
    def strategy_name(self) -> str:
        return "expert_hysteresis"

    def reset(self) -> None:
        """Begin an independent rollout, including a fresh identity contract."""
        self.active_slot: tuple[int, int] | None = None
        self.candidate_slot: tuple[int, int] | None = None
        self.candidate_count = 0
        self._axis_lengths: tuple[int, int] | None = None
        self._manifest_identity: str | None = None
        self._empty_slots: set[tuple[int, int]] = set()
        self.last_decision: GatingDecision | None = None

    def bind_manifest(self, manifest: TwoDimensionalSelectionManifest) -> None:
        """Reject changed axis order or Slot mapping within a rollout."""
        vol_labels = manifest.axes.volatility
        slope_labels = manifest.axes.slope
        if not vol_labels or not slope_labels or len(vol_labels) != len(slope_labels):
            raise ValueError("expert_hysteresis requires nonempty matching manifest axes")
        if len(manifest.slots) != len(vol_labels) * len(slope_labels):
            raise ValueError("expert_hysteresis manifest Slot mapping has invalid dimensions")
        if manifest.slot_index_formula != "volatility_index * num_labels + slope_index":
            raise ValueError("expert_hysteresis manifest Slot mapping formula is incompatible")
        for slot_id, slot in enumerate(manifest.slots):
            if slot.get("slot_id") != slot_id or slot.get("kind") not in (
                "model", "empty_model"
            ):
                raise ValueError("expert_hysteresis manifest Slot mapping is incompatible")
        identity = json.dumps(
            {"volatility": vol_labels, "slope": slope_labels, "slots": manifest.slots},
            sort_keys=True,
        )
        if self._manifest_identity is not None and identity != self._manifest_identity:
            raise ValueError("expert_hysteresis manifest axis labels or Slot mapping changed")
        self._manifest_identity = identity
        lengths = (len(vol_labels), len(slope_labels))
        if self._axis_lengths is not None and lengths != self._axis_lengths:
            raise ValueError("expert_hysteresis score axis dimensions do not match manifest")
        self._axis_lengths = lengths
        self._empty_slots = {
            divmod(i, len(slope_labels))
            for i, slot in enumerate(manifest.slots)
            if slot["kind"] == "empty_model"
        }

    def validate_scores(
        self,
        volatility_weights: Sequence[float] | np.ndarray,
        slope_weights: Sequence[float] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        arrays = (
            np.asarray(volatility_weights, dtype=float),
            np.asarray(slope_weights, dtype=float),
        )
        for axis, values in zip(("volatility", "slope"), arrays):
            if values.ndim != 1 or values.size == 0:
                raise ValueError(f"{axis} scores must be a nonempty vector")
            if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
                raise ValueError(f"{axis} scores must be finite and in [0, 1]")
        lengths = tuple(len(x) for x in arrays)
        if self._axis_lengths is not None and lengths != self._axis_lengths:
            raise ValueError(
                "expert_hysteresis score axis dimensions changed or mismatch manifest"
            )
        self._axis_lengths = lengths
        return arrays

    def clear_candidate(self) -> None:
        self.candidate_slot = None
        self.candidate_count = 0

    def decide(
        self,
        volatility_weights: Sequence[float] | np.ndarray,
        slope_weights: Sequence[float] | np.ndarray,
        current_position: float = 0.0,
    ) -> GatingDecision:
        vol, slope = self.validate_scores(volatility_weights, slope_weights)
        holding_vol = self.volatility_threshold * self.hysteresis_exit_ratio
        holding_slope = self.slope_threshold * self.hysteresis_exit_ratio
        active_valid = self.active_slot is not None and (
            vol[self.active_slot[0]] >= holding_vol
            and slope[self.active_slot[1]] >= holding_slope
        )
        highest = [
            (int(v), int(s))
            for v in np.flatnonzero(vol == np.max(vol))
            for s in np.flatnonzero(slope == np.max(slope))
        ]
        if active_valid and self.active_slot in highest:
            dominant = self.active_slot
        elif self.candidate_slot in highest:
            dominant = self.candidate_slot
        else:
            dominant = highest[0]
        metrics = {
            "max_volatility_weight": float(np.max(vol)),
            "max_slope_weight": float(np.max(slope)),
            "effective_volatility_threshold": (
                holding_vol if active_valid else self.volatility_threshold
            ),
            "effective_slope_threshold": (
                holding_slope if active_valid else self.slope_threshold
            ),
        }
        # Empty Slots retain their existing runtime rule/finite exit handling.
        if dominant in self._empty_slots:
            self.active_slot = None
            self.clear_candidate()
            decision = GatingDecision(False, *dominant, "empty_model", metrics)
        elif active_valid:
            # Atomic replacement confirmation is added by the switching ticket.
            self.clear_candidate()
            decision = GatingDecision(False, *self.active_slot, "expert_retained", metrics)
        else:
            self.active_slot = None
            eligible = (
                vol[dominant[0]] >= self.volatility_threshold
                and slope[dominant[1]] >= self.slope_threshold
            )
            if eligible:
                self.candidate_count = (
                    self.candidate_count + 1 if self.candidate_slot == dominant else 1
                )
                self.candidate_slot = dominant
                if self.candidate_count >= self.expert_confirm_steps:
                    self.active_slot = dominant
                    self.clear_candidate()
            else:
                self.clear_candidate()
            if self.active_slot is None:
                reason = "entry_confirmation" if eligible else "entry_threshold"
                decision = GatingDecision(True, -1, -1, reason, metrics)
            else:
                decision = GatingDecision(False, *self.active_slot, "expert_entered", metrics)
        self.last_decision = decision
        return decision
