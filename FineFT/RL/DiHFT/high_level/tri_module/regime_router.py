from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class RegimeRoute:
    volatility_index: int
    slope_index: int
    slot_id: int
    evidence: dict[str, Any]


@dataclass(frozen=True)
class CandidateResolution:
    slot_id: int
    kind: str
    available: bool
    candidate_id: str | None
    slot: dict[str, Any]


class RegimeSlotRouter:
    """Return the best two-dimensional slot without admission authority."""

    def __init__(self, num_labels: int) -> None:
        self.num_labels = int(num_labels)
        if self.num_labels <= 0:
            raise ValueError("num_labels must be positive")

    @staticmethod
    def _axis_evidence(values: Sequence[float], axis: str) -> tuple[int, dict[str, Any]]:
        array = np.asarray(values, dtype=float).reshape(-1)
        if array.size == 0 or not np.all(np.isfinite(array)):
            raise ValueError(f"{axis} weights must be finite and non-empty")
        total = float(np.sum(array))
        if total <= 0.0:
            probabilities = np.full(array.shape, 1.0 / array.size, dtype=float)
        else:
            probabilities = array / total
        index = int(np.argmax(array))
        ordered = np.sort(probabilities)[::-1]
        margin = float(ordered[0] - ordered[1]) if ordered.size > 1 else 1.0
        return index, {
            f"{axis}_weights": array.tolist(),
            f"{axis}_probabilities": probabilities.tolist(),
            f"{axis}_max_probability": float(probabilities[index]),
            f"{axis}_margin": margin,
        }

    def route(
        self,
        volatility_weights: Sequence[float],
        slope_weights: Sequence[float],
    ) -> RegimeRoute:
        if len(volatility_weights) != self.num_labels:
            raise ValueError("volatility weights do not match num_labels")
        if len(slope_weights) != self.num_labels:
            raise ValueError("slope weights do not match num_labels")
        volatility_index, volatility_evidence = self._axis_evidence(
            volatility_weights, "volatility"
        )
        slope_index, slope_evidence = self._axis_evidence(slope_weights, "slope")
        slot_id = volatility_index * self.num_labels + slope_index
        return RegimeRoute(
            volatility_index=volatility_index,
            slope_index=slope_index,
            slot_id=slot_id,
            evidence={**volatility_evidence, **slope_evidence},
        )


class CandidateResolver:
    """Resolve a routed slot to a candidate without judging trade utility."""

    def __init__(self, selection_manifest: Any) -> None:
        self.selection_manifest = selection_manifest
        slots = list(selection_manifest.slots)
        if int(selection_manifest.slot_count) != len(slots):
            raise ValueError("selection manifest slot_count does not match slots")
        self._slots = {int(slot["slot_id"]): dict(slot) for slot in slots}

    def resolve(self, route: RegimeRoute) -> CandidateResolution:
        if route.slot_id not in self._slots:
            raise ValueError(f"routed slot {route.slot_id} is absent from manifest")
        slot = self._slots[route.slot_id]
        kind = str(slot.get("kind", ""))
        if kind == "empty_model":
            return CandidateResolution(
                slot_id=route.slot_id,
                kind="null_candidate",
                available=False,
                candidate_id=None,
                slot=slot,
            )
        if kind != "model":
            raise ValueError(f"unsupported candidate slot kind {kind!r}")
        candidate_id = slot.get("candidate_id")
        if candidate_id is None:
            candidate_id = f"slot_{route.slot_id}"
        return CandidateResolution(
            slot_id=route.slot_id,
            kind="model",
            available=True,
            candidate_id=str(candidate_id),
            slot=slot,
        )
