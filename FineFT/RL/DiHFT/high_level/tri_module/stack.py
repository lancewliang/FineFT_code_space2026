from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from RL.DiHFT.high_level.trading_utility.gate import TradingUtilityGate
from RL.DiHFT.high_level.tri_module.ood_gate import (
    ExtremeOODDecision,
    ExtremeOODGate,
    InputValidity,
)
from RL.DiHFT.high_level.tri_module.regime_router import (
    CandidateResolution,
    CandidateResolver,
    RegimeRoute,
    RegimeSlotRouter,
)


@dataclass(frozen=True)
class TriModuleMarketDecision:
    ood: ExtremeOODDecision
    route: RegimeRoute | None
    candidate: CandidateResolution | None


class TriModuleGatingStack:
    stack_name = "tri_module"

    def __init__(
        self,
        *,
        num_labels: int,
        selection_manifest: object,
        utility_gate: TradingUtilityGate,
        recovery_steps: int,
        enter_threshold: float = 0.005,
        recover_threshold: float = 0.01,
    ) -> None:
        self.ood_gate = ExtremeOODGate(
            enter_threshold=enter_threshold,
            recover_threshold=recover_threshold,
            recovery_steps=recovery_steps,
        )
        self.router = RegimeSlotRouter(num_labels)
        self.candidate_resolver = CandidateResolver(selection_manifest)
        self.utility_gate = utility_gate

    def reset(self) -> None:
        self.ood_gate.reset()

    def evaluate_market(
        self,
        *,
        volatility_weights: Sequence[float],
        slope_weights: Sequence[float],
        input_validity: InputValidity,
    ) -> TriModuleMarketDecision:
        ood = self.ood_gate.evaluate(
            volatility_weights,
            slope_weights,
            input_validity,
        )
        if ood.degraded:
            return TriModuleMarketDecision(ood=ood, route=None, candidate=None)
        route = self.router.route(volatility_weights, slope_weights)
        candidate = self.candidate_resolver.resolve(route)
        return TriModuleMarketDecision(ood=ood, route=route, candidate=candidate)
