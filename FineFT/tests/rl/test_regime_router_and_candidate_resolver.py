from __future__ import annotations

from dataclasses import dataclass

from RL.DiHFT.high_level.tri_module.regime_router import (
    CandidateResolver,
    RegimeSlotRouter,
)


@dataclass
class _Manifest:
    slot_count: int
    slots: list[dict]


def test_router_returns_best_slot_even_when_margin_is_ambiguous() -> None:
    router = RegimeSlotRouter(num_labels=3)

    route = router.route(
        volatility_weights=[0.20, 0.21, 0.05],
        slope_weights=[0.30, 0.29, 0.01],
    )

    assert route.volatility_index == 1
    assert route.slope_index == 0
    assert route.slot_id == 3
    assert 0.0 <= route.evidence["volatility_margin"] < 0.03
    assert 0.0 <= route.evidence["slope_margin"] < 0.03


def test_candidate_resolver_distinguishes_model_and_null_candidate() -> None:
    router = RegimeSlotRouter(num_labels=2)
    manifest = _Manifest(
        slot_count=4,
        slots=[
            {"slot_id": 0, "kind": "model", "candidate_id": "agent-a"},
            {"slot_id": 1, "kind": "empty_model", "candidate_id": None},
            {"slot_id": 2, "kind": "model", "candidate_id": "agent-b"},
            {"slot_id": 3, "kind": "model", "candidate_id": "agent-c"},
        ],
    )
    resolver = CandidateResolver(manifest)

    model_route = router.route([0.8, 0.2], [0.9, 0.1])
    null_route = router.route([0.8, 0.2], [0.1, 0.9])

    model = resolver.resolve(model_route)
    null = resolver.resolve(null_route)

    assert model.kind == "model"
    assert model.candidate_id == "agent-a"
    assert model.available is True
    assert null.kind == "null_candidate"
    assert null.candidate_id is None
    assert null.available is False
