"""Atomic replacement contracts exercised through public router decisions."""

import numpy as np
import pytest
from common import ActionDecisionReasons
from RL.DiHFT.high_level import vae_routing_util as vru
from tests.rl.test_expert_hysteresis_routing import act, make_router


def switching_router(k=2, lead=0.125):
    router = make_router()
    router.gating_strategy = vru.create_gating_strategy(
        "expert_hysteresis",
        slope_threshold=0.5,
        volatility_threshold=0.5,
        hysteresis_exit_ratio=0.5,
        expert_confirm_steps=k,
        expert_switch_lead=lead,
    )
    # Different experts make observably different decisions. Each kept expert
    # still runs on every step, so identity and action both prove the source.
    router.agent_act = lambda state, info: 2 if router.selected_agent_index == 5 else 0
    for _ in range(k - 1):
        assert act(router) == 1
    assert act(router) == 2
    assert router.audit_predictions[-1]["routing_slot"] == 5
    return router


@pytest.mark.parametrize(
    "vol,slope,target",
    [
        ([0.1, 0.75, 0.1], [0.75, 0.1, 0.5], 3),  # slope only
        ([0.75, 0.5, 0.1], [0.1, 0.1, 0.75], 2),  # volatility only
        ([0.75, 0.5, 0.1], [0.75, 0.1, 0.5], 0),  # both axes
    ],
)
def test_kth_step_switches_whole_slot_and_action(vol, slope, target):
    router = switching_router()
    assert act(router, vol, slope) == 2
    decision = router.gating_strategy.last_decision
    assert decision.active_slot == (1, 2)
    assert decision.candidate_slot == divmod(target, 3)
    assert decision.candidate_count == 1
    assert decision.reject_reason == "switch_confirmation"
    assert act(router, vol, slope) == 0
    decision = router.gating_strategy.last_decision
    assert decision.active_slot == divmod(target, 3)
    assert decision.candidate_slot is None
    assert decision.candidate_count == 0
    assert decision.reject_reason == "expert_replaced"
    assert [p["routing_slot"] for p in router.audit_predictions[-2:]] == [5, target]
    assert [p["selected_action"] for p in router.audit_predictions[-2:]] == [2, 0]


@pytest.mark.parametrize(
    "vol,slope",
    [
        ([0.1, 0.49, 0.1], [0.75, 0.1, 0.5]),  # unchanged vol misses E
        ([0.75, 0.5, 0.1], [0.1, 0.1, 0.49]),  # unchanged slope misses E
        ([0.1, 0.75, 0.1], [0.49, 0.1, 0.3]),  # changed slope misses E
        ([0.49, 0.3, 0.1], [0.1, 0.1, 0.75]),  # changed vol misses E
        ([0.75, 0.6875, 0.1], [0.75, 0.1, 0.5]),  # only vol lead misses Delta
        ([0.75, 0.5, 0.1], [0.75, 0.1, 0.6875]),  # only slope lead misses Delta
    ],
)
def test_each_required_entry_and_changed_axis_lead_is_independent(vol, slope):
    router = switching_router()
    for _ in range(3):
        assert act(router, vol, slope) == 2
        decision = router.gating_strategy.last_decision
        assert decision.active_slot == (1, 2)
        assert decision.candidate_slot is None
        assert decision.candidate_count == 0


def test_candidate_changes_cannot_pool_axis_or_slot_confirmation():
    router = switching_router(k=3)
    # Volatility-only candidate, then both axes, then slope-only. These are
    # three different complete Slots; no per-axis confirmation is transferable.
    sequences = [
        ([0.75, 0.5, 0.1], [0.1, 0.1, 0.75], (0, 2)),
        ([0.75, 0.5, 0.1], [0.75, 0.1, 0.5], (0, 0)),
        ([0.1, 0.75, 0.1], [0.75, 0.1, 0.5], (1, 0)),
    ]
    for vol, slope, candidate in sequences:
        assert act(router, vol, slope) == 2
        decision = router.gating_strategy.last_decision
        assert decision.candidate_slot == candidate
        assert decision.candidate_count == 1
    vol, slope, _ = sequences[-1]
    assert act(router, vol, slope) == 2
    assert act(router, vol, slope) == 0
    assert router.audit_predictions[-1]["routing_slot"] == 3


def test_ineligible_step_interrupts_confirmed_candidate_evidence():
    router = switching_router(k=3)
    vol, slope = [0.75, 0.5, 0.1], [0.1, 0.1, 0.75]
    assert act(router, vol, slope) == 2
    assert act(router, vol, slope) == 2
    # Same highest Slot now has an insufficient changed-axis lead.
    assert act(router, [0.75, 0.7, 0.1], slope) == 2
    assert router.gating_strategy.last_decision.candidate_count == 0
    assert act(router, vol, slope) == 2
    assert act(router, vol, slope) == 2
    assert act(router, vol, slope) == 0


def test_candidate_evidence_crosses_invalidation_with_same_step_takeover():
    router = switching_router(k=3)
    vol, slope = [0.75, 0.5, 0.1], [0.75, 0.1, 0.5]
    assert act(router, vol, slope) == 2
    assert act(router, vol, slope) == 2
    # Old Slot loses its slope H. Candidate already has K-1 steps, so this
    # decision directly switches to the new expert and its Short action.
    assert act(router, vol, [0.75, 0.1, np.nextafter(0.25, 0.0)]) == 0
    assert router.gating_strategy.last_decision.reject_reason == "expert_replaced"
    assert router.audit_predictions[-1]["routing_slot"] == 0
    assert router.action_decision_reason_history[-1] == ActionDecisionReasons.POLICY_INFERENCE


def test_invalidation_before_k_keeps_evidence_and_routes_rules_until_ready():
    router = switching_router(k=3)
    vol, slope = [0.75, 0.5, 0.1], [0.75, 0.1, 0.5]
    assert act(router, vol, slope) == 2
    # Still only two qualified steps: no valid incumbent, so use rules.
    assert act(router, [0.75, 0.2, 0.1], slope) == 1
    decision = router.gating_strategy.last_decision
    assert decision.active_slot is None
    assert decision.candidate_slot == (0, 0)
    assert decision.candidate_count == 2
    # Recovery does not require a runner-up margin; nearly tied scores pass E.
    assert act(router, [0.75, 0.749, 0.1], [0.75, 0.749, 0.1]) == 0
    assert router.audit_predictions[-1]["routing_slot"] == 0


def test_active_tied_highest_wins_cartesian_product_and_clears_candidate():
    router = switching_router(k=2, lead=0.0)
    assert act(router, [0.75, 0.5, 0.1], [0.75, 0.1, 0.5]) == 2
    for _ in range(3):
        assert act(router, [0.75, 0.75, 0.1], [0.75, 0.1, 0.75]) == 2
        decision = router.gating_strategy.last_decision
        assert decision.dominant_slot == decision.active_slot == (1, 2)
        assert decision.candidate_slot is None
    # The old one-step candidate cannot survive a tied-incumbent interruption.
    assert act(router, [0.75, 0.5, 0.1], [0.75, 0.1, 0.5]) == 2
    assert act(router, [0.75, 0.5, 0.1], [0.75, 0.1, 0.5]) == 0


def test_existing_higher_id_candidate_survives_lower_id_tie():
    router = switching_router(k=3)
    assert act(router, [0.1, 0.5, 0.75], [0.1, 0.75, 0.5]) == 2
    # Candidate (2,1) stays preferred within the Cartesian highest set even
    # though the newly tied (0,0) has a lower stable Slot ID.
    vol, slope = [0.75, 0.5, 0.75], [0.75, 0.75, 0.5]
    assert act(router, vol, slope) == 2
    decision = router.gating_strategy.last_decision
    assert decision.dominant_slot == decision.candidate_slot == (2, 1)
    assert decision.candidate_count == 2
    assert act(router, vol, slope) == 0
    assert router.audit_predictions[-1]["routing_slot"] == 7


def test_initial_tie_has_stable_slot_order_and_repeatable_external_actions():
    results = []
    for _ in range(2):
        router = make_router(confirm_steps=3)
        actions = [act(router, [0.8, 0.8, 0.1], [0.6, 0.1, 0.6]) for _ in range(3)]
        results.append((actions, [p["routing_slot"] for p in router.audit_predictions]))
    assert results == [([1, 1, 2], [0]), ([1, 1, 2], [0])]


def test_delta_equality_and_adjacent_float_boundary():
    router = switching_router(k=2, lead=0.25)
    slope = [0.75, 0.1, 0.5]
    # A single nextafter below the exact positive Delta cannot count.
    assert act(router, [np.nextafter(0.75, 0.0), 0.5, 0.1], slope) == 2
    assert router.gating_strategy.last_decision.candidate_count == 0
    # Exactly .75 - .5 == .25 qualifies in both changed axes.
    assert act(router, [0.75, 0.5, 0.1], slope) == 2
    assert act(router, [0.75, 0.5, 0.1], slope) == 0


def test_zero_delta_still_requires_strict_lead_on_every_changed_axis():
    router = switching_router(k=2, lead=0.0)
    # Dominant Slot 0 changes both axes, but volatility only ties. The active
    # Slot is not itself tied highest because its slope is lower.
    vol, slope = [0.75, 0.75, 0.1], [0.75, 0.1, 0.5]
    assert act(router, vol, slope) == 2
    assert act(router, vol, slope) == 2
    assert router.gating_strategy.last_decision.candidate_count == 0
    vol[0] = np.nextafter(0.75, 1.0)
    assert act(router, vol, slope) == 2
    assert act(router, vol, slope) == 0


def test_unchanged_axis_at_entry_and_incumbent_axis_at_hold_are_inclusive():
    router = switching_router(k=2)
    # Volatility is unchanged and exactly E; old slope equals H and remains
    # valid while the candidate clears its own slope E.
    vol, slope = [0.1, 0.5, 0.1], [0.5, 0.1, 0.25]
    assert act(router, vol, slope) == 2
    assert router.gating_strategy.last_decision.reject_reason == "switch_confirmation"
    assert act(router, vol, slope) == 0
    assert router.audit_predictions[-1]["routing_slot"] == 3
