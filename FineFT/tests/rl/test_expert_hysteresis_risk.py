"""Risk interruption and empty-Slot behavior at the public action boundary."""

import numpy as np
import pytest
from common import ActionDecisionReasons
from RL.DiHFT.high_level import vae_routing_util as vru
from tests.rl.test_expert_hysteresis_routing import make_router

BASE_VOL = (0.1, 0.8, 0.2)
BASE_SLOPE = (0.1, 0.2, 0.6)
NEW_VOL = (0.95, 0.8, 0.2)
NEW_SLOPE = (0.9, 0.2, 0.6)


def invoke(router, vol=BASE_VOL, slope=BASE_SLOPE, position=0.0, mask=None, tier=1.0, **risk):
    router.calculate_axis_window_result = lambda axis: {"volatility": vol, "slope": slope}[axis]
    router.step_idx += 1
    return router.get_action(
        {"previous_action": router.action, "avaliable_action": mask or [1, 1, 1]},
        np.array([tier]),
        position,
        1,
        **risk,
    )


def prepared(k=3, *, empty=False):
    router = make_router(confirm_steps=k)
    if empty:
        router.selection_manifest.slots[0]["kind"] = "empty_model"
    calls = []

    def expert(state, info):
        calls.append((router.selected_agent_index, dict(info)))
        return 2

    router.agent_act = expert
    for _ in range(k - 1):
        assert invoke(router) == 1
    assert invoke(router) == 2
    return router, calls


@pytest.mark.parametrize("interruption", ["stop", "circuit", "non_main"])
def test_pre_inference_risk_breaks_k_minus_one_replacement_evidence(interruption):
    router, calls = prepared()
    assert invoke(router, NEW_VOL, NEW_SLOPE) == 2
    assert invoke(router, NEW_VOL, NEW_SLOPE) == 2
    query_count = len(calls)
    kwargs = {}
    if interruption == "stop":
        router.stop_loss_cooldown_steps = 0
        router.circuit_breaker_consecutive_stops = 0
        kwargs = {"position": 1.0, "current_markprice": 100.0, "current_unrealized_pnl": -2.0}
        reason = ActionDecisionReasons.HARD_STOP_LOSS
    elif interruption == "circuit":
        router.circuit_breaker_remaining_steps = 1
        reason = ActionDecisionReasons.CIRCUIT_BREAKER_SUSPENSION
    else:
        router.enable_non_main_contract_defense = True
        router.role_tier_index = 0
        kwargs = {"tier": 0.0}
        reason = ActionDecisionReasons.DEFENSIVE_PREEMPTION
    assert invoke(router, NEW_VOL, NEW_SLOPE, **kwargs) == 1
    assert len(calls) == query_count
    decision = router.gating_strategy.last_decision
    assert decision.candidate_slot is None
    assert decision.candidate_count == 0
    assert decision.queried_slot is None
    assert router.action_decision_reason_history[-1] == reason
    # Blocking ended: no reuse of the two previously eligible steps.
    assert invoke(router, NEW_VOL, NEW_SLOPE) == 2
    assert router.gating_strategy.last_decision.active_slot == (1, 2)
    assert invoke(router, NEW_VOL, NEW_SLOPE) == 2
    assert router.gating_strategy.last_decision.active_slot == (1, 2)
    assert invoke(router, NEW_VOL, NEW_SLOPE) == 2
    assert router.gating_strategy.last_decision.active_slot == (0, 0)


@pytest.mark.parametrize("pending_steps", [1, 2])
@pytest.mark.parametrize(
    "interruption", ["cooldown", "mask", "non_main_reversal", "non_main_expansion"]
)
def test_post_inference_risk_rolls_back_provisional_kth_admission(pending_steps, interruption):
    router, _ = prepared()
    for _ in range(pending_steps):
        assert invoke(router, NEW_VOL, NEW_SLOPE) == 2
    kwargs = {}
    expected = 1
    if interruption == "cooldown":
        router.cooldown_remaining_steps = 1
        router.last_stopped_position = 1.0
        reason = ActionDecisionReasons.STOP_LOSS_COOLDOWN
    elif interruption == "mask":
        kwargs = {"mask": [1, 1, 0]}
        reason = ActionDecisionReasons.ACTION_UNAVAILABLE_BREAK
    else:
        router.enable_non_main_contract_defense = True
        router.non_main_defense_mode = "graceful_hold_exit"
        router.role_tier_index = 0
        kwargs = {"position": 1.0, "tier": 0.0}
        router.agent_act = lambda state, info: 0
        reason = ActionDecisionReasons.DEFENSIVE_RULE_CLOSE
        if interruption == "non_main_expansion":
            router.position_list = [-1.0, -0.5, 0.0, 0.5, 1.0]
            router.zero_position_action = router.flat_action = 2
            router.agent_act = lambda state, info: 4
            kwargs.update(position=0.5, mask=[1] * 5)
            expected = 3  # retain real half-sized position
            reason = ActionDecisionReasons.ACTION_PERSISTENCE
    assert invoke(router, NEW_VOL, NEW_SLOPE, **kwargs) == expected
    decision = router.gating_strategy.last_decision
    assert decision.active_slot == (1, 2)
    assert decision.candidate_slot is None
    assert decision.candidate_count == 0
    assert decision.queried_slot == ((0, 0) if pending_steps == 2 else (1, 2))
    assert decision.proposed_action is not None
    assert router.action_decision_reason_history[-1] == reason
    # The blocked expert never becomes the retained expert on the next step.
    router.agent_act = lambda state, info: 2
    assert invoke(router, NEW_VOL, NEW_SLOPE) == 2
    assert router.gating_strategy.last_decision.active_slot == (1, 2)
    assert router.gating_strategy.last_decision.candidate_count == 1


def test_blocked_initial_kth_step_requires_new_confirmation():
    router = make_router(confirm_steps=2)
    assert invoke(router) == 1
    router.cooldown_remaining_steps = 1
    router.last_stopped_position = 1.0
    assert invoke(router) == 1
    assert router.gating_strategy.last_decision.active_slot is None
    assert router.gating_strategy.last_decision.queried_slot == (1, 2)
    assert invoke(router) == 1
    assert invoke(router) == 2


def test_blocked_kth_replacement_never_restores_an_invalid_incumbent():
    router, _ = prepared()
    assert invoke(router, NEW_VOL, NEW_SLOPE) == 2
    assert invoke(router, NEW_VOL, NEW_SLOPE) == 2
    router.cooldown_remaining_steps = 1
    router.last_stopped_position = 1.0
    invalid_old_slope = (0.9, 0.2, 0.1)
    assert invoke(router, NEW_VOL, invalid_old_slope) == 1
    decision = router.gating_strategy.last_decision
    assert decision.active_slot is None
    assert decision.queried_slot == (0, 0)
    assert decision.candidate_count == 0
    assert invoke(router, NEW_VOL, invalid_old_slope) == 1
    assert invoke(router, NEW_VOL, invalid_old_slope) == 1
    assert invoke(router, NEW_VOL, invalid_old_slope) == 2


def test_cooldown_allows_opposite_direction_and_non_main_allows_hold_and_reduction():
    router, _ = prepared(k=2)
    router.cooldown_remaining_steps = 1
    router.last_stopped_position = 1.0
    router.agent_act = lambda state, info: 0
    assert invoke(router) == 0
    assert router.gating_strategy.last_decision.reject_reason == "expert_retained"
    router.enable_non_main_contract_defense = True
    router.non_main_defense_mode = "graceful_hold_exit"
    router.role_tier_index = 0
    router.agent_act = lambda state, info: 2
    assert invoke(router, position=1.0, tier=0.0) == 2
    router.agent_act = lambda state, info: 1
    assert invoke(router, position=1.0, tier=0.0) == 1
    assert router.gating_strategy.last_decision.active_slot == (1, 2)
    router.position_list = [-1.0, -0.5, 0.0, 0.5, 1.0]
    router.zero_position_action = router.flat_action = 2
    router.agent_act = lambda state, info: 3
    assert invoke(router, position=1.0, tier=0.0, mask=[1] * 5) == 3


@pytest.mark.parametrize("position,hold", [(1.0, 2), (-1.0, 0)])
def test_empty_slot_uses_finite_buffer_and_never_queries_stale_expert(position, hold):
    router, calls = prepared(k=2, empty=True)
    query_count = len(calls)
    # Existing buffer is four steps; every empty observation consumes one.
    for remaining in [3, 2, 1, 0]:
        assert invoke(router, NEW_VOL, NEW_SLOPE, position=position) == hold
        assert router.remaining_persist == remaining
        assert router.gating_strategy.last_decision.active_slot is None
        assert router.gating_strategy.last_decision.candidate_slot is None
        assert router.last_prediction_id == ""
    for _ in range(2):
        assert invoke(router, NEW_VOL, NEW_SLOPE, position=position) == 1
        assert router.remaining_persist == 0
    assert len(calls) == query_count
    assert invoke(router) == 1
    assert invoke(router) == 2


def test_empty_slot_flat_and_unavailable_hold_have_no_new_risk_or_stale_queries():
    router, calls = prepared(k=2, empty=True)
    count = len(calls)
    assert invoke(router, NEW_VOL, NEW_SLOPE) == 1
    assert router.remaining_persist == 0
    # A fresh real expert grants a finite buffer again, but an unavailable
    # empty-Slot hold action must immediately take the rule path.
    assert invoke(router) == 1
    assert invoke(router) == 2
    count += 1
    assert invoke(router, NEW_VOL, NEW_SLOPE, position=1.0, mask=[1, 1, 0]) == 1
    assert router.remaining_persist == 0
    assert len(calls) == count


@pytest.mark.parametrize("risk", ["stop", "circuit", "non_main"])
def test_required_risk_preempts_empty_hold_buffer(risk):
    router, calls = prepared(k=2, empty=True)
    assert invoke(router, NEW_VOL, NEW_SLOPE, position=1.0) == 2
    assert router.remaining_persist == 3
    kwargs = {"position": 1.0}
    if risk == "stop":
        kwargs.update(current_markprice=100.0, current_unrealized_pnl=-2.0)
    elif risk == "circuit":
        router.circuit_breaker_remaining_steps = 1
    else:
        router.enable_non_main_contract_defense = True
        router.role_tier_index = 0
        kwargs["tier"] = 0.0
    assert invoke(router, NEW_VOL, NEW_SLOPE, **kwargs) == 1
    assert router.remaining_persist == 0
    assert len(calls) == 1


def test_empty_slot_interrupts_existing_candidate_before_recovery():
    router, _ = prepared(k=3, empty=True)
    # Slot3 can be an ordinary pending candidate; Slot0 remains unavailable.
    slope_only = (0.9, 0.2, 0.6)
    assert invoke(router, BASE_VOL, slope_only) == 2
    assert invoke(router, BASE_VOL, slope_only) == 2
    assert router.gating_strategy.last_decision.candidate_count == 2
    assert invoke(router, NEW_VOL, NEW_SLOPE) == 1
    assert router.gating_strategy.last_decision.candidate_count == 0
    assert invoke(router, BASE_VOL, slope_only) == 1
    assert invoke(router, BASE_VOL, slope_only) == 1
    assert invoke(router, BASE_VOL, slope_only) == 2


def test_rule_exit_remains_limited_by_real_liquidity_and_does_not_mutate_account():
    router, _ = prepared(k=2)
    router.position_list = [-1.0, -0.5, 0.0, 0.5, 1.0]
    router.zero_position_action = router.flat_action = 2
    router._defensive_action = vru.vae_risk_aware_routing._defensive_action.__get__(router)
    info = {
        "previous_action": 4,
        "avaliable_action": [0, 0, 0, 1, 1],
        "avaiable_action_list": [3, 4],
        "ask_qyts": [0.5],
        "bid_qyts": [0.5],
    }
    # Hard stop requests only the executable half-position reduction.
    action = router.get_action(info, np.array([1.0]), 1.0, 1, -2.0, 100.0)
    assert action == 3
    assert router.previous_step_position == 1.0
    assert router.gating_strategy.last_decision.reject_reason == "hard_stop_loss"


def test_model_errors_and_out_of_space_actions_are_not_hidden():
    router = make_router(confirm_steps=1)
    router.agent_act = lambda state, info: -1
    with pytest.raises(ValueError, match="outside the configured action space"):
        invoke(router)

    def broken(state, info):
        raise RuntimeError("failed expert")

    router.agent_act = broken
    with pytest.raises(RuntimeError, match="failed expert"):
        invoke(router)
