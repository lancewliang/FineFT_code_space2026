from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from RL.DiHFT.high_level import vae_routing_util as vru
from RL.DiHFT.high_level.gating import create_gating_strategy
from RL.DiHFT.high_level.trading_utility.gate import UtilityAdmissionDecision
from RL.DiHFT.high_level.tri_module import TriModuleGatingStack
from common import ActionDecisionReasons


class _StaticUtilityGate:
    def __init__(self, *, admitted: bool, raises: bool = False) -> None:
        self.admitted = admitted
        self.raises = raises

    def decide(self, **kwargs) -> UtilityAdmissionDecision:
        if self.raises:
            raise ValueError("feature schema mismatch")
        final_action = kwargs["candidate_action"] if self.admitted else kwargs["flat_action"]
        return UtilityAdmissionDecision(
            admitted=self.admitted,
            transition="open",
            candidate_action=kwargs["candidate_action"],
            denied_baseline_action=kwargs["flat_action"],
            final_action=final_action,
            channel_eligible=True,
            reject_stage=None if self.admitted else "utility_lower_bound",
            reject_reason=None if self.admitted else "utility_lower_not_positive",
            utility_raw_bps=2.0,
            utility_calibrated_bps=1.5,
            utility_lower_bps=1.0 if self.admitted else -0.1,
            win_probability_calibrated=0.55,
            utility_artifact_id="runtime-1",
        )


def _router(gate: _StaticUtilityGate):
    router = vru.vae_risk_aware_routing.__new__(vru.vae_risk_aware_routing)
    router.num_labels = 2
    router.slot_count = 4
    router.selection_manifest = SimpleNamespace(
        slot_count=4,
        slots=[
            {"slot_id": index, "kind": "model", "candidate_id": f"model-{index}"}
            for index in range(4)
        ],
    )
    router.tri_module_stack = TriModuleGatingStack(
        num_labels=2,
        selection_manifest=router.selection_manifest,
        utility_gate=gate,
        recovery_steps=2,
    )
    router.high_level_gating_stack = "tri_module"
    router.active_high_level_gating_stack = router.tri_module_stack
    router.utility_runtime_artifact_id = "runtime-1"
    router.gating_strategy = create_gating_strategy(
        "absolute", slope_threshold=0.2, volatility_threshold=0.2
    )
    router.calculate_axis_window_result = lambda axis: [0.8, 0.2]
    router.agent_act = lambda state, info: 2
    router.zero_position_action = 1
    router.flat_action = 1
    router.leverage_choices = [5]
    router.position_list = [-1.0, 0.0, 1.0]
    router.action_persistence = 1
    router.remaining_persist = 0
    router.current_action = 1
    router.action = 1
    router.enable_non_main_contract_defense = False
    router.role_tier_index = None
    router.previous_step_position = 0.0
    router.active_trade_stopped = False
    router.consecutive_stop_loss_count = 0
    router.circuit_breaker_remaining_steps = 0
    router.cooldown_remaining_steps = 0
    router.stop_loss_return_threshold = 0.0
    router.circuit_breaker_consecutive_stops = 0
    router.circuit_breaker_cooling_steps = 0
    router.stop_loss_cooldown_steps = 0
    router.last_stopped_position = 0.0
    router.macro_action_history = []
    router.action_decision_reason_history = []
    router.audit_predictions = []
    router.latest_action_values = [0.1, 0.2, 0.7]
    router.latest_tri_module_market_decision = None
    router.latest_utility_admission_decision = None
    router.df = None
    router.step_idx = 0
    router.run_id = "run-integration"
    router.current_contract_name = "FU"
    router.model_id = "model"
    router.feature_version = "features-v1"
    router.parameter_version = "params-v1"
    router.last_prediction_id = ""
    return router


def _info() -> dict[str, object]:
    return {"avaliable_action": np.asarray([1, 1, 1], dtype=int)}


def test_tri_module_rejection_changes_only_final_action_and_is_audited() -> None:
    router = _router(_StaticUtilityGate(admitted=False))

    action = router.get_action(_info(), np.asarray([0.25]), 0.0, 5)

    assert action == 1
    assert router.action_decision_reason_history[-1] == ActionDecisionReasons.UTILITY_REJECTION
    record = router.audit_predictions[-1]
    assert record["gating_stack"] == "tri_module"
    assert record["routing_slot"] == 0
    assert record["candidate_id"] == "model-0"
    assert record["candidate_kind"] == "model"
    assert record["candidate_action"] == 2
    assert record["denied_baseline_action"] == 1
    assert record["final_action"] == 1
    assert record["utility_reject_stage"] == "utility_lower_bound"


def test_tri_module_positive_utility_admits_candidate() -> None:
    router = _router(_StaticUtilityGate(admitted=True))

    action = router.get_action(_info(), np.asarray([0.25]), 0.0, 5)

    assert action == 2
    assert router.action_decision_reason_history[-1] == ActionDecisionReasons.POLICY_INFERENCE
    assert router.audit_predictions[-1]["final_action"] == 2
    assert router.audit_predictions[-1]["utility_lower"] == 1.0


def test_tri_module_runtime_validation_failure_rejects_risk_increase() -> None:
    router = _router(_StaticUtilityGate(admitted=True, raises=True))

    action = router.get_action(_info(), np.asarray([0.25]), 0.0, 5)

    assert action == 1
    decision = router.latest_utility_admission_decision
    assert decision.reject_stage == "asset_validation"
    assert decision.reject_reason == "runtime_validation:ValueError"
    assert router.audit_predictions[-1]["final_action"] == 1


def test_tri_module_invalid_input_degrades_before_candidate_query() -> None:
    router = _router(_StaticUtilityGate(admitted=True))
    called = False

    def _agent_act(state, info):
        nonlocal called
        called = True
        return 2

    router.agent_act = _agent_act
    info = _info()
    info["data_gap_detected"] = True

    action = router.get_action(info, np.asarray([0.25]), 0.0, 5)

    assert action == 1
    assert called is False
    assert router.latest_tri_module_market_decision.ood.reason == "data_invalid:data_gap_detected"
    record = router.audit_predictions[-1]
    assert record["ood_state"] == "enter"
    assert record["ood_reason"] == "data_invalid:data_gap_detected"
    assert record["candidate_action"] is None
    assert record["final_action"] == 1


def test_tri_module_null_candidate_is_audited_without_utility_call() -> None:
    gate = _StaticUtilityGate(admitted=True, raises=True)
    router = _router(gate)
    router.selection_manifest.slots[0] = {"slot_id": 0, "kind": "empty_model"}
    router.tri_module_stack = TriModuleGatingStack(
        num_labels=2,
        selection_manifest=router.selection_manifest,
        utility_gate=gate,
        recovery_steps=2,
    )

    action = router.get_action(_info(), np.asarray([0.25]), 0.0, 5)

    assert action == 1
    record = router.audit_predictions[-1]
    assert record["candidate_kind"] == "null_candidate"
    assert record["candidate_id"] is None
    assert record["candidate_action"] is None
