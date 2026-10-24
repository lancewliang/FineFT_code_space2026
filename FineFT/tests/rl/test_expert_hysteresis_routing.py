"""Expert ownership is observed through public actions and prediction provenance."""

import numpy as np
import pandas as pd
import pytest

from common import ActionDecisionReasons
from RL.DiHFT.high_level import vae_routing_util as vru
from tests.rl.test_vae_routing_action_persistence import _create_test_router


def make_router(confirm_steps=2, **kwargs):
    router = _create_test_router(action_persistence=5)
    router.gating_strategy = vru.create_gating_strategy(
        "expert_hysteresis",
        slope_threshold=0.6,
        volatility_threshold=0.8,
        hysteresis_exit_ratio=0.5,
        expert_confirm_steps=confirm_steps,
        **kwargs,
    )
    router.axis_window_lengths = {"slope": 2, "volatility": 2}
    router.audit_predictions = []
    router.latest_action_values = [0.0, 0.0, 1.0]
    router.agent_act = lambda state, info: 2
    return router


def act(router, vol=(0.1, 0.8, 0.2), slope=(0.1, 0.2, 0.6), position=0.0, info=None):
    router.calculate_axis_window_result = lambda axis: {"volatility": vol, "slope": slope}[axis]
    router.step_idx += 1
    return router.get_action(
        info or {"previous_action": router.action, "avaliable_action": [1, 1, 1], "trading_info": [position]},
        np.array([router.step_idx], dtype=float),
        position,
        1,
    )


def test_entry_confirmation_without_runner_up_margin_and_fresh_flat_retention():
    router = make_router(expert_switch_lead=0.9)
    seen = []
    outputs = iter([2, 1, 0])

    def expert(state, info):
        seen.append((state.copy(), dict(info), router.selected_agent_index))
        return next(outputs)

    router.agent_act = expert
    # Both axes exactly E; a nearly tied runner-up does not block entry.
    vol = [0.79, 0.8, 0.2]
    slope = [0.1, 0.59, 0.6]
    assert act(router, vol, slope) == 1
    assert router.audit_predictions == []
    assert act(router, vol, slope) == 2
    assert router.remaining_persist > 0
    # Own scores equal H: retain even while Flat, with a rival scoring higher.
    assert act(router, [0.95, 0.4, 0.1], [0.95, 0.1, 0.3], position=1.0) == 1
    info = {"previous_action": 1, "avaliable_action": [1, 1, 0], "trading_info": [0.0]}
    assert act(router, [0.95, 0.4, 0.1], [0.95, 0.1, 0.3], info=info) == 0
    assert [slot for _, _, slot in seen] == [5, 5, 5]
    assert [state[0] for state, _, _ in seen] == [2, 3, 4]
    assert seen[-1][1] == info
    assert [p["selected_action"] for p in router.audit_predictions] == [2, 1, 0]
    assert all(p["routing_slot"] == 5 for p in router.audit_predictions)
    assert router.action_decision_reason_history[-3:] == [ActionDecisionReasons.POLICY_INFERENCE] * 3


@pytest.mark.parametrize("axis", ["volatility", "slope"])
def test_current_own_axis_breach_revokes_immediately_then_recovers(axis):
    router = make_router()
    assert act(router) == 1
    assert act(router) == 2
    vol, slope = [0.99, 0.4, 0.1], [0.99, 0.1, 0.3]
    if axis == "volatility":
        vol[1] = np.nextafter(0.4, 0.0)
    else:
        slope[2] = np.nextafter(0.3, 0.0)
    # The rival's high scores cannot keep the old expert authorized.
    assert act(router, vol, slope, position=1.0) == 1
    assert len(router.audit_predictions) == 1
    assert act(router, vol, slope) == 2
    assert router.audit_predictions[-1]["routing_slot"] == 0


def test_changing_and_ineligible_candidates_restart_public_confirmation():
    router = make_router()
    assert act(router) == 1  # Slot 5, one eligible step
    assert act(router, [0.8, 0.1, 0.2]) == 1  # Slot 2 starts again
    assert act(router, [0.79, 0.1, 0.2]) == 1  # Below E interrupts
    assert act(router, [0.8, 0.1, 0.2]) == 1
    assert act(router, [0.8, 0.1, 0.2]) == 2
    assert [p["routing_slot"] for p in router.audit_predictions] == [2]
    # Completion clears the pending evidence: after invalidation recovery needs K.
    assert act(router, [0.1, 0.1, 0.2]) == 1
    assert act(router, [0.8, 0.1, 0.2]) == 1
    assert act(router, [0.8, 0.1, 0.2]) == 2


@pytest.mark.parametrize("bad", [[], [np.nan, 0.8, 0.2], [0.1, np.inf, 0.2], [-0.1, 0.8, 0.2], [0.1, 1.01, 0.2], [[0.1, 0.8, 0.2]], [0.1, 0.8]])
@pytest.mark.parametrize("axis", ["volatility", "slope"])
def test_invalid_quantiles_fail_at_public_boundary(bad, axis):
    router = make_router()
    act(router)
    with pytest.raises(ValueError, match="scores|dimensions"):
        act(router, vol=bad) if axis == "volatility" else act(router, slope=bad)


@pytest.mark.parametrize("mutation", ["labels", "kind", "slot_id", "formula"])
def test_manifest_identity_change_aborts_public_run(mutation):
    router = make_router()
    act(router)
    if mutation == "labels":
        router.selection_manifest.axes.slope.reverse()
    elif mutation == "formula":
        object.__setattr__(router.selection_manifest, "slot_index_formula", "changed")
    else:
        router.selection_manifest.slots[0][mutation] = "empty_model" if mutation == "kind" else 1
    with pytest.raises(ValueError, match="manifest|Slot mapping"):
        act(router)


@pytest.mark.parametrize("overrides", [
    {"slope_threshold": 0.0}, {"volatility_threshold": 1.01}, {"slope_threshold": np.nan},
    {"hysteresis_exit_ratio": 0.0}, {"hysteresis_exit_ratio": 1.0}, {"hysteresis_exit_ratio": np.nan},
    {"expert_switch_lead": -0.01}, {"expert_switch_lead": np.inf},
    {"expert_confirm_steps": 0}, {"expert_confirm_steps": 1.5}, {"expert_confirm_steps": True},
])
def test_invalid_public_configuration_is_rejected(overrides):
    with pytest.raises(ValueError):
        vru.create_gating_strategy("expert_hysteresis", **overrides)


def test_reset_and_reconfigure_clear_ownership_and_candidate_evidence():
    router = make_router()
    assert act(router) == 1
    router.reset_routing_state()
    assert act(router) == 1
    assert act(router) == 2
    router.save_artifacts = False
    router._resolve_test_path = lambda args: "/unused"
    args = vru.parser.parse_args(["--gating_strategy", "expert_hysteresis", "--expert_confirm_steps", "2"])
    vru.resolve_routing_parameters(args)
    args.slope_rule_base_threshold = 0.6
    args.volatility_rule_base_threshold = 0.8
    router.reconfigure_routing(args)
    assert act(router) == 1
    assert act(router) == 2


def test_rollout_warmup_and_independent_contracts_do_not_share_confirmation(tmp_path, monkeypatch):
    router = make_router()
    router.save_artifacts = False
    router.initial_wallet_balance = 100.0
    router.vae_slope_indicators = []
    router.vae_volatility_indicators = []
    router.tech_indicator_list = []
    router.initial_rollout_window_length = 2
    router.max_holding_number = 1
    router.position_choices = 3
    router.long_estimated_rate = router.short_estimated_rate = router.transcation_cost = 0.0
    router.maintenance_margin_ratio_dict = {}
    router.early_stop = 0
    router.initial_state = (100.0, 0.0, 0.0, 0.0, 1)
    router.order_book_depth = 5
    router.calculate_axis_window_result = lambda axis: {"volatility": [0.1, 0.8, 0.2], "slope": [0.1, 0.2, 0.6]}[axis]
    router.get_quantiles = lambda *args: None
    router._record_trade_audit = lambda **kwargs: None
    environments = []

    class FakeEnv:
        position = unrealized_pnl = 0.0
        current_markprice = 100.0
        leverage = 1
        initial_margin_history = maintain_marigine_history = new_position_required_money_history = [0.0]
        unrealized_pnl_history = [0.0]
        wallet_balance_history = [100.0]
        account_audit_history = []

        def reset(self):
            self.micro_action_history = []
            self.executed_position_history = []
            self.margine_balance_history = [100.0]
            return np.array([0.0]), {"previous_action": 1, "avaliable_action": [1, 1, 1]}

        def step(self, action):
            self.micro_action_history.append(action)
            self.executed_position_history.append(0.0)
            self.margine_balance_history.append(100.0)
            return np.array([0.0]), 0.0, len(self.micro_action_history) == 4, {"previous_action": action, "avaliable_action": [1, 1, 1]}

    def environment(**kwargs):
        env = FakeEnv()
        environments.append(env)
        return env

    monkeypatch.setattr(vru, "initiate_base_env", environment)
    monkeypatch.setattr(vru, "rule_based_close", lambda *args: 1)
    monkeypatch.setattr(vru, "calculate_required_money", lambda *args: 0.0)
    for contract in ["contract_a", "contract_b", "contract_a"]:
        router.run_single_valid_df(pd.DataFrame({"price": [100.0] * 5}), str(tmp_path), contract)
        # Two warm-up steps, then two ordinary entry confirmations.
        assert environments[-1].micro_action_history == [1, 1, 1, 2]
        assert len(router.audit_predictions) == 1
        assert router.audit_predictions[0]["contract"] == contract
