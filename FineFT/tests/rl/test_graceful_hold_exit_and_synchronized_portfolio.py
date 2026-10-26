import json
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from analysis.diagnostics.synchronized_portfolio_diagnostics import (
    SynchronizedPortfolioCalculator,
)
from analysis.pick_agent.FineFT_two_dimensional_agent_selector import (
    TwoDimensionalSelectionManifest,
)
from common import ActionDecisionReasons, ArtifactNames, HistoryArtifactNames
from RL.DiHFT.high_level import vae_routing_util as vru


def _sample_manifest(num_labels: int = 3) -> TwoDimensionalSelectionManifest:
    labels = [f"label_{i}" for i in range(num_labels)]
    payload = {
        "schema_version": 1,
        "selection_method": "test",
        "candidate_root": "/tmp",
        "valid_root": "/tmp",
        "axes": {"volatility": labels, "slope": labels},
        "slot_count": num_labels * num_labels,
        "slot_index_formula": "volatility_index * num_labels + slope_index",
        "null_policy": {"logical_kind": "empty_model", "intended_runtime_behavior": "flat_position", "model_assembly_status": "built_as_flat_qnet"},
        "candidate_scope": {"common_epochs": [1], "discovered_candidate_count": 1, "complete_candidate_count": 1, "excluded_incomplete_candidate_count": 0, "initial_actions": [0]},
        "selection_config": {},
        "metric_definition": {"return": "mean", "aggregation": "mean", "lcb": "lcb", "pair_score": "pair", "joint_context_note": "test"},
        "artifacts": {"model_assembly": "/tmp/model.pth", "high_level_model_change": "none"},
        "slots": [{"slot_id": i, "kind": "model"} for i in range(num_labels * num_labels)],
    }
    return TwoDimensionalSelectionManifest.from_dict(payload)


def _build_router(non_main_mode: str = "graceful_hold_exit") -> vru.vae_risk_aware_routing:
    r = vru.vae_risk_aware_routing.__new__(vru.vae_risk_aware_routing)
    r.num_labels = 3
    r.slot_count = 9
    r.rule_base_threshold = 0.2
    r.axis_thresholds = {"volatility": 0.2, "slope": 0.2}
    r.gating_strategy = vru.create_gating_strategy("absolute", slope_threshold=0.2, volatility_threshold=0.2)
    r.selection_manifest = _sample_manifest(3)
    r.enable_non_main_contract_defense = True
    r.non_main_defense_mode = non_main_mode
    r.role_tier_index = 0
    r.zero_position_action = 1  # [-1.0, 0.0, 1.0] -> actions: 0: -1.0, 1: 0.0, 2: 1.0
    r.leverage_choices = [1]
    r.position_list = [-1.0, 0.0, 1.0]
    r.action = 1
    r.flat_action = 1
    r.current_action = 1
    r.remaining_persist = 0
    r.action_persistence = 1
    r.cooldown_remaining_steps = 0
    r.circuit_breaker_remaining_steps = 0
    r.stop_loss_return_threshold = 0.0
    r.previous_step_position = 0.0
    r.consecutive_stop_loss_count = 0
    r.active_trade_stopped = False
    r.last_stopped_position = 0.0
    r.macro_action_history = []
    r.action_decision_reason_history = []
    r.calculate_axis_window_result = lambda axis: {"volatility": [0.1, 0.8, 0.1], "slope": [0.1, 0.1, 0.8]}[axis]
    return r


def test_graceful_hold_exit_forbids_opening_when_flat():
    router = _build_router(non_main_mode="graceful_hold_exit")
    router.agent_act = lambda s, info: 2  # candidate wants to open long (+1.0)

    # s[0] = 0.0 (non-main)
    state = np.array([0.0, 100.0, 200.0])
    info = {"avaliable_action": [1, 1, 1], "ask_qyts": [1000], "bid_qyts": [1000], "avaiable_action_list": [0, 1, 2]}

    # current_position = 0.0
    action = router.get_action(info, state, current_position=0.0, current_leverage=1)
    # Must remain flat (action 1 -> pos 0.0)
    assert action == 1
    assert router.action_decision_reason_history[-1] == ActionDecisionReasons.DEFENSIVE_RULE_CLOSE


def test_graceful_hold_exit_allows_holding_existing_position():
    router = _build_router(non_main_mode="graceful_hold_exit")
    router.agent_act = lambda s, info: 2  # candidate wants to hold long (+1.0)

    state = np.array([0.0, 100.0, 200.0])
    info = {"avaliable_action": [1, 1, 1], "ask_qyts": [1000], "bid_qyts": [1000], "avaiable_action_list": [0, 1, 2]}

    # current_position = 1.0
    action = router.get_action(info, state, current_position=1.0, current_leverage=1)
    # Holding is allowed!
    assert action == 2
    assert router.action_decision_reason_history[-1] == ActionDecisionReasons.POLICY_INFERENCE


def test_graceful_hold_exit_allows_closing_existing_position():
    router = _build_router(non_main_mode="graceful_hold_exit")
    router.agent_act = lambda s, info: 1  # candidate wants to close to 0.0

    state = np.array([0.0, 100.0, 200.0])
    info = {"avaliable_action": [1, 1, 1], "ask_qyts": [1000], "bid_qyts": [1000], "avaiable_action_list": [0, 1, 2]}

    # current_position = 1.0
    action = router.get_action(info, state, current_position=1.0, current_leverage=1)
    # Closing is allowed!
    assert action == 1
    assert router.action_decision_reason_history[-1] == ActionDecisionReasons.POLICY_INFERENCE


def test_graceful_hold_exit_intercepts_reversal_to_flat():
    router = _build_router(non_main_mode="graceful_hold_exit")
    router.agent_act = lambda s, info: 0  # candidate wants to reverse to short (-1.0)

    state = np.array([0.0, 100.0, 200.0])
    info = {"avaliable_action": [1, 1, 1], "ask_qyts": [1000], "bid_qyts": [1000], "avaiable_action_list": [0, 1, 2]}

    # current_position = 1.0
    action = router.get_action(info, state, current_position=1.0, current_leverage=1)
    # Reversal intercepted to flat 0.0 (action 1)
    assert action == 1
    assert router.action_decision_reason_history[-1] == ActionDecisionReasons.DEFENSIVE_RULE_CLOSE


def test_synchronized_portfolio_calculator_e2e(tmp_path):
    data_dir = tmp_path / "valid"
    data_dir.mkdir(parents=True)
    result_dir = tmp_path / "final_result"
    contracts_dir = result_dir / "contracts"
    contracts_dir.mkdir(parents=True)

    # Contract 1: Main for first 3 steps, then sub
    c1_dir = contracts_dir / "c1"
    c1_dir.mkdir()
    np.save(c1_dir / HistoryArtifactNames.EXECUTED_POSITION_HISTORY_NPY, np.array([1.0, 1.0, 0.0, 0.0]))
    np.save(c1_dir / HistoryArtifactNames.REWARD_HISTORY_NPY, np.array([10.0, 20.0, 0.0, 0.0]))
    np.save(c1_dir / HistoryArtifactNames.MICRO_ACTION_HISTORY_NPY, np.array([2, 2, 1, 1]))
    np.save(c1_dir / HistoryArtifactNames.MACRO_ACTION_HISTORY_NPY, np.array([0, 0, 9, 9]))
    np.save(c1_dir / HistoryArtifactNames.ACTION_DECISION_REASON_HISTORY_NPY, np.array([0, 0, 3, 3]))

    df_c1 = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01 09:00", periods=5, freq="30min"),
        "prev_day_contract_role_tier": [1.0, 1.0, 0.5, 0.5, 0.0],
        "mark_price": [100.0, 105.0, 110.0, 110.0, 110.0],
    })
    df_c1.to_feather(data_dir / "c1.feather")

    # Contract 2: Sub for first 2 steps, then Main
    c2_dir = contracts_dir / "c2"
    c2_dir.mkdir()
    np.save(c2_dir / HistoryArtifactNames.EXECUTED_POSITION_HISTORY_NPY, np.array([0.0, 1.0, 1.0, 0.0]))
    np.save(c2_dir / HistoryArtifactNames.REWARD_HISTORY_NPY, np.array([0.0, 15.0, 25.0, 0.0]))
    np.save(c2_dir / HistoryArtifactNames.MICRO_ACTION_HISTORY_NPY, np.array([1, 2, 2, 1]))
    np.save(c2_dir / HistoryArtifactNames.MACRO_ACTION_HISTORY_NPY, np.array([9, 0, 0, 9]))
    np.save(c2_dir / HistoryArtifactNames.ACTION_DECISION_REASON_HISTORY_NPY, np.array([3, 0, 0, 3]))

    df_c2 = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01 09:00", periods=5, freq="30min"),
        "prev_day_contract_role_tier": [0.5, 0.5, 1.0, 1.0, 0.0],
        "mark_price": [200.0, 202.0, 205.0, 208.0, 208.0],
    })
    df_c2.to_feather(data_dir / "c2.feather")

    calc = SynchronizedPortfolioCalculator(
        result_dir=result_dir,
        data_dir=data_dir,
        single_contract_capital=6000.0,
        contract_unit=1.0,
        freq=2,
        eval_stage="valid",
    )
    res = calc.run()

    assert "portfolio" in res
    assert "contracts" in res
    p = res["portfolio"]
    assert p["initial_capital"] == 12000.0
    # Step 0: c1 pnl = 10, c2 = 0 -> 10
    # Step 1: c1 pnl = 20, c2 = 15 -> 35
    # Step 2: c1 pnl = 0, c2 = 25 -> 25
    # Step 3: c1 pnl = 0, c2 = 0 -> 0
    # Total net PnL = 70
    assert p["total_net_pnl"] == pytest.approx(70.0)
    assert p["tr"] == pytest.approx(70.0 / 12000.0)
    assert p["max_concurrent_contracts"] <= 2

    timeline_csv = result_dir / ArtifactNames.PORTFOLIO_ACTION_TIMELINE_CSV
    assert timeline_csv.is_file()
    df_t = pd.read_csv(timeline_csv)
    assert len(df_t) == 4
    assert (df_t["holding_contract_count"] <= 2).all()


def test_synchronized_portfolio_uses_explicit_frozen_account_total(tmp_path):
    """A frozen replay must not silently create a second wallet per contract."""
    result_dir = tmp_path / "result"
    data_dir = tmp_path / "valid"
    contracts_dir = result_dir / "contracts"
    contracts_dir.mkdir(parents=True)
    data_dir.mkdir()
    for contract, reward in (("c1", 10.0), ("c2", 20.0)):
        contract_dir = contracts_dir / contract
        contract_dir.mkdir()
        np.save(contract_dir / HistoryArtifactNames.EXECUTED_POSITION_HISTORY_NPY, np.array([0.0]))
        np.save(contract_dir / HistoryArtifactNames.REWARD_HISTORY_NPY, np.array([reward]))
        np.save(contract_dir / HistoryArtifactNames.MICRO_ACTION_HISTORY_NPY, np.array([1]))
        np.save(contract_dir / HistoryArtifactNames.MACRO_ACTION_HISTORY_NPY, np.array([9]))
        np.save(contract_dir / HistoryArtifactNames.ACTION_DECISION_REASON_HISTORY_NPY, np.array([3]))
        pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=2, freq="10min"),
                "prev_day_contract_role_tier": [1.0, 1.0],
                "mark_price": [100.0, 100.0],
            }
        ).to_feather(data_dir / f"{contract}.feather")

    result = SynchronizedPortfolioCalculator(
        result_dir=result_dir,
        data_dir=data_dir,
        single_contract_capital=6000.0,
        portfolio_initial_capital=12_000.0,
        freq=1,
    ).run()

    assert result["portfolio"]["initial_capital"] == 12_000.0
    assert result["portfolio"]["tr"] == pytest.approx(30.0 / 12_000.0)


def test_log_initialized_attributes_runs_cleanly(caplog):
    import logging
    router = _build_router(non_main_mode="graceful_hold_exit")
    with caplog.at_level(logging.INFO):
        router._log_initialized_attributes()
    assert "初始化完成属性列表" in caplog.text
    assert "enable_non_main_contract_defense" in caplog.text
    assert "non_main_defense_mode" in caplog.text
