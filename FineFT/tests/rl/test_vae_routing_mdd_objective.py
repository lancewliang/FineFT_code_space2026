import types
import numpy as np
import pandas as pd
import pytest

from RL.DiHFT.high_level import vae_routing_util as vru
from common.metric_columns import MetricColumns


def test_mdd_penalty_calculation_low_vs_high_drawdown(monkeypatch):
    monkeypatch.setattr(pd, "read_feather", lambda p: pd.DataFrame({"a": [1]}))
    routing = vru.vae_risk_aware_routing.__new__(vru.vae_risk_aware_routing)
    routing.initial_wallet_balance = 6000.0
    routing.test_path = "/tmp/test_vae"
    routing.eval_stage = "valid"
    routing.save_artifacts = False
    routing.enable_mdd_penalty = True
    routing.target_mdd = 0.05
    routing.mdd_penalty_weight = 4.0
    routing.min_trades_guard = 60

    ts = pd.date_range("2024-01-01", periods=100, freq="10min")

    contract_low_mdd = [
        {
            "contract": "fu2409",
            "source_file": "path1",
            "rows": 100,
            "reward_sum": 600.0,
            "require_money": 6000.0,
            "return_rate": 0.10,
            "mdd": 0.02,
            "trades": 80,
            "timestamps": ts.values,
            "step_net_pnls": np.full(100, 6.0),
        }
    ]

    routing.find_contract_files = lambda: [("fu2409", "path1")]
    routing.reset_routing_state = lambda: None
    routing.run_single_valid_df = lambda df, p, contract_name: contract_low_mdd[0]

    score_low_mdd = routing.test()

    expected_base = 600.0 / 12000.0  # Unified 12k portfolio capital
    expected_mdd_factor = 1.0 + 2.0 * (0.05 - 0.0)
    expected_trade_guard = 1.0
    assert score_low_mdd == pytest.approx(expected_base * expected_mdd_factor * expected_trade_guard)

    contract_high_mdd = [
        {
            "contract": "fu2409",
            "source_file": "path1",
            "rows": 100,
            "reward_sum": 600.0,
            "require_money": 6000.0,
            "return_rate": 0.10,
            "mdd": 0.10,
            "trades": 80,
            "timestamps": ts.values,
            "step_net_pnls": np.concatenate([np.full(50, -30.0), np.full(50, 42.0)]),
        }
    ]
    routing.run_single_valid_df = lambda df, p, contract_name: contract_high_mdd[0]
    score_high_mdd = routing.test()

    assert score_high_mdd < score_low_mdd
    assert score_high_mdd < expected_base * 0.5


def test_mdd_penalty_trade_guard_penalizes_inactivity(monkeypatch):
    monkeypatch.setattr(pd, "read_feather", lambda p: pd.DataFrame({"a": [1]}))
    routing = vru.vae_risk_aware_routing.__new__(vru.vae_risk_aware_routing)
    routing.initial_wallet_balance = 6000.0
    routing.test_path = "/tmp/test_vae"
    routing.eval_stage = "valid"
    routing.save_artifacts = False
    routing.enable_mdd_penalty = True
    routing.target_mdd = 0.05
    routing.mdd_penalty_weight = 4.0
    routing.min_trades_guard = 100

    ts = pd.date_range("2024-01-01", periods=50, freq="10min")

    contract = [
        {
            "contract": "fu2409",
            "source_file": "path1",
            "rows": 50,
            "reward_sum": 600.0,
            "require_money": 6000.0,
            "return_rate": 0.10,
            "mdd": 0.02,
            "trades": 25,
            "timestamps": ts.values,
            "step_net_pnls": np.full(50, 12.0),
        }
    ]

    routing.find_contract_files = lambda: [("fu2409", "path1")]
    routing.reset_routing_state = lambda: None
    routing.run_single_valid_df = lambda df, p, contract_name: contract[0]

    score = routing.test()

    base_obj = 600.0 / 12000.0  # Unified 12k portfolio capital
    mdd_factor = 1.0 + 2.0 * (0.05 - 0.0)
    expected_trade_guard = 25.0 / 100.0
    assert score == pytest.approx(base_obj * mdd_factor * expected_trade_guard)


def test_mdd_penalty_disabled_reverts_to_legacy_objective(monkeypatch):
    monkeypatch.setattr(pd, "read_feather", lambda p: pd.DataFrame({"a": [1]}))
    routing = vru.vae_risk_aware_routing.__new__(vru.vae_risk_aware_routing)
    routing.initial_wallet_balance = 6000.0
    routing.test_path = "/tmp/test_vae"
    routing.eval_stage = "valid"
    routing.save_artifacts = False
    routing.enable_mdd_penalty = False

    ts = pd.date_range("2024-01-01", periods=50, freq="10min")

    contract = [
        {
            "contract": "fu2409",
            "source_file": "path1",
            "rows": 50,
            "reward_sum": 600.0,
            "require_money": 6000.0,
            "return_rate": 0.10,
            "mdd": 0.15,
            "trades": 10,
            "timestamps": ts.values,
            "step_net_pnls": np.full(50, 12.0),
        }
    ]

    routing.find_contract_files = lambda: [("fu2409", "path1")]
    routing.reset_routing_state = lambda: None
    routing.run_single_valid_df = lambda df, p, contract_name: contract[0]

    score = routing.test()
    assert score == pytest.approx(600.0 / 12000.0)  # Unified 12k portfolio capital
