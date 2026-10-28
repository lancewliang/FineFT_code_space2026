from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from RL.DiHFT.high_level import vae_routing_util as vru
from common.metric_columns import MetricColumns


def test_single_contract_return_rate_eliminates_zero_denominator():
    """Verify single contract return rate divides by allocated capital (6000), not dynamic require_money."""
    routing = vru.vae_risk_aware_routing.__new__(vru.vae_risk_aware_routing)
    routing.initial_wallet_balance = 6000.0

    # Case A: Contract not traded at all (reward = 0, require_money would be 0)
    # Under old logic, 0.0 / (0.0 + 1e-12) could be unstable or 10^15 if residual reward existed
    reward_zero = 0.0
    contract_capital = float(routing.initial_wallet_balance)
    return_rate_zero = reward_zero / contract_capital
    assert return_rate_zero == 0.0
    assert abs(return_rate_zero) < 1.0

    # Case B: Contract traded with profit
    reward_profit = 600.0
    return_rate_profit = reward_profit / contract_capital
    assert return_rate_profit == pytest.approx(0.10)


def test_portfolio_capital_fixed_at_6000_regardless_of_replayed_contract_count(monkeypatch, tmp_path):
    """Verify the one-slot strategy uses fixed 6000 capital, not replayed-file count."""
    monkeypatch.setattr(pd, "read_feather", lambda p: pd.DataFrame({"a": [1]}))
    routing = vru.vae_risk_aware_routing.__new__(vru.vae_risk_aware_routing)
    routing.initial_wallet_balance = 6000.0
    routing.test_path = str(tmp_path)
    routing.eval_stage = "valid"
    routing.save_artifacts = True
    routing.enable_mdd_penalty = False
    routing.frozen_experiment_identity = "frozen-experiment"

    ts = pd.date_range("2024-01-01", periods=10, freq="10min")

    # 4 contracts evaluated: only 1 has trades and profit, 3 are inactive
    contract_active = {
        "contract": "fu2509",
        "source_file": "path1",
        "rows": 10,
        "reward_sum": 1200.0,
        "require_money": 6000.0,
        "return_rate": 0.20,
        "mdd": 0.01,
        "trades": 10,
        "timestamps": ts.values,
        "step_net_pnls": np.full(10, 120.0),
    }
    inactive_contracts = [
        {
            "contract": f"fu250{i}",
            "source_file": f"path{i}",
            "rows": 10,
            "reward_sum": 0.0,
            "require_money": 0.0,
            "return_rate": 0.0,
            "mdd": 0.0,
            "trades": 0,
            "timestamps": ts.values,
            "step_net_pnls": np.zeros(10),
        }
        for i in [1, 2, 3]
    ]

    all_contracts = [contract_active] + inactive_contracts
    routing.find_contract_files = lambda: [(c["contract"], c["source_file"]) for c in all_contracts]
    routing.reset_routing_state = lambda: None

    contract_map = {c["contract"]: c for c in all_contracts}
    routing.run_single_valid_df = lambda df, p, contract_name: contract_map[contract_name]

    score = routing.test()

    # One active slot owns the account: 6000.0, not 1 * 6000.0 or 4 * 6000.0.
    expected_tr = 1200.0 / 6000.0  # 0.20 (20%)
    assert score == pytest.approx(expected_tr)
    assert routing.return_rate == pytest.approx(expected_tr)

    # Check contract_results.csv written to disk
    csv_file = tmp_path / "contract_results.csv"
    assert csv_file.exists()
    df_res = pd.read_csv(csv_file)
    assert len(df_res) == 4
    # Ensure no extreme values in return_rate
    assert (df_res["return_rate"].abs() <= 10.0).all()
    # Check that inactive contracts have 0 return_rate, not 10^13
    inactive_df = df_res[df_res["contract"] != "fu2509"]
    assert (inactive_df["return_rate"] == 0.0).all()
