import argparse
import os
import types
from pathlib import Path

from RL.DiHFT.high_level import vae_routing_util as vru
import numpy as np
import pandas as pd
import pytest

from analysis.pick_agent.DiHFT_high_level_heurstic import Picker


def _create_fake_trial_dir(trial_path: Path) -> None:
    contract_dir = trial_path / "contracts" / "fu2409"
    contract_dir.mkdir(parents=True, exist_ok=True)
    n = 30
    np.save(contract_dir / "reward_history.npy", np.ones(n) * 2.0)
    np.save(contract_dir / "initial_margin_history.npy", np.ones(n) * 100.0)
    np.save(contract_dir / "maintain_marigine_history.npy", np.ones(n) * 50.0)
    np.save(contract_dir / "new_position_required_money_history.npy", np.ones(n) * 10.0)
    np.save(contract_dir / "micro_action_history.npy", np.ones(n, dtype=int))
    np.save(contract_dir / "total_asset_history.npy", np.ones(n) * 1000.0)
    np.save(contract_dir / "unrealized_pnl_history.npy", np.zeros(n))
    np.save(contract_dir / "wallet_balance_history.npy", np.ones(n) * 1000.0)


def test_analysis_all_epoch_extracts_and_merges_optuna_parameters(tmp_path: Path):
    dataset_name = "fu"
    exp_name = "10min_test"

    model_root = (
        tmp_path / "result" / "DiHFT" / "high_level" / dataset_name / exp_name / "vae_risk_aware_routing"
    )
    optuna_root = (
        tmp_path / "result" / "DiHFT" / "high_level" / dataset_name / exp_name / "vae_risk_aware_routing_optuna"
    )
    optuna_root.mkdir(parents=True, exist_ok=True)

    # 1. Create optuna_results.csv with 2 trials
    optuna_csv = optuna_root / "optuna_results.csv"
    df_optuna = pd.DataFrame(
        [
            {
                "number": 0,
                "params_slope_window_length": 60,
                "params_volatility_window_length": 40,
                "params_slope_gamma": 0.95,
                "params_volatility_gamma": 0.92,
                "params_slope_rule_base_threshold": 0.30,
                "params_volatility_rule_base_threshold": 0.40,
                "state": "COMPLETE",
            },
            {
                "number": 1,
                "params_slope_window_length": 80,
                "params_volatility_window_length": 50,
                "params_slope_gamma": 0.98,
                "params_volatility_gamma": 0.96,
                "params_slope_rule_base_threshold": 0.25,
                "params_volatility_rule_base_threshold": 0.35,
                "state": "COMPLETE",
            },
        ]
    )
    df_optuna.to_csv(optuna_csv, index=False)

    # 2. Create corresponding trial directories
    trial_0 = model_root / "gamma_0.95_window_60_threshold_0.3_trial_0"
    trial_1 = model_root / "gamma_0.98_window_80_threshold_0.25_trial_1"
    _create_fake_trial_dir(trial_0)
    _create_fake_trial_dir(trial_1)

    save_path = tmp_path / "analysis_result" / "high_level_heurstic"

    args = types.SimpleNamespace(
        base_path=str(tmp_path / "dataset"),
        dataset_name=dataset_name,
        experiment_name=exp_name,
        save_path=str(save_path),
        optuna_csv=str(optuna_csv),
        early_stop=0,
        selection_metric="tr",
        result_path=str(tmp_path / "result" / "DiHFT" / "high_level"),
    )

    picker = Picker(args)
    picker.analysis_all_epoch()

    assert hasattr(picker, "result_df")
    assert len(picker.result_df) == 2

    # Check that the 6 dual-axis parameters are present in result_df
    assert "trial_id" in picker.result_df.columns
    assert "slope_window_length" in picker.result_df.columns
    assert "volatility_window_length" in picker.result_df.columns
    assert "slope_gamma" in picker.result_df.columns
    assert "volatility_gamma" in picker.result_df.columns
    assert "slope_rule_base_threshold" in picker.result_df.columns
    assert "volatility_rule_base_threshold" in picker.result_df.columns

    row_0 = picker.result_df[picker.result_df["trial_id"] == 0].iloc[0]
    assert int(row_0["slope_window_length"]) == 60
    assert int(row_0["volatility_window_length"]) == 40
    assert pytest.approx(float(row_0["slope_gamma"])) == 0.95
    assert pytest.approx(float(row_0["volatility_gamma"])) == 0.92
    assert pytest.approx(float(row_0["slope_rule_base_threshold"])) == 0.30
    assert pytest.approx(float(row_0["volatility_rule_base_threshold"])) == 0.40

    row_1 = picker.result_df[picker.result_df["trial_id"] == 1].iloc[0]
    assert int(row_1["slope_window_length"]) == 80
    assert int(row_1["volatility_window_length"]) == 50

    # Verify saved CSV
    saved_csv = Path(picker.save_path) / "result.csv"
    assert saved_csv.exists()
    df_saved = pd.read_csv(saved_csv)
    assert "slope_window_length" in df_saved.columns
    assert "volatility_gamma" in df_saved.columns


def test_create_best_agent_writes_self_contained_2d_parameters(tmp_path: Path):
    dataset_name = "fu"
    exp_name = "10min_test"

    result_root = tmp_path / "result" / "DiHFT"
    model_root = result_root / "high_level" / dataset_name / exp_name / "vae_risk_aware_routing"
    optuna_root = result_root / "high_level" / dataset_name / exp_name / "vae_risk_aware_routing_optuna"
    optuna_root.mkdir(parents=True, exist_ok=True)

    optuna_csv = optuna_root / "optuna_results.csv"
    df_optuna = pd.DataFrame(
        [
            {
                "number": 0,
                "params_slope_window_length": 60,
                "params_volatility_window_length": 40,
                "params_slope_gamma": 0.95,
                "params_volatility_gamma": 0.92,
                "params_slope_rule_base_threshold": 0.30,
                "params_volatility_rule_base_threshold": 0.40,
                "state": "COMPLETE",
            },
            {
                "number": 1,
                "params_slope_window_length": 80,
                "params_volatility_window_length": 50,
                "params_slope_gamma": 0.98,
                "params_volatility_gamma": 0.96,
                "params_slope_rule_base_threshold": 0.25,
                "params_volatility_rule_base_threshold": 0.35,
                "state": "COMPLETE",
            },
        ]
    )
    df_optuna.to_csv(optuna_csv, index=False)

    trial_0 = model_root / "gamma_0.95_window_60_threshold_0.3_trial_0"
    trial_1 = model_root / "gamma_0.98_window_80_threshold_0.25_trial_1"
    _create_fake_trial_dir(trial_0)
    _create_fake_trial_dir(trial_1)

    save_path = tmp_path / "analysis_result" / "high_level_heurstic"

    args = types.SimpleNamespace(
        base_path=str(tmp_path / "dataset"),
        dataset_name=dataset_name,
        experiment_name=exp_name,
        save_path=str(save_path),
        optuna_csv=str(optuna_csv),
        early_stop=0,
        selection_metric="tr",
        result_path=str(result_root / "high_level"),
    )

    picker = Picker(args)
    picker.analysis_all_epoch()
    picker.analysis_best_epoch()
    picker.create_best_agent()

    final_result_dir = result_root / "final_result" / dataset_name / exp_name
    para_file = final_result_dir / "high_level_agent_para.txt"
    assert para_file.exists()

    para_content = para_file.read_text(encoding="utf-8").strip()
    # Self-contained format must contain 6 explicit dual-axis parameter tags
    for tag in ["ws_", "wv_", "gs_", "gv_", "ts_", "tv_"]:
        assert tag in para_content, f"Missing parameter tag {tag} in {para_content}"

    # Verify resolve_routing_parameters can parse it even without optuna_csv
    mock_final_args = types.SimpleNamespace(
        para_file=str(para_file),
        optuna_csv=None,
        slope_window_length=None,
        volatility_window_length=None,
        slope_gamma=None,
        volatility_gamma=None,
        slope_rule_base_threshold=None,
        volatility_rule_base_threshold=None,
        window_length=10,
        gamma=0.9,
        rule_base_threshold=0.2,
    )
    resolved_args = vru.resolve_routing_parameters(mock_final_args)
    assert resolved_args.slope_window_length in (60, 80)
    assert resolved_args.volatility_window_length in (40, 50)


def _create_custom_contract_dir(contract_dir: Path, reward_val: float, req_val: float, n: int = 30):
    contract_dir.mkdir(parents=True, exist_ok=True)
    np.save(contract_dir / "reward_history.npy", np.ones(n) * reward_val)
    np.save(contract_dir / "initial_margin_history.npy", np.ones(n) * req_val)
    np.save(contract_dir / "maintain_marigine_history.npy", np.ones(n) * req_val)
    np.save(contract_dir / "new_position_required_money_history.npy", np.ones(n) * req_val)
    np.save(contract_dir / "micro_action_history.npy", np.ones(n, dtype=int))
    np.save(contract_dir / "total_asset_history.npy", np.ones(n) * 1000.0)
    np.save(contract_dir / "unrealized_pnl_history.npy", np.zeros(n))
    np.save(contract_dir / "wallet_balance_history.npy", np.ones(n) * req_val)


def test_selection_metric_prioritizes_metric_and_picks_best_agent(tmp_path: Path):
    dataset_name = "fu"
    exp_name = "10min_test"

    result_root = tmp_path / "result" / "DiHFT"
    model_root = result_root / "high_level" / dataset_name / exp_name / "vae_risk_aware_routing"
    optuna_root = result_root / "high_level" / dataset_name / exp_name / "vae_risk_aware_routing_optuna"
    optuna_root.mkdir(parents=True, exist_ok=True)

    optuna_csv = optuna_root / "optuna_results.csv"
    df_optuna = pd.DataFrame(
        [
            {
                "number": 0,
                "params_slope_window_length": 60,
                "params_volatility_window_length": 40,
                "params_slope_gamma": 0.95,
                "params_volatility_gamma": 0.92,
                "params_slope_rule_base_threshold": 0.30,
                "params_volatility_rule_base_threshold": 0.40,
                "state": "COMPLETE",
            },
            {
                "number": 1,
                "params_slope_window_length": 80,
                "params_volatility_window_length": 50,
                "params_slope_gamma": 0.98,
                "params_volatility_gamma": 0.96,
                "params_slope_rule_base_threshold": 0.25,
                "params_volatility_rule_base_threshold": 0.35,
                "state": "COMPLETE",
            },
        ]
    )
    df_optuna.to_csv(optuna_csv, index=False)

    trial_0 = model_root / "trial_0"
    trial_1 = model_root / "trial_1"

    # Trial 0: higher tr (4.83), lower portfolio_tr (0.15)
    _create_custom_contract_dir(trial_0 / "contracts" / "fu_a", reward_val=10.0, req_val=10.0)
    _create_custom_contract_dir(trial_0 / "contracts" / "fu_b", reward_val=0.0, req_val=1000.0)

    # Trial 1: lower tr (1.12), higher portfolio_tr (1.20)
    _create_custom_contract_dir(trial_1 / "contracts" / "fu_a", reward_val=4.0, req_val=50.0)
    _create_custom_contract_dir(trial_1 / "contracts" / "fu_b", reward_val=4.0, req_val=50.0)

    save_path = tmp_path / "analysis_result" / "high_level_heurstic"

    # Case 1: Pick with selection_metric="portfolio_tr" -> should select trial 1
    args_portfolio = types.SimpleNamespace(
        base_path=str(tmp_path / "dataset"),
        dataset_name=dataset_name,
        experiment_name=exp_name,
        save_path=str(save_path),
        optuna_csv=str(optuna_csv),
        early_stop=0,
        selection_metric="portfolio_tr",
        result_path=str(result_root / "high_level"),
    )
    picker_portfolio = Picker(args_portfolio)
    picker_portfolio.analysis_all_epoch()
    picker_portfolio.analysis_best_epoch()
    picker_portfolio.create_best_agent()

    para_file = result_root / "final_result" / dataset_name / exp_name / "high_level_agent_para.txt"
    assert "trial_1" in para_file.read_text(encoding="utf-8")
    assert picker_portfolio.best_result_df.iloc[0]["indicator"] == "portfolio_tr"
    assert "portfolio_tr" in picker_portfolio.best_result_df["indicator"].values

    # Case 2: Pick with selection_metric="tr" -> should select trial 0
    args_tr = types.SimpleNamespace(
        base_path=str(tmp_path / "dataset"),
        dataset_name=dataset_name,
        experiment_name=exp_name,
        save_path=str(save_path),
        optuna_csv=str(optuna_csv),
        early_stop=0,
        selection_metric="tr",
        result_path=str(result_root / "high_level"),
    )
    picker_tr = Picker(args_tr)
    picker_tr.analysis_all_epoch()
    picker_tr.analysis_best_epoch()
    picker_tr.create_best_agent()

    assert "trial_0" in para_file.read_text(encoding="utf-8")
    assert picker_tr.best_result_df.iloc[0]["indicator"] == "tr"


def test_high_level_heurstic_fu_10_script_configuration():
    script_path = Path(__file__).resolve().parents[2] / "script" / "analysis" / "pick_agent" / "high_level_heurstic_fu_10.sh"
    assert script_path.exists()
    content = script_path.read_text(encoding="utf-8")

    assert "BASE_PATH=${BASE_PATH:-dataset/10min}" in content
    assert "DATASET_NAME=${DATASET_NAME:-fu}" in content
    assert "EXPERIMENT_NAME=${EXPERIMENT_NAME:-10min_parallel}" in content
    assert "RESULT_PATH=${RESULT_PATH:-result/DiHFT/high_level}" in content
    assert "SELECTION_METRIC=${SELECTION_METRIC:-tr}" in content
    assert "EARLY_STOP=${EARLY_STOP:-0}" in content
    assert "FOREGROUND=${FOREGROUND:-0}" in content
    assert "--selection_metric" in content
    assert "FineFT/analysis/pick_agent/DiHFT_high_level_heurstic.py" in content
