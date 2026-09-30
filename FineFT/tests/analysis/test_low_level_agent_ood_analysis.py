from __future__ import annotations

import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
FINEFT_ROOT = REPO_ROOT / "FineFT"
if str(FINEFT_ROOT) not in sys.path:
    sys.path.insert(0, str(FINEFT_ROOT))

from common import ArtifactNames
from model.low_level import ensemble_Qnet
from analysis.feature.low_level_agent_ood_analysis import (
    BaselineStats,
    compute_step_ood_metrics,
    forward_ensemble_with_latents,
    load_ensemble_model_from_checkpoint,
    main,
)


def test_compute_step_ood_metrics_unanimous():
    batch_size = 3
    ensemble_size = 5
    n_actions = 4
    hidden_dim = 8

    # All 5 ensemble models predict identical Q values: action 2 has highest Q value
    q_single = torch.tensor([1.0, 2.0, 10.0, 3.0], dtype=torch.float32)
    q_values = q_single.repeat(batch_size, ensemble_size, 1)  # (B, M, A)
    mean_latents = torch.zeros((batch_size, hidden_dim), dtype=torch.float32)

    metrics = compute_step_ood_metrics(q_values, mean_latents, baseline=None)

    # Q variance must be strictly 0.0 because all models agree
    assert np.allclose(metrics["q_variance"], 0.0)
    # Disagreement rate must be strictly 0.0
    assert np.allclose(metrics["disagreement_rate"], 0.0)
    # Consensus action is action 2
    assert (metrics["consensus_action"] == 2).all()


def test_compute_step_ood_metrics_divergent_and_mahalanobis():
    batch_size = 1
    ensemble_size = 4
    n_actions = 4
    hidden_dim = 2

    # Model 0 prefers action 0, Model 1 prefers 1, Model 2 prefers 2, Model 3 prefers 3
    q_values = torch.zeros((batch_size, ensemble_size, n_actions), dtype=torch.float32)
    q_values[0, 0, 0] = 10.0
    q_values[0, 1, 1] = 10.0
    q_values[0, 2, 2] = 10.0
    q_values[0, 3, 3] = 10.0

    mean_latents = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

    # Baseline with mu = [0, 0] and precision = diag([4.0, 1.0])
    baseline = BaselineStats(
        mu_latent=np.zeros(2),
        precision_matrix=np.diag([4.0, 1.0]),
        q_var_quantiles={90: 1.0, 95: 2.0, 99: 3.0},
        disagree_quantiles={90: 0.5, 95: 0.7, 99: 0.8},
        mahalanobis_quantiles={90: 2.0, 95: 3.0, 99: 4.0},
        baseline_mahalanobis_values=np.array([1.0, 2.0]),
        sample_count=2,
    )

    metrics = compute_step_ood_metrics(q_values, mean_latents, baseline=baseline)

    # Disagreement rate: 4 different votes -> max_vote=1 -> 1.0 - 1/4 = 0.75
    assert metrics["disagreement_rate"][0] == pytest.approx(0.75, abs=1e-5)
    # Q variance must be positive
    assert metrics["q_variance"][0] > 0.0
    # Mahalanobis distance = sqrt(1.0^2 * 4.0 + 2.0^2 * 1.0) = sqrt(4 + 4) = sqrt(8)
    expected_mahal = np.sqrt(8.0)
    assert metrics["mahalanobis_distance"][0] == pytest.approx(expected_mahal, abs=1e-5)


def test_forward_ensemble_with_latents_shapes():
    n_states = 6
    n_actions = 3
    hidden_nodes = 16
    time_info_dim = 2
    ensemble_size = 4
    batch_size = 5

    model = ensemble_Qnet(
        N_STATES=n_states,
        N_ACTIONS=n_actions,
        hidden_nodes=hidden_nodes,
        TIME_INFO_DIM=time_info_dim,
        ensemble_number=ensemble_size,
        TRADING_INFO_DIM=4,
    )

    state = torch.randn(batch_size, n_states)
    time_input = torch.randn(batch_size, time_info_dim)
    prev_act = torch.zeros(batch_size, 1)
    avail_act = torch.ones(batch_size, n_actions)
    trading_info = torch.zeros(batch_size, 4)

    q_vals, latents = forward_ensemble_with_latents(
        model, state, time_input, prev_act, avail_act, trading_info
    )

    assert q_vals.shape == (batch_size, ensemble_size, n_actions)
    assert latents.shape == (batch_size, hidden_nodes)


def test_load_ensemble_model_from_checkpoint(tmp_path: Path):
    n_states = 8
    n_actions = 3
    hidden_nodes = 32
    time_info_dim = 2
    ensemble_size = 3

    model = ensemble_Qnet(
        N_STATES=n_states,
        N_ACTIONS=n_actions,
        hidden_nodes=hidden_nodes,
        TIME_INFO_DIM=time_info_dim,
        ensemble_number=ensemble_size,
        TRADING_INFO_DIM=4,
    )

    ckpt_path = tmp_path / "trained_model.pkl"
    torch.save(model.state_dict(), ckpt_path)

    loaded_model, config = load_ensemble_model_from_checkpoint(ckpt_path, torch.device("cpu"))

    assert config["ensemble_number"] == ensemble_size
    assert config["hidden_nodes"] == hidden_nodes
    assert config["n_states"] == n_states
    assert config["n_actions"] == n_actions
    assert config["time_info_dim"] == time_info_dim
    assert isinstance(loaded_model, ensemble_Qnet)


def test_low_level_agent_ood_analysis_end_to_end(tmp_path: Path, monkeypatch):
    n_states = 4
    n_actions = 3
    hidden_nodes = 16
    ensemble_size = 2
    feature_names = [f"feat_{i}" for i in range(n_states)]

    # 1. Save dummy feature list
    feature_file = tmp_path / ArtifactNames.RL_STATE_FEATURES_NPY
    np.save(feature_file, np.array(feature_names))

    # 2. Save dummy model checkpoint
    model = ensemble_Qnet(
        N_STATES=n_states,
        N_ACTIONS=n_actions,
        hidden_nodes=hidden_nodes,
        TIME_INFO_DIM=2,
        ensemble_number=ensemble_size,
        TRADING_INFO_DIM=4,
    )
    model_dir = tmp_path / "model_epoch"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "trained_model.pkl"
    torch.save(model.state_dict(), model_path)

    # 3. Save dummy buffer_diverse.pkl with 9 grids
    buffer_payload = {}
    for g in range(9):
        n_samples = 30
        buffer_payload[g] = {
            "states": torch.randn(n_samples, n_states),
            "actions": torch.zeros(n_samples, 1, dtype=torch.int64),
            "rewards": torch.zeros(n_samples, 1),
            "next_states": torch.randn(n_samples, n_states),
            "dones": torch.zeros(n_samples, 1),
            "infos": {
                "avaliable_action": torch.ones(n_samples, n_actions),
                "previous_action": torch.zeros(n_samples),
                "funding_count_down_hour": torch.zeros(n_samples),
                "funding_count_down_minute": torch.zeros(n_samples),
                "trading_info": torch.zeros(n_samples, 4),
                "q_value": torch.zeros(n_samples, n_actions),
            },
        }
    buffer_path = model_dir / "buffer_diverse.pkl"
    torch.save(buffer_payload, buffer_path)

    # 4. Save dummy test split data
    data_dir = tmp_path / "data"
    test_dir = data_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    n_rows = 20
    t_range = pd.date_range("2026-01-01", periods=n_rows, freq="10min")
    f_range = pd.date_range("2026-01-01 08:00:00", periods=n_rows, freq="10min")
    contract_df = {
        "timestamp": t_range,
        "mark_price": [100.0 + i * 0.1 for i in range(n_rows)],
        "funding_rate": [0.0] * n_rows,
        "funding_timestamp": f_range,
        "funding_count_down_hour": [8.0] * n_rows,
        "funding_count_down_minute": [0.0] * n_rows,
    }
    for level in range(1, 26):
        contract_df[f"bid{level}_price"] = [99.0] * n_rows
        contract_df[f"ask{level}_price"] = [101.0] * n_rows
        contract_df[f"bid{level}_size"] = [10.0] * n_rows
        contract_df[f"ask{level}_size"] = [10.0] * n_rows

    for f_name in feature_names:
        contract_df[f_name] = np.random.randn(n_rows).tolist()

    df_test = pd.DataFrame(contract_df)
    feather_file = test_dir / "fu2601.feather"
    df_test.to_feather(feather_file)

    output_dir = tmp_path / "output_ood"

    # Test env_rollout mode
    test_args = [
        "low_level_agent_ood_analysis.py",
        "--model_path", str(model_path),
        "--buffer_path", str(buffer_path),
        "--data_dir", str(data_dir),
        "--feature_path", str(feature_file),
        "--split", "test",
        "--symbol", "fu",
        "--target_freq", "10min",
        "--eval_mode", "env_rollout",
        "--output_dir", str(output_dir),
        "--device", "cpu",
        "--max_samples_per_grid", "20",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    main()

    # Verify output artifacts
    assert (output_dir / "agent_ood_summary.csv").is_file()
    assert (output_dir / "agent_epistemic_uncertainty_timeseries.png").is_file()
    assert (output_dir / "action_agreement_distribution.png").is_file()
    assert (output_dir / "latent_mahalanobis_kde.png").is_file()

    summary_df = pd.read_csv(output_dir / "agent_ood_summary.csv")
    assert len(summary_df) == 1
    assert summary_df["contract"].iloc[0] == "fu2601"
    assert "q_var_mean" in summary_df.columns
    assert "disagreement_mean" in summary_df.columns
    assert "mahalanobis_mean" in summary_df.columns

    # Test static_scan mode
    output_static = tmp_path / "output_static"
    test_static_args = [
        "low_level_agent_ood_analysis.py",
        "--model_path", str(model_path),
        "--buffer_path", str(buffer_path),
        "--data_dir", str(data_dir),
        "--feature_path", str(feature_file),
        "--split", "test",
        "--symbol", "fu",
        "--target_freq", "10min",
        "--eval_mode", "static_scan",
        "--output_dir", str(output_static),
        "--device", "cpu",
        "--max_samples_per_grid", "20",
    ]
    monkeypatch.setattr(sys, "argv", test_static_args)
    main()

    assert (output_static / "agent_ood_summary.csv").is_file()


def test_low_level_agent_ood_analysis_default_feature_path(tmp_path: Path, monkeypatch):
    n_states = 4
    n_actions = 3
    hidden_nodes = 16
    ensemble_size = 2
    feature_names = [f"feat_{i}" for i in range(n_states)]

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / ArtifactNames.RL_STATE_FEATURES_NPY, np.array(feature_names))

    model = ensemble_Qnet(
        N_STATES=n_states,
        N_ACTIONS=n_actions,
        hidden_nodes=hidden_nodes,
        TIME_INFO_DIM=2,
        ensemble_number=ensemble_size,
        TRADING_INFO_DIM=4,
    )
    model_dir = tmp_path / "model_epoch"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "trained_model.pkl"
    torch.save(model.state_dict(), model_path)

    buffer_payload = {}
    for g in range(9):
        n_samples = 30
        buffer_payload[g] = {
            "states": torch.randn(n_samples, n_states),
            "actions": torch.zeros(n_samples, 1, dtype=torch.int64),
            "rewards": torch.zeros(n_samples, 1),
            "next_states": torch.randn(n_samples, n_states),
            "dones": torch.zeros(n_samples, 1),
            "infos": {
                "avaliable_action": torch.ones(n_samples, n_actions),
                "previous_action": torch.zeros(n_samples),
                "funding_count_down_hour": torch.zeros(n_samples),
                "funding_count_down_minute": torch.zeros(n_samples),
                "trading_info": torch.zeros(n_samples, 4),
                "q_value": torch.zeros(n_samples, n_actions),
            },
        }
    buffer_path = model_dir / "buffer_diverse.pkl"
    torch.save(buffer_payload, buffer_path)

    split_dir = data_dir / "test"
    split_dir.mkdir(parents=True, exist_ok=True)
    n_rows = 20
    t_range = pd.date_range("2026-01-01", periods=n_rows, freq="10min")
    f_range = pd.date_range("2026-01-01 08:00:00", periods=n_rows, freq="10min")
    df_data = {
        "timestamp": t_range,
        "mark_price": [100.0 + i * 0.1 for i in range(n_rows)],
        "funding_rate": [0.0] * n_rows,
        "funding_timestamp": f_range,
        "funding_count_down_hour": [8.0] * n_rows,
        "funding_count_down_minute": [0.0] * n_rows,
    }
    for level in range(1, 26):
        df_data[f"bid{level}_price"] = [99.0] * n_rows
        df_data[f"ask{level}_price"] = [101.0] * n_rows
        df_data[f"bid{level}_size"] = [10.0] * n_rows
        df_data[f"ask{level}_size"] = [10.0] * n_rows

    for i in range(n_states):
        df_data[f"feat_{i}"] = np.random.randn(n_rows).tolist()

    df_test = pd.DataFrame(df_data)
    df_test.to_feather(split_dir / "fu2601.feather")

    output_dir = tmp_path / "output_default_path"
    test_args = [
        "low_level_agent_ood_analysis.py",
        "--model_path", str(model_path),
        "--buffer_path", str(buffer_path),
        "--data_dir", str(data_dir),
        "--split", "test",
        "--symbol", "fu",
        "--target_freq", "10min",
        "--eval_mode", "env_rollout",
        "--output_dir", str(output_dir),
        "--device", "cpu",
        "--max_samples_per_grid", "20",
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    main()

    assert (output_dir / "agent_ood_summary.csv").is_file()


def test_low_level_agent_ood_analysis_missing_feature_fails_fast(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / "empty_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    model_dir = tmp_path / "model_epoch"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "trained_model.pkl"
    torch.save({}, model_path)

    test_args = [
        "low_level_agent_ood_analysis.py",
        "--model_path", str(model_path),
        "--buffer_path", str(tmp_path / "buffer.pkl"),
        "--data_dir", str(data_dir),
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    with pytest.raises(FileNotFoundError, match="Feature file not found"):
        main()
