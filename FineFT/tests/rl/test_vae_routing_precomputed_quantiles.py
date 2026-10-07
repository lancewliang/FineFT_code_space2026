import os
import shutil
import types
import numpy as np
import pandas as pd
import pytest
import torch

from RL.DiHFT.VAE.vae import MLP_VAE
from RL.DiHFT.high_level import vae_routing_optuna as vro
from RL.DiHFT.high_level import vae_routing_util as vru
from analysis.pick_agent.FineFT_two_dimensional_agent_selector import (
    TwoDimensionalSelectionManifest,
)
from common import ArtifactNames, MetricColumns, RoutingParamColumns
from tests.rl.test_vae_routing_allow_reverse_position import _sample_manifest_payload


def test_compute_vae_losses_batch():
    input_dim = 4
    z_dim = 2
    model = MLP_VAE(
        INPUT_DIM=input_dim,
        Z_DIM=z_dim,
        hidden_dims=[8, 4],
        loss_func="gaussian",
    )
    data = np.random.randn(20, input_dim).astype(np.float32)
    device = "cpu"

    losses = vru.compute_vae_losses_batch(model, data, device, batch_size=8)
    assert isinstance(losses, np.ndarray)
    assert losses.shape == (20,)
    assert not np.isnan(losses).any()


def test_ensure_precomputed_vae_quantiles(tmp_path):
    dataset_name = "fu"
    eval_stage = "valid"
    stage_dir = tmp_path / "data" / dataset_name / eval_stage
    stage_dir.mkdir(parents=True)

    slope_cols = ["slope_feat_0", "slope_feat_1"]
    vol_cols = ["vol_feat_0", "vol_feat_1"]
    data_dir = tmp_path / "data" / dataset_name
    np.save(data_dir / ArtifactNames.VAE_SLOPE_STATE_FEATURES_NPY, np.array(slope_cols))
    np.save(data_dir / ArtifactNames.VAE_VOLATILITY_STATE_FEATURES_NPY, np.array(vol_cols))

    t_len = 15
    df1 = pd.DataFrame({
        "slope_feat_0": np.random.randn(t_len),
        "slope_feat_1": np.random.randn(t_len),
        "vol_feat_0": np.random.randn(t_len),
        "vol_feat_1": np.random.randn(t_len),
    })
    df1.to_feather(stage_dir / "fu2501.feather")

    vae_root = tmp_path / "vae" / dataset_name / "exp1"
    num_labels = 2
    for axis, cols in [("slope", slope_cols), ("volatility", vol_cols)]:
        for i in range(num_labels):
            label_dir = vae_root / axis / f"label_{i}"
            label_dir.mkdir(parents=True)
            model = MLP_VAE(
                INPUT_DIM=len(cols),
                Z_DIM=2,
                hidden_dims=[8, 4],
                loss_func="gaussian",
            )
            torch.save(model.state_dict(), label_dir / ArtifactNames.MODEL_LATEST_PTH)
            np.save(label_dir / ArtifactNames.ID_LOGPX_NPY, np.linspace(-10, 0, 50))

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        __import__("json").dumps(_sample_manifest_payload(num_labels=num_labels)),
        encoding="utf-8",
    )

    quantiles_dir = tmp_path / "quantiles_cache"
    args = types.SimpleNamespace(
        base_path=str(tmp_path / "data"),
        dataset_name=dataset_name,
        eval_stage=eval_stage,
        experiment_name="exp1",
        vae_path=str(tmp_path / "vae"),
        selection_manifest=str(manifest_path),
        precomputed_quantiles_dir=str(quantiles_dir),
        z_dim=2,
        vae_hidden_dims=[8, 4],
        loss_type="gaussian",
    )

    resolved_dir = vru.ensure_precomputed_vae_quantiles(args)
    assert resolved_dir == str(quantiles_dir)
    saved_file = quantiles_dir / "fu2501.npy"
    assert saved_file.is_file()

    arr = np.load(saved_file)
    assert arr.shape == (t_len, 2 * num_labels)
    assert np.all((arr >= 0.0) & (arr <= 1.0))

    resolved_dir2 = vru.ensure_precomputed_vae_quantiles(args)
    assert resolved_dir2 == str(quantiles_dir)


def test_vae_routing_skips_vae_models_when_cache_present(tmp_path, monkeypatch):
    routing = vru.vae_risk_aware_routing.__new__(vru.vae_risk_aware_routing)
    routing.dataset_name = "fu"
    routing.experiment_name = "exp1"
    routing.eval_stage = "valid"
    routing.num_labels = 2

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    np.save(cache_dir / "fu2501.npy", np.zeros((10, 4), dtype=np.float32))

    args = types.SimpleNamespace(
        precomputed_quantiles_dir=str(cache_dir),
        save_artifacts=False,
    )
    resolved = routing._resolve_precomputed_quantiles_dir(args)
    assert resolved == str(cache_dir)


def test_vae_routing_get_quantiles_from_mmap():
    routing = vru.vae_risk_aware_routing.__new__(vru.vae_risk_aware_routing)
    routing.num_labels = 2
    routing.quantiles = {
        "slope": [__import__("collections").deque(maxlen=10) for _ in range(2)],
        "volatility": [__import__("collections").deque(maxlen=10) for _ in range(2)],
    }
    dummy_matrix = np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
    ], dtype=np.float32)
    routing.current_contract_quantiles = dummy_matrix
    routing.step_idx = 1

    quants = routing.get_quantiles(None, None)
    assert quants["slope"][0][-1] == pytest.approx(0.5)
    assert quants["slope"][1][-1] == pytest.approx(0.6)
    assert quants["volatility"][0][-1] == pytest.approx(0.7)
    assert quants["volatility"][1][-1] == pytest.approx(0.8)


def test_apply_best_trial_parameters_hierarchical():
    best_params = {
        RoutingParamColumns.SLOPE_WINDOW_LENGTH: 50,
        RoutingParamColumns.VOLATILITY_WINDOW_LENGTH: 80,
        RoutingParamColumns.SLOPE_GAMMA: 0.95,
        RoutingParamColumns.VOLATILITY_GAMMA: 0.92,
        RoutingParamColumns.OOD_THRESHOLD: 0.01,
        RoutingParamColumns.SLOPE_MARGIN_THRESHOLD: 0.15,
        RoutingParamColumns.VOLATILITY_MARGIN_THRESHOLD: 0.18,
    }
    search_args = types.SimpleNamespace(gating_strategy="hierarchical")
    trial_args = types.SimpleNamespace()

    res = vro.apply_best_trial_parameters(trial_args, best_params, search_args)
    assert res.slope_window_length == 50
    assert res.volatility_window_length == 80
    assert res.window_length == 80
    assert res.slope_gamma == pytest.approx(0.95)
    assert res.volatility_gamma == pytest.approx(0.92)
    assert res.ood_threshold == pytest.approx(0.01)
    assert res.slope_margin_threshold == pytest.approx(0.15)
    assert res.volatility_margin_threshold == pytest.approx(0.18)


def test_apply_best_trial_parameters_absolute():
    best_params = {
        RoutingParamColumns.SLOPE_WINDOW_LENGTH: 30,
        RoutingParamColumns.VOLATILITY_WINDOW_LENGTH: 40,
        RoutingParamColumns.SLOPE_GAMMA: 0.98,
        RoutingParamColumns.VOLATILITY_GAMMA: 0.96,
        RoutingParamColumns.SLOPE_RULE_BASE_THRESHOLD: 0.25,
        RoutingParamColumns.VOLATILITY_RULE_BASE_THRESHOLD: 0.35,
        RoutingParamColumns.HYSTERESIS_EXIT_RATIO: 0.70,
    }
    search_args = types.SimpleNamespace(gating_strategy="absolute")
    trial_args = types.SimpleNamespace()

    res = vro.apply_best_trial_parameters(trial_args, best_params, search_args)
    assert res.slope_window_length == 30
    assert res.volatility_window_length == 40
    assert res.window_length == 40
    assert res.slope_rule_base_threshold == pytest.approx(0.25)
    assert res.volatility_rule_base_threshold == pytest.approx(0.35)
    assert res.rule_base_threshold == pytest.approx(0.25)
    assert res.hysteresis_exit_ratio == pytest.approx(0.70)


def test_lean_rollout_skips_disk_artifacts(tmp_path, monkeypatch):
    routing = vru.vae_risk_aware_routing.__new__(vru.vae_risk_aware_routing)
    routing.save_artifacts = False
    routing.eval_stage = "valid"
    routing.initial_wallet_balance = 10000.0
    routing.test_path = str(tmp_path / "trial_0")
    routing.find_contract_files = lambda: [
        ("c1", str(tmp_path / "c1.feather")),
        ("c2", str(tmp_path / "c2.feather")),
    ]
    routing.reset_routing_state = lambda: None

    def fake_run(df, path, contract_name=None):
        return {
            "rows": 100,
            "reward_sum": 10.0,
            "require_money": 100.0,
            "return_rate": 0.1,
        }

    monkeypatch.setattr(pd, "read_feather", lambda p: pd.DataFrame({"a": [1]}))
    routing.run_single_valid_df = fake_run

    ret = routing.test()
    assert ret == pytest.approx(0.001)

    assert not (tmp_path / "trial_0" / ArtifactNames.CONTRACT_RESULTS_CSV).exists()
    assert not (tmp_path / "trial_0" / ArtifactNames.TRADING_INFO_NPY).exists()
