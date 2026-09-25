import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from FineFT.analysis.feature.vae_feature_ood_analysis import (
    compute_feature_gaussian_nll,
    analyze_feature_vae_ood,
)
from FineFT.RL.DiHFT.VAE.vae import MLP_VAE, softclip, gaussian_nll


class DummyVAE(nn.Module):
    """Minimal dummy VAE that predicts zero mean and fixed logvar for testing."""
    def __init__(self, input_dim: int, fixed_logvar: float = -1.0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.fixed_logvar = fixed_logvar

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        recon_mu = torch.zeros_like(x)
        recon_logvar = torch.full_like(x, fill_value=self.fixed_logvar)
        mu = torch.zeros(batch_size, 2, device=x.device)
        logvar = torch.zeros(batch_size, 2, device=x.device)
        return recon_mu, recon_logvar, mu, logvar


def test_compute_feature_gaussian_nll_mathematical_consistency():
    device = "cpu"
    batch_size = 20
    feature_dim = 5
    model = DummyVAE(input_dim=feature_dim, fixed_logvar=-1.0)
    data = torch.randn(batch_size, feature_dim)

    feature_nll = compute_feature_gaussian_nll(model, data, device)

    assert feature_nll.shape == (batch_size, feature_dim)

    # Compute expected using exact formula with clipped logvar
    recon_mu, recon_logvar, _, _ = model(data)
    clipped_logvar = softclip(recon_logvar, -6.0)
    clipped_logvar = -softclip(-clipped_logvar, 0.0)
    expected_tensor = gaussian_nll(recon_mu, 0.5 * clipped_logvar, data)
    expected_nll = expected_tensor.numpy()

    np.testing.assert_allclose(feature_nll, expected_nll, rtol=1e-5, atol=1e-5)
    # Assert additivity: sum across features equals total gaussian NLL
    np.testing.assert_allclose(
        feature_nll.sum(axis=1),
        expected_tensor.sum(dim=1).numpy(),
        rtol=1e-5,
        atol=1e-5,
    )


def test_analyze_feature_vae_ood_end_to_end(tmp_path: Path):
    feature_names = ["feat_stable", "feat_mean_shift", "feat_var_shift", "feat_noise"]
    input_dim = len(feature_names)
    num_samples = 100

    np.random.seed(42)
    # Valid distribution: mean 0, std 1
    valid_df = pd.DataFrame(
        np.random.normal(0.0, 1.0, size=(num_samples, input_dim)),
        columns=feature_names,
    )
    # Train distribution: similar to valid
    train_df = pd.DataFrame(
        np.random.normal(0.0, 1.0, size=(num_samples, input_dim)),
        columns=feature_names,
    )
    # Test distribution: feat_mean_shift has severe mean drift (+8.0), feat_var_shift has high variance
    test_mat = np.random.normal(0.0, 1.0, size=(num_samples, input_dim))
    test_mat[:, 1] += 8.0
    test_mat[:, 2] *= 4.0
    test_df = pd.DataFrame(test_mat, columns=feature_names)

    # Set up directory structure
    dataset_name = "mock_symbol"
    experiment_name = "mock_exp"

    data_dir = tmp_path / "dataset" / "10min" / dataset_name
    for split, df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        split_dir = data_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        df.to_feather(split_dir / "c1.feather")

    np.save(data_dir / "state_features.npy", np.array(feature_names))

    vae_dir = tmp_path / "result" / "DiHFT" / "vae_results" / dataset_name / experiment_name
    hidden_dims = [16, 8]
    z_dim = 4

    for axis in ("slope", "volatility"):
        for label_idx in range(3):
            label_dir = vae_dir / axis / f"label_{label_idx}"
            label_dir.mkdir(parents=True, exist_ok=True)
            vae_model = MLP_VAE(INPUT_DIM=input_dim, Z_DIM=z_dim, hidden_dims=hidden_dims, loss_func="NLL")
            torch.save(vae_model.state_dict(), label_dir / "model_latest.pth")

    output_dir = tmp_path / "analysis_result" / "test_ood"

    args = argparse.Namespace(
        base_path=str(tmp_path / "dataset" / "10min"),
        dataset_name=dataset_name,
        experiment_name=experiment_name,
        vae_path=str(tmp_path / "result" / "DiHFT" / "vae_results"),
        output_dir=str(output_dir),
        sample_size=100,
        device="cpu",
        per_contract=False,
        z_dim=z_dim,
        hidden_dims=hidden_dims,
    )

    results = analyze_feature_vae_ood(args)

    assert "summary_df" in results
    summary_df = results["summary_df"]

    assert len(summary_df) == input_dim
    assert (output_dir / "feature_ood_summary.csv").is_file()

    # The severely drifted features must rank ahead of stable features
    top_feature = summary_df.iloc[0]["feature"]
    second_feature = summary_df.iloc[1]["feature"]
    assert {top_feature, second_feature} == {"feat_mean_shift", "feat_var_shift"}
    assert summary_df.iloc[0]["delta_nll_test_vs_valid"] > 0


def test_analyze_feature_vae_ood_per_contract(tmp_path: Path):
    feature_names = ["feat_a", "feat_b"]
    input_dim = len(feature_names)
    num_samples = 50

    np.random.seed(123)
    valid_df = pd.DataFrame(np.random.randn(num_samples, input_dim), columns=feature_names)
    train_df = pd.DataFrame(np.random.randn(num_samples, input_dim), columns=feature_names)
    test_df_c1 = pd.DataFrame(np.random.randn(num_samples, input_dim), columns=feature_names)
    test_df_c2 = pd.DataFrame(np.random.randn(num_samples, input_dim), columns=feature_names)

    dataset_name = "mock_sym2"
    experiment_name = "mock_exp2"
    data_dir = tmp_path / "dataset" / "10min" / dataset_name

    (data_dir / "valid").mkdir(parents=True, exist_ok=True)
    valid_df.to_feather(data_dir / "valid" / "v1.feather")
    (data_dir / "train").mkdir(parents=True, exist_ok=True)
    train_df.to_feather(data_dir / "train" / "tr1.feather")
    (data_dir / "test").mkdir(parents=True, exist_ok=True)
    test_df_c1.to_feather(data_dir / "test" / "c1.feather")
    test_df_c2.to_feather(data_dir / "test" / "c2.feather")

    np.save(data_dir / "state_features.npy", np.array(feature_names))

    vae_dir = tmp_path / "result" / "DiHFT" / "vae_results" / dataset_name / experiment_name
    hidden_dims = [8, 4]
    z_dim = 2

    for axis in ("slope", "volatility"):
        for label_idx in range(3):
            label_dir = vae_dir / axis / f"label_{label_idx}"
            label_dir.mkdir(parents=True, exist_ok=True)
            vae_model = MLP_VAE(INPUT_DIM=input_dim, Z_DIM=z_dim, hidden_dims=hidden_dims, loss_func="NLL")
            torch.save(vae_model.state_dict(), label_dir / "model_latest.pth")

    output_dir = tmp_path / "analysis_result" / "test_ood_per_contract"

    args = argparse.Namespace(
        base_path=str(tmp_path / "dataset" / "10min"),
        dataset_name=dataset_name,
        experiment_name=experiment_name,
        vae_path=str(tmp_path / "result" / "DiHFT" / "vae_results"),
        output_dir=str(output_dir),
        sample_size=100,
        device="cpu",
        per_contract=True,
        z_dim=z_dim,
        hidden_dims=hidden_dims,
    )

    results = analyze_feature_vae_ood(args)

    assert "per_contract_summaries" in results
    per_contract = results["per_contract_summaries"]
    assert "c1" in per_contract
    assert "c2" in per_contract
    assert (output_dir / "contracts" / "c1_ood.csv").is_file()
    assert (output_dir / "contracts" / "c2_ood.csv").is_file()
