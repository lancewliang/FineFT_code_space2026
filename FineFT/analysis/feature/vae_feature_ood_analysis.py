from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Ensure root directory is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
FINEFT_ROOT = REPO_ROOT / "FineFT"
if str(FINEFT_ROOT) not in sys.path:
    sys.path.insert(0, str(FINEFT_ROOT))

from RL.DiHFT.VAE.vae import MLP_VAE, softclip, gaussian_nll

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("VAEFeatureOOD")


def compute_feature_gaussian_nll(
    model: nn.Module,
    data: torch.Tensor,
    device: str = "cpu",
    batch_size: int = 512,
) -> np.ndarray:
    """Compute per-sample, per-feature Gaussian NLL from a VAE model."""
    model.eval()
    data = data.to(device)
    total_samples = data.shape[0]
    nll_chunks: list[np.ndarray] = []

    with torch.no_grad():
        for start_idx in range(0, total_samples, batch_size):
            batch_x = data[start_idx : start_idx + batch_size]
            recon_mu, recon_logvar, _, _ = model(batch_x)
            recon_logvar = softclip(recon_logvar, -6.0)
            recon_logvar = -softclip(-recon_logvar, 0.0)
            feat_nll = gaussian_nll(recon_mu, 0.5 * recon_logvar, batch_x)
            nll_chunks.append(feat_nll.cpu().numpy())

    return np.concatenate(nll_chunks, axis=0)


def load_data_split(
    base_path: Path,
    dataset_name: str,
    split: str,
    feature_names: list[str],
    sample_size: int | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Load and sample feather datasets for a given split."""
    split_dir = base_path / dataset_name / split
    feather_files = sorted(split_dir.glob("*.feather"))
    if not feather_files:
        single_feather = base_path / dataset_name / f"{split}.feather"
        if single_feather.is_file():
            feather_files = [single_feather]
        else:
            raise FileNotFoundError(
                f"No feather files found for split '{split}' in {split_dir}"
            )

    contract_dfs: dict[str, pd.DataFrame] = {}
    pooled_dfs: list[pd.DataFrame] = []

    for file_path in feather_files:
        contract_name = file_path.stem
        df = pd.read_feather(file_path)[feature_names]
        contract_dfs[contract_name] = df
        pooled_dfs.append(df)

    combined_df = pd.concat(pooled_dfs, ignore_index=True)
    if sample_size is not None and len(combined_df) > sample_size:
        combined_df = combined_df.sample(n=sample_size, random_state=42).reset_index(
            drop=True
        )

    return combined_df, contract_dfs


def load_vae_models(
    vae_path: Path,
    dataset_name: str,
    experiment_name: str,
    input_dim: int,
    device: str = "cpu",
    z_dim: int = 512,
    hidden_dims: list[int] | None = None,
) -> dict[str, list[nn.Module]]:
    """Load dual-axis VAE models across labels."""
    if hidden_dims is None:
        hidden_dims = [4096, 2048, 1024, 1024]

    root = vae_path / dataset_name / experiment_name
    axes_models: dict[str, list[nn.Module]] = {}

    for axis in ("slope", "volatility"):
        axes_models[axis] = []
        for label_idx in range(3):
            label_name = f"label_{label_idx}"
            model_file = root / axis / label_name / "model_latest.pth"
            if not model_file.is_file():
                raise FileNotFoundError(f"Missing VAE model checkpoint at {model_file}")

            model = MLP_VAE(
                INPUT_DIM=input_dim,
                Z_DIM=z_dim,
                hidden_dims=hidden_dims,
                loss_func="NLL",
            ).to(device)
            model.load_state_dict(torch.load(model_file, map_location=device))
            model.eval()
            axes_models[axis].append(model)

    return axes_models


def calculate_distribution_drift(
    ref_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_names: list[str],
) -> dict[str, dict[str, float]]:
    """Compute empirical statistical distribution drift metrics per feature."""
    ref_vals = ref_df[feature_names].values
    target_vals = target_df[feature_names].values

    ref_means = np.mean(ref_vals, axis=0)
    ref_stds = np.std(ref_vals, axis=0)
    target_means = np.mean(target_vals, axis=0)
    target_stds = np.std(target_vals, axis=0)
    target_mins = np.min(target_vals, axis=0)
    target_maxs = np.max(target_vals, axis=0)

    drift_metrics: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(feature_names):
        ref_mean = float(ref_means[idx])
        ref_std = float(ref_stds[idx])
        target_mean = float(target_means[idx])
        target_std = float(target_stds[idx])

        std_guard = ref_std if ref_std > 1e-8 else 1e-8
        mean_shift = abs(target_mean - ref_mean) / std_guard
        var_ratio = (target_std**2) / (std_guard**2)

        drift_metrics[name] = {
            "ref_mean": ref_mean,
            "ref_std": ref_std,
            "target_mean": target_mean,
            "target_std": target_std,
            "mean_shift": mean_shift,
            "var_ratio": var_ratio,
            "target_min": float(target_mins[idx]),
            "target_max": float(target_maxs[idx]),
        }

    return drift_metrics


def analyze_feature_vae_ood(args: argparse.Namespace) -> dict[str, Any]:
    """Execute complete feature-level VAE OOD diagnosis pipeline."""
    base_path = Path(args.base_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_file = base_path / args.dataset_name / "state_features.npy"
    if not feature_file.is_file():
        raise FileNotFoundError(f"Missing state features file: {feature_file}")
    feature_names: list[str] = list(np.load(feature_file))
    feature_dim = len(feature_names)

    logger.info("Loaded %d features from %s", feature_dim, feature_file)

    # 1. Load data splits
    logger.info("Loading valid dataset (in-distribution reference baseline)...")
    valid_df, _ = load_data_split(
        base_path, args.dataset_name, "valid", feature_names, args.sample_size
    )

    logger.info("Loading test dataset...")
    test_df, test_contract_dfs = load_data_split(
        base_path, args.dataset_name, "test", feature_names, args.sample_size
    )

    logger.info("Loading train dataset...")
    train_df, _ = load_data_split(
        base_path, args.dataset_name, "train", feature_names, args.sample_size
    )

    valid_tensor = torch.tensor(valid_df.values, dtype=torch.float32)
    test_tensor = torch.tensor(test_df.values, dtype=torch.float32)
    train_tensor = torch.tensor(train_df.values, dtype=torch.float32)

    # 2. Load VAE models
    logger.info("Loading VAE checkpoints from %s...", args.vae_path)
    vae_path = Path(args.vae_path)
    models_dict = load_vae_models(
        vae_path=vae_path,
        dataset_name=args.dataset_name,
        experiment_name=args.experiment_name,
        input_dim=feature_dim,
        device=args.device,
        z_dim=args.z_dim,
        hidden_dims=args.hidden_dims,
    )

    # 3. Compute per-model, per-feature NLL across splits
    logger.info("Computing per-feature Gaussian NLL across all VAE regime models...")
    model_keys = []
    valid_nlls = []
    test_nlls = []
    train_nlls = []

    axis_delta_test: dict[str, list[np.ndarray]] = {"slope": [], "volatility": []}
    axis_delta_train: dict[str, list[np.ndarray]] = {"slope": [], "volatility": []}

    for axis, model_list in models_dict.items():
        for label_idx, model in enumerate(model_list):
            key = f"{axis}_label_{label_idx}"
            model_keys.append(key)

            v_nll = compute_feature_gaussian_nll(model, valid_tensor, args.device)
            t_nll = compute_feature_gaussian_nll(model, test_tensor, args.device)
            tr_nll = compute_feature_gaussian_nll(model, train_tensor, args.device)

            v_mean = v_nll.mean(axis=0)
            t_mean = t_nll.mean(axis=0)
            tr_mean = tr_nll.mean(axis=0)

            valid_nlls.append(v_mean)
            test_nlls.append(t_mean)
            train_nlls.append(tr_mean)

            d_test = t_mean - v_mean
            d_train = tr_mean - v_mean

            axis_delta_test[axis].append(d_test)
            axis_delta_train[axis].append(d_train)

    # Global average across all 6 models
    avg_valid_nll = np.mean(valid_nlls, axis=0)
    avg_test_nll = np.mean(test_nlls, axis=0)
    avg_train_nll = np.mean(train_nlls, axis=0)

    delta_nll_test = avg_test_nll - avg_valid_nll
    delta_nll_train = avg_train_nll - avg_valid_nll

    total_delta_test = float(np.sum(delta_nll_test))
    norm_factor_test = total_delta_test if total_delta_test > 1e-8 else 1.0
    contrib_pct_test = (delta_nll_test / norm_factor_test) * 100.0

    total_delta_train = float(np.sum(delta_nll_train))
    norm_factor_train = total_delta_train if total_delta_train > 1e-8 else 1.0
    contrib_pct_train = (delta_nll_train / norm_factor_train) * 100.0

    # Axis-level averages
    slope_delta_test = np.mean(axis_delta_test["slope"], axis=0)
    volatility_delta_test = np.mean(axis_delta_test["volatility"], axis=0)

    # 4. Statistical distribution drift metrics
    test_drift = calculate_distribution_drift(valid_df, test_df, feature_names)
    train_drift = calculate_distribution_drift(valid_df, train_df, feature_names)

    # 5. Build summary table
    summary_rows = []
    for idx, feat in enumerate(feature_names):
        t_stat = test_drift[feat]
        tr_stat = train_drift[feat]

        summary_rows.append(
            {
                "feature": feat,
                "delta_nll_test_vs_valid": float(delta_nll_test[idx]),
                "contrib_pct_test": float(contrib_pct_test[idx]),
                "delta_nll_train_vs_valid": float(delta_nll_train[idx]),
                "contrib_pct_train": float(contrib_pct_train[idx]),
                "slope_delta_nll_test": float(slope_delta_test[idx]),
                "volatility_delta_nll_test": float(volatility_delta_test[idx]),
                "valid_mean": t_stat["ref_mean"],
                "valid_std": t_stat["ref_std"],
                "test_mean": t_stat["target_mean"],
                "test_std": t_stat["target_std"],
                "test_mean_shift": t_stat["mean_shift"],
                "test_var_ratio": t_stat["var_ratio"],
                "test_min": t_stat["target_min"],
                "test_max": t_stat["target_max"],
                "train_mean": tr_stat["target_mean"],
                "train_std": tr_stat["target_std"],
                "train_mean_shift": tr_stat["mean_shift"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(
        by="delta_nll_test_vs_valid", ascending=False
    ).reset_index(drop=True)
    summary_df["rank"] = summary_df.index + 1

    # Reorder columns
    cols = ["rank", "feature", "delta_nll_test_vs_valid", "contrib_pct_test"] + [
        c
        for c in summary_df.columns
        if c not in ["rank", "feature", "delta_nll_test_vs_valid", "contrib_pct_test"]
    ]
    summary_df = summary_df[cols]

    # Save primary CSV
    summary_csv_path = output_dir / "feature_ood_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    logger.info("Saved feature OOD summary to %s", summary_csv_path)

    # 6. Per-contract breakdown if requested
    per_contract_summaries = {}
    if args.per_contract:
        logger.info("Executing per-contract OOD breakdown for test contracts...")
        contracts_out_dir = output_dir / "contracts"
        contracts_out_dir.mkdir(parents=True, exist_ok=True)

        for c_name, c_df in test_contract_dfs.items():
            c_tensor = torch.tensor(c_df.values, dtype=torch.float32)
            c_test_nlls = []
            for axis, model_list in models_dict.items():
                for model in model_list:
                    c_nll = compute_feature_gaussian_nll(model, c_tensor, args.device)
                    c_test_nlls.append(c_nll.mean(axis=0))

            avg_c_nll = np.mean(c_test_nlls, axis=0)
            c_delta = avg_c_nll - avg_valid_nll
            c_sum_delta = float(np.sum(c_delta))
            c_norm = c_sum_delta if c_sum_delta > 1e-8 else 1.0

            c_drift = calculate_distribution_drift(valid_df, c_df, feature_names)

            c_rows = []
            for idx, feat in enumerate(feature_names):
                c_rows.append(
                    {
                        "feature": feat,
                        "delta_nll": float(c_delta[idx]),
                        "contrib_pct": float((c_delta[idx] / c_norm) * 100.0),
                        "mean_shift": c_drift[feat]["mean_shift"],
                        "contract_mean": c_drift[feat]["target_mean"],
                        "contract_std": c_drift[feat]["target_std"],
                        "valid_mean": c_drift[feat]["ref_mean"],
                        "valid_std": c_drift[feat]["ref_std"],
                    }
                )

            c_summary_df = pd.DataFrame(c_rows).sort_values(
                by="delta_nll", ascending=False
            ).reset_index(drop=True)
            c_summary_df["rank"] = c_summary_df.index + 1
            c_csv = contracts_out_dir / f"{c_name}_ood.csv"
            c_summary_df.to_csv(c_csv, index=False)
            per_contract_summaries[c_name] = c_summary_df

    # 7. Print formatted top summary
    print("\n" + "=" * 96)
    print(" TOP 15 FEATURES CAUSING VAE OOD COLLAPSE (Test vs Valid Baseline)")
    print("=" * 96)
    print_cols = [
        "rank",
        "feature",
        "delta_nll_test_vs_valid",
        "contrib_pct_test",
        "valid_mean",
        "test_mean",
        "test_mean_shift",
        "test_var_ratio",
    ]
    display_df = summary_df.head(15)[print_cols].copy()
    display_df["delta_nll_test_vs_valid"] = display_df[
        "delta_nll_test_vs_valid"
    ].round(2)
    display_df["contrib_pct_test"] = display_df["contrib_pct_test"].round(2)
    display_df["valid_mean"] = display_df["valid_mean"].round(3)
    display_df["test_mean"] = display_df["test_mean"].round(3)
    display_df["test_mean_shift"] = display_df["test_mean_shift"].round(2)
    display_df["test_var_ratio"] = display_df["test_var_ratio"].round(2)

    print(display_df.to_string(index=False))
    print("=" * 96 + "\n")

    return {
        "summary_df": summary_df,
        "per_contract_summaries": per_contract_summaries,
        "total_delta_test": total_delta_test,
        "total_delta_train": total_delta_train,
        "output_dir": output_dir,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze per-feature VAE OOD contributions between valid and test/train splits."
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default="dataset/10min",
        help="Base path to dataset directory (e.g. dataset/10min)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="fu",
        help="Dataset / symbol name (e.g. fu, al)",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="10min_parallel",
        help="Experiment name under vae_results (e.g. 10min_parallel)",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default="result/DiHFT/vae_results",
        help="Path to VAE checkpoints directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save analysis CSV and diagnostic artifacts",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=5000,
        help="Maximum number of rows to sample per split (None for all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda:0 or cpu)",
    )
    parser.add_argument(
        "--z_dim",
        type=int,
        default=512,
        help="VAE latent dimension (default: 512)",
    )
    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        default=[4096, 2048, 1024, 1024],
        help="VAE hidden layer dimensions",
    )
    parser.add_argument(
        "--per_contract",
        action="store_true",
        default=False,
        help="Enable per-contract OOD breakdown for each test contract",
    )

    args = parser.parse_args()

    if not args.output_dir:
        args.output_dir = (
            f"analysis_result/DiHFT/feature_ood/{args.dataset_name}/{args.experiment_name}"
        )

    analyze_feature_vae_ood(args)


if __name__ == "__main__":
    main()
