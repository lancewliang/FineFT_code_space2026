from __future__ import annotations
from common import ArtifactNames

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
    axis: str = "slope",
    device: str = "cpu",
    z_dim: int = 512,
    hidden_dims: list[int] | None = None,
) -> dict[str, list[nn.Module]]:
    """Load dual-axis VAE models across labels."""
    if hidden_dims is None:
        hidden_dims = [4096, 2048, 1024, 1024]

    root = vae_path / dataset_name / experiment_name
    axes_models: dict[str, list[nn.Module]] = {}

    axes_to_load = [axis] if axis in ("slope", "volatility") else ["slope", "volatility"]
    for axis in axes_to_load:
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
    """Execute complete feature-level VAE OOD multi-perspective diagnosis pipeline."""
    base_path = Path(args.base_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    axis = args.axis
    feature_filename = (
        ArtifactNames.VAE_SLOPE_STATE_FEATURES_NPY
        if axis == "slope"
        else ArtifactNames.VAE_VOLATILITY_STATE_FEATURES_NPY
    )
    feature_file = base_path / args.dataset_name / feature_filename
    if not feature_file.is_file():
        raise FileNotFoundError(f"Missing state features file: {feature_file}")
    feature_names: list[str] = list(np.load(feature_file))
    feature_dim = len(feature_names)

    logger.info("Loaded %d features from %s", feature_dim, feature_file)

    # 1. Load data splits (train is true in-distribution baseline per ADR-0021 / ADR-0026)
    logger.info("Loading train dataset (in-distribution reference baseline)...")
    train_df, _ = load_data_split(
        base_path, args.dataset_name, "train", feature_names, args.sample_size
    )

    logger.info("Loading valid dataset...")
    valid_df, _ = load_data_split(
        base_path, args.dataset_name, "valid", feature_names, args.sample_size
    )

    logger.info("Loading test dataset...")
    test_df, test_contract_dfs = load_data_split(
        base_path, args.dataset_name, "test", feature_names, args.sample_size
    )

    train_tensor = torch.tensor(train_df.values, dtype=torch.float32)
    valid_tensor = torch.tensor(valid_df.values, dtype=torch.float32)
    test_tensor = torch.tensor(test_df.values, dtype=torch.float32)

    # 2. Load VAE models
    logger.info("Loading VAE checkpoints from %s...", args.vae_path)
    vae_path = Path(args.vae_path)
    models_dict = load_vae_models(
        vae_path=vae_path,
        dataset_name=args.dataset_name,
        experiment_name=args.experiment_name,
        input_dim=feature_dim,
        axis=axis,
        device=args.device,
        z_dim=args.z_dim,
        hidden_dims=args.hidden_dims,
    )

    # 3. Compute per-model, per-feature NLL across splits
    logger.info("Computing per-feature Gaussian NLL across all VAE regime models...")
    model_keys = []
    train_nlls = []
    valid_nlls = []
    test_nlls = []

    axis_delta_test_vs_train: dict[str, list[np.ndarray]] = {"slope": [], "volatility": []}
    axis_delta_valid_vs_train: dict[str, list[np.ndarray]] = {"slope": [], "volatility": []}
    axis_delta_test_vs_valid: dict[str, list[np.ndarray]] = {"slope": [], "volatility": []}

    for axis, model_list in models_dict.items():
        for label_idx, model in enumerate(model_list):
            key = f"{axis}_label_{label_idx}"
            model_keys.append(key)

            tr_nll = compute_feature_gaussian_nll(model, train_tensor, args.device)
            v_nll = compute_feature_gaussian_nll(model, valid_tensor, args.device)
            t_nll = compute_feature_gaussian_nll(model, test_tensor, args.device)

            tr_mean = tr_nll.mean(axis=0)
            v_mean = v_nll.mean(axis=0)
            t_mean = t_nll.mean(axis=0)

            train_nlls.append(tr_mean)
            valid_nlls.append(v_mean)
            test_nlls.append(t_mean)

            axis_delta_test_vs_train[axis].append(t_mean - tr_mean)
            axis_delta_valid_vs_train[axis].append(v_mean - tr_mean)
            axis_delta_test_vs_valid[axis].append(t_mean - v_mean)

    # Global average across all 6 models
    avg_train_nll = np.mean(train_nlls, axis=0)
    avg_valid_nll = np.mean(valid_nlls, axis=0)
    avg_test_nll = np.mean(test_nlls, axis=0)

    # 1. Perspective: valid_vs_train (Validation drift relative to training baseline)
    delta_nll_valid_vs_train = avg_valid_nll - avg_train_nll
    total_delta_valid_vs_train = float(np.sum(delta_nll_valid_vs_train))
    norm_factor_v_vs_tr = (
        total_delta_valid_vs_train if total_delta_valid_vs_train > 1e-8 else 1.0
    )
    contrib_pct_valid_vs_train = (
        delta_nll_valid_vs_train / norm_factor_v_vs_tr
    ) * 100.0

    # 2. Perspective: test_vs_train (True VAE OOD breakdown from training dynamics)
    delta_nll_test_vs_train = avg_test_nll - avg_train_nll
    total_delta_test_vs_train = float(np.sum(delta_nll_test_vs_train))
    norm_factor_t_vs_tr = (
        total_delta_test_vs_train if total_delta_test_vs_train > 1e-8 else 1.0
    )
    contrib_pct_test_vs_train = (
        delta_nll_test_vs_train / norm_factor_t_vs_tr
    ) * 100.0

    # 3. Perspective: test_vs_valid (Generalization degradation from validation set)
    delta_nll_test_vs_valid = avg_test_nll - avg_valid_nll
    total_delta_test_vs_valid = float(np.sum(delta_nll_test_vs_valid))
    norm_factor_t_vs_v = (
        total_delta_test_vs_valid if total_delta_test_vs_valid > 1e-8 else 1.0
    )
    contrib_pct_test_vs_valid = (
        delta_nll_test_vs_valid / norm_factor_t_vs_v
    ) * 100.0

    # Axis-level averages for primary OOD benchmark (test_vs_train)
    slope_delta_test_vs_train = (
        np.mean(axis_delta_test_vs_train["slope"], axis=0)
        if len(axis_delta_test_vs_train["slope"]) > 0
        else np.zeros(feature_dim)
    )
    volatility_delta_test_vs_train = (
        np.mean(axis_delta_test_vs_train["volatility"], axis=0)
        if len(axis_delta_test_vs_train["volatility"]) > 0
        else np.zeros(feature_dim)
    )

    # 4. Statistical distribution drift metrics (independently calculated per pair)
    drift_valid_vs_train = calculate_distribution_drift(
        ref_df=train_df, target_df=valid_df, feature_names=feature_names
    )
    drift_test_vs_train = calculate_distribution_drift(
        ref_df=train_df, target_df=test_df, feature_names=feature_names
    )
    drift_test_vs_valid = calculate_distribution_drift(
        ref_df=valid_df, target_df=test_df, feature_names=feature_names
    )

    # 5. Build perspective tables
    # Table 1: valid_vs_train
    rows_v_tr = []
    for idx, feat in enumerate(feature_names):
        st = drift_valid_vs_train[feat]
        rows_v_tr.append(
            {
                "feature": feat,
                "delta_nll_valid_vs_train": float(delta_nll_valid_vs_train[idx]),
                "contrib_pct_valid_vs_train": float(contrib_pct_valid_vs_train[idx]),
                "train_mean": st["ref_mean"],
                "train_std": st["ref_std"],
                "valid_mean": st["target_mean"],
                "valid_std": st["target_std"],
                "mean_shift": st["mean_shift"],
                "var_ratio": st["var_ratio"],
                "valid_min": st["target_min"],
                "valid_max": st["target_max"],
            }
        )
    df_valid_vs_train = (
        pd.DataFrame(rows_v_tr)
        .sort_values(by="delta_nll_valid_vs_train", ascending=False)
        .reset_index(drop=True)
    )
    df_valid_vs_train["rank"] = df_valid_vs_train.index + 1
    cols_v_tr = [
        "rank",
        "feature",
        "delta_nll_valid_vs_train",
        "contrib_pct_valid_vs_train",
    ] + [
        c
        for c in df_valid_vs_train.columns
        if c not in ["rank", "feature", "delta_nll_valid_vs_train", "contrib_pct_valid_vs_train"]
    ]
    df_valid_vs_train = df_valid_vs_train[cols_v_tr]
    csv_valid_vs_train = output_dir / "feature_ood_valid_vs_train.csv"
    df_valid_vs_train.to_csv(csv_valid_vs_train, index=False)
    logger.info("Saved valid vs train OOD summary to %s", csv_valid_vs_train)

    # Table 2: test_vs_train (Primary OOD Benchmark)
    rows_t_tr = []
    for idx, feat in enumerate(feature_names):
        st = drift_test_vs_train[feat]
        rows_t_tr.append(
            {
                "feature": feat,
                "delta_nll_test_vs_train": float(delta_nll_test_vs_train[idx]),
                "contrib_pct_test_vs_train": float(contrib_pct_test_vs_train[idx]),
                "train_mean": st["ref_mean"],
                "train_std": st["ref_std"],
                "test_mean": st["target_mean"],
                "test_std": st["target_std"],
                "mean_shift": st["mean_shift"],
                "var_ratio": st["var_ratio"],
                "test_min": st["target_min"],
                "test_max": st["target_max"],
            }
        )
    df_test_vs_train = (
        pd.DataFrame(rows_t_tr)
        .sort_values(by="delta_nll_test_vs_train", ascending=False)
        .reset_index(drop=True)
    )
    df_test_vs_train["rank"] = df_test_vs_train.index + 1
    cols_t_tr = [
        "rank",
        "feature",
        "delta_nll_test_vs_train",
        "contrib_pct_test_vs_train",
    ] + [
        c
        for c in df_test_vs_train.columns
        if c not in ["rank", "feature", "delta_nll_test_vs_train", "contrib_pct_test_vs_train"]
    ]
    df_test_vs_train = df_test_vs_train[cols_t_tr]
    csv_test_vs_train = output_dir / "feature_ood_test_vs_train.csv"
    df_test_vs_train.to_csv(csv_test_vs_train, index=False)
    logger.info("Saved test vs train OOD summary to %s", csv_test_vs_train)

    # Table 3: test_vs_valid
    rows_t_v = []
    for idx, feat in enumerate(feature_names):
        st = drift_test_vs_valid[feat]
        rows_t_v.append(
            {
                "feature": feat,
                "delta_nll_test_vs_valid": float(delta_nll_test_vs_valid[idx]),
                "contrib_pct_test_vs_valid": float(contrib_pct_test_vs_valid[idx]),
                "valid_mean": st["ref_mean"],
                "valid_std": st["ref_std"],
                "test_mean": st["target_mean"],
                "test_std": st["target_std"],
                "mean_shift": st["mean_shift"],
                "var_ratio": st["var_ratio"],
                "test_min": st["target_min"],
                "test_max": st["target_max"],
            }
        )
    df_test_vs_valid = (
        pd.DataFrame(rows_t_v)
        .sort_values(by="delta_nll_test_vs_valid", ascending=False)
        .reset_index(drop=True)
    )
    df_test_vs_valid["rank"] = df_test_vs_valid.index + 1
    cols_t_v = [
        "rank",
        "feature",
        "delta_nll_test_vs_valid",
        "contrib_pct_test_vs_valid",
    ] + [
        c
        for c in df_test_vs_valid.columns
        if c not in ["rank", "feature", "delta_nll_test_vs_valid", "contrib_pct_test_vs_valid"]
    ]
    df_test_vs_valid = df_test_vs_valid[cols_t_v]
    csv_test_vs_valid = output_dir / "feature_ood_test_vs_valid.csv"
    df_test_vs_valid.to_csv(csv_test_vs_valid, index=False)
    logger.info("Saved test vs valid OOD summary to %s", csv_test_vs_valid)

    # 6. Build and save master consolidated summary table
    summary_rows = []
    for idx, feat in enumerate(feature_names):
        v_tr = drift_valid_vs_train[feat]
        t_tr = drift_test_vs_train[feat]
        t_v = drift_test_vs_valid[feat]

        summary_rows.append(
            {
                "feature": feat,
                "delta_nll_test_vs_train": float(delta_nll_test_vs_train[idx]),
                "contrib_pct_test_vs_train": float(contrib_pct_test_vs_train[idx]),
                "delta_nll_valid_vs_train": float(delta_nll_valid_vs_train[idx]),
                "contrib_pct_valid_vs_train": float(contrib_pct_valid_vs_train[idx]),
                "delta_nll_test_vs_valid": float(delta_nll_test_vs_valid[idx]),
                "contrib_pct_test_vs_valid": float(contrib_pct_test_vs_valid[idx]),
                "slope_delta_nll_test_vs_train": float(slope_delta_test_vs_train[idx]),
                "volatility_delta_nll_test_vs_train": float(
                    volatility_delta_test_vs_train[idx]
                ),
                "train_mean": t_tr["ref_mean"],
                "train_std": t_tr["ref_std"],
                "valid_mean": v_tr["target_mean"],
                "valid_std": v_tr["target_std"],
                "valid_mean_shift_vs_train": v_tr["mean_shift"],
                "valid_var_ratio_vs_train": v_tr["var_ratio"],
                "test_mean": t_tr["target_mean"],
                "test_std": t_tr["target_std"],
                "test_mean_shift_vs_train": t_tr["mean_shift"],
                "test_var_ratio_vs_train": t_tr["var_ratio"],
                "test_mean_shift_vs_valid": t_v["mean_shift"],
                "test_var_ratio_vs_valid": t_v["var_ratio"],
                "test_min": t_tr["target_min"],
                "test_max": t_tr["target_max"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(
        by="delta_nll_test_vs_train", ascending=False
    ).reset_index(drop=True)
    summary_df["rank"] = summary_df.index + 1

    cols_master = [
        "rank",
        "feature",
        "delta_nll_test_vs_train",
        "contrib_pct_test_vs_train",
    ] + [
        c
        for c in summary_df.columns
        if c
        not in [
            "rank",
            "feature",
            "delta_nll_test_vs_train",
            "contrib_pct_test_vs_train",
        ]
    ]
    summary_df = summary_df[cols_master]

    summary_csv_path = output_dir / "feature_ood_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    logger.info("Saved master feature OOD summary to %s", summary_csv_path)

    # 7. Per-contract breakdown if requested
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
            c_delta_vs_train = avg_c_nll - avg_train_nll
            c_delta_vs_valid = avg_c_nll - avg_valid_nll

            c_sum_delta_train = float(np.sum(c_delta_vs_train))
            c_norm_train = c_sum_delta_train if c_sum_delta_train > 1e-8 else 1.0

            c_drift_train = calculate_distribution_drift(train_df, c_df, feature_names)
            c_drift_valid = calculate_distribution_drift(valid_df, c_df, feature_names)

            c_rows = []
            for idx, feat in enumerate(feature_names):
                dt_tr = c_drift_train[feat]
                dt_v = c_drift_valid[feat]
                c_rows.append(
                    {
                        "feature": feat,
                        "delta_nll_vs_train": float(c_delta_vs_train[idx]),
                        "contrib_pct_vs_train": float(
                            (c_delta_vs_train[idx] / c_norm_train) * 100.0
                        ),
                        "delta_nll_vs_valid": float(c_delta_vs_valid[idx]),
                        "contract_mean": dt_tr["target_mean"],
                        "contract_std": dt_tr["target_std"],
                        "train_mean": dt_tr["ref_mean"],
                        "train_std": dt_tr["ref_std"],
                        "valid_mean": dt_v["ref_mean"],
                        "valid_std": dt_v["ref_std"],
                        "mean_shift_vs_train": dt_tr["mean_shift"],
                        "var_ratio_vs_train": dt_tr["var_ratio"],
                        "mean_shift_vs_valid": dt_v["mean_shift"],
                        "var_ratio_vs_valid": dt_v["var_ratio"],
                    }
                )

            c_summary_df = (
                pd.DataFrame(c_rows)
                .sort_values(by="delta_nll_vs_train", ascending=False)
                .reset_index(drop=True)
            )
            c_summary_df["rank"] = c_summary_df.index + 1
            c_cols = [
                "rank",
                "feature",
                "delta_nll_vs_train",
                "contrib_pct_vs_train",
            ] + [
                c
                for c in c_summary_df.columns
                if c
                not in [
                    "rank",
                    "feature",
                    "delta_nll_vs_train",
                    "contrib_pct_vs_train",
                ]
            ]
            c_summary_df = c_summary_df[c_cols]
            c_csv = contracts_out_dir / f"{c_name}_ood.csv"
            c_summary_df.to_csv(c_csv, index=False)
            per_contract_summaries[c_name] = c_summary_df

    # 8. Print formatted multi-perspective summaries
    top_k = args.top_k
    print("\n" + "=" * 105)
    print(
        f" TABLE 1: TOP {top_k} FEATURES WITH DISTRIBUTION SHIFT (Valid vs Train Baseline)"
    )
    print("=" * 105)
    cols_display_t1 = [
        "rank",
        "feature",
        "delta_nll_valid_vs_train",
        "contrib_pct_valid_vs_train",
        "train_mean",
        "valid_mean",
        "mean_shift",
        "var_ratio",
    ]
    disp_t1 = df_valid_vs_train.head(top_k)[cols_display_t1].copy()
    disp_t1["delta_nll_valid_vs_train"] = disp_t1["delta_nll_valid_vs_train"].round(2)
    disp_t1["contrib_pct_valid_vs_train"] = disp_t1[
        "contrib_pct_valid_vs_train"
    ].round(2)
    disp_t1["train_mean"] = disp_t1["train_mean"].round(3)
    disp_t1["valid_mean"] = disp_t1["valid_mean"].round(3)
    disp_t1["mean_shift"] = disp_t1["mean_shift"].round(2)
    disp_t1["var_ratio"] = disp_t1["var_ratio"].round(2)
    print(disp_t1.to_string(index=False))

    print("\n" + "=" * 105)
    print(
        f" TABLE 2: TOP {top_k} FEATURES CAUSING TRUE VAE OOD COLLAPSE (Test vs Train Baseline)"
    )
    print("=" * 105)
    cols_display_t2 = [
        "rank",
        "feature",
        "delta_nll_test_vs_train",
        "contrib_pct_test_vs_train",
        "train_mean",
        "test_mean",
        "mean_shift",
        "var_ratio",
    ]
    disp_t2 = df_test_vs_train.head(top_k)[cols_display_t2].copy()
    disp_t2["delta_nll_test_vs_train"] = disp_t2["delta_nll_test_vs_train"].round(2)
    disp_t2["contrib_pct_test_vs_train"] = disp_t2["contrib_pct_test_vs_train"].round(2)
    disp_t2["train_mean"] = disp_t2["train_mean"].round(3)
    disp_t2["test_mean"] = disp_t2["test_mean"].round(3)
    disp_t2["mean_shift"] = disp_t2["mean_shift"].round(2)
    disp_t2["var_ratio"] = disp_t2["var_ratio"].round(2)
    print(disp_t2.to_string(index=False))

    print("\n" + "=" * 105)
    print(
        f" TABLE 3: TOP {top_k} FEATURES WITH GENERALIZATION SHIFT (Test vs Valid Baseline)"
    )
    print("=" * 105)
    cols_display_t3 = [
        "rank",
        "feature",
        "delta_nll_test_vs_valid",
        "contrib_pct_test_vs_valid",
        "valid_mean",
        "test_mean",
        "mean_shift",
        "var_ratio",
    ]
    disp_t3 = df_test_vs_valid.head(top_k)[cols_display_t3].copy()
    disp_t3["delta_nll_test_vs_valid"] = disp_t3["delta_nll_test_vs_valid"].round(2)
    disp_t3["contrib_pct_test_vs_valid"] = disp_t3["contrib_pct_test_vs_valid"].round(2)
    disp_t3["valid_mean"] = disp_t3["valid_mean"].round(3)
    disp_t3["test_mean"] = disp_t3["test_mean"].round(3)
    disp_t3["mean_shift"] = disp_t3["mean_shift"].round(2)
    disp_t3["var_ratio"] = disp_t3["var_ratio"].round(2)
    print(disp_t3.to_string(index=False))
    print("=" * 105 + "\n")

    return {
        "summary_df": summary_df,
        "summary_valid_vs_train": df_valid_vs_train,
        "summary_test_vs_train": df_test_vs_train,
        "summary_test_vs_valid": df_test_vs_valid,
        "per_contract_summaries": per_contract_summaries,
        "total_delta_test_vs_train": total_delta_test_vs_train,
        "total_delta_valid_vs_train": total_delta_valid_vs_train,
        "total_delta_test_vs_valid": total_delta_test_vs_valid,
        "output_dir": output_dir,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze per-feature VAE OOD contributions across train, valid, and test splits."
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
        "--axis",
        type=str,
        default="slope",
        choices=["slope", "volatility"],
        help="Target VAE axis to diagnose (slope or volatility, default: slope)",
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
        "--top_k",
        type=int,
        default=10,
        help="Number of top features to display in each terminal table (default: 10)",
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
