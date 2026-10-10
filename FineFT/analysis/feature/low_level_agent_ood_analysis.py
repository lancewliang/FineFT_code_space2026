from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure root directory and FineFT are in sys.path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
FINEFT_ROOT = REPO_ROOT / "FineFT"
if str(FINEFT_ROOT) not in sys.path:
    sys.path.insert(0, str(FINEFT_ROOT))

from env.env_initiate.base_initiate import initiate_base_env
from model.low_level import Qnet, ensemble_Qnet
from common import ArtifactNames

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("LowLevelAgentOOD")


@dataclass
class BaselineStats:
    """In-distribution baseline statistics and quantiles derived from training replay buffer."""

    mu_latent: np.ndarray  # shape: (D,)
    precision_matrix: np.ndarray  # shape: (D, D)
    q_var_quantiles: dict[int, float]  # {90: ..., 95: ..., 99: ...}
    disagree_quantiles: dict[int, float]
    mahalanobis_quantiles: dict[int, float]
    baseline_mahalanobis_values: np.ndarray
    sample_count: int


def load_ensemble_model_from_checkpoint(
    model_path: Path,
    device: torch.device,
) -> tuple[ensemble_Qnet, dict[str, int]]:
    """Inspect state_dict and instantiate matching ensemble_Qnet without manual hyperparameter guesswork."""
    logger.info("Loading model weights from: %s", model_path)
    state_dict = torch.load(model_path, map_location=device)

    # Deduce ensemble size and layer shapes
    qnet_indices = set()
    for key in state_dict.keys():
        if key.startswith("qnet_list."):
            qnet_indices.add(int(key.split(".")[1]))

    ensemble_number = len(qnet_indices)
    hidden_nodes = state_dict["qnet_list.0.fc1.weight"].shape[0]
    n_states = state_dict["qnet_list.0.fc1.weight"].shape[1]
    n_actions = state_dict["qnet_list.0.out.weight"].shape[0]
    time_info_dim = state_dict["qnet_list.0.fc_time.weight"].shape[1]
    trading_info_dim = state_dict["qnet_list.0.fc_trading.weight"].shape[1]

    config = {
        "ensemble_number": ensemble_number,
        "hidden_nodes": hidden_nodes,
        "n_states": n_states,
        "n_actions": n_actions,
        "time_info_dim": time_info_dim,
        "trading_info_dim": trading_info_dim,
    }
    logger.info("Inferred ensemble configuration: %s", config)

    model = ensemble_Qnet(
        N_STATES=n_states,
        N_ACTIONS=n_actions,
        hidden_nodes=hidden_nodes,
        TIME_INFO_DIM=time_info_dim,
        ensemble_number=ensemble_number,
        TRADING_INFO_DIM=trading_info_dim,
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()
    return model, config


def forward_qnet_with_latent(
    qnet: Qnet,
    state: torch.Tensor,
    time: torch.Tensor,
    previous_action: torch.Tensor,
    avaliable_action: torch.Tensor,
    trading_info: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass through single Qnet, extracting post-fc2 latent vector."""
    state_hidden = F.relu(qnet.fc1(state))
    t_hidden = qnet.fc_time(time)
    prev_act_hidden = F.relu(qnet.fc3(previous_action))
    trading_hidden = F.relu(qnet.fc_trading(trading_info))
    info_hidden = torch.cat(
        [state_hidden, prev_act_hidden, t_hidden, trading_hidden], dim=1
    )
    latent = qnet.fc2(info_hidden)
    action = qnet.out(latent)
    masked_action = action + (avaliable_action - 1) * qnet.max_punish
    return masked_action, latent


def forward_ensemble_with_latents(
    ensemble: ensemble_Qnet,
    state: torch.Tensor,
    time: torch.Tensor,
    previous_action: torch.Tensor,
    avaliable_action: torch.Tensor,
    trading_info: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass across all subnets, returning (batch, M, N_ACTIONS) and mean latent (batch, D)."""
    q_list: list[torch.Tensor] = []
    latent_list: list[torch.Tensor] = []
    for qnet in ensemble.qnet_list:
        q_val, latent = forward_qnet_with_latent(
            qnet, state, time, previous_action, avaliable_action, trading_info
        )
        q_list.append(q_val)
        latent_list.append(latent)

    q_values = torch.stack(q_list, dim=1)  # (batch, ensemble_number, N_ACTIONS)
    mean_latent = torch.stack(latent_list, dim=1).mean(dim=1)  # (batch, hidden_nodes)
    return q_values, mean_latent


def compute_step_ood_metrics(
    q_values: torch.Tensor,
    mean_latents: torch.Tensor,
    baseline: BaselineStats | None = None,
) -> dict[str, np.ndarray]:
    """Calculate Ensemble Q variance, Action disagreement rate, and Latent Mahalanobis distance."""
    # 1. Epistemic Q variance around consensus greedy action
    # q_values: (B, M, A)
    consensus_q = q_values.mean(dim=1)  # (B, A)
    greedy_action = consensus_q.argmax(dim=-1, keepdim=True)  # (B, 1)

    # Gather Q values for the consensus action across all M models: (B, M, 1)
    gathered_q = q_values.gather(dim=-1, index=greedy_action.unsqueeze(1).expand(-1, q_values.shape[1], -1))
    q_variance = gathered_q.squeeze(-1).var(dim=1, unbiased=False).cpu().numpy()  # (B,)

    # 2. Optimal action decision disagreement rate
    sub_optimal_actions = q_values.argmax(dim=-1).cpu().numpy()  # (B, M)
    batch_size, ensemble_size = sub_optimal_actions.shape
    disagreement_rates = np.zeros(batch_size, dtype=float)

    for i in range(batch_size):
        actions_row = sub_optimal_actions[i]
        _, counts = np.unique(actions_row, return_counts=True)
        max_vote = counts.max()
        disagreement_rates[i] = 1.0 - (float(max_vote) / float(ensemble_size))

    # 3. Latent Mahalanobis distance
    latent_np = mean_latents.cpu().numpy()  # (B, D)
    if baseline is not None:
        diff = latent_np - baseline.mu_latent  # (B, D)
        # Mahalanobis D_M = sqrt(diff @ precision @ diff.T)
        left = np.dot(diff, baseline.precision_matrix)  # (B, D)
        quad = np.sum(left * diff, axis=1)  # (B,)
        mahalanobis_dist = np.sqrt(np.maximum(quad, 0.0))
    else:
        mahalanobis_dist = np.zeros(batch_size, dtype=float)

    return {
        "q_variance": q_variance,
        "disagreement_rate": disagreement_rates,
        "mahalanobis_distance": mahalanobis_dist,
        "consensus_action": greedy_action.squeeze(-1).cpu().numpy(),
    }


def fit_in_distribution_baseline(
    ensemble: ensemble_Qnet,
    buffer_path: Path,
    device: torch.device,
    max_samples_per_grid: int = 2000,
    batch_size: int = 512,
) -> BaselineStats:
    """Extract stratified transitions from buffer_diverse.pkl and fit reference statistics."""
    logger.info("Fitting in-distribution baseline from buffer: %s", buffer_path)
    buffer_payload: dict[int, dict[str, Any]] = torch.load(buffer_path, map_location="cpu")

    all_states: list[torch.Tensor] = []
    all_times: list[torch.Tensor] = []
    all_prev_acts: list[torch.Tensor] = []
    all_avail_acts: list[torch.Tensor] = []
    all_trading_infos: list[torch.Tensor] = []

    for grid_id, grid_data in buffer_payload.items():
        grid_n = grid_data["states"].shape[0]
        if grid_n == 0:
            continue
        sample_n = min(grid_n, max_samples_per_grid)
        indices = np.random.choice(grid_n, size=sample_n, replace=False)

        states = grid_data["states"][indices].float()
        infos = grid_data["infos"]
        avail_act = infos["avaliable_action"][indices].float()
        prev_act = infos["previous_action"][indices].float().unsqueeze(-1)
        h_down = infos["funding_count_down_hour"][indices].float().unsqueeze(-1)
        m_down = infos["funding_count_down_minute"][indices].float().unsqueeze(-1)
        time_input = torch.cat([h_down, m_down], dim=1)
        trading_info = infos["trading_info"][indices].float()

        all_states.append(states)
        all_times.append(time_input)
        all_prev_acts.append(prev_act)
        all_avail_acts.append(avail_act)
        all_trading_infos.append(trading_info)

    stacked_states = torch.cat(all_states, dim=0)
    stacked_times = torch.cat(all_times, dim=0)
    stacked_prev_acts = torch.cat(all_prev_acts, dim=0)
    stacked_avail_acts = torch.cat(all_avail_acts, dim=0)
    stacked_trading_infos = torch.cat(all_trading_infos, dim=0)

    total_samples = stacked_states.shape[0]
    logger.info("Stratified baseline transition pool size: %d samples", total_samples)

    q_var_list: list[np.ndarray] = []
    disagree_list: list[np.ndarray] = []
    latent_list: list[np.ndarray] = []

    with torch.no_grad():
        for start_idx in range(0, total_samples, batch_size):
            end_idx = min(start_idx + batch_size, total_samples)
            s = stacked_states[start_idx:end_idx].to(device)
            t = stacked_times[start_idx:end_idx].to(device)
            pa = stacked_prev_acts[start_idx:end_idx].to(device)
            aa = stacked_avail_acts[start_idx:end_idx].to(device)
            ti = stacked_trading_infos[start_idx:end_idx].to(device)

            q_vals, latents = forward_ensemble_with_latents(ensemble, s, t, pa, aa, ti)
            metrics = compute_step_ood_metrics(q_vals, latents, baseline=None)

            q_var_list.append(metrics["q_variance"])
            disagree_list.append(metrics["disagreement_rate"])
            latent_list.append(latents.cpu().numpy())

    all_q_vars = np.concatenate(q_var_list, axis=0)
    all_disagrees = np.concatenate(disagree_list, axis=0)
    all_latents = np.concatenate(latent_list, axis=0)

    # Fit Ledoit-Wolf precision matrix on latent embeddings
    logger.info("Fitting Ledoit-Wolf covariance shrinkage on %d latent embeddings...", total_samples)
    lw = LedoitWolf(assume_centered=False).fit(all_latents)
    mu_latent = lw.location_
    precision_matrix = lw.precision_

    # Calculate Mahalanobis distances on baseline
    diff = all_latents - mu_latent
    left = np.dot(diff, precision_matrix)
    base_mahalanobis = np.sqrt(np.maximum(np.sum(left * diff, axis=1), 0.0))

    q_var_quantiles = {
        90: float(np.percentile(all_q_vars, 90)),
        95: float(np.percentile(all_q_vars, 95)),
        99: float(np.percentile(all_q_vars, 99)),
    }
    disagree_quantiles = {
        90: float(np.percentile(all_disagrees, 90)),
        95: float(np.percentile(all_disagrees, 95)),
        99: float(np.percentile(all_disagrees, 99)),
    }
    mahalanobis_quantiles = {
        90: float(np.percentile(base_mahalanobis, 90)),
        95: float(np.percentile(base_mahalanobis, 95)),
        99: float(np.percentile(base_mahalanobis, 99)),
    }

    logger.info("Baseline Q-variance thresholds: 90%%=%.4f, 95%%=%.4f, 99%%=%.4f",
                q_var_quantiles[90], q_var_quantiles[95], q_var_quantiles[99])
    logger.info("Baseline Disagreement thresholds: 90%%=%.4f, 95%%=%.4f, 99%%=%.4f",
                disagree_quantiles[90], disagree_quantiles[95], disagree_quantiles[99])
    logger.info("Baseline Mahalanobis thresholds: 90%%=%.4f, 95%%=%.4f, 99%%=%.4f",
                mahalanobis_quantiles[90], mahalanobis_quantiles[95], mahalanobis_quantiles[99])

    return BaselineStats(
        mu_latent=mu_latent,
        precision_matrix=precision_matrix,
        q_var_quantiles=q_var_quantiles,
        disagree_quantiles=disagree_quantiles,
        mahalanobis_quantiles=mahalanobis_quantiles,
        baseline_mahalanobis_values=base_mahalanobis,
        sample_count=total_samples,
    )


def run_contract_env_rollout(
    ensemble: ensemble_Qnet,
    df: pd.DataFrame,
    feature_names: list[str],
    baseline: BaselineStats,
    device: torch.device,
    position_choices: int = 3,
    max_holding_number: int = 1,
    leverage_choices: list[int] | None = None,
    holding_duration_norm_steps: int = 180,
    order_book_depth: int = 0,
    maintenance_margin_ratio_dict: dict[str, list[float]] | None = None,
) -> pd.DataFrame:
    """Run interactive step-by-step evaluation rollout in Base_Env, capturing sequential OOD dynamics."""
    if leverage_choices is None:
        leverage_choices = [1]

    if order_book_depth <= 0:
        detected_depth = 0
        while f"bid{detected_depth + 1}_price" in df.columns and f"ask{detected_depth + 1}_price" in df.columns:
            detected_depth += 1
        effective_depth = detected_depth if detected_depth > 0 else 5
    else:
        effective_depth = order_book_depth

    env_kwargs: dict[str, Any] = {
        "df": df,
        "feature_list": feature_names,
        "max_holding_number": max_holding_number,
        "position_choices": position_choices,
        "leverage_choice": leverage_choices,
        "holding_duration_norm_steps": holding_duration_norm_steps,
        "order_book_depth": effective_depth,
        "initial_state": (1e5, 0, 0, 0, leverage_choices[0]),
    }
    if maintenance_margin_ratio_dict is not None:
        env_kwargs["maintenance_margin_ratio_dict"] = maintenance_margin_ratio_dict

    env = initiate_base_env(**env_kwargs)

    state, info = env.reset()
    done = False
    step_idx = 0

    records: list[dict[str, Any]] = []

    with torch.no_grad():
        while not done:
            state_tensor = torch.unsqueeze(torch.FloatTensor(state).reshape(-1), 0).to(device)
            prev_act_tensor = torch.unsqueeze(torch.tensor([info["previous_action"]]).float().to(device), 0)
            avail_act_tensor = torch.unsqueeze(torch.tensor(info["avaliable_action"]).to(device), 0)
            h_count = torch.unsqueeze(torch.tensor([info["funding_count_down_hour"]]), 0).float().to(device)
            m_count = torch.unsqueeze(torch.tensor([info["funding_count_down_minute"]]), 0).float().to(device)
            time_tensor = torch.cat([h_count, m_count], dim=1)
            trading_tensor = torch.from_numpy(info["trading_info"]).float().reshape(1, -1).to(device)

            q_vals, latent = forward_ensemble_with_latents(
                ensemble, state_tensor, time_tensor, prev_act_tensor, avail_act_tensor, trading_tensor
            )
            step_metrics = compute_step_ood_metrics(q_vals, latent, baseline)

            # Consensus decision with soft variance penalty matching test_agent_average
            mean_q = q_vals.mean(dim=1)
            std_q = q_vals.std(dim=1)
            chosen_q = mean_q - 0.0001 * std_q
            action = int(torch.max(chosen_q, 1)[1].item())

            mark_price = float(df["mark_price"].iloc[step_idx]) if "mark_price" in df.columns else 0.0
            timestamp = df["timestamp"].iloc[step_idx] if "timestamp" in df.columns else step_idx

            q_var = float(step_metrics["q_variance"][0])
            disagree = float(step_metrics["disagreement_rate"][0])
            mahal = float(step_metrics["mahalanobis_distance"][0])

            records.append({
                "step": step_idx,
                "timestamp": timestamp,
                "mark_price": mark_price,
                "action": action,
                "position": float(info["previous_action"]),
                "q_variance": q_var,
                "disagreement_rate": disagree,
                "mahalanobis_distance": mahal,
                "is_q_var_ood_95": bool(q_var > baseline.q_var_quantiles[95]),
                "is_disagree_ood_95": bool(disagree > baseline.disagree_quantiles[95]),
                "is_mahalanobis_ood_95": bool(mahal > baseline.mahalanobis_quantiles[95]),
            })

            state, _, done, info = env.step(action)
            step_idx += 1

    return pd.DataFrame(records)


def run_contract_static_scan(
    ensemble: ensemble_Qnet,
    df: pd.DataFrame,
    feature_names: list[str],
    baseline: BaselineStats,
    device: torch.device,
    batch_size: int = 512,
    n_actions: int = 3,
) -> pd.DataFrame:
    """Run offline static feature scan without environment execution, using neutral defaults."""
    total_samples = len(df)
    state_vals = df[feature_names].values.astype(float)

    h_down = df["funding_count_down_hour"].values if "funding_count_down_hour" in df.columns else np.zeros(total_samples)
    m_down = df["funding_count_down_minute"].values if "funding_count_down_minute" in df.columns else np.zeros(total_samples)

    records: list[dict[str, Any]] = []

    with torch.no_grad():
        for start_idx in range(0, total_samples, batch_size):
            end_idx = min(start_idx + batch_size, total_samples)
            b = end_idx - start_idx

            s = torch.tensor(state_vals[start_idx:end_idx], dtype=torch.float32, device=device)
            t = torch.tensor(np.stack([h_down[start_idx:end_idx], m_down[start_idx:end_idx]], axis=1), dtype=torch.float32, device=device)
            pa = torch.zeros((b, 1), dtype=torch.float32, device=device)
            aa = torch.ones((b, n_actions), dtype=torch.float32, device=device)
            ti = torch.zeros((b, 4), dtype=torch.float32, device=device)

            q_vals, latents = forward_ensemble_with_latents(ensemble, s, t, pa, aa, ti)
            metrics = compute_step_ood_metrics(q_vals, latents, baseline)

            for i in range(b):
                idx = start_idx + i
                q_var = float(metrics["q_variance"][i])
                disagree = float(metrics["disagreement_rate"][i])
                mahal = float(metrics["mahalanobis_distance"][i])
                action = int(metrics["consensus_action"][i])
                mark_price = float(df["mark_price"].iloc[idx]) if "mark_price" in df.columns else 0.0
                timestamp = df["timestamp"].iloc[idx] if "timestamp" in df.columns else idx

                records.append({
                    "step": idx,
                    "timestamp": timestamp,
                    "mark_price": mark_price,
                    "action": action,
                    "position": 0.0,
                    "q_variance": q_var,
                    "disagreement_rate": disagree,
                    "mahalanobis_distance": mahal,
                    "is_q_var_ood_95": bool(q_var > baseline.q_var_quantiles[95]),
                    "is_disagree_ood_95": bool(disagree > baseline.disagree_quantiles[95]),
                    "is_mahalanobis_ood_95": bool(mahal > baseline.mahalanobis_quantiles[95]),
                })

    return pd.DataFrame(records)


def plot_epistemic_uncertainty_timeseries(
    contract_df: pd.DataFrame,
    baseline: BaselineStats,
    save_path: Path,
    contract_name: str,
):
    """Plot dual-axis time series showing price, agent action, Q-variance, and 95% threshold."""
    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(3, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [2, 1.5, 1.5]})

    steps = contract_df["step"].values
    price = contract_df["mark_price"].values
    q_var = contract_df["q_variance"].values
    disagree = contract_df["disagreement_rate"].values

    # 1. Price and Actions
    ax_top.plot(steps, price, color="black", label="Mark Price", alpha=0.8)
    ax_top.set_ylabel("Price")
    ax_top.set_title(f"Agent Epistemic Uncertainty & Policy Diagnostics: {contract_name}", fontsize=14)
    ax_top.grid(True, linestyle="--", alpha=0.5)

    # 2. Q-Variance with 95% threshold
    ax_mid.plot(steps, q_var, color="#1f77b4", label="Q Epistemic Variance", linewidth=1.2)
    ax_mid.axhline(baseline.q_var_quantiles[95], color="red", linestyle="--", label=f"95% OOD Threshold ({baseline.q_var_quantiles[95]:.3f})")
    ax_mid.set_ylabel("Q Variance")
    ax_mid.legend(loc="upper left")
    ax_mid.grid(True, linestyle="--", alpha=0.5)

    # 3. Action Disagreement Rate
    ax_bot.plot(steps, disagree, color="#ff7f0e", label="Action Disagreement Rate", linewidth=1.2)
    ax_bot.axhline(baseline.disagree_quantiles[95], color="red", linestyle="--", label=f"95% OOD Threshold ({baseline.disagree_quantiles[95]:.3f})")
    ax_bot.set_ylabel("Disagreement")
    ax_bot.set_xlabel("Contract Steps")
    ax_bot.legend(loc="upper left")
    ax_bot.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    logger.info("Saved time-series diagnostic plot to: %s", save_path)


def plot_action_agreement_distribution(
    all_df: pd.DataFrame,
    baseline: BaselineStats,
    save_path: Path,
):
    """Plot histogram and CDF of action disagreement rates."""
    fig, ax = plt.subplots(figsize=(8, 5))
    disagree_vals = all_df["disagreement_rate"].values

    ax.hist(disagree_vals, bins=20, density=True, color="#2ca02c", alpha=0.7, edgecolor="black")
    ax.axvline(baseline.disagree_quantiles[95], color="red", linestyle="--", linewidth=2, label=f"95% Baseline ({baseline.disagree_quantiles[95]:.3f})")
    ax.set_title("Action Decision Disagreement Distribution", fontsize=12)
    ax.set_xlabel("Disagreement Rate [0, 1]")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    logger.info("Saved action agreement distribution plot to: %s", save_path)


def plot_latent_mahalanobis_kde(
    baseline_mahal: np.ndarray,
    eval_mahal: np.ndarray,
    save_path: Path,
):
    """Plot overlay comparing baseline vs eval latent Mahalanobis distance distributions."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(baseline_mahal, bins=40, density=True, alpha=0.5, color="blue", label="Training In-Distribution")
    ax.hist(eval_mahal, bins=40, density=True, alpha=0.5, color="orange", label="Evaluation Split")
    ax.set_title("Latent Policy Manifold Mahalanobis Distance Shift", fontsize=12)
    ax.set_xlabel("Mahalanobis Distance D_M(s)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    logger.info("Saved latent Mahalanobis distribution plot to: %s", save_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Low-Level Agent Policy-Level OOD Diagnostic Analysis"
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained_model.pkl")
    parser.add_argument("--buffer_path", type=str, default=None, help="Path to buffer_diverse.pkl snapshot")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to scaled feather data root")
    parser.add_argument(
        "--feature_path",
        type=str,
        default=None,
        help=f"Path to rl_state_features.npy (defaults to <data_dir>/{ArtifactNames.RL_STATE_FEATURES_NPY})",
    )
    parser.add_argument("--split", type=str, default="test", choices=["test", "valid", "train"], help="Split to evaluate")
    parser.add_argument("--symbol", type=str, default="fu", help="Commodity symbol")
    parser.add_argument("--target_freq", type=str, default="10min", help="Target bar frequency")
    parser.add_argument("--eval_mode", type=str, default="env_rollout", choices=["env_rollout", "static_scan"], help="Evaluation mode")
    parser.add_argument("--output_dir", type=str, default=None, help="Custom output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device (cpu, cuda)")
    parser.add_argument("--max_samples_per_grid", type=int, default=2000, help="Max transitions sampled per grid for baseline")
    parser.add_argument("--position_choices", type=int, default=3, help="Number of discrete position choices (e.g. 3 for short/flat/long)")
    parser.add_argument("--max_holding_number", type=int, default=1, help="Max holding units")
    parser.add_argument("--order_book_depth", type=int, default=0, help="Order book depth (0 to auto-detect from dataframe columns)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    logger.info("Running low-level agent OOD analysis with device=%s, eval_mode=%s", device, args.eval_mode)

    model_path = Path(args.model_path)
    if model_path.is_dir():
        model_file = model_path / "trained_model.pkl"
    else:
        model_file = model_path

    if not model_file.is_file():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    # Resolve buffer_path
    if args.buffer_path:
        buffer_file = Path(args.buffer_path)
    else:
        # Search parent folders of model_file for buffer_diverse.pkl
        candidate = model_file.parent.parent / "buffer_diverse.pkl"
        if candidate.is_file():
            buffer_file = candidate
        else:
            candidate_same = model_file.parent / "buffer_diverse.pkl"
            if candidate_same.is_file():
                buffer_file = candidate_same
            else:
                raise FileNotFoundError("buffer_diverse.pkl not specified and not found near model_path")

    # Load feature names
    if args.feature_path:
        feature_path = Path(args.feature_path)
    else:
        feature_path = Path(args.data_dir) / ArtifactNames.RL_STATE_FEATURES_NPY
    if not feature_path.is_file():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")
    feature_names = list(np.load(feature_path, allow_pickle=True))
    logger.info("Loaded %d state features from %s", len(feature_names), feature_path)

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = REPO_ROOT / f"analysis_result/DiHFT/agent_ood/{args.symbol}/{args.target_freq}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Model
    ensemble, model_cfg = load_ensemble_model_from_checkpoint(model_file, device)

    # 2. Fit In-Distribution Baseline
    baseline = fit_in_distribution_baseline(
        ensemble=ensemble,
        buffer_path=buffer_file,
        device=device,
        max_samples_per_grid=args.max_samples_per_grid,
    )

    # 3. Discover Split Feather Files
    data_dir = Path(args.data_dir) / args.split
    feather_files = sorted(data_dir.glob("*.feather"))
    if not feather_files:
        raise FileNotFoundError(f"No feather files found in split directory: {data_dir}")

    logger.info("Found %d contracts in %s split", len(feather_files), args.split)

    margin_dict: dict[str, list[float]] | None = None
    for p in [Path(args.data_dir) / "maintenance_margin_ratio_dict.npy", Path(args.data_dir).parent / "maintenance_margin_ratio_dict.npy"]:
        if p.is_file():
            margin_dict = np.load(p, allow_pickle=True).item()
            logger.info("Loaded maintenance margin ratio dict from: %s", p)
            break

    summary_rows: list[dict[str, Any]] = []
    contract_results: dict[str, pd.DataFrame] = {}

    for file_path in feather_files:
        contract_name = file_path.stem
        logger.info("Evaluating contract: %s (%s)", contract_name, file_path.name)
        df_contract = pd.read_feather(file_path)

        if args.eval_mode == "env_rollout":
            res_df = run_contract_env_rollout(
                ensemble=ensemble,
                df=df_contract,
                feature_names=feature_names,
                baseline=baseline,
                device=device,
                position_choices=args.position_choices,
                max_holding_number=args.max_holding_number,
                order_book_depth=args.order_book_depth,
                maintenance_margin_ratio_dict=margin_dict,
            )
        else:
            res_df = run_contract_static_scan(
                ensemble=ensemble,
                df=df_contract,
                feature_names=feature_names,
                baseline=baseline,
                device=device,
                n_actions=model_cfg["n_actions"],
            )

        contract_results[contract_name] = res_df

        q_var_vals = res_df["q_variance"].values
        disagree_vals = res_df["disagreement_rate"].values
        mahal_vals = res_df["mahalanobis_distance"].values

        q_var_ood_ratio = float(np.mean(q_var_vals > baseline.q_var_quantiles[95]))
        disagree_ood_ratio = float(np.mean(disagree_vals > baseline.disagree_quantiles[95]))
        mahal_ood_ratio = float(np.mean(mahal_vals > baseline.mahalanobis_quantiles[95]))

        summary_rows.append({
            "contract": contract_name,
            "steps": len(res_df),
            "q_var_mean": float(np.mean(q_var_vals)),
            "q_var_p95": float(np.percentile(q_var_vals, 95)),
            "q_var_ood_ratio_95": q_var_ood_ratio,
            "disagreement_mean": float(np.mean(disagree_vals)),
            "disagreement_p95": float(np.percentile(disagree_vals, 95)),
            "disagreement_ood_ratio_95": disagree_ood_ratio,
            "mahalanobis_mean": float(np.mean(mahal_vals)),
            "mahalanobis_p95": float(np.percentile(mahal_vals, 95)),
            "mahalanobis_ood_ratio_95": mahal_ood_ratio,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_dir / "agent_ood_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    logger.info("Saved agent OOD summary table to: %s", summary_csv)

    # 4. Generate Visual Analytics
    # Timeseries plot for the first contract (or representative contract)
    first_contract = list(contract_results.keys())[0]
    ts_plot_path = output_dir / "agent_epistemic_uncertainty_timeseries.png"
    plot_epistemic_uncertainty_timeseries(
        contract_results[first_contract],
        baseline,
        ts_plot_path,
        contract_name=first_contract,
    )

    # Distribution plot for all contracts combined
    combined_df = pd.concat(list(contract_results.values()), ignore_index=True)
    agree_plot_path = output_dir / "action_agreement_distribution.png"
    plot_action_agreement_distribution(combined_df, baseline, agree_plot_path)

    # Latent Mahalanobis distribution shift
    kde_plot_path = output_dir / "latent_mahalanobis_kde.png"
    plot_latent_mahalanobis_kde(
        baseline.baseline_mahalanobis_values,
        combined_df["mahalanobis_distance"].values,
        kde_plot_path,
    )

    logger.info("Agent policy OOD analysis completed successfully! Deliverables in: %s", output_dir)


if __name__ == "__main__":
    main()
