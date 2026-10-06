# the frequency of the high level agent is the same as the low level agent
# based on a sequence of high level actions
import argparse
from collections import deque
import json
import logging
import os
import random
import shutil
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append(".")

logger = logging.getLogger(__name__)
if not logger.handlers and not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

from env.env_initiate.base_initiate import initiate_base_env, Base_Env
from env.env_class.futures_util import (
    map_action_to_position_leverage,
    map_position_leverage_to_action,
    rule_based_close,
)
from RL.DiHFT.VAE.vae import MLP_VAE, analyze_single_sample, gaussian_nll, softclip
from RL.DiHFT.high_level.gating import create_gating_strategy

from analysis.pick_agent.FineFT_two_dimensional_agent_selector import (
    TwoDimensionalSelectionManifest,
)
from common import (
    ActionDecisionReasons,
    ArtifactNames,
    HistoryArtifactNames,
    MetricColumns,
    RoutingParamColumns,
)
from model.low_level import ensemble_Qnet
from model.high_level import RankBasedQNetwork
from RL.util.update import disable_gradients, get_rank
from analysis.calculate_metric.calculate_metric import (
    calculate_differences,
    calculate_required_money,
)

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"
parser = argparse.ArgumentParser()
# * Env setting
parser.add_argument(
    "--base_path",
    type=str,
    default="dataset",
    help="the number of action we have in the training and testing env",
)
parser.add_argument(
    "--dataset_name",
    type=str,
    default="BTCUSDT",
    help="training data chunk",
)
parser.add_argument(
    "--experiment_name",
    type=str,
    default="default",
    help="experiment name",
)
parser.add_argument(
    "--max_holding_number",
    type=float,
    default=8,
    help="the transcation cost of not holding the same action as before",
)
parser.add_argument(
    "--position_choices",
    type=int,
    default=9,
    help="the transcation cost of not holding the same action as before",
)
parser.add_argument(
    "--leverage_choices",
    action="append",
    type=int,
    default=[5],
    help="the transaction cost of not holding the same action as before",
)
parser.add_argument(
    "--long_estimated_rate",
    type=float,
    default=0.0005,
    help="the transcation cost of not holding the same action as before",
)
parser.add_argument(
    "--short_estimated_rate",
    type=float,
    default=0,
    help="the transcation cost of not holding the same action as before",
)
parser.add_argument(
    "--transcation_cost",
    type=float,
    default=0.0002,
    help="the transcation cost of not holding the same action as before",
)

parser.add_argument(
    "--early_stop",
    type=int,
    default=0,
    help="the transcation cost of not holding the same action as before",
)
parser.add_argument(
    "--initial_wallet_balance",
    type=float,
    default=1e5,
    help="wallet balance",
)
parser.add_argument(
    "--initial_margin",
    type=float,
    default=0,
    help="initial margin",
)
parser.add_argument(
    "--initial_unrealized_pnL",
    type=float,
    default=0,
    help="unrealized pnL",
)
parser.add_argument(
    "--initial_position",
    type=float,
    default=0,
    help="unrealized pnL",
)
parser.add_argument(
    "--initial_leverage",
    type=float,
    default=5,
    help="initial leverage",
)
parser.add_argument(
    "--order_book_depth",
    type=int,
    default=25,
    help="number of bid/ask price levels available in the order book",
)
parser.add_argument(
    "--allow_reverse_position",
    action="store_true",
    help="allow reverse position in single step",
)
parser.add_argument(
    "--action_persistence",
    type=int,
    default=3,
    help="number of consecutive steps a non-flat action persists before re-evaluating policy",
)
parser.add_argument(
    "--stop_loss_abs_threshold",
    type=float,
    default=50.0,
    help="unrealized PnL hard stop-loss absolute threshold in quote currency (0.0 to disable)",
)
parser.add_argument(
    "--stop_loss_cooldown_steps",
    type=int,
    default=12,
    help="directional lockout cooldown steps after stop-loss trigger (0 to disable)",
)
parser.add_argument(
    "--circuit_breaker_consecutive_stops",
    type=int,
    default=2,
    help="consecutive stop-loss trigger count to trip contract-level circuit breaker (0 to disable)",
)
parser.add_argument(
    "--circuit_breaker_cooling_steps",
    type=int,
    default=72,
    help="steps to suspend trading on circuit breaker (-1 for permanent suspension)",
)
# low level network setting
parser.add_argument(
    "--hidden_nodes",
    type=int,
    default=128,
    help="the number of the hidden nodes",
)

parser.add_argument(
    "--time_info_dim",
    type=int,
    default=2,
    help="context number",
)
# VAE network path
parser.add_argument(
    "--vae_path",
    type=str,
    default="result/DiHFT/vae_results",
    help="the path for storing the test result",
)
# vae related
parser.add_argument(
    "--z_dim",
    type=int,
    default=512,
    help="the sequency length",
)
parser.add_argument(
    "--vae_hidden_dims",
    type=list,
    default=[4096, 2048, 1024, 1024],
    help="the sequency length",
)
parser.add_argument(
    "--loss_type",
    type=str,
    default="NLL",
    help="the sequency length",
)
parser.add_argument(
    "--vae_results",
    type=str,
    default="result/DiHFT/vae_results",
    help="the sequency length",
)

# high level network setting
parser.add_argument(
    "--result_path",
    type=str,
    default="result/DiHFT/high_level",
    help="the path for storing the test result",
)
parser.add_argument(
    "--window_length",
    type=int,
    default=64,
    help="the path for storing the test result",
)
parser.add_argument(
    "--gamma",
    type=float,
    default=0.9,
    help="the path for storing the test result",
)
# 判断是rule base，且之前的down deviation以及超过5% 切成rule based result 等五个step
parser.add_argument(
    "--rule_base_threshold",
    type=float,
    default=0.2,
    help="the sequency length",
)
parser.add_argument(
    "--selection_manifest",
    type=str,
    default=None,
    help="two-dimensional low-level selection manifest",
)
parser.add_argument(
    "--enable_non_main_contract_defense",
    action="store_true",
    default=False,
    help="enable defensive gating on non-main and non-sub-main contracts",
)
parser.add_argument(
    "--eval_stage",
    type=str,
    default="valid",
    choices=["valid", "test"],
    help="evaluation dataset stage (valid or test)",
)
parser.add_argument(
    "--para_file",
    type=str,
    default=None,
    help="path to high_level_agent_para.txt",
)
parser.add_argument(
    "--optuna_csv",
    type=str,
    default=None,
    help="path to optuna_results.csv",
)
parser.add_argument(
    "--slope_window_length",
    type=int,
    default=None,
    help="slope rolling window length",
)
parser.add_argument(
    "--volatility_window_length",
    type=int,
    default=None,
    help="volatility rolling window length",
)
parser.add_argument(
    "--slope_gamma",
    type=float,
    default=None,
    help="slope decay gamma",
)
parser.add_argument(
    "--volatility_gamma",
    type=float,
    default=None,
    help="volatility decay gamma",
)
parser.add_argument(
    "--slope_rule_base_threshold",
    type=float,
    default=None,
    help="slope rule base threshold",
)
parser.add_argument(
    "--volatility_rule_base_threshold",
    type=float,
    default=None,
    help="volatility rule base threshold",
)
parser.add_argument(
    "--gating_strategy",
    type=str,
    default="absolute",
    choices=["absolute", "hierarchical"],
    help="gating strategy type (absolute or hierarchical)",
)
parser.add_argument(
    "--ood_threshold",
    type=float,
    default=0.005,
    help="OOD circuit breaker threshold for hierarchical gating",
)
parser.add_argument(
    "--slope_margin_threshold",
    type=float,
    default=0.12,
    help="slope top-1 vs top-2 probability margin threshold for hierarchical gating",
)
parser.add_argument(
    "--volatility_margin_threshold",
    type=float,
    default=0.12,
    help="volatility top-1 vs top-2 probability margin threshold for hierarchical gating",
)
parser.add_argument(
    "--trial_number",
    type=int,
    default=None,
    help="Optuna trial number used to isolate result artifacts",
)

parser.add_argument(
    "--gpu_index",
    type=int,
    default=0,
    help="the transcation cost of not holding the same action as before",
)
parser.add_argument(
    "--precomputed_quantiles_dir",
    type=str,
    default=None,
    help="directory containing precomputed VAE quantiles for contracts",
)
parser.add_argument(
    "--save_artifacts",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="save simulation history arrays and contract results to disk",
)


def seed_torch(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


import re




def compute_vae_losses_batch(
    model: MLP_VAE,
    data: np.ndarray,
    device: str,
    batch_size: int = 512,
) -> np.ndarray:
    model.eval()
    losses = []
    data_tensor = torch.from_numpy(data).float()
    with torch.no_grad():
        for start_idx in range(0, len(data_tensor), batch_size):
            batch = data_tensor[start_idx : start_idx + batch_size].to(device)
            recon_mu, recon_logsigma, mu, logvar = model(batch)
            recon_logvar = softclip(recon_logsigma, -6.0)
            recon_logvar = -softclip(-recon_logvar, 0.0)
            rec = gaussian_nll(recon_mu, 0.5 * recon_logvar, batch).sum(dim=-1)
            kld = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1, dim=-1)
            batch_loss = -(rec + kld).cpu().numpy()
            losses.append(batch_loss)
    return np.concatenate(losses)


def default_precomputed_quantiles_dir(
    dataset_name: str,
    experiment_name: str,
    eval_stage: str,
) -> str:
    return os.path.join(
        "analysis_result",
        "DiHFT",
        "high_level",
        dataset_name,
        experiment_name,
        "vae_quantiles",
        eval_stage,
    )


def ensure_precomputed_vae_quantiles(args) -> str:
    """Precompute VAE routing quantiles for all contracts in stage if missing.

    Runs in a single process (optionally GPU or CPU batch) before worker processes
    are launched, saving (T, 2*num_labels) float32 arrays into analysis_result/...
    """
    target_dir = args.precomputed_quantiles_dir or default_precomputed_quantiles_dir(
        args.dataset_name, args.experiment_name, args.eval_stage
    )
    os.makedirs(target_dir, exist_ok=True)

    stage_dir = os.path.join(args.base_path, args.dataset_name, args.eval_stage)
    single_file = os.path.join(
        args.base_path, args.dataset_name, f"{args.eval_stage}.feather"
    )

    contract_files = []
    if os.path.isdir(stage_dir):
        for filename in sorted(os.listdir(stage_dir)):
            if filename.endswith(".feather"):
                contract_files.append(
                    (os.path.splitext(filename)[0], os.path.join(stage_dir, filename))
                )
    elif os.path.isfile(single_file):
        contract_files.append((args.eval_stage, single_file))

    if not contract_files:
        raise ValueError(f"No contract files found in {stage_dir} or {single_file}")

    all_exist = all(
        os.path.isfile(os.path.join(target_dir, f"{contract}.npy"))
        for contract, _ in contract_files
    )
    if all_exist:
        logger.info(
            "Found complete precomputed VAE quantiles in %s for %d contracts",
            target_dir,
            len(contract_files),
        )
        return target_dir

    logger.info(
        "Precomputing VAE quantiles for %d contracts into %s ...",
        len(contract_files),
        target_dir,
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    slope_indicators = np.load(
        os.path.join(
            args.base_path, args.dataset_name, ArtifactNames.VAE_SLOPE_STATE_FEATURES_NPY
        )
    )
    vol_indicators = np.load(
        os.path.join(
            args.base_path,
            args.dataset_name,
            ArtifactNames.VAE_VOLATILITY_STATE_FEATURES_NPY,
        )
    )

    manifest = load_two_dimensional_selection_manifest(args.selection_manifest)
    num_labels = len(manifest.axes.volatility)

    default_vae_root = os.path.join(
        args.vae_path,
        args.dataset_name,
        args.experiment_name,
    )
    vae_roots = {
        "slope": os.path.join(default_vae_root, "slope"),
        "volatility": os.path.join(default_vae_root, "volatility"),
    }
    axis_indicators = {
        "slope": slope_indicators,
        "volatility": vol_indicators,
    }

    vae_models = {}
    sorted_bases = {}
    for axis, root in vae_roots.items():
        vae_models[axis] = []
        sorted_bases[axis] = []
        for i in range(num_labels):
            label = f"label_{i}"
            path = os.path.join(root, label, ArtifactNames.MODEL_LATEST_PTH)
            id_path = os.path.join(root, label, ArtifactNames.ID_LOGPX_NPY)
            model = MLP_VAE(
                INPUT_DIM=len(axis_indicators[axis]),
                Z_DIM=args.z_dim,
                hidden_dims=args.vae_hidden_dims,
                loss_func=args.loss_type,
            ).to(device)
            model.load_state_dict(
                torch.load(path, map_location=torch.device(device))
            )
            model.eval()
            disable_gradients(model)
            vae_models[axis].append(model)
            sorted_bases[axis].append(np.sort(np.load(id_path).reshape(-1)))

    for contract, file_path in contract_files:
        save_file = os.path.join(target_dir, f"{contract}.npy")
        if os.path.isfile(save_file):
            continue

        df = pd.read_feather(file_path)
        t_len = len(df)
        quantiles_matrix = np.zeros((t_len, 2 * num_labels), dtype=np.float32)

        for axis_idx, axis in enumerate(("slope", "volatility")):
            axis_data = df[axis_indicators[axis]].values.astype(np.float32)
            for label_idx in range(num_labels):
                col_idx = axis_idx * num_labels + label_idx
                model = vae_models[axis][label_idx]
                sorted_base = sorted_bases[axis][label_idx]

                losses = compute_vae_losses_batch(model, axis_data, device)
                quantiles = np.searchsorted(sorted_base, losses, side="right") / float(
                    len(sorted_base)
                )
                quantiles[losses < sorted_base[0]] = 0.0
                quantiles[losses > sorted_base[-1]] = 1.0
                quantiles_matrix[:, col_idx] = quantiles.astype(np.float32)

        np.save(save_file, quantiles_matrix)
        logger.info(
            "Saved precomputed quantiles for contract '%s' (shape: %s) -> %s",
            contract,
            quantiles_matrix.shape,
            save_file,
        )

    del vae_models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("All precomputed VAE quantiles ready in %s", target_dir)
    return target_dir


def resolve_routing_parameters(args):
    """Resolve dual-axis routing parameters and gating strategy from CLI arguments, para_file, or optuna_csv."""
    slope_window_length = args.slope_window_length
    volatility_window_length = args.volatility_window_length
    slope_gamma = args.slope_gamma
    volatility_gamma = args.volatility_gamma
    slope_rule_base_threshold = args.slope_rule_base_threshold
    volatility_rule_base_threshold = args.volatility_rule_base_threshold

    gating_strategy = args.gating_strategy
    ood_threshold = args.ood_threshold
    slope_margin_threshold = args.slope_margin_threshold
    volatility_margin_threshold = args.volatility_margin_threshold

    para_file = args.para_file
    optuna_csv = args.optuna_csv

    para_str = ""
    if para_file and os.path.exists(para_file):
        with open(para_file, encoding="utf-8") as f:
            para_str = f.readline().strip()

    # 1. Try resolving via trial ID in optuna_csv
    trial_match = re.search(r"trial_(\d+)", para_str)
    if trial_match and optuna_csv and os.path.exists(optuna_csv):
        trial_id = int(trial_match.group(1))
        df = pd.read_csv(optuna_csv)
        row = df[df[RoutingParamColumns.NUMBER] == trial_id]
        if not row.empty:
            r = row.iloc[0]
            if slope_window_length is None:
                slope_window_length = int(r[RoutingParamColumns.PARAMS_SLOPE_WINDOW_LENGTH])
            if volatility_window_length is None:
                volatility_window_length = int(r[RoutingParamColumns.PARAMS_VOLATILITY_WINDOW_LENGTH])
            if slope_gamma is None:
                slope_gamma = float(r[RoutingParamColumns.PARAMS_SLOPE_GAMMA])
            if volatility_gamma is None:
                volatility_gamma = float(r[RoutingParamColumns.PARAMS_VOLATILITY_GAMMA])
            if slope_rule_base_threshold is None:
                slope_rule_base_threshold = float(r[RoutingParamColumns.PARAMS_SLOPE_RULE_BASE_THRESHOLD])
            if volatility_rule_base_threshold is None and RoutingParamColumns.PARAMS_VOLATILITY_RULE_BASE_THRESHOLD in df.columns:
                volatility_rule_base_threshold = float(r[RoutingParamColumns.PARAMS_VOLATILITY_RULE_BASE_THRESHOLD])
            if RoutingParamColumns.PARAMS_GATING_STRATEGY in df.columns:
                gating_strategy = str(r[RoutingParamColumns.PARAMS_GATING_STRATEGY])
            if RoutingParamColumns.PARAMS_OOD_THRESHOLD in df.columns:
                ood_threshold = float(r[RoutingParamColumns.PARAMS_OOD_THRESHOLD])
                gating_strategy = "hierarchical"
            if RoutingParamColumns.PARAMS_SLOPE_MARGIN_THRESHOLD in df.columns:
                slope_margin_threshold = float(r[RoutingParamColumns.PARAMS_SLOPE_MARGIN_THRESHOLD])
            if RoutingParamColumns.PARAMS_VOLATILITY_MARGIN_THRESHOLD in df.columns:
                volatility_margin_threshold = float(r[RoutingParamColumns.PARAMS_VOLATILITY_MARGIN_THRESHOLD])

    # 2. Try resolving via explicit ws_ / wv_ format in para_str
    if para_str:
        ws_match = re.search(r"ws_(\d+)", para_str)
        wv_match = re.search(r"wv_(\d+)", para_str)
        gs_match = re.search(r"gs_([0-9.]+)", para_str)
        gv_match = re.search(r"gv_([0-9.]+)", para_str)
        ts_match = re.search(r"ts_([0-9.]+)", para_str)
        tv_match = re.search(r"tv_([0-9.]+)", para_str)

        if ws_match and slope_window_length is None:
            slope_window_length = int(ws_match.group(1))
        if wv_match and volatility_window_length is None:
            volatility_window_length = int(wv_match.group(1))
        if gs_match and slope_gamma is None:
            slope_gamma = float(gs_match.group(1))
        if gv_match and volatility_gamma is None:
            volatility_gamma = float(gv_match.group(1))
        if ts_match and slope_rule_base_threshold is None:
            slope_rule_base_threshold = float(ts_match.group(1))
        if tv_match and volatility_rule_base_threshold is None:
            volatility_rule_base_threshold = float(tv_match.group(1))

        if "strat_hierarchical" in para_str or "ood_" in para_str:
            gating_strategy = "hierarchical"
        elif "strat_absolute" in para_str:
            gating_strategy = "absolute"

        ood_match = re.search(r"ood_([0-9.]+)", para_str)
        ms_match = re.search(r"ms_([0-9.]+)", para_str)
        mv_match = re.search(r"mv_([0-9.]+)", para_str)
        if ood_match:
            ood_threshold = float(ood_match.group(1))
        if ms_match:
            slope_margin_threshold = float(ms_match.group(1))
        if mv_match:
            volatility_margin_threshold = float(mv_match.group(1))

    # 3. Fallback to single gamma_ / window_ / threshold_ format if present in para_str
    if para_str:
        gamma_m = re.search(r"gamma_([0-9.]+)", para_str)
        window_m = re.search(r"window_([0-9]+)", para_str)
        thresh_m = re.search(r"threshold_([0-9.]+)", para_str)
        if slope_gamma is None and gamma_m:
            slope_gamma = float(gamma_m.group(1))
        if volatility_gamma is None and gamma_m:
            volatility_gamma = float(gamma_m.group(1))
        if slope_window_length is None and window_m:
            slope_window_length = int(window_m.group(1))
        if volatility_window_length is None and window_m:
            volatility_window_length = int(window_m.group(1))
        if slope_rule_base_threshold is None and thresh_m:
            slope_rule_base_threshold = float(thresh_m.group(1))
        if volatility_rule_base_threshold is None and thresh_m:
            volatility_rule_base_threshold = float(thresh_m.group(1))

    # 4. Final fallbacks to general args.window_length / args.gamma / args.rule_base_threshold
    base_window = args.window_length
    base_gamma = args.gamma
    base_threshold = args.rule_base_threshold

    args.slope_window_length = slope_window_length if slope_window_length is not None else base_window
    args.volatility_window_length = volatility_window_length if volatility_window_length is not None else base_window
    args.slope_gamma = slope_gamma if slope_gamma is not None else base_gamma
    args.volatility_gamma = volatility_gamma if volatility_gamma is not None else base_gamma
    args.gating_strategy = gating_strategy
    args.ood_threshold = ood_threshold
    args.slope_margin_threshold = slope_margin_threshold
    args.volatility_margin_threshold = volatility_margin_threshold

    args.slope_rule_base_threshold = slope_rule_base_threshold if slope_rule_base_threshold is not None else base_threshold
    args.volatility_rule_base_threshold = volatility_rule_base_threshold if volatility_rule_base_threshold is not None else base_threshold

    args.window_length = max(args.slope_window_length, args.volatility_window_length)
    args.gamma = args.slope_gamma
    args.rule_base_threshold = min(args.slope_rule_base_threshold, args.volatility_rule_base_threshold)

    return args


def load_two_dimensional_selection_manifest(
    manifest_path: str | os.PathLike[str],
) -> TwoDimensionalSelectionManifest:
    """Load and validate the logical slope/volatility slot contract."""

    with open(manifest_path, encoding="utf-8") as file:
        data = json.load(file)

    manifest = TwoDimensionalSelectionManifest.from_dict(data)

    axes = manifest.axes
    volatility_labels = axes.volatility
    slope_labels = axes.slope
    if not volatility_labels or len(volatility_labels) != len(slope_labels):
        raise ValueError("volatility and slope axes must have the same non-zero size")

    num_labels = len(volatility_labels)
    expected_slot_count = num_labels * num_labels
    if manifest.slot_count != expected_slot_count:
        raise ValueError(
            "manifest slot_count does not match the two-dimensional axes: "
            f"expected {expected_slot_count}, got {manifest.slot_count}"
        )
    if manifest.slot_index_formula != (
        "volatility_index * num_labels + slope_index"
    ):
        raise ValueError("unsupported two-dimensional slot index formula")

    slots = manifest.slots
    if len(slots) != expected_slot_count:
        raise ValueError("manifest slots must contain every logical slot")
    slot_ids = [slot["slot_id"] for slot in slots]
    if sorted(slot_ids) != list(range(expected_slot_count)):
        raise ValueError("manifest slot_id values must be contiguous and start at zero")
    if any(slot["kind"] not in {"model", "empty_model"} for slot in slots):
        raise ValueError("manifest slot kind must be model or empty_model")
    manifest.slots = sorted(slots, key=lambda slot: slot["slot_id"])

    return manifest


class vae_risk_aware_routing:
    action_persistence: int = 3
    remaining_persist: int = 0
    current_action: int = 0
    flat_action: int = 0
    action_decision_reason_history: list[int]
    enable_non_main_contract_defense: bool = False
    role_tier_index: int | None = None
    save_artifacts: bool = True
    precomputed_quantiles_dir: str | None = None
    current_contract_quantiles: np.ndarray | None = None
    stop_loss_abs_threshold: float = 50.0
    stop_loss_cooldown_steps: int = 12
    circuit_breaker_consecutive_stops: int = 2
    circuit_breaker_cooling_steps: int = 72
    cooldown_remaining_steps: int = 0
    last_stopped_position: float = 0.0
    consecutive_stop_loss_count: int = 0
    circuit_breaker_remaining_steps: int = 0
    hard_stop_loss_count: int = 0
    cooldown_intercept_count: int = 0
    circuit_breaker_suspension_count: int = 0
    previous_step_position: float = 0.0
    active_trade_stopped: bool = False

    def __init__(self, args) -> None:
        # device
        if torch.cuda.is_available():
            self.device = "cuda:{}".format(args.gpu_index)
        else:
            self.device = "cpu"
        self.gamma = args.gamma
        self.rule_base_threshold = args.rule_base_threshold
        self.window_length = args.window_length
        manifest_path = args.selection_manifest
        if not manifest_path:
            raise ValueError("selection_manifest is required")
        self.selection_manifest: TwoDimensionalSelectionManifest = (
            load_two_dimensional_selection_manifest(manifest_path)
        )
        self.num_labels = len(self.selection_manifest.axes.volatility)
        self.slot_count = self.selection_manifest.slot_count
        self.axis_window_lengths = {
            "slope": args.slope_window_length,
            "volatility": args.volatility_window_length,
        }
        self.axis_gammas = {
            "slope": args.slope_gamma,
            "volatility": args.volatility_gamma,
        }
        self.axis_thresholds = {
            "slope": args.slope_rule_base_threshold,
            "volatility": args.volatility_rule_base_threshold,
        }
        self.stop_loss_abs_threshold = float(args.stop_loss_abs_threshold)
        if self.stop_loss_abs_threshold < 0:
            raise ValueError("stop_loss_abs_threshold must be non-negative")
        self.stop_loss_cooldown_steps = int(args.stop_loss_cooldown_steps)
        if self.stop_loss_cooldown_steps < 0:
            raise ValueError("stop_loss_cooldown_steps must be non-negative")
        self.circuit_breaker_consecutive_stops = int(args.circuit_breaker_consecutive_stops)
        if self.circuit_breaker_consecutive_stops < 0:
            raise ValueError("circuit_breaker_consecutive_stops must be non-negative")
        self.circuit_breaker_cooling_steps = int(args.circuit_breaker_cooling_steps)
        if self.circuit_breaker_cooling_steps < -1:
            raise ValueError("circuit_breaker_cooling_steps must be -1 or non-negative")
        self.gating_strategy = create_gating_strategy(
            args.gating_strategy,
            slope_threshold=self.axis_thresholds["slope"],
            volatility_threshold=self.axis_thresholds["volatility"],
            ood_threshold=args.ood_threshold,
            slope_margin_threshold=args.slope_margin_threshold,
            volatility_margin_threshold=args.volatility_margin_threshold,
        )
        self.initial_rollout_window_length = max(self.axis_window_lengths.values())
        self.experiment_name = args.experiment_name
        self.eval_stage = args.eval_stage
        self.save_artifacts = args.save_artifacts
        self.precomputed_quantiles_dir = self._resolve_precomputed_quantiles_dir(args)
        self.current_contract_quantiles = None

        self.test_path = self._resolve_test_path(args)
        if self.save_artifacts and not os.path.exists(self.test_path):
            os.makedirs(self.test_path, exist_ok=True)

        # trading environment setting
        self.base_path = args.base_path
        self.dataset_name = args.dataset_name
        self.allow_reverse_position = args.allow_reverse_position
        self.eval_stage_dir = os.path.join(self.base_path, self.dataset_name, self.eval_stage)
        self.single_data_path = os.path.join(
            self.base_path, self.dataset_name, f"{self.eval_stage}.feather"
        )
        self.test_data_path = self.single_data_path
        self.tech_indicator_list = np.load(
            os.path.join(self.base_path, self.dataset_name, ArtifactNames.RL_STATE_FEATURES_NPY)
        )
        self.vae_slope_indicators = np.load(
            os.path.join(self.base_path, self.dataset_name, ArtifactNames.VAE_SLOPE_STATE_FEATURES_NPY)
        )
        self.vae_volatility_indicators = np.load(
            os.path.join(self.base_path, self.dataset_name, ArtifactNames.VAE_VOLATILITY_STATE_FEATURES_NPY)
        )
        self.vae_vol_indicators = self.vae_volatility_indicators
        self.enable_non_main_contract_defense = args.enable_non_main_contract_defense
        self.stop_loss_abs_threshold = float(args.stop_loss_abs_threshold)
        if self.stop_loss_abs_threshold < 0:
            raise ValueError("stop_loss_abs_threshold must be non-negative")
        self.stop_loss_cooldown_steps = int(args.stop_loss_cooldown_steps)
        if self.stop_loss_cooldown_steps < 0:
            raise ValueError("stop_loss_cooldown_steps must be non-negative")
        self.circuit_breaker_consecutive_stops = int(args.circuit_breaker_consecutive_stops)
        if self.circuit_breaker_consecutive_stops < 0:
            raise ValueError("circuit_breaker_consecutive_stops must be non-negative")
        self.circuit_breaker_cooling_steps = int(args.circuit_breaker_cooling_steps)
        if self.circuit_breaker_cooling_steps < -1:
            raise ValueError("circuit_breaker_cooling_steps must be -1 or non-negative")
        role_tier_indices = np.where(
            self.tech_indicator_list == "prev_day_contract_role_tier"
        )[0]
        self.role_tier_index = (
            int(role_tier_indices[0]) if len(role_tier_indices) > 0 else None
        )
        self.maintenance_margin_ratio_dict = np.load(
            os.path.join(
                self.base_path, self.dataset_name, ArtifactNames.MAINTENANCE_MARGIN_RATIO_DICT_NPY
            ),
            allow_pickle=True,
        ).item()
        self.max_holding_number = args.max_holding_number
        self.position_choices = args.position_choices
        self.single_side_action_num = int((self.position_choices - 1) / 2)
        self.position_list = (
            [
                self.max_holding_number / self.single_side_action_num * i
                for i in range(1, self.single_side_action_num + 1)
            ]
            + [0]
            + [
                self.max_holding_number / self.single_side_action_num * -i
                for i in range(1, self.single_side_action_num + 1)
            ]
        )
        self.position_list.sort()
        self.leverage_choices = args.leverage_choices
        self.long_estimated_rate = args.long_estimated_rate
        self.short_estimated_rate = args.short_estimated_rate
        self.transcation_cost = args.transcation_cost
        self.early_stop = args.early_stop
        self.initial_wallet_balance = args.initial_wallet_balance
        self.initial_margin = args.initial_margin
        self.initial_unrealized_pnL = args.initial_unrealized_pnL
        self.initial_position = args.initial_position
        self.initial_leverage = args.initial_leverage
        self.order_book_depth = args.order_book_depth
        self.initial_state = (
            self.initial_wallet_balance,
            self.initial_margin,
            self.initial_unrealized_pnL,
            self.initial_position,
            self.initial_leverage,
        )
        self.initial_action = map_position_leverage_to_action(
            self.initial_position,
            self.initial_leverage,
            self.leverage_choices,
            self.position_list,
        )
        self.zero_position_action = len(self.leverage_choices) * (
            len(self.position_list) // 2
        )
        self.flat_action = self.zero_position_action
        self.action_persistence = int(args.action_persistence)
        if self.action_persistence <= 0:
            raise ValueError("action_persistence must be positive")
        self.remaining_persist = 0
        self.current_action = self.flat_action
        self.action_decision_reason_history = []
        self.cooldown_remaining_steps = 0
        self.last_stopped_position = 0.0
        self.consecutive_stop_loss_count = 0
        self.circuit_breaker_remaining_steps = 0
        self.hard_stop_loss_count = 0
        self.cooldown_intercept_count = 0
        self.circuit_breaker_suspension_count = 0
        self.previous_step_position = 0.0
        self.active_trade_stopped = False

        # low-level network
        self.time_info_dim = args.time_info_dim
        self.hidden_nodes = args.hidden_nodes
        self.N = self.slot_count
        self.N_ACTIONS = (self.position_choices - 1) * len(self.leverage_choices) + 1
        self.low_level_network = ensemble_Qnet(
            N_STATES=len(self.tech_indicator_list),
            N_ACTIONS=self.N_ACTIONS,
            hidden_nodes=self.hidden_nodes,
            TIME_INFO_DIM=self.time_info_dim,
            ensemble_number=self.N,
        ).to(self.device)
        low_level_model_path = self.selection_manifest.artifacts.model_assembly
        if not low_level_model_path:
            raise ValueError("manifest has no model_assembly artifact")
        self.low_level_network.load_state_dict(
            torch.load(low_level_model_path, map_location=torch.device(self.device))
        )
        self.low_level_network.to(self.device)
        self.low_level_network.eval()
        disable_gradients(self.low_level_network)
        # loss deque

        # label vae
        # VAE network path
        axis_indicators = {
            "slope": self.vae_slope_indicators,
            "volatility": self.vae_volatility_indicators,
        }

        def load_vae_axis(axis, root):
            indicators = axis_indicators[axis]
            label_list = [f"label_{i}" for i in range(self.num_labels)]
            model_list = []
            logpx_list = []
            for label in label_list:
                path = os.path.join(root, label, ArtifactNames.MODEL_LATEST_PTH)
                id_path = os.path.join(root, label, ArtifactNames.ID_LOGPX_NPY)
                vae_model = MLP_VAE(
                    INPUT_DIM=len(indicators),
                    Z_DIM=args.z_dim,
                    hidden_dims=args.vae_hidden_dims,
                    loss_func=args.loss_type,
                ).to(self.device)
                vae_model.load_state_dict(
                    torch.load(path, map_location=torch.device(self.device))
                )
                model_list.append(vae_model)
                logpx_list.append(np.load(id_path).reshape(-1))
            return model_list, logpx_list

        default_vae_root = os.path.join(
            args.vae_path,
            self.dataset_name,
            self.experiment_name,
        )
        vae_roots = {
            "slope": os.path.join(default_vae_root, "slope"),
            "volatility": os.path.join(default_vae_root, "volatility"),
        }
        self.quantiles = {}
        for axis in ("slope", "volatility"):
            self.quantiles[axis] = [
                deque(maxlen=self.axis_window_lengths[axis])
                for _ in range(self.num_labels)
            ]
        if self.precomputed_quantiles_dir is not None:
            self.vae_models = None
            self.in_ds_logpx = None
            logger.info(
                "Using precomputed VAE quantiles from %s; skipping VAE models loading.",
                self.precomputed_quantiles_dir,
            )
        else:
            self.vae_models = {}
            self.in_ds_logpx = {}
            for axis, root in vae_roots.items():
                self.vae_models[axis], logpx_list = load_vae_axis(axis, root)
                # pre-sort once here so find_quantile only needs searchsorted per step
                self.in_ds_logpx[axis] = [np.sort(logpx) for logpx in logpx_list]
        self.action = self.zero_position_action
        self.macro_action_history = []

    def reset_routing_state(self):
        self.quantiles = {
            axis: [
                deque(maxlen=self.axis_window_lengths[axis])
                for _ in range(self.num_labels)
            ]
            for axis in ("slope", "volatility")
        }
        self.action = self.zero_position_action
        self.macro_action_history = []
        self.flat_action = self.zero_position_action
        self.remaining_persist = 0
        self.current_action = self.flat_action
        self.action_decision_reason_history = []
        self.cooldown_remaining_steps = 0
        self.last_stopped_position = 0.0
        self.consecutive_stop_loss_count = 0
        self.circuit_breaker_remaining_steps = 0
        self.hard_stop_loss_count = 0
        self.cooldown_intercept_count = 0
        self.circuit_breaker_suspension_count = 0
        self.previous_step_position = 0.0
        self.active_trade_stopped = False

    def _resolve_precomputed_quantiles_dir(self, args) -> str | None:
        target_dir = args.precomputed_quantiles_dir
        if target_dir:
            if os.path.isdir(target_dir):
                return target_dir
            return None
        candidate = default_precomputed_quantiles_dir(
            dataset_name=self.dataset_name,
            experiment_name=self.experiment_name,
            eval_stage=self.eval_stage,
        )
        if os.path.isdir(candidate):
            files = [f for f in os.listdir(candidate) if f.endswith(".npy")]
            if files:
                return candidate
        return None

    def _resolve_test_path(self, args):
        if self.eval_stage == "test" or args.result_path.endswith("final_result"):
            return os.path.join(
                args.result_path,
                args.dataset_name,
                self.experiment_name,
            )
        self.model_path = os.path.join(
            args.result_path,
            args.dataset_name,
            self.experiment_name,
            "vae_risk_aware_routing",
        )
        trial_suffix = "" if args.trial_number is None else f"_trial_{args.trial_number}"
        if args.gating_strategy == "hierarchical":
            return os.path.join(
                self.model_path,
                f"strat_hierarchical_gamma_{self.gamma}_window_{self.window_length}_ood_{args.ood_threshold}_ms_{args.slope_margin_threshold}_mv_{args.volatility_margin_threshold}"
                + trial_suffix,
            )
        return os.path.join(
            self.model_path,
            "gamma_{}_window_{}_threshold_{}".format(
                self.gamma, self.window_length, self.rule_base_threshold
            ) + trial_suffix,
        )

    def reconfigure_routing(self, args):
        """Reset trial-dependent routing state while keeping loaded models.

        Avoids reloading the low-level network and VAE models between Optuna
        trials; only windows/gammas/thresholds, the output path and the
        routing deques change from trial to trial.
        """
        self.action_persistence = int(args.action_persistence)
        if self.action_persistence <= 0:
            raise ValueError("action_persistence must be positive")
        self.gamma = args.gamma
        self.rule_base_threshold = args.rule_base_threshold
        self.window_length = args.window_length
        self.axis_window_lengths = {
            "slope": args.slope_window_length,
            "volatility": args.volatility_window_length,
        }
        self.axis_gammas = {
            "slope": args.slope_gamma,
            "volatility": args.volatility_gamma,
        }
        self.axis_thresholds = {
            "slope": args.slope_rule_base_threshold,
            "volatility": args.volatility_rule_base_threshold,
        }
        self.gating_strategy = create_gating_strategy(
            args.gating_strategy,
            slope_threshold=self.axis_thresholds["slope"],
            volatility_threshold=self.axis_thresholds["volatility"],
            ood_threshold=args.ood_threshold,
            slope_margin_threshold=args.slope_margin_threshold,
            volatility_margin_threshold=args.volatility_margin_threshold,
        )
        self.initial_rollout_window_length = max(self.axis_window_lengths.values())
        self.enable_non_main_contract_defense = args.enable_non_main_contract_defense
        self.stop_loss_abs_threshold = float(args.stop_loss_abs_threshold)
        if self.stop_loss_abs_threshold < 0:
            raise ValueError("stop_loss_abs_threshold must be non-negative")
        self.stop_loss_cooldown_steps = int(args.stop_loss_cooldown_steps)
        if self.stop_loss_cooldown_steps < 0:
            raise ValueError("stop_loss_cooldown_steps must be non-negative")
        self.circuit_breaker_consecutive_stops = int(args.circuit_breaker_consecutive_stops)
        if self.circuit_breaker_consecutive_stops < 0:
            raise ValueError("circuit_breaker_consecutive_stops must be non-negative")
        self.circuit_breaker_cooling_steps = int(args.circuit_breaker_cooling_steps)
        if self.circuit_breaker_cooling_steps < -1:
            raise ValueError("circuit_breaker_cooling_steps must be -1 or non-negative")
        self.test_path = self._resolve_test_path(args)
        if self.save_artifacts and not os.path.exists(self.test_path):
            os.makedirs(self.test_path, exist_ok=True)
        self.reset_routing_state()

    def find_contract_files(self):
        stage = self.eval_stage
        stage_dir = self.eval_stage_dir
        if os.path.isdir(stage_dir):
            contract_files = []
            for filename in sorted(os.listdir(stage_dir)):
                path = os.path.join(stage_dir, filename)
                if os.path.isfile(path) and filename.endswith(".feather"):
                    contract_files.append((os.path.splitext(filename)[0], path))
            if contract_files:
                return contract_files
        if os.path.isfile(self.single_data_path):
            return [(stage, self.single_data_path)]
        return []

    def find_quantile(self, value, sorted_array):
        # sorted_array must be pre-sorted (see __init__: in_ds_logpx is sorted once)
        if value < sorted_array[0]:
            quantile = 0.0  # Value is below the minimum
        elif value > sorted_array[-1]:
            quantile = 1.0
        else:
            quantile = np.searchsorted(sorted_array, value, side="right") / len(
                sorted_array
            )
        return quantile

    def get_quantiles(self, vae_s_slope, vae_s_vol):
        if self.current_contract_quantiles is not None:
            idx = min(self.step_idx, len(self.current_contract_quantiles) - 1)
            row = self.current_contract_quantiles[idx]
            for label_idx in range(self.num_labels):
                self.quantiles["slope"][label_idx].append(float(row[label_idx]))
                self.quantiles["volatility"][label_idx].append(
                    float(row[self.num_labels + label_idx])
                )
            return self.quantiles

        axis_inputs = {
            "slope": vae_s_slope,
            "volatility": vae_s_vol,
        }
        for axis in ("slope", "volatility"):
            s_axis = axis_inputs[axis]
            loss_list = [
                analyze_single_sample(vae_model, s_axis, self.device)[1]
                for vae_model in self.vae_models[axis]
            ]
            for quantile_deque, loss, base_array in zip(
                self.quantiles[axis],
                loss_list,
                self.in_ds_logpx[axis],
            ):
                quantile_deque.append(self.find_quantile(loss, base_array))
        return self.quantiles

    def calculate_rolling_window(
        self, quantile_deque: deque, gamma=None, window_length=None
    ):
        gamma = self.gamma if gamma is None else gamma
        window_length = self.window_length if window_length is None else window_length
        if not quantile_deque:
            return 0.0
        weights = gamma ** np.arange(window_length)[::-1]
        values = np.asarray(quantile_deque, dtype=float)
        weights = weights[-len(values) :]
        weighted_sum = np.sum(values * weights)
        sum_of_weights = np.sum(weights)
        decay_average = weighted_sum / sum_of_weights
        return decay_average

    def calculate_axis_window_result(self, axis):
        return [
            self.calculate_rolling_window(
                quantile_deque,
                gamma=self.axis_gammas[axis],
                window_length=self.axis_window_lengths[axis],
            )
            for quantile_deque in self.quantiles[axis]
        ]

    def _defensive_action(self, info, current_position, current_leverage):
        return rule_based_close(
            info,
            self.zero_position_action,
            self.leverage_choices,
            self.position_list,
            current_position,
            current_leverage,
        )

    def _apply_defensive_action(
        self, info, current_position, current_leverage
    ) -> int:
        if self.cooldown_remaining_steps > 0:
            self.cooldown_remaining_steps -= 1
        reason = (
            ActionDecisionReasons.DEFENSIVE_PREEMPTION
            if self.remaining_persist > 0
            else ActionDecisionReasons.DEFENSIVE_RULE_CLOSE
        )
        self.remaining_persist = 0
        action = self._defensive_action(info, current_position, current_leverage)
        self.current_action = action
        self.macro_action_history.append(self.slot_count)
        self.action_decision_reason_history.append(reason)
        self.action = action
        self.previous_step_position = float(current_position)
        return action

    def _arm_persistence(self, action: int) -> None:
        if self.action_persistence > 1 and action != self.flat_action:
            self.remaining_persist = self.action_persistence - 1
        else:
            self.remaining_persist = 0

    def get_action(
        self,
        info,
        s,
        current_position,
        current_leverage,
        current_unrealized_pnl: float = 0.0,
    ) -> int:
        current_pos_float = float(current_position)

        # 0. Trade lifecycle tracking: detect trade exit and reset consecutive stop-out streak
        if self.previous_step_position == 0.0 and current_pos_float != 0.0:
            self.active_trade_stopped = False
        elif (
            self.previous_step_position != 0.0
            and (current_pos_float == 0.0 or self.previous_step_position * current_pos_float < 0)
        ):
            if not self.active_trade_stopped:
                self.consecutive_stop_loss_count = 0
            self.active_trade_stopped = False

        # 1. Tier 3: Contract-Level Circuit Breaker Suspension
        if self.circuit_breaker_remaining_steps != 0:
            if self.circuit_breaker_remaining_steps > 0:
                self.circuit_breaker_remaining_steps -= 1
            if self.cooldown_remaining_steps > 0:
                self.cooldown_remaining_steps -= 1
            self.remaining_persist = 0
            action = self._defensive_action(info, current_pos_float, current_leverage)
            self.current_action = action
            self.macro_action_history.append(self.slot_count)
            self.action_decision_reason_history.append(
                ActionDecisionReasons.CIRCUIT_BREAKER_SUSPENSION
            )
            self.circuit_breaker_suspension_count += 1
            self.action = action
            self.previous_step_position = current_pos_float
            return action

        # 2. Tier 1: Unrealized PnL Hard Stop-Loss
        if (
            self.stop_loss_abs_threshold > 0.0
            and current_pos_float != 0.0
            and current_unrealized_pnl <= -self.stop_loss_abs_threshold
        ):
            self.active_trade_stopped = True
            self.last_stopped_position = current_pos_float
            self.hard_stop_loss_count += 1
            self.consecutive_stop_loss_count += 1
            self.remaining_persist = 0

            if (
                self.circuit_breaker_consecutive_stops > 0
                and self.consecutive_stop_loss_count >= self.circuit_breaker_consecutive_stops
            ):
                self.circuit_breaker_remaining_steps = self.circuit_breaker_cooling_steps
                self.cooldown_remaining_steps = 0
            else:
                self.cooldown_remaining_steps = self.stop_loss_cooldown_steps

            action = self._defensive_action(info, current_pos_float, current_leverage)
            self.current_action = action
            self.macro_action_history.append(self.slot_count)
            self.action_decision_reason_history.append(ActionDecisionReasons.HARD_STOP_LOSS)
            self.action = action
            self.previous_step_position = current_pos_float
            return action

        # 3. Existing high-level defenses
        if (
            self.enable_non_main_contract_defense
            and self.role_tier_index is not None
            and float(s[self.role_tier_index]) < 0.5
        ):
            return self._apply_defensive_action(
                info, current_pos_float, current_leverage
            )

        volatility_weights = self.calculate_axis_window_result("volatility")
        slope_weights = self.calculate_axis_window_result("slope")
        decision = self.gating_strategy.decide(volatility_weights, slope_weights)
        if decision.is_defensive:
            return self._apply_defensive_action(
                info, current_pos_float, current_leverage
            )

        volatility_index = decision.volatility_index
        slope_index = decision.slope_index
        slot_id = volatility_index * self.num_labels + slope_index
        slot = self.selection_manifest.slots[slot_id]
        if slot["kind"] == "empty_model":
            return self._apply_defensive_action(
                info, current_pos_float, current_leverage
            )

        self.selected_agent_index = slot_id
        self.macro_action_history.append(slot_id)

        # 4. Candidate Action Query
        if self.remaining_persist > 0 and not bool(
            info["avaliable_action"][self.current_action]
        ):
            self.remaining_persist = 0
            candidate_action = self.agent_act(s, info)
            candidate_reason = ActionDecisionReasons.ACTION_UNAVAILABLE_BREAK
        elif self.remaining_persist > 0:
            candidate_action = self.current_action
            candidate_reason = ActionDecisionReasons.ACTION_PERSISTENCE
        else:
            candidate_action = self.agent_act(s, info)
            candidate_reason = ActionDecisionReasons.POLICY_INFERENCE

        # 5. Tier 2: Directional Cooldown Lockout Interception
        if self.cooldown_remaining_steps > 0:
            self.cooldown_remaining_steps -= 1
            target_pos, _ = map_action_to_position_leverage(
                candidate_action, self.leverage_choices, self.position_list
            )
            if target_pos * self.last_stopped_position > 0:
                self.remaining_persist = 0
                action = self._defensive_action(info, current_pos_float, current_leverage)
                self.current_action = action
                self.action_decision_reason_history.append(
                    ActionDecisionReasons.STOP_LOSS_COOLDOWN
                )
                self.cooldown_intercept_count += 1
                self.action = action
                self.previous_step_position = current_pos_float
                return action

        if candidate_reason == ActionDecisionReasons.ACTION_PERSISTENCE:
            self.remaining_persist -= 1
        else:
            self._arm_persistence(candidate_action)

        self.current_action = candidate_action
        self.action_decision_reason_history.append(candidate_reason)
        self.action = candidate_action
        self.previous_step_position = current_pos_float
        return candidate_action

    def agent_act(self, state, info):
        # low level agent
        state = torch.unsqueeze(torch.FloatTensor(state).reshape(-1), 0).to(self.device)
        previous_action = torch.unsqueeze(
            torch.tensor([info["previous_action"]]).float().to(self.device), 0
        ).to(self.device)
        avaliable_action = torch.unsqueeze(
            torch.tensor(info["avaliable_action"]).to(self.device), 0
        ).to(self.device)
        hour_count_down = (
            torch.unsqueeze(torch.tensor([info["funding_count_down_hour"]]), 0)
            .to(self.device)
            .float()
        )
        minute_count_down = (
            torch.unsqueeze(torch.tensor([info["funding_count_down_minute"]]), 0)
            .to(self.device)
            .float()
        )
        time_input = torch.cat([hour_count_down, minute_count_down], dim=1).to(
            self.device
        )
        trading_info = torch.unsqueeze(torch.tensor(info["trading_info"]).float().to(self.device), 0)
        actions_value = self.low_level_network(
            state=state,
            time=time_input,
            previous_action=previous_action,
            avaliable_action=avaliable_action,
            trading_info=trading_info,
        )
        action_value_chosen_index = actions_value[:, self.selected_agent_index, :]
        action = torch.max(action_value_chosen_index, 1)[1].data.cpu().numpy()
        action = action[0]

        return action

    def run_single_valid_df(self, df, save_path, contract_name: str | None = None):
        self.df = df
        self.vae_slope_array = self.df[self.vae_slope_indicators].values
        self.vae_vol_array = self.df[self.vae_volatility_indicators].values
        if self.precomputed_quantiles_dir is not None and contract_name is not None:
            quant_path = os.path.join(
                self.precomputed_quantiles_dir, f"{contract_name}.npy"
            )
            if os.path.isfile(quant_path):
                self.current_contract_quantiles = np.load(quant_path, mmap_mode="r")
            else:
                self.current_contract_quantiles = None
        else:
            self.current_contract_quantiles = None
        env = initiate_base_env(
            df=self.df,
            feature_list=self.tech_indicator_list,
            max_holding_number=self.max_holding_number,
            position_choices=self.position_choices,  # (must be an odd number, the minum of trading equals to (max_holder_number)/((action_dim-1)/2)s))
            leverage_choice=self.leverage_choices,  # recommend only use one leverage choice, because the leverage does not influence the return directly, the position
            # itself is enough to show the risk preference
            long_estimated_rate=self.long_estimated_rate,
            short_estimated_rate=self.short_estimated_rate,
            commission_rate=self.transcation_cost,
            # maten_mar_ratio_dict varies among different perpertual contracts, need to perform a config file for different perpertual
            # the default is for btcusdt perpetual contract
            maintenance_margin_ratio_dict=self.maintenance_margin_ratio_dict,
            early_stop=self.early_stop,
            # initial_personal_state
            initial_state=self.initial_state,
            order_book_depth=self.order_book_depth,
            allow_reverse_position=self.allow_reverse_position,
        )
        logger.info(
            "Environment initialized. Resetting environment with %d rows of data...",
            len(self.df),
        )
        s, info = env.reset()
        logger.info(
            "Environment reset complete. Initial wallet balance: %.2f, initial state: %s",
            self.initial_wallet_balance,
            self.initial_state,
        )
        episode_reward_sum = 0
        self.step_idx = 0
        env, s, r, done, info = self.initial_rollout(env, s, info)
        while not done:
            action = self.get_action(
                info,
                s,
                env.position,
                env.leverage,
                current_unrealized_pnl=float(env.unrealized_pnl),
            )
            s_, r, done, info = env.step(action)
            self.step_idx += 1
            vae_idx = min(self.step_idx, len(self.vae_slope_array) - 1)
            vae_s_slope = self.vae_slope_array[vae_idx]
            vae_s_vol = self.vae_vol_array[vae_idx]
            self.get_quantiles(vae_s_slope, vae_s_vol)
            episode_reward_sum += r
            if done:
                break
            s = s_
        total_asset_history = env.margine_balance_history
        reward_history = calculate_differences(total_asset_history)
        micro_action_history = env.micro_action_history
        reasons = np.array(self.action_decision_reason_history, dtype=np.int32)
        total_steps = len(reasons)
        persistence_held_steps = int(np.sum(reasons == ActionDecisionReasons.ACTION_PERSISTENCE))
        policy_inference_steps = int(
            np.sum(
                (reasons == ActionDecisionReasons.POLICY_INFERENCE)
                | (reasons == ActionDecisionReasons.ACTION_UNAVAILABLE_BREAK)
            )
        )
        defensive_preemptions = int(np.sum(reasons == ActionDecisionReasons.DEFENSIVE_PREEMPTION))
        action_unavailable_breaks = int(np.sum(reasons == ActionDecisionReasons.ACTION_UNAVAILABLE_BREAK))
        skip_inference_ratio = (
            float(persistence_held_steps / total_steps) if total_steps > 0 else 0.0
        )
        hard_stop_loss_steps = int(np.sum(reasons == ActionDecisionReasons.HARD_STOP_LOSS))
        stop_loss_cooldown_steps = int(np.sum(reasons == ActionDecisionReasons.STOP_LOSS_COOLDOWN))
        circuit_breaker_suspension_steps = int(np.sum(reasons == ActionDecisionReasons.CIRCUIT_BREAKER_SUSPENSION))
        trading_info = {
            "return rate": total_asset_history[-1] / self.initial_wallet_balance,
            "total_steps": total_steps,
            "inference_steps": policy_inference_steps,
            "persistence_held_steps": persistence_held_steps,
            "skip_inference_ratio": skip_inference_ratio,
            "defensive_preemptions": defensive_preemptions,
            "action_unavailable_breaks": action_unavailable_breaks,
            "hard_stop_loss_steps": hard_stop_loss_steps,
            "stop_loss_cooldown_steps": stop_loss_cooldown_steps,
            "circuit_breaker_suspension_steps": circuit_breaker_suspension_steps,
            "hard_stop_loss_count": self.hard_stop_loss_count,
            "cooldown_intercept_count": self.cooldown_intercept_count,
            "circuit_breaker_suspension_count": self.circuit_breaker_suspension_count,
        }
        logger.info(
            "[Trading Diagnostics] steps=%d, inference=%d, persisted=%d, "
            "skip_ratio=%.2f%%, def_preemptions=%d, unavail_breaks=%d",
            total_steps,
            policy_inference_steps,
            persistence_held_steps,
            skip_inference_ratio * 100.0,
            defensive_preemptions,
            action_unavailable_breaks,
        )

        if self.save_artifacts:
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            np.save(os.path.join(save_path, HistoryArtifactNames.REWARD_HISTORY_NPY), reward_history)
            np.save(
                os.path.join(save_path, HistoryArtifactNames.TOTAL_ASSET_HISTORY_NPY), total_asset_history
            )
            np.save(
                os.path.join(save_path, HistoryArtifactNames.MICRO_ACTION_HISTORY_NPY),
                micro_action_history,
            )
            np.save(
                os.path.join(save_path, HistoryArtifactNames.ACTION_DECISION_REASON_HISTORY_NPY),
                reasons,
            )
            np.save(os.path.join(save_path, ArtifactNames.TRADING_INFO_NPY), trading_info)
            np.save(
                os.path.join(save_path, HistoryArtifactNames.INITIAL_MARGIN_HISTORY_NPY),
                env.initial_margin_history,
            )
            np.save(
                os.path.join(save_path, HistoryArtifactNames.WALLET_BALANCE_HISTORY_NPY),
                env.wallet_balance_history,
            )
            np.save(
                os.path.join(save_path, HistoryArtifactNames.UNREALIZED_PNL_HISTORY_NPY),
                env.unrealized_pnl_history,
            )
            np.save(
                os.path.join(save_path, HistoryArtifactNames.MAINTAIN_MARGIN_HISTORY_NPY),
                env.maintain_marigine_history,
            )
            np.save(
                os.path.join(save_path, HistoryArtifactNames.NEW_POSITION_REQUIRED_MONEY_HISTORY_NPY),
                env.new_position_required_money_history,
            )
            np.save(
                os.path.join(save_path, HistoryArtifactNames.MACRO_ACTION_NPY),
                self.macro_action_history,
            )
            np.save(
                os.path.join(save_path, HistoryArtifactNames.MACRO_ACTION_HISTORY_NPY),
                self.macro_action_history,
            )

        self.current_contract_quantiles = None
        require_money = calculate_required_money(
            np.array(env.initial_margin_history),
            np.array(env.maintain_marigine_history),
            np.array(env.new_position_required_money_history),
            np.array(env.unrealized_pnl_history),
            np.array(env.wallet_balance_history),
        )
        reward_sum = np.sum(reward_history)
        self.return_rate = reward_sum / (require_money + 1e-12)
        if self.save_artifacts:
            logger.info(
                "[Artifacts] Saved simulation history to %s | rows: %d, reward_sum: %.4f, require_money: %.4f, return_rate: %.6f",
                save_path,
                len(self.df),
                reward_sum,
                require_money,
                self.return_rate,
            )
        return {
            "rows": len(self.df),
            "reward_sum": float(reward_sum),
            "require_money": float(require_money),
            "return_rate": float(self.return_rate),
        }

    def test(self):
        logger.info("[Test Start] Starting VAE routing test...")
        logger.info("[Test Config] Test path: %s", self.test_path)
        contract_files = self.find_contract_files()
        if not contract_files:
            logger.info(
                "[Test] Multi-contract directory not found in '%s', evaluating single dataset file: %s",
                self.eval_stage_dir,
                self.test_data_path,
            )
            self.reset_routing_state()
            contract_name = os.path.splitext(os.path.basename(self.test_data_path))[0]
            result = self.run_single_valid_df(
                pd.read_feather(self.test_data_path),
                self.test_path,
                contract_name=contract_name,
            )
            logger.info(
                "[Test End] Single dataset test completed | return_rate: %.6f | artifacts: %s",
                result["return_rate"],
                self.test_path,
            )
            return result["return_rate"]

        logger.info(
            "[Test] Found %d contracts to evaluate in stage '%s'.",
            len(contract_files),
            self.eval_stage,
        )
        contract_results = []
        for idx, (contract, path) in enumerate(contract_files, start=1):
            logger.info(
                "[Test Contract] [%d/%d] Evaluating contract '%s' (%s) ...",
                idx,
                len(contract_files),
                contract,
                path,
            )
            self.reset_routing_state()
            result = self.run_single_valid_df(
                pd.read_feather(path),
                os.path.join(self.test_path, "contracts", contract),
                contract_name=contract,
            )
            result["contract"] = contract
            result["source_file"] = path
            contract_results.append(result)
            logger.info(
                "[Test Contract Done] [%d/%d] Completed '%s': reward_sum=%.4f, require_money=%.4f, return_rate=%.6f",
                idx,
                len(contract_files),
                contract,
                result["reward_sum"],
                result["require_money"],
                result["return_rate"],
            )

        result_df = pd.DataFrame(contract_results)
        result_df = result_df[
            [
                MetricColumns.CONTRACT,
                MetricColumns.SOURCE_FILE,
                MetricColumns.ROWS,
                MetricColumns.REWARD_SUM,
                MetricColumns.REQUIRE_MONEY,
                MetricColumns.RETURN_RATE,
            ]
        ]
        if self.save_artifacts:
            first_contract_dir = os.path.join(self.test_path, "contracts", contract_results[0]["contract"])
            if os.path.isdir(first_contract_dir):
                for f_name in os.listdir(first_contract_dir):
                    if f_name.endswith(".npy") or f_name.endswith(".csv"):
                        shutil.copy2(os.path.join(first_contract_dir, f_name), os.path.join(self.test_path, f_name))

            csv_path = os.path.join(self.test_path, ArtifactNames.CONTRACT_RESULTS_CSV)
            result_df.to_csv(csv_path, index=False)
            logger.info("[Artifacts] Saved contract results summary to %s", csv_path)

        total_reward_sum = float(result_df[MetricColumns.REWARD_SUM].sum())
        traded_mask = (result_df[MetricColumns.REWARD_SUM] != 0) | (result_df[MetricColumns.RETURN_RATE] != 0)
        traded_count = int(traded_mask.sum())
        total_initial_capital = (
            self.initial_wallet_balance * traded_count
            if traded_count > 0
            else self.initial_wallet_balance * len(contract_results)
        )
        portfolio_return_rate = total_reward_sum / (total_initial_capital + 1e-12)
        win_rate = float((result_df[MetricColumns.RETURN_RATE] > 0).mean())
        self.return_rate = portfolio_return_rate * win_rate

        if self.save_artifacts:
            trading_info = {
                "return_rate": self.return_rate,
                "portfolio_return_rate": portfolio_return_rate,
                "win_rate": win_rate,
                "equal_weighted_mean_return": float(result_df[MetricColumns.RETURN_RATE].mean()),
                "total_reward_sum": total_reward_sum,
                "aggregation": "option2_portfolio_return_times_win_rate",
                "contract_count": len(contract_results),
            }
            trading_info_path = os.path.join(self.test_path, ArtifactNames.TRADING_INFO_NPY)
            np.save(trading_info_path, trading_info)
            logger.info("[Artifacts] Saved aggregated trading info to %s", trading_info_path)

        logger.info(
            "[Test End] Multi-contract test completed | Contracts: %d | Total Reward Sum: %.4f | "
            "Portfolio Return: %.6f | Win Rate: %.4f (%.1f%%) | Final Return Rate: %.6f | Output Dir: %s",
            len(contract_results),
            total_reward_sum,
            portfolio_return_rate,
            win_rate,
            win_rate * 100.0,
            self.return_rate,
            self.test_path,
        )
        return self.return_rate

    def initial_rollout(self, env: Base_Env, s, info):
        done = False
        r = 0
        rollout_window_length = self.initial_rollout_window_length
        for i in range(rollout_window_length):
            action = rule_based_close(
                info,
                self.zero_position_action,
                self.leverage_choices,
                self.position_list,
                env.position,
                env.leverage,
            )
            self.action_decision_reason_history.append(
                ActionDecisionReasons.DEFENSIVE_RULE_CLOSE
            )
            s, r, done, info = env.step(action)
            self.step_idx += 1
            vae_idx = min(self.step_idx, len(self.vae_slope_array) - 1)
            vae_s_slope = self.vae_slope_array[vae_idx]
            vae_s_vol = self.vae_vol_array[vae_idx]
            self.get_quantiles(vae_s_slope, vae_s_vol)
            if done:
                break
        return env, s, r, done, info


if __name__ == "__main__":
    seed_torch(42)
    args = parser.parse_args()
    logger.info("Starting VAE risk-aware routing test with args: %s", args)
    vae_routing = vae_risk_aware_routing(args)
    final_return_rate = vae_routing.test()
    logger.info("Test finished with final return rate: %.6f", final_return_rate)
