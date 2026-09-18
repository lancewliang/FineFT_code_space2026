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
from RL.DiHFT.VAE.vae import MLP_VAE, analyze_single_sample

from analysis.pick_agent.FineFT_two_dimensional_agent_selector import (
    TwoDimensionalSelectionManifest,
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


def resolve_routing_parameters(args):
    """Resolve 6 dual-axis routing parameters from CLI arguments, para_file, or optuna_csv."""
    slope_window_length = args.slope_window_length
    volatility_window_length = args.volatility_window_length
    slope_gamma = args.slope_gamma
    volatility_gamma = args.volatility_gamma
    slope_rule_base_threshold = args.slope_rule_base_threshold
    volatility_rule_base_threshold = args.volatility_rule_base_threshold

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
        row = df[df["number"] == trial_id]
        if not row.empty:
            r = row.iloc[0]
            if slope_window_length is None:
                slope_window_length = int(r["params_slope_window_length"])
            if volatility_window_length is None:
                volatility_window_length = int(r["params_volatility_window_length"])
            if slope_gamma is None:
                slope_gamma = float(r["params_slope_gamma"])
            if volatility_gamma is None:
                volatility_gamma = float(r["params_volatility_gamma"])
            if slope_rule_base_threshold is None:
                slope_rule_base_threshold = float(r["params_slope_rule_base_threshold"])
            if volatility_rule_base_threshold is None:
                volatility_rule_base_threshold = float(r["params_volatility_rule_base_threshold"])

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
        self.initial_rollout_window_length = max(self.axis_window_lengths.values())
        self.experiment_name = args.experiment_name
        self.eval_stage = args.eval_stage

        self.test_path = self._resolve_test_path(args)
        if not os.path.exists(self.test_path):
            os.makedirs(self.test_path, exist_ok=True)
            #
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
            os.path.join(self.base_path, self.dataset_name, "state_features.npy")
        )
        self.maintenance_margin_ratio_dict = np.load(
            os.path.join(
                self.base_path, self.dataset_name, "maintenance_margin_ratio_dict.npy"
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
        def load_vae_axis(root):
            label_list = [f"label_{i}" for i in range(self.num_labels)]
            model_list = []
            logpx_list = []
            for label in label_list:
                path = os.path.join(root, label, "model_latest.pth")
                id_path = os.path.join(root, label, "id_logpx.npy")
                vae_model = MLP_VAE(
                    INPUT_DIM=len(self.tech_indicator_list),
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
        self.vae_models = {}
        self.in_ds_logpx = {}
        self.quantiles = {}
        for axis, root in vae_roots.items():
            self.vae_models[axis], logpx_list = load_vae_axis(root)
            # pre-sort once here so find_quantile only needs searchsorted per step
            self.in_ds_logpx[axis] = [np.sort(logpx) for logpx in logpx_list]
            self.quantiles[axis] = [
                deque(maxlen=self.axis_window_lengths[axis])
                for _ in range(self.num_labels)
            ]
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
        self.initial_rollout_window_length = max(self.axis_window_lengths.values())
        self.test_path = self._resolve_test_path(args)
        if not os.path.exists(self.test_path):
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

    def get_quantiles(self, s):
        for axis in ("slope", "volatility"):
            loss_list = [
                analyze_single_sample(vae_model, s, self.device)[1]
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

    def get_action(self, info, s, current_position, current_leverage):
        volatility_weights = self.calculate_axis_window_result("volatility")
        slope_weights = self.calculate_axis_window_result("slope")
        if (
            max(volatility_weights) < self.axis_thresholds["volatility"]
            or max(slope_weights) < self.axis_thresholds["slope"]
        ):
            action = self._defensive_action(info, current_position, current_leverage)
            self.macro_action_history.append(self.slot_count)
        else:
            volatility_index = int(np.argmax(volatility_weights))
            slope_index = int(np.argmax(slope_weights))
            slot_id = volatility_index * self.num_labels + slope_index
            slot = self.selection_manifest.slots[slot_id]
            if slot["kind"] == "empty_model":
                action = self._defensive_action(info, current_position, current_leverage)
                self.macro_action_history.append(self.slot_count)
            else:
                self.selected_agent_index = slot_id
                action = self.agent_act(s, info)
                self.macro_action_history.append(slot_id)
        self.action = action
        return action

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

    def run_single_valid_df(self, df, save_path):
        self.df = df
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
        env, s, r, done, info = self.initial_rollout(env, s, info)
        while not done:
            action = self.get_action(info, s, env.position, env.leverage)
            s_, r, done, info = env.step(action)
            self.get_quantiles(s_)
            episode_reward_sum += r
            if done:
                break
            s = s_
        total_asset_history = env.margine_balance_history
        reward_history = calculate_differences(total_asset_history)
        micro_action_history = env.micro_action_history
        trading_info = {
            "return rate": total_asset_history[-1] / self.initial_wallet_balance
        }

        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, "reward_history.npy"), reward_history)
        np.save(
            os.path.join(save_path, "total_asset_history.npy"), total_asset_history
        )
        np.save(
            os.path.join(save_path, "micro_action_history.npy"),
            micro_action_history,
        )
        np.save(os.path.join(save_path, "trading_info.npy"), trading_info)
        np.save(
            os.path.join(save_path, "initial_margin_history.npy"),
            env.initial_margin_history,
        )
        np.save(
            os.path.join(save_path, "wallet_balance_history.npy"),
            env.wallet_balance_history,
        )
        np.save(
            os.path.join(save_path, "unrealized_pnl_history.npy"),
            env.unrealized_pnl_history,
        )
        np.save(
            os.path.join(save_path, "maintain_marigine_history.npy"),
            env.maintain_marigine_history,
        )
        np.save(
            os.path.join(save_path, "new_position_required_money_history.npy"),
            env.new_position_required_money_history,
        )
        np.save(
            os.path.join(save_path, "macro_action.npy"),
            self.macro_action_history,
        )
        np.save(
            os.path.join(save_path, "macro_action_history.npy"),
            self.macro_action_history,
        )
        require_money = calculate_required_money(
            np.array(env.initial_margin_history),
            np.array(env.maintain_marigine_history),
            np.array(env.new_position_required_money_history),
            np.array(env.unrealized_pnl_history),
            np.array(env.wallet_balance_history),
        )
        reward_sum = np.sum(reward_history)
        self.return_rate = reward_sum / (require_money + 1e-12)
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
            result = self.run_single_valid_df(
                pd.read_feather(self.test_data_path), self.test_path
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

        first_contract_dir = os.path.join(self.test_path, "contracts", contract_results[0]["contract"])
        if os.path.isdir(first_contract_dir):
            for f_name in os.listdir(first_contract_dir):
                if f_name.endswith(".npy") or f_name.endswith(".csv"):
                    shutil.copy2(os.path.join(first_contract_dir, f_name), os.path.join(self.test_path, f_name))

        result_df = pd.DataFrame(contract_results)
        result_df = result_df[
            [
                "contract",
                "source_file",
                "rows",
                "reward_sum",
                "require_money",
                "return_rate",
            ]
        ]
        csv_path = os.path.join(self.test_path, "contract_results.csv")
        result_df.to_csv(csv_path, index=False)
        logger.info("[Artifacts] Saved contract results summary to %s", csv_path)

        total_reward_sum = float(result_df["reward_sum"].sum())
        total_initial_capital = self.initial_wallet_balance * len(contract_results)
        portfolio_return_rate = total_reward_sum / (total_initial_capital + 1e-12)
        win_rate = float((result_df["return_rate"] > 0).mean())
        self.return_rate = portfolio_return_rate * win_rate
        trading_info = {
            "return_rate": self.return_rate,
            "portfolio_return_rate": portfolio_return_rate,
            "win_rate": win_rate,
            "equal_weighted_mean_return": float(result_df["return_rate"].mean()),
            "total_reward_sum": total_reward_sum,
            "aggregation": "option2_portfolio_return_times_win_rate",
            "contract_count": len(contract_results),
        }
        trading_info_path = os.path.join(self.test_path, "trading_info.npy")
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
            s, r, done, info = env.step(action)
            self.get_quantiles(s)
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
