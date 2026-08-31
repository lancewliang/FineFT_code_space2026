# the frequency of the high level agent is the same as the low level agent
# based on a sequence of high level actions
import argparse
from collections import deque
import json
import logging
import os
import random
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
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True


def load_two_dimensional_selection_manifest(manifest_path):
    """Load and validate the logical slope/volatility slot contract."""

    with open(manifest_path, encoding="utf-8") as file:
        manifest = json.load(file)

    axes = manifest.get("axes")
    if not isinstance(axes, dict):
        raise ValueError("two-dimensional manifest must contain an axes object")
    volatility_labels = axes.get("volatility")
    slope_labels = axes.get("slope")
    if not isinstance(volatility_labels, list) or not isinstance(slope_labels, list):
        raise ValueError("manifest axes must contain volatility and slope lists")
    if not volatility_labels or len(volatility_labels) != len(slope_labels):
        raise ValueError("volatility and slope axes must have the same non-zero size")

    num_labels = len(volatility_labels)
    expected_slot_count = num_labels * num_labels
    if manifest.get("slot_count") != expected_slot_count:
        raise ValueError(
            "manifest slot_count does not match the two-dimensional axes: "
            f"expected {expected_slot_count}, got {manifest.get('slot_count')}"
        )
    if manifest.get("slot_index_formula") != (
        "volatility_index * num_labels + slope_index"
    ):
        raise ValueError("unsupported two-dimensional slot index formula")

    slots = manifest.get("slots")
    if not isinstance(slots, list) or len(slots) != expected_slot_count:
        raise ValueError("manifest slots must contain every logical slot")
    slot_ids = [slot.get("slot_id") for slot in slots]
    if sorted(slot_ids) != list(range(expected_slot_count)):
        raise ValueError("manifest slot_id values must be contiguous and start at zero")
    if any(slot.get("kind") not in {"model", "empty_model"} for slot in slots):
        raise ValueError("manifest slot kind must be model or empty_model")
    manifest["slots"] = sorted(slots, key=lambda slot: slot["slot_id"])

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
        manifest_path = getattr(args, "selection_manifest", None)
        if not manifest_path:
            raise ValueError("selection_manifest is required")
        self.selection_manifest = load_two_dimensional_selection_manifest(
            manifest_path
        )
        self.num_labels = len(self.selection_manifest["axes"]["volatility"])
        self.slot_count = self.selection_manifest["slot_count"]
        self.axis_window_lengths = {
            "slope": getattr(args, "slope_window_length", self.window_length),
            "volatility": getattr(
                args, "volatility_window_length", self.window_length
            ),
        }
        self.axis_gammas = {
            "slope": getattr(args, "slope_gamma", self.gamma),
            "volatility": getattr(args, "volatility_gamma", self.gamma),
        }
        self.axis_thresholds = {
            "slope": getattr(
                args, "slope_rule_base_threshold", self.rule_base_threshold
            ),
            "volatility": getattr(
                args,
                "volatility_rule_base_threshold",
                self.rule_base_threshold,
            ),
        }
        self.initial_rollout_window_length = max(self.axis_window_lengths.values())
        self.experiment_name = getattr(args, "experiment_name", "default")
        self.model_path = os.path.join(
                args.result_path,
                args.dataset_name,
                self.experiment_name,
                "vae_risk_aware_routing",
            )
  

        trial_number = getattr(args, "trial_number", None)
        trial_suffix = "" if trial_number is None else f"_trial_{trial_number}"
        self.test_path = os.path.join(
            self.model_path,
            "gamma_{}_window_{}_threshold_{}".format(
                self.gamma, self.window_length, self.rule_base_threshold
            ) + trial_suffix,
        )
        if not os.path.exists(self.test_path):
            os.makedirs(self.test_path, exist_ok=True)
            #
        # trading environment setting
        self.base_path = args.base_path
        self.dataset_name = args.dataset_name
        self.allow_reverse_position = getattr(args, "allow_reverse_position", False)
        self.valid_data_path = os.path.join(self.base_path, self.dataset_name, "valid")
        self.test_data_path = os.path.join(
            self.base_path, self.dataset_name, "valid.feather"
        )
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
        self.order_book_depth = getattr(args, "order_book_depth", 25)
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
        low_level_model_path = self.selection_manifest.get("artifacts", {}).get(
            "model_assembly"
        )
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
            self.vae_models[axis], self.in_ds_logpx[axis] = load_vae_axis(root)
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

    def valid_contract_files(self):
        if not os.path.isdir(self.valid_data_path):
            return []
        contract_files = []
        for filename in sorted(os.listdir(self.valid_data_path)):
            path = os.path.join(self.valid_data_path, filename)
            if os.path.isfile(path) and filename.endswith(".feather"):
                contract_files.append((os.path.splitext(filename)[0], path))
        return contract_files

    def find_quantile(self, value, array):
        sorted_array = np.sort(array)

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

    def _defensive_action(self, info):
        return rule_based_close(
            info,
            self.zero_position_action,
            self.leverage_choices,
            self.position_list,
        )

    def get_action(self, info, s):
        volatility_weights = self.calculate_axis_window_result("volatility")
        slope_weights = self.calculate_axis_window_result("slope")
        if (
            max(volatility_weights) < self.axis_thresholds["volatility"]
            or max(slope_weights) < self.axis_thresholds["slope"]
        ):
            action = self._defensive_action(info)
            self.macro_action_history.append(self.slot_count)
        else:
            volatility_index = int(np.argmax(volatility_weights))
            slope_index = int(np.argmax(slope_weights))
            slot_id = volatility_index * self.num_labels + slope_index
            slot = self.selection_manifest["slots"][slot_id]
            if slot["kind"] == "empty_model":
                action = self._defensive_action(info)
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
            order_book_depth=getattr(self, "order_book_depth", 25),
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
            action = self.get_action(info, s)
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
        contract_files = self.valid_contract_files()
        if not contract_files:
            logger.info(
                "[Test] Multi-contract directory not found in '%s', evaluating single dataset file: %s",
                getattr(self, "valid_data_path", ""),
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
            "[Test] Found %d contracts in '%s' to evaluate.",
            len(contract_files),
            self.valid_data_path,
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
        initial_wallet = getattr(self, "initial_wallet_balance", 10000.0)
        total_initial_capital = initial_wallet * len(contract_results)
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
        rollout_window_length = getattr(
            self, "initial_rollout_window_length", self.window_length
        )
        for i in range(rollout_window_length):
            action = rule_based_close(
                info,
                self.zero_position_action,
                self.leverage_choices,
                self.position_list,
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
