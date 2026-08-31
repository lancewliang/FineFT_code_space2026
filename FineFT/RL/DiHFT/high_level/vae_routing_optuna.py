import copy
import argparse
import json
import os
import random
import sys

import numpy as np
import optuna
import torch

sys.path.append(".")
from RL.DiHFT.high_level.vae_routing_util import vae_risk_aware_routing

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
parser_all = argparse.ArgumentParser()
# * Env setting
parser_all.add_argument(
    "--dataset_name",
    type=str,
    default="BTCUSDT",
    help="training data chunk",
)
parser_all.add_argument(
    "--experiment_name",
    type=str,
    default="default",
    help="experiment name",
)
parser_all.add_argument(
    "--max_holding_number",
    type=float,
    default=8,
    help="the transcation cost of not holding the same action as before",
)
parser_all.add_argument(
    "--order_book_depth",
    type=int,
    default=25,
    help="number of bid/ask price levels available in the order book",
)
parser_all.add_argument(
    "--window_length_max",
    type=int,
    default=150,
    help="the transcation cost of not holding the same action as before",
)
parser_all.add_argument(
    "--window_length_min",
    type=int,
    default=50,
    help="the transcation cost of not holding the same action as before",
)


parser_all.add_argument(
    "--gamma_max",
    type=float,
    default=0.98,
    help="the transcation cost of not holding the same action as before",
)
parser_all.add_argument(
    "--gamma_min",
    type=float,
    default=0.92,
    help="the transcation cost of not holding the same action as before",
)
parser_all.add_argument(
    "--rule_base_threshold_min",
    type=float,
    default=0.2,
    help="the transcation cost of not holding the same action as before",
)
parser_all.add_argument(
    "--rule_base_threshold_max",
    type=float,
    default=0.5,
    help="the transcation cost of not holding the same action as before",
)
parser_all.add_argument(
    "--allow_reverse_position",
    action="store_true",
    help="allow reverse position in single step",
)
parser_all.add_argument(
    "--selection_manifest",
    type=str,
    default=None,
    help="two-dimensional low-level selection manifest",
)
parser_all.add_argument(
    "--n_trials",
    type=int,
    default=128,
    help="number of Optuna trials",
)
parser_all.add_argument(
    "--n_jobs",
    type=int,
    default=None,
    help="parallel Optuna jobs; defaults to the available GPU count",
)


def default_selection_manifest_path(args):
    return os.path.join(
        "analysis_result",
        "DiHFT",
        "low_level",
        args.dataset_name,
        args.experiment_name,
        "two_dimensional_selection",
        "two_dimensional_selection_manifest.json",
    )


def prepare_base_args(args_1, args_2):
    """Apply CLI-level routing configuration without sharing mutable trial state."""

    base_args = copy.deepcopy(args_1)
    base_args.dataset_name = args_2.dataset_name
    base_args.max_holding_number = args_2.max_holding_number
    base_args.order_book_depth = args_2.order_book_depth
    base_args.experiment_name = getattr(args_2, "experiment_name", "default") or getattr(
        base_args, "experiment_name", "default"
    )
    base_args.allow_reverse_position = getattr(
        args_2, "allow_reverse_position", False
    ) or getattr(base_args, "allow_reverse_position", False)
    manifest_path = getattr(args_2, "selection_manifest", None)
    manifest_path = manifest_path or default_selection_manifest_path(base_args)
    with open(manifest_path, encoding="utf-8") as file:
        manifest = json.load(file)
    if not manifest.get("artifacts", {}).get("model_assembly"):
        raise ValueError("two-dimensional manifest has no model_assembly artifact")
    base_args.selection_manifest = manifest_path
    return base_args


def suggest_trial_parameters(trial, trial_args, search_args):
    """Apply independent slope and volatility parameters to one trial."""

    trial_args.slope_window_length = trial.suggest_int(
        "slope_window_length",
        search_args.window_length_min,
        search_args.window_length_max,
    )
    trial_args.volatility_window_length = trial.suggest_int(
        "volatility_window_length",
        search_args.window_length_min,
        search_args.window_length_max,
    )
    trial_args.slope_gamma = trial.suggest_float(
        "slope_gamma",
        search_args.gamma_min,
        search_args.gamma_max,
        log=True,
    )
    trial_args.volatility_gamma = trial.suggest_float(
        "volatility_gamma",
        search_args.gamma_min,
        search_args.gamma_max,
        log=True,
    )
    trial_args.slope_rule_base_threshold = trial.suggest_float(
        "slope_rule_base_threshold",
        search_args.rule_base_threshold_min,
        search_args.rule_base_threshold_max,
    )
    trial_args.volatility_rule_base_threshold = trial.suggest_float(
        "volatility_rule_base_threshold",
        search_args.rule_base_threshold_min,
        search_args.rule_base_threshold_max,
    )
    trial_args.window_length = max(
        trial_args.slope_window_length,
        trial_args.volatility_window_length,
    )
    trial_args.gamma = trial_args.slope_gamma
    trial_args.rule_base_threshold = min(
        trial_args.slope_rule_base_threshold,
        trial_args.volatility_rule_base_threshold,
    )
    return trial_args


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


def tune(args_1, args_2):
    # args 1 from orginal trader
    # args 2 from here
    seed_torch(12345)
    base_args = prepare_base_args(args_1, args_2)
    print("change parameters:", base_args, args_2)

    def objective(trial):
        trial_args = copy.deepcopy(base_args)
        gpu_id = trial.number % max(torch.cuda.device_count(), 1)
        trial_args.gpu_index = gpu_id
        trial_args.trial_number = trial.number
        print("gpu_id:", gpu_id)
        trial_args = suggest_trial_parameters(trial, trial_args, args_2)
        vae_routing = vae_risk_aware_routing(trial_args)
        return_rate = vae_routing.test()
        return return_rate

    print("define objective")
    study = optuna.create_study(direction="maximize")
    n_jobs = getattr(args_2, "n_jobs", None)
    if n_jobs is None:
        n_jobs = max(torch.cuda.device_count(), 1)
    study.optimize(objective, n_trials=args_2.n_trials, n_jobs=n_jobs)

    print("Number of finished trials: ", len(study.trials))
    print("BEST TRAIL: ", study.best_trial.params)
    df = study.trials_dataframe()
    optunal_path = os.path.join(
            "result/DiHFT/high_level/",
            base_args.dataset_name,
            base_args.experiment_name,
            "vae_risk_aware_routing_optuna",
        )
    if not os.path.exists(optunal_path):
        os.makedirs(optunal_path)
    df.to_csv(os.path.join(optunal_path, "optuna_results.csv"))


if __name__ == "__main__":
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
    from RL.DiHFT.high_level.vae_routing_util import parser

    args_1, _ = parser.parse_known_args()
    args_2, _ = parser_all.parse_known_args()
    tune(args_1, args_2)
    print("Done!")
