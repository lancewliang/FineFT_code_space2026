import copy
import sys

sys.path.append(".")
from RL.DiHFT.high_level.vae_routing_util import vae_risk_aware_routing
import optuna
import argparse
import torch
import numpy as np
import os
import random
import os

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
    args_1.dataset_name = args_2.dataset_name
    args_1.max_holding_number = args_2.max_holding_number
    args_1.order_book_depth = args_2.order_book_depth
    args_1.experiment_name = getattr(
        args_2, "experiment_name", "default"
    ) or getattr(args_1, "experiment_name", "default")
    args_1.allow_reverse_position = getattr(
        args_2, "allow_reverse_position", False
    ) or getattr(args_1, "allow_reverse_position", False)
    print("change parameters:", args_1, args_2)

    def objective(trail):
        trial_args = copy.deepcopy(args_1)
        gpu_id = trail.number % max(torch.cuda.device_count(), 1)
        trial_args.gpu_index = gpu_id
        print("gpu_id:", gpu_id)
        trial_args.window_length = trail.suggest_int(
            "window_length", args_2.window_length_min, args_2.window_length_max
        )
        trial_args.gamma = trail.suggest_float(
            "gamma", args_2.gamma_min, args_2.gamma_max, log=True
        )
        trial_args.rule_base_threshold = trail.suggest_float(
            "rule_base_threshold",
            args_2.rule_base_threshold_min,
            args_2.rule_base_threshold_max,
        )
        vae_routing = vae_risk_aware_routing(trial_args)
        return_rate = vae_routing.test()
        return return_rate

    print("define objective")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=128, n_jobs=16)

    print("Number of finished trials: ", len(study.trials))
    print("BEST TRAIL: ", study.best_trial.params)
    df = study.trials_dataframe()
    optunal_path = os.path.join(
            "result/DiHFT/high_level/",
            args_1.dataset_name,
            args_1.experiment_name,
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
