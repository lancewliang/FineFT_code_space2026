import copy
import argparse
import multiprocessing
import os
import random
import sys

# keep every worker process single-threaded; must be set before importing torch
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import optuna  # noqa: E402
import torch  # noqa: E402

sys.path.append(".")
from common import ArtifactNames, RoutingParamColumns
from RL.DiHFT.high_level.vae_routing_util import (  # noqa: E402
    ensure_precomputed_vae_quantiles,
    load_two_dimensional_selection_manifest,
    vae_risk_aware_routing,
)

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
    "--enable_non_main_contract_defense",
    action="store_true",
    default=False,
    help="enable defensive gating on non-main and non-sub-main contracts",
)
parser_all.add_argument(
    "--n_trials",
    type=int,
    default=128,
    help="number of Optuna trials",
)
parser_all.add_argument(
    "--n_workers",
    type=int,
    default=32,
    help="number of parallel worker processes; each runs trials independently "
    "on CPU with a single thread and shares the study via sqlite storage",
)
parser_all.add_argument(
    "--action_persistence",
    type=int,
    default=3,
    help="number of consecutive steps a non-flat action persists before re-evaluating policy",
)
parser_all.add_argument(
    "--stop_loss_abs_threshold",
    type=float,
    default=50.0,
    help="unrealized PnL hard stop-loss absolute threshold in quote currency (0.0 to disable)",
)
parser_all.add_argument(
    "--stop_loss_cooldown_steps",
    type=int,
    default=12,
    help="directional lockout cooldown steps after stop-loss trigger (0 to disable)",
)
parser_all.add_argument(
    "--circuit_breaker_consecutive_stops",
    type=int,
    default=2,
    help="consecutive stop-loss trigger count to trip contract-level circuit breaker (0 to disable)",
)
parser_all.add_argument(
    "--circuit_breaker_cooling_steps",
    type=int,
    default=72,
    help="steps to suspend trading on circuit breaker (-1 for permanent suspension)",
)
parser_all.add_argument(
    "--gating_strategy",
    type=str,
    default="absolute",
    choices=["absolute", "hierarchical"],
    help="gating strategy type (absolute or hierarchical)",
)
parser_all.add_argument(
    "--ood_threshold_min",
    type=float,
    default=0.001,
    help="minimum OOD threshold for hierarchical gating",
)
parser_all.add_argument(
    "--ood_threshold_max",
    type=float,
    default=0.02,
    help="maximum OOD threshold for hierarchical gating",
)
parser_all.add_argument(
    "--margin_threshold_min",
    type=float,
    default=0.05,
    help="minimum margin threshold for hierarchical gating",
)
parser_all.add_argument(
    "--margin_threshold_max",
    type=float,
    default=0.30,
    help="maximum margin threshold for hierarchical gating",
)


def default_selection_manifest_path(args):
    return os.path.join(
        "analysis_result",
        "DiHFT",
        "low_level",
        args.dataset_name,
        args.experiment_name,
        "two_dimensional_selection",
        ArtifactNames.TWO_DIMENSIONAL_SELECTION_MANIFEST_JSON,
    )


def prepare_base_args(args_1, args_2):
    """Apply CLI-level routing configuration without sharing mutable trial state."""

    base_args = copy.deepcopy(args_1)
    base_args.dataset_name = args_2.dataset_name
    base_args.max_holding_number = args_2.max_holding_number
    base_args.order_book_depth = args_2.order_book_depth
    base_args.experiment_name = (
        args_2.experiment_name or base_args.experiment_name
    )
    base_args.allow_reverse_position = (
        args_2.allow_reverse_position or base_args.allow_reverse_position
    )
    base_args.enable_non_main_contract_defense = (
        args_2.enable_non_main_contract_defense
        or base_args.enable_non_main_contract_defense
    )
    base_args.gating_strategy = args_2.gating_strategy
    base_args.action_persistence = args_2.action_persistence
    base_args.stop_loss_abs_threshold = args_2.stop_loss_abs_threshold
    base_args.stop_loss_cooldown_steps = args_2.stop_loss_cooldown_steps
    base_args.circuit_breaker_consecutive_stops = args_2.circuit_breaker_consecutive_stops
    base_args.circuit_breaker_cooling_steps = args_2.circuit_breaker_cooling_steps
    base_args.ood_threshold = 0.005
    base_args.slope_margin_threshold = 0.12
    base_args.volatility_margin_threshold = 0.12
    manifest_path = args_2.selection_manifest or default_selection_manifest_path(base_args)
    manifest = load_two_dimensional_selection_manifest(manifest_path)
    if not manifest.artifacts.model_assembly:
        raise ValueError("two-dimensional manifest has no model_assembly artifact")
    base_args.selection_manifest = manifest_path
    return base_args


def suggest_trial_parameters(trial, trial_args, search_args):
    """Apply independent slope and volatility parameters to one trial."""

    trial_args.slope_window_length = trial.suggest_int(
        RoutingParamColumns.SLOPE_WINDOW_LENGTH,
        search_args.window_length_min,
        search_args.window_length_max,
    )
    trial_args.volatility_window_length = trial.suggest_int(
        RoutingParamColumns.VOLATILITY_WINDOW_LENGTH,
        search_args.window_length_min,
        search_args.window_length_max,
    )
    trial_args.slope_gamma = trial.suggest_float(
        RoutingParamColumns.SLOPE_GAMMA,
        search_args.gamma_min,
        search_args.gamma_max,
        log=True,
    )
    trial_args.volatility_gamma = trial.suggest_float(
        RoutingParamColumns.VOLATILITY_GAMMA,
        search_args.gamma_min,
        search_args.gamma_max,
        log=True,
    )
    trial_args.window_length = max(
        trial_args.slope_window_length,
        trial_args.volatility_window_length,
    )
    trial_args.gamma = trial_args.slope_gamma
    trial_args.gating_strategy = search_args.gating_strategy
    trial.suggest_categorical(
        RoutingParamColumns.GATING_STRATEGY,
        [search_args.gating_strategy],
    )

    if search_args.gating_strategy == "hierarchical":
        trial_args.ood_threshold = trial.suggest_float(
            RoutingParamColumns.OOD_THRESHOLD,
            search_args.ood_threshold_min,
            search_args.ood_threshold_max,
            log=True,
        )
        trial_args.slope_margin_threshold = trial.suggest_float(
            RoutingParamColumns.SLOPE_MARGIN_THRESHOLD,
            search_args.margin_threshold_min,
            search_args.margin_threshold_max,
        )
        trial_args.volatility_margin_threshold = trial.suggest_float(
            RoutingParamColumns.VOLATILITY_MARGIN_THRESHOLD,
            search_args.margin_threshold_min,
            search_args.margin_threshold_max,
        )
        trial_args.slope_rule_base_threshold = 0.0
        trial_args.volatility_rule_base_threshold = 0.0
        trial_args.rule_base_threshold = 0.0
    else:
        trial_args.slope_rule_base_threshold = trial.suggest_float(
            RoutingParamColumns.SLOPE_RULE_BASE_THRESHOLD,
            search_args.rule_base_threshold_min,
            search_args.rule_base_threshold_max,
        )
        trial_args.volatility_rule_base_threshold = trial.suggest_float(
            RoutingParamColumns.VOLATILITY_RULE_BASE_THRESHOLD,
            search_args.rule_base_threshold_min,
            search_args.rule_base_threshold_max,
        )
        trial_args.ood_threshold = 0.005
        trial_args.slope_margin_threshold = 0.12
        trial_args.volatility_margin_threshold = 0.12
        trial_args.rule_base_threshold = min(
            trial_args.slope_rule_base_threshold,
            trial_args.volatility_rule_base_threshold,
        )
    return trial_args




def apply_best_trial_parameters(trial_args, best_params, search_args):
    """Apply best Optuna parameters to trial_args for canonical replay."""
    trial_args.slope_window_length = int(
        best_params[RoutingParamColumns.SLOPE_WINDOW_LENGTH]
    )
    trial_args.volatility_window_length = int(
        best_params[RoutingParamColumns.VOLATILITY_WINDOW_LENGTH]
    )
    trial_args.slope_gamma = float(best_params[RoutingParamColumns.SLOPE_GAMMA])
    trial_args.volatility_gamma = float(
        best_params[RoutingParamColumns.VOLATILITY_GAMMA]
    )
    trial_args.window_length = max(
        trial_args.slope_window_length,
        trial_args.volatility_window_length,
    )
    trial_args.gamma = trial_args.slope_gamma
    trial_args.gating_strategy = search_args.gating_strategy

    if search_args.gating_strategy == "hierarchical":
        trial_args.ood_threshold = float(
            best_params[RoutingParamColumns.OOD_THRESHOLD]
        )
        trial_args.slope_margin_threshold = float(
            best_params[RoutingParamColumns.SLOPE_MARGIN_THRESHOLD]
        )
        trial_args.volatility_margin_threshold = float(
            best_params[RoutingParamColumns.VOLATILITY_MARGIN_THRESHOLD]
        )
        trial_args.slope_rule_base_threshold = 0.0
        trial_args.volatility_rule_base_threshold = 0.0
        trial_args.rule_base_threshold = 0.0
    else:
        trial_args.slope_rule_base_threshold = float(
            best_params[RoutingParamColumns.SLOPE_RULE_BASE_THRESHOLD]
        )
        trial_args.volatility_rule_base_threshold = float(
            best_params[RoutingParamColumns.VOLATILITY_RULE_BASE_THRESHOLD]
        )
        trial_args.ood_threshold = 0.005
        trial_args.slope_margin_threshold = 0.12
        trial_args.volatility_margin_threshold = 0.12
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
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def optuna_result_path(base_args):
    return os.path.join(
        "result/DiHFT/high_level/",
        base_args.dataset_name,
        base_args.experiment_name,
        "vae_risk_aware_routing_optuna",
    )


def create_study_storage(base_args):
    """Create a fresh sqlite storage shared by all worker processes."""
    optunal_path = optuna_result_path(base_args)
    os.makedirs(optunal_path, exist_ok=True)
    storage_path = os.path.join(optunal_path, "optuna_study.db")
    if os.path.exists(storage_path):
        # each invocation tunes from scratch, matching the old in-memory study
        os.remove(storage_path)
    return "sqlite:///" + storage_path


def make_rdb_storage(storage_url):
    # 60s busy timeout so concurrent workers never fail on sqlite locks
    return optuna.storages.RDBStorage(
        url=storage_url,
        engine_kwargs={"connect_args": {"timeout": 60}},
    )


def run_optuna_worker(base_args, args_2, study_name, storage_url, n_trials):
    """Run n_trials Optuna trials in one process, reusing loaded models."""
    seed_torch(12345)
    torch.set_num_threads(1)
    study = optuna.load_study(
        study_name=study_name,
        storage=make_rdb_storage(storage_url),
    )
    router = None

    def objective(trial):
        nonlocal router
        trial_args = copy.deepcopy(base_args)
        gpu_id = trial.number % max(torch.cuda.device_count(), 1)
        trial_args.gpu_index = gpu_id
        trial_args.trial_number = trial.number
        trial_args.save_artifacts = False
        print("gpu_id:", gpu_id)
        trial_args = suggest_trial_parameters(trial, trial_args, args_2)
        if router is None:
            router = vae_risk_aware_routing(trial_args)
        else:
            router.reconfigure_routing(trial_args)
        return router.test()

    study.optimize(objective, n_trials=n_trials, n_jobs=1)


def tune(args_1, args_2):
    # args 1 from orginal trader
    # args 2 from here
    seed_torch(12345)
    base_args = prepare_base_args(args_1, args_2)
    print("change parameters:", base_args, args_2)

    base_args.precomputed_quantiles_dir = ensure_precomputed_vae_quantiles(base_args)

    storage_url = create_study_storage(base_args)
    study_name = "vae_risk_aware_routing"
    optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=make_rdb_storage(storage_url),
        load_if_exists=True,
    )

    n_workers = min(args_2.n_workers, args_2.n_trials)
    base_count, extra = divmod(args_2.n_trials, n_workers)
    worker_trial_counts = [
        base_count + (1 if worker_index < extra else 0)
        for worker_index in range(n_workers)
    ]
    print(
        "launching {} worker processes for {} trials".format(
            n_workers, args_2.n_trials
        )
    )

    if n_workers == 1:
        run_optuna_worker(
            base_args, args_2, study_name, storage_url, worker_trial_counts[0]
        )
    else:
        spawn_context = multiprocessing.get_context("spawn")
        processes = [
            spawn_context.Process(
                target=run_optuna_worker,
                args=(base_args, args_2, study_name, storage_url, count),
            )
            for count in worker_trial_counts
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join()

    study = optuna.load_study(
        study_name=study_name,
        storage=make_rdb_storage(storage_url),
    )
    print("Number of finished trials: ", len(study.trials))
    print("BEST TRAIL: ", study.best_trial.params)
    df = study.trials_dataframe()
    optunal_path = optuna_result_path(base_args)
    if not os.path.exists(optunal_path):
        os.makedirs(optunal_path)
    df.to_csv(os.path.join(optunal_path, ArtifactNames.OPTUNA_RESULTS_CSV))

    print("Replaying best trial to save canonical artifacts...")
    best_args = copy.deepcopy(base_args)
    best_args.save_artifacts = True
    best_args.trial_number = study.best_trial.number
    best_args = apply_best_trial_parameters(
        best_args, study.best_trial.params, args_2
    )
    best_router = vae_risk_aware_routing(best_args)
    best_return_rate = best_router.test()
    print("Best trial evaluation completed with return rate: {:.6f}".format(best_return_rate))


if __name__ == "__main__":
    from RL.DiHFT.high_level.vae_routing_util import parser

    args_1, _ = parser.parse_known_args()
    args_2, _ = parser_all.parse_known_args()
    tune(args_1, args_2)
    print("Done!")
