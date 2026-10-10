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
    help="交易品种或数据集名称（例如 BTCUSDT），用于确定数据与参数存储路径",
)
parser_all.add_argument(
    "--experiment_name",
    type=str,
    default="default",
    help="实验名称，用于命名超参数寻优结果目录与隔离实验产物",
)
parser_all.add_argument(
    "--max_holding_number",
    type=float,
    default=8,
    help="最大允许持仓头寸上限，用于根据离散动作数划分单侧持仓档位",
)
parser_all.add_argument(
    "--order_book_depth",
    type=int,
    default=25,
    help="订单簿（LOB）买卖盘的深度档位数（例如 25 档买卖价量信息）",
)
parser_all.add_argument(
    "--window_length_max",
    type=int,
    default=150,
    help="Optuna 搜索空间中滚动特征窗口长度（window_length）的上限",
)
parser_all.add_argument(
    "--window_length_min",
    type=int,
    default=50,
    help="Optuna 搜索空间中滚动特征窗口长度（window_length）的下限",
)


parser_all.add_argument(
    "--gamma_max",
    type=float,
    default=0.98,
    help="Optuna 搜索空间中指数衰减因子（gamma）的上限",
)
parser_all.add_argument(
    "--gamma_min",
    type=float,
    default=0.92,
    help="Optuna 搜索空间中指数衰减因子（gamma）的下限",
)
parser_all.add_argument(
    "--rule_base_threshold_min",
    type=float,
    default=0.2,
    help="Optuna 搜索空间中规则基底阈值（rule_base_threshold）的下限",
)
parser_all.add_argument(
    "--rule_base_threshold_max",
    type=float,
    default=0.5,
    help="Optuna 搜索空间中规则基底阈值（rule_base_threshold）的上限",
)
parser_all.add_argument(
    "--selection_manifest",
    type=str,
    default=None,
    help="二维低层子代理选择清单（manifest JSON）的文件路径",
)
parser_all.add_argument(
    "--enable_non_main_contract_defense",
    action="store_true",
    default=False,
    help="是否在非主力与非次主力合约上启用防御性门控机制",
)
parser_all.add_argument(
    "--n_trials",
    type=int,
    default=128,
    help="Optuna 超参数优化的总试验评估次数（trials 数量）",
)
parser_all.add_argument(
    "--n_workers",
    type=int,
    default=32,
    help="并行工作进程数；各进程在 CPU 上独立单线程运行 trial 并通过 SQLite 数据库共享 Study",
)
parser_all.add_argument(
    "--action_persistence",
    type=int,
    default=3,
    help="非平仓动作在重新评估策略前持续保持的连续步数（动作持久化机制）",
)
parser_all.add_argument(
    "--stop_loss_return_threshold",
    type=float,
    default=0.015,
    help="仓位收益率硬止损阈值，例如 0.015 表示价格不利变动达 1.5%% 时触发止损（设为 0.0 则禁用）",
)
parser_all.add_argument(
    "--stop_loss_cooldown_steps",
    type=int,
    default=12,
    help="触发止损后的同向开仓冷却锁进步数（设为 0 则禁用）",
)
parser_all.add_argument(
    "--circuit_breaker_consecutive_stops",
    type=int,
    default=2,
    help="触发合约级别熔断机制所需的连续止损次数阈值（设为 0 则禁用）",
)
parser_all.add_argument(
    "--circuit_breaker_cooling_steps",
    type=int,
    default=72,
    help="触发熔断后暂停交易的步数冷却时间（-1 表示永久熔断暂停交易）",
)
parser_all.add_argument(
    "--gating_strategy",
    type=str,
    default="absolute",
    choices=["absolute", "hierarchical"],
    help="门控路由策略类型，可选绝对阈值门控（absolute）或分层门控（hierarchical）",
)
parser_all.add_argument(
    "--ood_threshold_min",
    type=float,
    default=0.001,
    help="分层门控策略中分布外（OOD）检测阈值的搜索下限",
)
parser_all.add_argument(
    "--ood_threshold_max",
    type=float,
    default=0.02,
    help="分层门控策略中分布外（OOD）检测阈值的搜索上限",
)
parser_all.add_argument(
    "--margin_threshold_min",
    type=float,
    default=0.05,
    help="分层门控策略中 Top-1 与 Top-2 候选专家概率裕度（margin）阈值的搜索下限",
)
parser_all.add_argument(
    "--margin_threshold_max",
    type=float,
    default=0.30,
    help="分层门控策略中 Top-1 与 Top-2 候选专家概率裕度（margin）阈值的搜索上限",
)
parser_all.add_argument(
    "--hysteresis_exit_ratio",
    type=float,
    default=0.65,
    help="绝对门控策略中迟滞退出比率的默认值（退出阈值相对入场阈值的比例）",
)
parser_all.add_argument(
    "--hysteresis_exit_ratio_min",
    type=float,
    default=0.50,
    help="绝对门控策略中迟滞退出比率（hysteresis exit ratio）的搜索下限",
)
parser_all.add_argument(
    "--hysteresis_exit_ratio_max",
    type=float,
    default=0.80,
    help="绝对门控策略中迟滞退出比率（hysteresis exit ratio）的搜索上限",
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
    base_args.enable_non_main_contract_defense = (
        args_2.enable_non_main_contract_defense
        or base_args.enable_non_main_contract_defense
    )
    base_args.gating_strategy = args_2.gating_strategy
    base_args.action_persistence = args_2.action_persistence
    base_args.stop_loss_return_threshold = args_2.stop_loss_return_threshold
    base_args.stop_loss_cooldown_steps = args_2.stop_loss_cooldown_steps
    base_args.circuit_breaker_consecutive_stops = args_2.circuit_breaker_consecutive_stops
    base_args.circuit_breaker_cooling_steps = args_2.circuit_breaker_cooling_steps
    base_args.ood_threshold = 0.005
    base_args.slope_margin_threshold = 0.12
    base_args.volatility_margin_threshold = 0.12
    base_args.hysteresis_exit_ratio = float(args_2.hysteresis_exit_ratio)
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
        trial_args.hysteresis_exit_ratio = 1.0
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
        trial_args.hysteresis_exit_ratio = trial.suggest_float(
            RoutingParamColumns.HYSTERESIS_EXIT_RATIO,
            search_args.hysteresis_exit_ratio_min,
            search_args.hysteresis_exit_ratio_max,
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
        trial_args.hysteresis_exit_ratio = 1.0
    else:
        trial_args.slope_rule_base_threshold = float(
            best_params[RoutingParamColumns.SLOPE_RULE_BASE_THRESHOLD]
        )
        trial_args.volatility_rule_base_threshold = float(
            best_params[RoutingParamColumns.VOLATILITY_RULE_BASE_THRESHOLD]
        )
        trial_args.hysteresis_exit_ratio = float(
            best_params[RoutingParamColumns.HYSTERESIS_EXIT_RATIO]
            if RoutingParamColumns.HYSTERESIS_EXIT_RATIO in best_params
            else 0.65
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
