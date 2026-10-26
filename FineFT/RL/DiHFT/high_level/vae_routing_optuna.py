import argparse
import copy
import hashlib
import json
import multiprocessing
import os
import random
import re
import sys
from pathlib import Path

# keep every worker process single-threaded; must be set before importing torch
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import optuna  # noqa: E402
import torch  # noqa: E402

sys.path.append(".")
from common import ArtifactNames, RoutingParamColumns
from RL.DiHFT.high_level.frozen_experiment import (
    SCHEMA_VERSION,
    SEARCH_FIELDS,
    STRATEGY_VERSION,
    FrozenRoutingArgumentParser,
    effective_trial_configuration,
    export_frozen_experiment,
    prepare_experiment,
    validate_trial_configuration,
)
from RL.DiHFT.high_level.vae_routing_util import (  # noqa: E402
    ensure_precomputed_vae_quantiles,
    load_two_dimensional_selection_manifest,
    vae_risk_aware_routing,
)

OBJECTIVE_NAME = "expert_hysteresis_calibration_score"
OBJECTIVE_VERSION = "v1"
OBJECTIVE_DIRECTION = "maximize"
STUDY_NAMESPACE = "expert_hysteresis_calibration"
SELECTION_EVIDENCE_FILENAME = "selection_evidence.json"
FROZEN_CONFIG_FILENAME = "selected_expert_hysteresis_config.json"
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_MANIFEST_SEARCH_OVERRIDE_FIELDS = {
    "window_length_min",
    "window_length_max",
    "gamma_min",
    "gamma_max",
    "rule_base_threshold_min",
    "rule_base_threshold_max",
    "hysteresis_exit_ratio_min",
    "hysteresis_exit_ratio_max",
}


parser_all = FrozenRoutingArgumentParser()
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
    default=1,
    help="最大允许持仓头寸上限，用于根据离散动作数划分单侧持仓档位",
)
parser_all.add_argument(
    "--position_choices",
    type=int,
    default=3,
    help="离散目标仓位选择数量（例如 3 档，包含空头档位、空仓及多头档位）",
)
parser_all.add_argument(
    "--leverage_choices",
    nargs="+",
    type=int,
    default=[1],
    help="交易可选的杠杆倍数列表",
)
parser_all.add_argument(
    "--initial_leverage",
    type=float,
    default=1,
    help="交易环境初始化的初始杠杆倍数",
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
    "--non_main_defense_mode",
    type=str,
    default="strict_instant_close",
    choices=["strict_instant_close", "graceful_hold_exit"],
    help="非主力合约防御模式：graceful_hold_exit（禁开仓但允许持有和平仓）或 strict_instant_close（即时规则平仓）",
)
parser_all.add_argument(
    "--enable_mdd_penalty",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="是否在 Optuna 目标函数中启用 MDD 屏障惩罚与交易活跃度守卫",
)
parser_all.add_argument(
    "--target_mdd",
    type=float,
    default=0.05,
    help="目标最大回撤阈值（默认 0.05 即 5%%）",
)
parser_all.add_argument(
    "--mdd_penalty_weight",
    type=float,
    default=4.0,
    help="超出目标 MDD 时的指数惩罚强度系数",
)
parser_all.add_argument(
    "--min_trades_guard",
    type=int,
    default=60,
    help="最低有效交易次数门限，防止空仓作弊",
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
    "--calibration_run_id",
    type=str,
    default="default",
    help="Expert-hysteresis calibration artifact namespace; existing runs are never replaced",
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
    default="expert_hysteresis",
    choices=["expert_hysteresis"],
    help="版本化高层专家迟滞路由策略",
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

    if args_2.gating_strategy == "expert_hysteresis":
        explicit = set(getattr(args_2, "_explicit_routing_options", ()))
        legacy_overrides = sorted(explicit & _MANIFEST_SEARCH_OVERRIDE_FIELDS)
        if legacy_overrides:
            raise ValueError(
                "expert_hysteresis search bounds come from experiment_manifest; "
                f"CLI range overrides are unsupported: {', '.join(legacy_overrides)}"
            )
        calibration_run_id = getattr(args_2, "calibration_run_id", "default")
        if not _RUN_ID_PATTERN.fullmatch(calibration_run_id):
            raise ValueError("calibration_run_id must be a safe nonempty artifact identifier")
    base_args = copy.deepcopy(args_1)
    base_args.dataset_name = args_2.dataset_name
    base_args.max_holding_number = getattr(
        args_2,
        "max_holding_number",
        getattr(args_1, "max_holding_number", 1),
    )
    if hasattr(args_2, "position_choices") and args_2.position_choices is not None:
        base_args.position_choices = args_2.position_choices
    if hasattr(args_2, "leverage_choices") and args_2.leverage_choices is not None:
        base_args.leverage_choices = args_2.leverage_choices
    if hasattr(args_2, "initial_leverage") and args_2.initial_leverage is not None:
        base_args.initial_leverage = args_2.initial_leverage
    base_args.order_book_depth = args_2.order_book_depth
    base_args.experiment_name = (
        args_2.experiment_name or base_args.experiment_name
    )
    base_args.enable_non_main_contract_defense = (
        args_2.enable_non_main_contract_defense
        or base_args.enable_non_main_contract_defense
    )
    base_args.non_main_defense_mode = args_2.non_main_defense_mode
    base_args.enable_mdd_penalty = args_2.enable_mdd_penalty
    base_args.target_mdd = args_2.target_mdd
    base_args.mdd_penalty_weight = args_2.mdd_penalty_weight
    base_args.min_trades_guard = args_2.min_trades_guard
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
    base_args.calibration_run_id = getattr(args_2, "calibration_run_id", "default")
    manifest_path = args_2.selection_manifest or default_selection_manifest_path(base_args)
    manifest = load_two_dimensional_selection_manifest(manifest_path)
    if not manifest.artifacts.model_assembly:
        raise ValueError("two-dimensional manifest has no model_assembly artifact")
    base_args.selection_manifest = manifest_path
    if base_args.gating_strategy == "expert_hysteresis":
        prepare_experiment(base_args, "calibration", validate_trial=False)
    return base_args


def _set_trial_user_attr(trial, name, value):
    if hasattr(trial, "set_user_attr"):
        trial.set_user_attr(name, value)


def _suggest_expert_trial_parameters(trial, trial_args):
    """Sample the manifest-declared nine-dimensional expert configuration."""
    search_space = trial_args.search_space
    for field in SEARCH_FIELDS:
        descriptor = search_space[field]
        if descriptor["type"] == "int":
            value = trial.suggest_int(field, descriptor["low"], descriptor["high"])
        else:
            value = trial.suggest_float(
                field,
                descriptor["low"],
                descriptor["high"],
                log=bool(descriptor.get("log", False)),
            )
        setattr(trial_args, field, value)
    validate_trial_configuration(trial_args)
    effective = effective_trial_configuration(trial_args)
    _set_trial_user_attr(trial, "strategy_version", STRATEGY_VERSION)
    _set_trial_user_attr(trial, "experiment_identity", trial_args.frozen_experiment_identity)
    _set_trial_user_attr(trial, "raw_score_source_identity", trial_args.raw_score_source_identity)
    _set_trial_user_attr(trial, "effective_configuration", effective)
    _set_trial_user_attr(trial, "calibration_warmup_steps", trial_args.calibration_warmup_steps)
    return trial_args


def suggest_trial_parameters(trial, trial_args, search_args):
    """Apply one independent trial configuration without sharing routing state."""

    trial_args.gating_strategy = search_args.gating_strategy
    if search_args.gating_strategy == "expert_hysteresis":
        return _suggest_expert_trial_parameters(trial, trial_args)

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
    if search_args.gating_strategy == "expert_hysteresis":
        if set(best_params) != set(SEARCH_FIELDS):
            missing = sorted(set(SEARCH_FIELDS) - set(best_params))
            extra = sorted(set(best_params) - set(SEARCH_FIELDS))
            raise ValueError(
                "best expert Trial must contain exactly nine parameters; "
                f"missing={missing}, extra={extra}"
            )
        integer_fields = {
            "slope_window_length",
            "volatility_window_length",
            "expert_confirm_steps",
        }
        for field in SEARCH_FIELDS:
            value = best_params[field]
            setattr(
                trial_args,
                field,
                int(value) if field in integer_fields else float(value),
            )
        trial_args.gating_strategy = "expert_hysteresis"
        return validate_trial_configuration(trial_args)

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
    """Return the isolated expert calibration namespace for one declared run."""
    if base_args.gating_strategy != "expert_hysteresis":
        raise ValueError("legacy VAE Optuna targets are not valid step-7 outputs")
    if not _RUN_ID_PATTERN.fullmatch(base_args.experiment_id):
        raise ValueError("experiment_id must be a safe artifact identifier")
    return os.path.join(
        base_args.result_path,
        base_args.dataset_name,
        base_args.experiment_name,
        "expert_hysteresis",
        base_args.experiment_id,
        "valid",
        "calibration",
        base_args.calibration_run_id,
    )


def create_study_storage(base_args):
    """Create a new expert-only SQLite target without deleting prior studies."""
    calibration_path = Path(optuna_result_path(base_args)).resolve()
    calibration_path.mkdir(parents=True, exist_ok=True)
    storage_path = calibration_path / "expert_hysteresis_optuna.sqlite3"
    if storage_path.exists():
        raise FileExistsError(
            "refusing to replace an existing expert calibration study: "
            f"{storage_path}"
        )
    return "sqlite:///" + str(storage_path)


def _storage_path(storage_url):
    prefix = "sqlite:///"
    if not storage_url.startswith(prefix):
        raise ValueError("step 7 requires isolated local SQLite study storage")
    return str(Path(storage_url[len(prefix) :]).resolve())


def _identity(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _write_json_once(path, document):
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to replace calibration evidence: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return str(destination)


def make_rdb_storage(storage_url):
    # 60s busy timeout so concurrent workers never fail on sqlite locks
    return optuna.storages.RDBStorage(
        url=storage_url,
        engine_kwargs={"connect_args": {"timeout": 60}},
    )


def _objective_metrics(router, objective_value):
    return _validate_objective_metrics(
        getattr(router, "last_objective_metrics", {}),
        objective_value,
    )


def _validate_objective_metrics(metrics, objective_value):
    metrics = dict(metrics)
    if not metrics:
        raise ValueError("calibration rollout did not publish raw objective metrics")
    metrics["objective_value"] = float(objective_value)
    required = {"objective_value", "raw_return_rate", "portfolio_mdd", "total_trades"}
    if not required.issubset(metrics):
        raise ValueError("calibration rollout raw objective metrics are incomplete")
    for name, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"objective metric must be numeric: {name}")
        if not np.isfinite(float(value)):
            raise ValueError(f"objective metric must be finite: {name}")
    return metrics


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
        objective_value = float(router.test())
        _set_trial_user_attr(trial, "raw_metrics", _objective_metrics(router, objective_value))
        _set_trial_user_attr(
            trial,
            "objective",
            {
                "name": OBJECTIVE_NAME,
                "version": OBJECTIVE_VERSION,
                "direction": OBJECTIVE_DIRECTION,
            },
        )
        return objective_value

    study.optimize(objective, n_trials=n_trials, n_jobs=1)


def _trial_state_name(trial):
    state = getattr(trial, "state", None)
    return getattr(state, "name", str(state))


def _objective_definition(base_args):
    with Path(base_args.experiment_manifest).open(encoding="utf-8") as source:
        calibration_period = json.load(source)["periods"]["calibration"]
    return {
        "name": OBJECTIVE_NAME,
        "version": OBJECTIVE_VERSION,
        "direction": OBJECTIVE_DIRECTION,
        "raw_metric": "raw_return_rate",
        "mdd_penalty_enabled": bool(base_args.enable_mdd_penalty),
        "target_mdd": float(base_args.target_mdd),
        "mdd_penalty_weight": float(base_args.mdd_penalty_weight),
        "min_trades_guard": int(base_args.min_trades_guard),
        "penalty_definition": {
            "non_negative_return": "raw_return_rate * mdd_factor * trade_guard",
            "negative_return": "raw_return_rate / max(mdd_factor, 1e-4)",
            "trade_guard": "min(1, total_trades / min_trades_guard)",
            "mdd_factor_above_target": (
                "exp(-mdd_penalty_weight * "
                "(portfolio_mdd - target_mdd) / target_mdd)"
            ),
        },
        "evaluation_boundary": {
            "period_start": calibration_period["start"],
            "period_end": calibration_period["end"],
            "warmup_steps": int(base_args.calibration_warmup_steps),
        },
    }


def _selection_evidence(base_args, best_trial, frozen_path, replay_router, replay_value):
    lean_metrics = _validate_objective_metrics(
        best_trial.user_attrs.get("raw_metrics", {}),
        best_trial.value,
    )
    replay_metrics = _objective_metrics(replay_router, replay_value)
    document = {
        "schema_version": SCHEMA_VERSION,
        "document_type": "expert_hysteresis_calibration_selection",
        "strategy_version": STRATEGY_VERSION,
        "experiment_id": base_args.experiment_id,
        "experiment_identity": base_args.frozen_experiment_identity,
        "stage": "valid",
        "purpose": "calibration_selection",
        "study": {
            "name": base_args.study_name,
            "storage_identity": base_args.study_storage_identity,
            "storage_path": _storage_path(base_args.study_storage_url),
        },
        "objective": _objective_definition(base_args),
        "selected_trial": {
            "number": int(best_trial.number),
            "state": _trial_state_name(best_trial),
            "objective_value": float(best_trial.value),
            "parameters": {field: best_trial.params[field] for field in SEARCH_FIELDS},
            "effective_configuration": effective_trial_configuration(base_args),
            "raw_metrics": lean_metrics,
        },
        "frozen_config": {
            "path": str(Path(frozen_path).resolve()),
            "config_hash": base_args.frozen_config_hash,
        },
        "best_replay": {
            "artifact_path": str(Path(replay_router.test_path).resolve()),
            "objective_value": float(replay_value),
            "raw_metrics": replay_metrics,
            "matches_lean_trial": True,
        },
        "raw_score_source_identity": base_args.raw_score_source_identity,
    }
    document["evidence_hash"] = _identity(document)
    return document


def tune(args_1, args_2):
    """Run lean expert calibration and one exact, artifact-producing best replay."""
    if args_2.gating_strategy != "expert_hysteresis":
        raise ValueError("step 7 only supports gating_strategy=expert_hysteresis")
    if args_2.n_trials <= 0 or args_2.n_workers <= 0:
        raise ValueError("n_trials and n_workers must be positive")
    seed_torch(12345)
    base_args = prepare_base_args(args_1, args_2)
    print("change parameters:", base_args, args_2)

    base_args.precomputed_quantiles_dir = ensure_precomputed_vae_quantiles(base_args)

    storage_url = create_study_storage(base_args)
    study_name = ".".join(
        (STUDY_NAMESPACE, base_args.experiment_id, base_args.calibration_run_id)
    )
    optuna.create_study(
        direction=OBJECTIVE_DIRECTION,
        study_name=study_name,
        storage=make_rdb_storage(storage_url),
        load_if_exists=False,
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
        failed = [process.exitcode for process in processes if process.exitcode != 0]
        if failed:
            raise RuntimeError(f"expert calibration workers failed: exit_codes={failed}")

    study = optuna.load_study(
        study_name=study_name,
        storage=make_rdb_storage(storage_url),
    )
    print("Number of finished trials: ", len(study.trials))
    print("BEST TRAIL: ", study.best_trial.params)
    best_trial = study.best_trial
    if _trial_state_name(best_trial) != "COMPLETE":
        raise ValueError("selected expert Trial must have state COMPLETE")
    df = study.trials_dataframe()
    calibration_path = Path(optuna_result_path(base_args))
    results_csv = calibration_path / ArtifactNames.OPTUNA_RESULTS_CSV
    if results_csv.exists():
        raise FileExistsError(f"refusing to replace calibration Trial results: {results_csv}")
    df.to_csv(results_csv, index=False)

    print("Replaying best trial to save canonical artifacts...")
    best_args = copy.deepcopy(base_args)
    best_args.save_artifacts = True
    best_args.trial_number = best_trial.number
    best_args = apply_best_trial_parameters(
        best_args, best_trial.params, args_2
    )
    best_args.study_name = study_name
    best_args.study_storage_url = storage_url
    best_args.study_storage_identity = _identity(
        {
            "experiment_identity": best_args.frozen_experiment_identity,
            "study_name": study_name,
            "storage_path": _storage_path(storage_url),
        }
    )
    best_args.trial_state = "COMPLETE"
    best_args.objective_name = OBJECTIVE_NAME
    best_args.objective_version = OBJECTIVE_VERSION
    best_args.objective_direction = OBJECTIVE_DIRECTION
    best_args.objective_value = float(best_trial.value)
    best_args.output_sub_dir = os.path.join(
        "calibration",
        best_args.calibration_run_id,
        "best_replay",
        f"trial_{best_trial.number}",
    )
    # The replay's audit metadata must carry the exact frozen configuration
    # identity that downstream publication verifies.  Exporting before replay is
    # safe: the selected Trial/configuration are already final, and the export
    # rejects any later attempt to replace different evidence.
    frozen_path = export_frozen_experiment(
        best_args,
        calibration_path / FROZEN_CONFIG_FILENAME,
    )
    best_args.frozen_configuration_identity = best_args.frozen_config_hash
    best_router = vae_risk_aware_routing(best_args)
    best_return_rate = float(best_router.test())
    if not np.isclose(best_return_rate, float(best_trial.value), rtol=1e-12, atol=1e-12):
        raise ValueError(
            "best replay objective differs from the selected lean Trial: "
            f"lean={best_trial.value}, replay={best_return_rate}"
        )
    evidence = _selection_evidence(
        best_args,
        best_trial,
        frozen_path,
        best_router,
        best_return_rate,
    )
    evidence_path = _write_json_once(
        calibration_path / SELECTION_EVIDENCE_FILENAME,
        evidence,
    )
    print("Best trial evaluation completed with return rate: {:.6f}".format(best_return_rate))
    return {
        "frozen_config": frozen_path,
        "selection_evidence": evidence_path,
        "best_replay_path": best_router.test_path,
        "objective_value": best_return_rate,
    }


if __name__ == "__main__":
    from RL.DiHFT.high_level.vae_routing_util import parser

    args_1, _ = parser.parse_known_args()
    args_2, _ = parser_all.parse_known_args()
    tune(args_1, args_2)
    print("Done!")
