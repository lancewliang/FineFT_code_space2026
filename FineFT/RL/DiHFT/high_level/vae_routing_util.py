# the frequency of the high level agent is the same as the low level agent
# based on a sequence of high level actions
import argparse
from collections import deque
from typing import Any
import json
import logging
import os
import random
import shutil
import sys
import time

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
from RL.DiHFT.high_level.gating import ExpertHysteresisGating, create_gating_strategy
from RL.DiHFT.high_level.frozen_experiment import (
    FrozenRoutingArgumentParser,
    load_frozen_experiment,
    prepare_experiment,
    resolve_expert_parameters,
    validate_trial_configuration,
    validate_quantile_cache,
    write_quantile_cache_manifest,
)

from analysis.pick_agent.FineFT_two_dimensional_agent_selector import (
    TwoDimensionalSelectionManifest,
)
from common import (
    ActionDecisionReasons,
    ArtifactNames,
    AuditArtifactNames,
    HistoryArtifactNames,
    MetricColumns,
    RoutingParamColumns,
)
from common.audit_records import (
    AccountRecord,
    PredictionRecord,
    TradeRecord,
    build_account_snapshot_id,
    build_decision_id,
    build_prediction_id,
    build_trade_id,
    calculate_action_value_margin,
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
parser = FrozenRoutingArgumentParser()
# * Env setting
parser.add_argument(
    "--base_path",
    type=str,
    default="dataset",
    help="数据存储的基础根目录路径",
)
parser.add_argument(
    "--dataset_name",
    type=str,
    default="BTCUSDT",
    help="交易品种或数据集名称（例如 BTCUSDT）",
)
parser.add_argument(
    "--experiment_name",
    type=str,
    default="default",
    help="实验名称，用于定位模型、配置及输出结果的专属目录",
)
parser.add_argument(
    "--max_holding_number",
    type=float,
    default=1,
    help="最大允许持仓头寸上限，用于根据离散动作数划分单侧持仓档位",
)
parser.add_argument(
    "--position_choices",
    type=int,
    default=3,
    help="离散目标仓位选择数量（例如 9 档，包含空头档位、空仓及多头档位）",
)
parser.add_argument(
    "--leverage_choices",
    nargs="+",
    type=int,
    default=[1],
    help="交易可选的杠杆倍数列表",
)
parser.add_argument(
    "--long_estimated_rate",
    type=float,
    default=0.0005,
    help="多头开仓预估滑点与附加成本费率",
)
parser.add_argument(
    "--short_estimated_rate",
    type=float,
    default=0,
    help="空头开仓预估滑点与附加成本费率",
)
parser.add_argument(
    "--transcation_cost",
    type=float,
    default=0.0002,
    help="单次调仓交易手续费率 / 交易成本",
)

parser.add_argument(
    "--early_stop",
    type=int,
    default=0,
    help="单轮测试交互的最大早停步数（设为 0 表示不启用早停）",
)
parser.add_argument(
    "--initial_wallet_balance",
    type=float,
    default=1e5,
    help="交易环境初始化的账户钱包初始余额",
)
parser.add_argument(
    "--initial_margin",
    type=float,
    default=0,
    help="交易环境初始化的初始保证金金额",
)
parser.add_argument(
    "--initial_unrealized_pnL",
    type=float,
    default=0,
    help="交易环境初始化的初始未实现盈亏",
)
parser.add_argument(
    "--initial_position",
    type=float,
    default=0,
    help="交易环境初始化的初始持仓量",
)
parser.add_argument(
    "--initial_leverage",
    type=float,
    default=1,
    help="交易环境初始化的初始杠杆倍数",
)
parser.add_argument(
    "--order_book_depth",
    type=int,
    default=25,
    help="订单簿（LOB）买卖盘的深度档位数（例如 25 档买卖价量信息）",
)
parser.add_argument(
    "--action_persistence",
    type=int,
    default=3,
    help="非平仓动作在重新评估策略前持续保持的连续步数（动作持久化机制）",
)
parser.add_argument(
    "--stop_loss_return_threshold",
    type=float,
    default=0.015,
    help="仓位收益率硬止损阈值，例如 0.015 表示价格不利变动达 1.5%% 时触发止损（设为 0.0 则禁用）",
)
parser.add_argument(
    "--stop_loss_cooldown_steps",
    type=int,
    default=12,
    help="触发止损后的同向开仓冷却锁进步数（设为 0 则禁用）",
)
parser.add_argument(
    "--circuit_breaker_consecutive_stops",
    type=int,
    default=2,
    help="触发合约级别熔断机制所需的连续止损次数阈值（设为 0 则禁用）",
)
parser.add_argument(
    "--circuit_breaker_cooling_steps",
    type=int,
    default=72,
    help="触发熔断后暂停交易的步数冷却时间（-1 表示永久熔断暂停交易）",
)
# low level network setting
parser.add_argument(
    "--hidden_nodes",
    type=int,
    default=128,
    help="低层策略网络隐藏层的神经元节点数量",
)

parser.add_argument(
    "--time_info_dim",
    type=int,
    default=2,
    help="时间特征向量的维度（包含日内时间周期与进度特征）",
)
# VAE network path
parser.add_argument(
    "--vae_path",
    type=str,
    default="result/DiHFT/vae_results",
    help="预训练 VAE 模型权重的读取路径",
)
# vae related
parser.add_argument(
    "--z_dim",
    type=int,
    default=512,
    help="VAE 潜在空间（latent space）的潜在向量维度",
)
parser.add_argument(
    "--vae_hidden_dims",
    type=list,
    default=[4096, 2048, 1024, 1024],
    help="VAE 编码器与解码器隐藏层的各层神经元维度列表",
)
parser.add_argument(
    "--loss_type",
    type=str,
    default="NLL",
    help="VAE 重构损失的计算类型（如 NLL 负对数似然或 MSE 均方误差）",
)
parser.add_argument(
    "--vae_results",
    type=str,
    default="result/DiHFT/vae_results",
    help="VAE 训练与评估结果的输出存储目录",
)

# high level network setting
parser.add_argument(
    "--result_path",
    type=str,
    default="result/DiHFT/high_level",
    help="高层路由策略测试结果与日志的存储路径",
)
parser.add_argument(
    "--strategy_position_unit",
    type=str,
    default="tonne",
    choices=["tonne"],
    help="策略仓位数量口径；当前 fu 研究固定为吨",
)
parser.add_argument(
    "--diagnostic_contract_unit",
    type=float,
    default=1.0,
    help="最终诊断中将策略数量映射到名义金额的单位系数",
)
parser.add_argument(
    "--output_sub_dir",
    type=str,
    default=None,
    help="结果保存的子目录名（例如 main_sub）",
)
parser.add_argument(
    "--window_length",
    type=int,
    default=64,
    help="高层状态统计特征平滑计算的默认滚动时间窗口长度",
)
parser.add_argument(
    "--gamma",
    type=float,
    default=0.9,
    help="高层状态平滑计算中的默认指数加权衰减系数",
)
# 判断是rule base，且之前的down deviation以及超过5% 切成rule based result 等五个step
parser.add_argument(
    "--rule_base_threshold",
    type=float,
    default=0.2,
    help="切换为规则基底决策模式时的默认状态阈值（如超过该阈值时启用防御规则）",
)
parser.add_argument(
    "--selection_manifest",
    type=str,
    default=None,
    help="二维低层子代理选择清单（manifest JSON）的文件路径",
)
parser.add_argument(
    "--enable_non_main_contract_defense",
    action="store_true",
    default=False,
    help="是否在非主力与非次主力合约上启用防御性门控机制",
)
parser.add_argument(
    "--non_main_defense_mode",
    type=str,
    default="strict_instant_close",
    choices=["strict_instant_close", "graceful_hold_exit"],
    help="非主力合约防御模式：graceful_hold_exit（禁开仓但允许持有和平仓）或 strict_instant_close（即时规则平仓）",
)
parser.add_argument(
    "--enable_mdd_penalty",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="是否在目标函数中启用 MDD 屏障惩罚与交易活跃度守卫",
)
parser.add_argument(
    "--target_mdd",
    type=float,
    default=0.05,
    help="目标最大回撤阈值（默认 0.05 即 5%%）",
)
parser.add_argument(
    "--mdd_penalty_weight",
    type=float,
    default=4.0,
    help="超出目标 MDD 时的指数惩罚强度系数",
)
parser.add_argument(
    "--min_trades_guard",
    type=int,
    default=60,
    help="最低有效交易次数门限，防止空仓作弊",
)
parser.add_argument(
    "--eval_stage",
    type=str,
    default="valid",
    choices=["valid", "test"],
    help="评估所使用的数据集阶段划分，可选验证集（valid）或测试集（test）",
)
parser.add_argument(
    "--para_file",
    type=str,
    default=None,
    help="高层代理固定参数配置文件（high_level_agent_para.txt）的文件路径",
)
parser.add_argument(
    "--optuna_csv",
    type=str,
    default=None,
    help="Optuna 超参数优化搜索结果文件（optuna_results.csv）的文件路径",
)
parser.add_argument(
    "--slope_window_length",
    type=int,
    default=None,
    help="斜率/趋势维度统计特征计算的滚动时间窗口长度",
)
parser.add_argument(
    "--volatility_window_length",
    type=int,
    default=None,
    help="波动率维度统计特征计算的滚动时间窗口长度",
)
parser.add_argument(
    "--slope_gamma",
    type=float,
    default=None,
    help="斜率/趋势维度统计特征的指数加权衰减系数（gamma）",
)
parser.add_argument(
    "--volatility_gamma",
    type=float,
    default=None,
    help="波动率维度统计特征的指数加权衰减系数（gamma）",
)
parser.add_argument(
    "--slope_rule_base_threshold",
    type=float,
    default=None,
    help="斜率/趋势维度触发规则基底（防御）模式的判定阈值",
)
parser.add_argument(
    "--volatility_rule_base_threshold",
    type=float,
    default=None,
    help="波动率维度触发规则基底（防御）模式的判定阈值",
)
parser.add_argument(
    "--gating_strategy",
    type=str,
    default="expert_hysteresis",
    choices=["expert_hysteresis"],
    help="高层专家迟滞路由策略（expert_hysteresis）",
)
parser.add_argument("--expert_switch_lead", type=float, default=0.0)
parser.add_argument("--expert_confirm_steps", type=int, default=1)
parser.add_argument("--strategy_version", type=str, default=None)
parser.add_argument("--experiment_manifest", type=str, default=None)
parser.add_argument("--frozen_config", type=str, default=None)
parser.add_argument(
    "--ood_threshold",
    type=float,
    default=0.005,
    help="分层门控策略中分布外（OOD）熔断保护的阈值",
)
parser.add_argument(
    "--slope_margin_threshold",
    type=float,
    default=0.12,
    help="分层门控策略中斜率维度 Top-1 与 Top-2 候选专家的概率裕度（margin）阈值",
)
parser.add_argument(
    "--volatility_margin_threshold",
    type=float,
    default=0.12,
    help="分层门控策略中波动率维度 Top-1 与 Top-2 候选专家的概率裕度（margin）阈值",
)
parser.add_argument(
    "--hysteresis_exit_ratio",
    type=float,
    default=0.65,
    help="迟滞退出阈值比例，退出阈值相对于入场阈值的比例（0.0 至 1.0，用于防止频繁抖动切换）",
)
parser.add_argument(
    "--trial_number",
    type=int,
    default=None,
    help="指定提取参数或隔离结果产物的 Optuna Trial 序号",
)

parser.add_argument(
    "--gpu_index",
    type=int,
    default=0,
    help="指定用于计算与推理的 GPU 设备索引号（例如 0 表示 cuda:0）",
)
parser.add_argument(
    "--precomputed_quantiles_dir",
    type=str,
    default=None,
    help="存储各合约预计算 VAE 分位数统计特征（quantiles）的目录路径",
)
parser.add_argument(
    "--save_artifacts",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="是否将仿真历史轨迹数组及各合约回测指标结果保存至磁盘",
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


def default_expert_quantiles_dir(
    dataset_name: str,
    experiment_name: str,
    eval_stage: str,
) -> str:
    """Return the isolated raw-score cache namespace for the upgraded strategy."""
    return os.path.join(
        "analysis_result",
        "DiHFT",
        "high_level",
        dataset_name,
        experiment_name,
        "expert_hysteresis",
        "raw_quantiles",
        eval_stage,
    )


def ensure_precomputed_vae_quantiles(args) -> str:
    """Precompute VAE routing quantiles for all contracts in stage if missing.

    Runs in a single process (optionally GPU or CPU batch) before worker processes
    are launched, saving (T, 2*num_labels) float32 arrays into analysis_result/...
    """
    expert_experiment = getattr(args, "gating_strategy", None) == "expert_hysteresis"
    if expert_experiment:
        prepare_experiment(
            args,
            "calibration" if args.eval_stage == "valid" else "final",
            validate_trial=False,
        )
    default_cache = (
        default_expert_quantiles_dir
        if expert_experiment
        else default_precomputed_quantiles_dir
    )
    target_dir = args.precomputed_quantiles_dir or default_cache(
        args.dataset_name,
        args.experiment_name,
        args.eval_stage,
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
        if expert_experiment:
            validate_quantile_cache(args, target_dir)
        logger.info(
            "Found complete precomputed VAE quantiles in %s for %d contracts",
            target_dir,
            len(contract_files),
        )
        return target_dir

    if expert_experiment:
        if any(name.endswith(".npy") for name in os.listdir(target_dir)):
            raise ValueError(
                "partial quantile cache cannot establish frozen scoring provenance"
            )
        seed_torch(args.raw_scoring_protocol["seed"])

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

                losses = compute_vae_losses_batch(
                    model,
                    axis_data,
                    device,
                    batch_size=(
                        args.raw_scoring_protocol["batch_size"]
                        if expert_experiment
                        else 512
                    ),
                )
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

    if expert_experiment:
        write_quantile_cache_manifest(args, target_dir)
    logger.info("All precomputed VAE quantiles ready in %s", target_dir)
    return target_dir


def resolve_routing_parameters(args):
    """Resolve dual-axis routing parameters and gating strategy from CLI arguments, para_file, or optuna_csv."""
    if getattr(args, "frozen_config", None):
        if args.para_file or args.optuna_csv:
            raise ValueError("frozen_config cannot be combined with legacy para_file/optuna_csv")
        return load_frozen_experiment(args, args.frozen_config)
    if args.gating_strategy == "expert_hysteresis":
        if args.para_file or args.optuna_csv:
            raise ValueError("expert_hysteresis uses frozen_config; legacy para/CSV cannot supply it")
        return resolve_expert_parameters(args, require_explicit=False)
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
    hysteresis_exit_ratio = args.hysteresis_exit_ratio

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
            if RoutingParamColumns.PARAMS_HYSTERESIS_EXIT_RATIO in df.columns:
                hysteresis_exit_ratio = float(r[RoutingParamColumns.PARAMS_HYSTERESIS_EXIT_RATIO])

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
        hr_match = re.search(r"hr_([0-9.]+)", para_str)
        if ood_match:
            ood_threshold = float(ood_match.group(1))
        if ms_match:
            slope_margin_threshold = float(ms_match.group(1))
        if mv_match:
            volatility_margin_threshold = float(mv_match.group(1))
        if hr_match:
            hysteresis_exit_ratio = float(hr_match.group(1))

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
    if gating_strategy == "expert_hysteresis" or "strat_expert_hysteresis" in para_str:
        raise ValueError("expert_hysteresis requires a complete frozen_config snapshot")
    args.gating_strategy = gating_strategy
    args.ood_threshold = ood_threshold
    args.slope_margin_threshold = slope_margin_threshold
    args.volatility_margin_threshold = volatility_margin_threshold
    args.hysteresis_exit_ratio = hysteresis_exit_ratio

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
    non_main_defense_mode: str = "strict_instant_close"
    role_tier_index: int | None = None
    save_artifacts: bool = True
    precomputed_quantiles_dir: str | None = None
    current_contract_quantiles: np.ndarray | None = None
    stop_loss_return_threshold: float = 0.015
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
    max_holding_number: int = 1
    position_choices: int = 3
    _position_list: list[float] | None = None
    enable_mdd_penalty: bool = False
    target_mdd: float = 0.05
    mdd_penalty_weight: float = 4.0
    min_trades_guard: int = 60
    run_id: str = "unspecified_run"
    current_contract_name: str = "default_contract"
    model_id: str = "dihft_high_level_vae_routing"
    feature_version: str = "v1"
    parameter_version: str = "v1"
    step_idx: int = 0
    last_prediction_id: str = ""
    latest_action_values: list[float] = [0.0, 0.0, 0.0]
    audit_predictions: list[dict[str, Any]] = []
    audit_trades: list[dict[str, Any]] = []
    df: pd.DataFrame | None = None
    is_terminal_defense_step: np.ndarray | None = None

    def __init__(self, args) -> None:
        if args.gating_strategy == "expert_hysteresis":
            prepare_experiment(args, "calibration" if args.eval_stage == "valid" else "final")
            self.frozen_experiment_identity = args.frozen_experiment_identity
            self._raw_scoring_protocol = dict(args.raw_scoring_protocol)
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
        self.stop_loss_return_threshold = float(args.stop_loss_return_threshold)
        if self.stop_loss_return_threshold < 0:
            raise ValueError("stop_loss_return_threshold must be non-negative")
        self.stop_loss_cooldown_steps = int(args.stop_loss_cooldown_steps)
        if self.stop_loss_cooldown_steps < 0:
            raise ValueError("stop_loss_cooldown_steps must be non-negative")
        self.circuit_breaker_consecutive_stops = int(args.circuit_breaker_consecutive_stops)
        if self.circuit_breaker_consecutive_stops < 0:
            raise ValueError("circuit_breaker_consecutive_stops must be non-negative")
        self.circuit_breaker_cooling_steps = int(args.circuit_breaker_cooling_steps)
        if self.circuit_breaker_cooling_steps < -1:
            raise ValueError("circuit_breaker_cooling_steps must be -1 or non-negative")
        self.hysteresis_exit_ratio = float(args.hysteresis_exit_ratio)
        self.gating_strategy = create_gating_strategy(
            args.gating_strategy,
            slope_threshold=self.axis_thresholds["slope"],
            volatility_threshold=self.axis_thresholds["volatility"],
            hysteresis_exit_ratio=self.hysteresis_exit_ratio,
            expert_switch_lead=args.expert_switch_lead,
            expert_confirm_steps=args.expert_confirm_steps,
        )
        self.initial_rollout_window_length = max(self.axis_window_lengths.values())
        self.base_path = args.base_path
        self.dataset_name = args.dataset_name
        self.experiment_name = args.experiment_name
        self.eval_stage = args.eval_stage
        self.save_artifacts = args.save_artifacts
        self.precomputed_quantiles_dir = self._resolve_precomputed_quantiles_dir(args)
        self.current_contract_quantiles = None

        self.test_path = self._resolve_test_path(args)
        if self.save_artifacts and not os.path.exists(self.test_path):
            os.makedirs(self.test_path, exist_ok=True)

        # trading environment setting
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
        self.enable_non_main_contract_defense = bool(args.enable_non_main_contract_defense)
        self.non_main_defense_mode = args.non_main_defense_mode
        self.enable_mdd_penalty = bool(args.enable_mdd_penalty)
        self.target_mdd = float(args.target_mdd)
        self.mdd_penalty_weight = float(args.mdd_penalty_weight)
        self.min_trades_guard = int(args.min_trades_guard)
        self.stop_loss_return_threshold = float(args.stop_loss_return_threshold)
        if self.stop_loss_return_threshold < 0:
            raise ValueError("stop_loss_return_threshold must be non-negative")
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
        mmr_path = os.path.join(
            self.base_path, self.dataset_name, ArtifactNames.MAINTENANCE_MARGIN_RATIO_DICT_NPY
        )
        if os.path.exists(mmr_path):
            self.maintenance_margin_ratio_dict = np.load(
                mmr_path,
                allow_pickle=True,
            ).item()
        else:
            candidates = [
                os.path.join(self.base_path, "fu", ArtifactNames.MAINTENANCE_MARGIN_RATIO_DICT_NPY),
                os.path.join(os.path.dirname(self.base_path), "fu", ArtifactNames.MAINTENANCE_MARGIN_RATIO_DICT_NPY),
                os.path.join("dataset", "fu", ArtifactNames.MAINTENANCE_MARGIN_RATIO_DICT_NPY),
            ]
            loaded = False
            for cand in candidates:
                if os.path.exists(cand):
                    self.maintenance_margin_ratio_dict = np.load(cand, allow_pickle=True).item()
                    loaded = True
                    break
            if not loaded:
                self.maintenance_margin_ratio_dict = {"1000000000000": [0.5, 0]}
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
        self.leverage_choices = (
            [int(x) for x in args.leverage_choices]
            if isinstance(args.leverage_choices, list)
            else [int(args.leverage_choices)]
        )
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
        self.run_id = f"run_{self.experiment_name}_{int(time.time())}"
        self.current_contract_name = "default_contract"
        self.model_id = "dihft_high_level_vae_routing"
        self.feature_version = "v1"
        self.parameter_version = "v1"
        self.step_idx = 0
        self.audit_predictions = []
        self.audit_trades = []
        self.last_prediction_id = ""
        self.latest_action_values = [0.0, 0.0, 0.0]
        self.df = None
        self._position_list = None
        self._log_initialized_attributes()

    def _log_initialized_attributes(self) -> None:
        """Log all initialized instance attributes and their values."""
        logger.info("=" * 80)
        logger.info("📌 [vae_risk_aware_routing] 初始化完成属性列表 (Initialized Attributes):")
        for key in sorted(self.__dict__.keys()):
            val = self.__dict__[key]
            val_str = self._format_attribute_for_logging(val)
            logger.info("  • %-35s : %s", key, val_str)
        logger.info("=" * 80)

    @staticmethod
    def _format_attribute_for_logging(val: Any) -> str:
        """Format an attribute value concisely for logging."""
        if isinstance(val, (int, float, bool, str)) or val is None:
            return repr(val)
        if isinstance(val, np.ndarray):
            if val.ndim == 1 and len(val) <= 5:
                return f"ndarray(shape={val.shape}, dtype={val.dtype}, items={list(val)})"
            return f"ndarray(shape={val.shape}, dtype={val.dtype})"
        if isinstance(val, torch.nn.Module):
            param_count = sum(p.numel() for p in val.parameters())
            return f"{val.__class__.__name__}(parameters={param_count:,})"
        if isinstance(val, dict):
            if len(val) <= 5 and all(isinstance(v, (int, float, str, bool)) for v in val.values()):
                return repr(val)
            return f"dict(len={len(val)}, keys={list(val.keys())[:6]})"
        if isinstance(val, (list, tuple)):
            if len(val) <= 8 and all(isinstance(v, (int, float, str, bool)) for v in val):
                return repr(val)
            return f"{type(val).__name__}(len={len(val)}, items={list(val[:4])}...)"
        if isinstance(val, deque):
            return f"deque(maxlen={val.maxlen}, len={len(val)})"
        repr_str = repr(val)
        if len(repr_str) > 120:
            return f"{val.__class__.__name__}({repr_str[:90]}...)"
        return repr_str

    def reset_routing_state(self):
        if isinstance(getattr(self, "gating_strategy", None), ExpertHysteresisGating):
            self.gating_strategy.reset()
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
        self.step_idx = 0
        self.audit_predictions = []
        self.audit_trades = []
        self.last_prediction_id = ""
        self.latest_action_values = [0.0, 0.0, 0.0]

    @property
    def position_list(self):
        if self._position_list is None:
            max_h = self.max_holding_number
            choices = self.position_choices
            self._position_list = np.linspace(-max_h, max_h, choices).tolist()
            self._position_list.sort()
        return self._position_list

    @position_list.setter
    def position_list(self, value):
        self._position_list = value

    def _resolve_precomputed_quantiles_dir(self, args) -> str | None:
        target_dir = args.precomputed_quantiles_dir
        if getattr(args, "gating_strategy", None) == "expert_hysteresis":
            target_dir = target_dir or default_expert_quantiles_dir(
                self.dataset_name, self.experiment_name, self.eval_stage,
            )
            return validate_quantile_cache(args, target_dir)
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
        sub_dir = args.output_sub_dir
        if self.eval_stage == "test" or args.result_path.endswith("final_result"):
            base = os.path.join(
                args.result_path,
                args.dataset_name,
                self.experiment_name,
            )
            return os.path.join(base, sub_dir) if sub_dir else base
        self.model_path = os.path.join(
            args.result_path,
            args.dataset_name,
            self.experiment_name,
            "vae_risk_aware_routing",
        )
        trial_suffix = "" if args.trial_number is None else f"_trial_{args.trial_number}"
        if args.gating_strategy == "hierarchical":
            target = os.path.join(
                self.model_path,
                f"strat_hierarchical_gamma_{self.gamma}_window_{self.window_length}_ood_{args.ood_threshold}_ms_{args.slope_margin_threshold}_mv_{args.volatility_margin_threshold}"
                + trial_suffix,
            )
        else:
            target = os.path.join(
            self.model_path,
            "gamma_{}_window_{}_threshold_{}".format(
                self.gamma, self.window_length, self.rule_base_threshold
            ) + trial_suffix,
        )
        return os.path.join(target, sub_dir) if sub_dir else target

    def reconfigure_routing(self, args):
        """Reset trial-dependent routing state while keeping loaded models.

        Avoids reloading the low-level network and VAE models between Optuna
        trials; only windows/gammas/thresholds, the output path and the
        routing deques change from trial to trial.
        """
        if getattr(args, "gating_strategy", None) == "expert_hysteresis":
            if hasattr(self, "frozen_experiment_identity"):
                validate_trial_configuration(args)
                if args.frozen_experiment_identity != self.frozen_experiment_identity:
                    raise ValueError(
                        "trial reconfiguration cannot change frozen experiment identity"
                    )
                if args.raw_scoring_protocol != self._raw_scoring_protocol:
                    raise ValueError("trial reconfiguration cannot change raw scoring sources")
            else:
                # Lightweight routers built at the public test seam have no
                # experiment files, but still exercise the same state reset.
                resolve_expert_parameters(args)
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
        self.hysteresis_exit_ratio = float(args.hysteresis_exit_ratio)
        self.gating_strategy = create_gating_strategy(
            args.gating_strategy,
            slope_threshold=self.axis_thresholds["slope"],
            volatility_threshold=self.axis_thresholds["volatility"],
            hysteresis_exit_ratio=self.hysteresis_exit_ratio,
            expert_switch_lead=args.expert_switch_lead,
            expert_confirm_steps=args.expert_confirm_steps,
        )
        self.initial_rollout_window_length = max(self.axis_window_lengths.values())
        self.enable_non_main_contract_defense = bool(args.enable_non_main_contract_defense)
        self.stop_loss_return_threshold = float(args.stop_loss_return_threshold)
        if self.stop_loss_return_threshold < 0:
            raise ValueError("stop_loss_return_threshold must be non-negative")
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
        if float(current_position) == 0.0:
            return self.zero_position_action
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
        current_markprice: float = 0.0,
    ) -> int:
        current_pos_float = float(current_position)
        expert_hysteresis = isinstance(self.gating_strategy, ExpertHysteresisGating)
        if expert_hysteresis:
            self.gating_strategy.bind_manifest(self.selection_manifest)
            volatility_weights = self.calculate_axis_window_result("volatility")
            slope_weights = self.calculate_axis_window_result("slope")
            # Invalid scoring inputs fail even when risk would otherwise preempt.
            self.gating_strategy.validate_scores(volatility_weights, slope_weights)

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

        # 2. Tier 1: Position Return Rate Hard Stop-Loss (Scheme A)
        loss_rate = 0.0
        if (
            current_pos_float != 0.0
            and current_markprice > 0.0
            and current_unrealized_pnl < 0.0
        ):
            notional = abs(current_pos_float) * current_markprice
            loss_rate = -current_unrealized_pnl / notional

        if (
            self.stop_loss_return_threshold > 0.0
            and current_pos_float != 0.0
            and loss_rate >= self.stop_loss_return_threshold
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
        is_downgraded_hold = False
        if (
            self.enable_non_main_contract_defense
            and self.role_tier_index is not None
            and float(s[self.role_tier_index]) < 0.5
        ):
            mode = self.non_main_defense_mode
            if mode == "strict_instant_close" or current_pos_float == 0.0:
                return self._apply_defensive_action(
                    info, current_pos_float, current_leverage
                )
            is_downgraded_hold = True

        if not expert_hysteresis:
            volatility_weights = self.calculate_axis_window_result("volatility")
            slope_weights = self.calculate_axis_window_result("slope")
        decision = self.gating_strategy.decide(
            volatility_weights, slope_weights, current_position=current_pos_float
        )
        if decision.is_defensive:
            return self._apply_defensive_action(
                info, current_pos_float, current_leverage
            )

        volatility_index = decision.volatility_index
        slope_index = decision.slope_index
        slot_id = volatility_index * self.num_labels + slope_index
        slot = self.selection_manifest.slots[slot_id]
        if slot["kind"] == "empty_model":
            # ADR-0061: Graceful hold exit and No New Risk semantics
            if current_pos_float != 0.0 and self.remaining_persist > 0:
                hold_action = map_position_leverage_to_action(
                    current_pos_float,
                    current_leverage,
                    self.leverage_choices,
                    self.position_list,
                )
                if bool(info["avaliable_action"][hold_action]):
                    if self.cooldown_remaining_steps > 0:
                        self.cooldown_remaining_steps -= 1
                    self.remaining_persist -= 1
                    self.current_action = hold_action
                    self.selected_agent_index = slot_id
                    self.macro_action_history.append(slot_id)
                    self.action_decision_reason_history.append(
                        ActionDecisionReasons.ACTION_PERSISTENCE
                    )
                    self.action = hold_action
                    self.previous_step_position = current_pos_float
                    return hold_action
                self.remaining_persist = 0
            return self._apply_defensive_action(
                info, current_pos_float, current_leverage
            )

        self.selected_agent_index = slot_id
        self.macro_action_history.append(slot_id)

        # 4. Candidate Action Query
        if expert_hysteresis:
            candidate_action = self.agent_act(s, info)
            candidate_reason = ActionDecisionReasons.POLICY_INFERENCE
        elif self.remaining_persist > 0 and not bool(
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

        # 6. Graceful Downgrade: forbid opening new positions or expanding exposure on non-main contracts
        if is_downgraded_hold:
            target_pos, _ = map_action_to_position_leverage(
                candidate_action, self.leverage_choices, self.position_list
            )
            if target_pos * current_pos_float < 0:
                self.remaining_persist = 0
                action = self._defensive_action(info, current_pos_float, current_leverage)
                self.current_action = action
                self.action_decision_reason_history.append(
                    ActionDecisionReasons.DEFENSIVE_RULE_CLOSE
                )
                self.action = action
                self.previous_step_position = current_pos_float
                return action
            elif abs(target_pos) > abs(current_pos_float):
                self.remaining_persist = 0
                action = map_position_leverage_to_action(
                    current_pos_float, current_leverage, self.leverage_choices, self.position_list
                )
                self.current_action = action
                self.action_decision_reason_history.append(
                    ActionDecisionReasons.ACTION_PERSISTENCE
                )
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

        if candidate_reason in (
            ActionDecisionReasons.POLICY_INFERENCE,
            ActionDecisionReasons.ACTION_UNAVAILABLE_BREAK,
        ):
            step_i = int(self.step_idx)
            run_id = str(self.run_id)
            contract = str(self.current_contract_name)
            pred_id = build_prediction_id(run_id, contract, step_i)
            dec_id = build_decision_id(run_id, contract, step_i)
            row_idx = min(step_i, len(self.df) - 1) if self.df is not None else 0
            event_time = (
                str(self.df["timestamp"].iloc[row_idx])
                if self.df is not None and "timestamp" in self.df.columns
                else ""
            )
            trading_day = (
                str(self.df["trading_day"].iloc[row_idx])
                if self.df is not None and "trading_day" in self.df.columns
                else ""
            )
            actions_vals = self.latest_action_values
            q_short = float(actions_vals[0]) if len(actions_vals) > 0 else 0.0
            q_flat = float(actions_vals[1]) if len(actions_vals) > 1 else 0.0
            q_long = float(actions_vals[2]) if len(actions_vals) > 2 else 0.0
            q_margin = calculate_action_value_margin(actions_vals)
            vol_w = [float(x) for x in volatility_weights]
            slope_w = [float(x) for x in slope_weights]
            avail_actions = (
                str([int(x) for x in info["avaliable_action"]])
                if "avaliable_action" in info
                else "[]"
            )
            pred = PredictionRecord(
                prediction_id=pred_id,
                decision_id=dec_id,
                run_id=run_id,
                model_id=str(self.model_id),
                feature_version=str(self.feature_version),
                parameter_version=str(self.parameter_version),
                contract=contract,
                step_index=step_i,
                event_time=event_time,
                trading_day=trading_day,
                market_index=int(step_i),
                routing_slot=int(self.selected_agent_index),
                routing_evidence=json.dumps({"volatility": vol_w, "slope": slope_w}),
                action_value_short=q_short,
                action_value_flat=q_flat,
                action_value_long=q_long,
                action_value_margin=q_margin,
                available_actions=avail_actions,
                selected_action=int(candidate_action),
            )
            self.audit_predictions.append(pred.to_dict())
            self.last_prediction_id = pred_id

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
        self.latest_action_values = [
            float(x) for x in action_value_chosen_index[0].detach().cpu().numpy()
        ]
        action = torch.max(action_value_chosen_index, 1)[1].data.cpu().numpy()
        action = int(action[0])

        return action

    def run_single_valid_df(self, df, save_path, contract_name: str | None = None):
        if isinstance(getattr(self, "gating_strategy", None), ExpertHysteresisGating):
            self.reset_routing_state()
        self.df = df
        self.current_contract_name = contract_name or "default_contract"
        self.audit_predictions = []
        self.audit_trades = []
        self.last_prediction_id = ""
        self.latest_action_values = [0.0, 0.0, 0.0]
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
            step_idx = self.step_idx
            old_position = float(env.position)
            action = self.get_action(
                info,
                s,
                env.position,
                env.leverage,
                current_unrealized_pnl=float(env.unrealized_pnl),
                current_markprice=float(env.current_markprice),
            )
            target_position, _ = map_action_to_position_leverage(
                action, self.leverage_choices, self.position_list
            )
            s_, r, done, info = env.step(action)
            reason = (
                self.action_decision_reason_history[-1]
                if self.action_decision_reason_history
                else ActionDecisionReasons.POLICY_INFERENCE
            )
            pred_id = (
                self.last_prediction_id
                if reason in (
                    ActionDecisionReasons.POLICY_INFERENCE,
                    ActionDecisionReasons.ACTION_UNAVAILABLE_BREAK,
                    ActionDecisionReasons.ACTION_PERSISTENCE,
                )
                else ""
            )
            self._record_trade_audit(
                env=env,
                step_index=step_idx,
                action=action,
                old_position=old_position,
                target_position=target_position,
                decision_reason=reason,
                prediction_id=pred_id,
            )
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
            np.save(os.path.join(save_path, HistoryArtifactNames.EXECUTED_POSITION_HISTORY_NPY), env.executed_position_history)
            np.save(os.path.join(save_path, HistoryArtifactNames.COMMISSION_FEE_HISTORY_NPY), env.commission_fee_history)
            np.save(os.path.join(save_path, HistoryArtifactNames.MARKET_INDEX_HISTORY_NPY), env.market_index_history)
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

        run_id = str(self.run_id)
        contract = str(self.current_contract_name)
        audit_accounts: list[dict[str, Any]] = []

        acc_history = env.account_audit_history
        if not acc_history:
            init_eq = float(self.initial_wallet_balance)
            acc_history = [
                {
                    "step_index": 0,
                    "snapshot_type": "initial",
                    "wallet_balance": init_eq,
                    "unrealized_pnl": 0.0,
                    "equity": init_eq,
                    "margin": 0.0,
                    "available_balance": init_eq,
                    "position": 0.0,
                    "exposure": 0.0,
                    "leverage": 1.0,
                    "peak_equity": init_eq,
                    "drawdown": 0.0,
                    "reward": 0.0,
                }
            ]
        for k, snap in enumerate(acc_history):
            if k == 0:
                snap_id = build_account_snapshot_id(run_id, contract, 0, "initial")
                snap_type = "initial"
                step_i = 0
                r_val = 0.0
            else:
                snap_id = build_account_snapshot_id(run_id, contract, k - 1, "post")
                snap_type = "post_step"
                step_i = k
                prev_eq = acc_history[k - 1]["equity"]
                curr_eq = snap["equity"]
                r_val = float(curr_eq - prev_eq)

            row_i = min(k, len(self.df) - 1) if self.df is not None else 0
            e_time = (
                str(self.df["timestamp"].iloc[row_i])
                if self.df is not None and "timestamp" in self.df.columns
                else ""
            )
            t_day = (
                str(self.df["trading_day"].iloc[row_i])
                if self.df is not None and "trading_day" in self.df.columns
                else ""
            )

            acc_rec = AccountRecord(
                snapshot_id=snap_id,
                run_id=run_id,
                contract=contract,
                account_id=f"acct_{contract}",
                account_scope="independent_contract_wallet",
                step_index=step_i,
                snapshot_type=snap_type,
                event_time=e_time,
                trading_day=t_day,
                wallet_balance=float(snap["wallet_balance"]),
                unrealized_pnl=float(snap["unrealized_pnl"]),
                equity=float(snap["equity"]),
                margin=float(snap["margin"]),
                available_balance=float(snap["available_balance"]),
                position=float(snap["position"]),
                exposure=float(snap["exposure"]),
                leverage=float(snap["leverage"]),
                peak_equity=float(snap["peak_equity"]),
                drawdown=float(snap["drawdown"]),
                reward=r_val,
            )
            audit_accounts.append(acc_rec.to_dict())

        if self.save_artifacts:
            with open(
                os.path.join(save_path, AuditArtifactNames.AUDIT_PREDICTIONS_JSON),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(self.audit_predictions, f, indent=2)
            with open(
                os.path.join(save_path, AuditArtifactNames.AUDIT_TRADES_JSON),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(self.audit_trades, f, indent=2)
            with open(
                os.path.join(save_path, AuditArtifactNames.AUDIT_ACCOUNT_SNAPSHOTS_JSON),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(audit_accounts, f, indent=2)
            with open(
                os.path.join(save_path, AuditArtifactNames.AUDIT_METADATA_JSON),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    {
                        "run_id": run_id,
                        "model_id": str(self.model_id),
                        "feature_version": str(self.feature_version),
                        "parameter_version": str(self.parameter_version),
                        "contract": contract,
                    },
                    f,
                    indent=2,
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
        contract_capital = float(self.initial_wallet_balance) if self.initial_wallet_balance > 0 else 6000.0
        self.return_rate = float(reward_sum / contract_capital)
        equities = np.array(env.margine_balance_history, dtype=np.float64)
        if len(equities) > 0:
            peaks = np.maximum.accumulate(equities)
            drawdowns = (peaks - equities) / (peaks + 1e-12)
            contract_mdd = float(np.max(drawdowns))
        else:
            contract_mdd = 0.0

        executed_positions = np.array(env.executed_position_history, dtype=np.float64)
        if len(executed_positions) > 0:
            pos_diffs = np.abs(np.diff(np.r_[0.0, executed_positions]))
            trades_count = int(np.sum(pos_diffs > 1e-6))
        else:
            trades_count = 0

        if self.save_artifacts:
            logger.info(
                "[Artifacts] Saved simulation history to %s | rows: %d, reward_sum: %.4f, require_money: %.4f, return_rate: %.6f, mdd: %.4f, trades: %d",
                save_path,
                len(self.df),
                reward_sum,
                require_money,
                self.return_rate,
                contract_mdd,
                trades_count,
            )
        contract_timestamps = (
            self.df["timestamp"].values
            if self.df is not None and "timestamp" in self.df.columns
            else None
        )
        return {
            "rows": len(self.df),
            "reward_sum": float(reward_sum),
            "require_money": float(require_money),
            "return_rate": float(self.return_rate),
            "mdd": float(contract_mdd),
            "trades": int(trades_count),
            "timestamps": contract_timestamps,
            "step_net_pnls": reward_history,
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
            if self.enable_mdd_penalty:
                single_mdd = float(result.get("mdd", 0.0))
                single_trades = int(result.get("trades", 0))
                target_mdd = float(self.target_mdd)
                penalty_weight = float(self.mdd_penalty_weight)
                min_trades = int(self.min_trades_guard)

                if single_mdd <= target_mdd:
                    mdd_factor = 1.0 + 2.0 * (target_mdd - single_mdd)
                else:
                    excess_mdd = (single_mdd - target_mdd) / (target_mdd + 1e-12)
                    mdd_factor = float(np.exp(-penalty_weight * excess_mdd))

                trade_guard = min(1.0, float(single_trades / min_trades)) if min_trades > 0 and single_trades > 0 else (1.0 if min_trades <= 0 else 0.0)
                base_ret = result["return_rate"]
                if base_ret >= 0:
                    score = float(base_ret * mdd_factor * trade_guard)
                else:
                    score = float(base_ret / max(mdd_factor, 1e-4))
                self.return_rate = score
            else:
                self.return_rate = result["return_rate"]
            logger.info(
                "[Test End] Single dataset test completed | return_rate: %.6f | artifacts: %s",
                self.return_rate,
                self.test_path,
            )
            return self.return_rate

        logger.info(
            "[Test] Found %d contracts to evaluate in stage '%s'.",
            len(contract_files),
            self.eval_stage,
        )
        if self.save_artifacts:
            target_contracts_dir = os.path.join(self.test_path, "contracts")
            if os.path.exists(target_contracts_dir):
                shutil.rmtree(target_contracts_dir)
            os.makedirs(target_contracts_dir, exist_ok=True)
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
        for col_name, default_val in [
            (MetricColumns.MDD, 0.0),
            ("trades", 0),
        ]:
            if col_name not in result_df.columns:
                result_df[col_name] = default_val

        result_df = result_df[
            [
                MetricColumns.CONTRACT,
                MetricColumns.SOURCE_FILE,
                MetricColumns.ROWS,
                MetricColumns.REWARD_SUM,
                MetricColumns.REQUIRE_MONEY,
                MetricColumns.RETURN_RATE,
                MetricColumns.MDD,
                "trades",
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
        portfolio_capital = float(self.initial_wallet_balance * 2.0)
        portfolio_return_rate = float(total_reward_sum / (portfolio_capital + 1e-12))
        win_rate = float((result_df[MetricColumns.RETURN_RATE] > 0).mean())
        base_objective = portfolio_return_rate
        total_trades = int(result_df["trades"].sum())

        portfolio_mdd = None
        has_timestamps = all(
            res.get("timestamps") is not None and res.get("step_net_pnls") is not None
            for res in contract_results
        )
        if has_timestamps and len(contract_results) > 0:
            timeline_ts = [
                res["timestamps"][:len(res["step_net_pnls"])]
                for res in contract_results
                if len(res["step_net_pnls"]) > 0
            ]
            timeline_pnls = [
                res["step_net_pnls"]
                for res in contract_results
                if len(res["step_net_pnls"]) > 0
            ]
            if timeline_ts and timeline_pnls:
                concat_ts = np.concatenate(timeline_ts)
                concat_pnls = np.concatenate(timeline_pnls)
                timeline_df = pd.DataFrame({"ts": concat_ts, "pnl": concat_pnls})
                step_pnls = timeline_df.groupby("ts")["pnl"].sum().sort_index().to_numpy()
                portfolio_capital = self.initial_wallet_balance * 2.0
                equities = portfolio_capital + np.cumsum(step_pnls)
                peaks = np.maximum.accumulate(equities)
                portfolio_mdd = float(np.max((peaks - equities) / (peaks + 1e-12)))

        if portfolio_mdd is None:
            portfolio_mdd = float(result_df[MetricColumns.MDD].max()) if MetricColumns.MDD in result_df else 0.0

        if self.enable_mdd_penalty:
            target_mdd = float(self.target_mdd)
            penalty_weight = float(self.mdd_penalty_weight)
            min_trades = int(self.min_trades_guard)

            if portfolio_mdd <= target_mdd:
                mdd_factor = 1.0 + 2.0 * (target_mdd - portfolio_mdd)
            else:
                excess_mdd = (portfolio_mdd - target_mdd) / (target_mdd + 1e-12)
                mdd_factor = float(np.exp(-penalty_weight * excess_mdd))

            trade_guard = min(1.0, float(total_trades / min_trades)) if min_trades > 0 and total_trades > 0 else (1.0 if min_trades <= 0 else 0.0)
            if base_objective >= 0:
                self.return_rate = float(base_objective * mdd_factor * trade_guard)
            else:
                self.return_rate = float(base_objective / max(mdd_factor, 1e-4))
            aggregation_mode = "option3_hinged_mdd_barrier_and_trades_guard"
        else:
            mdd_factor = 1.0
            trade_guard = 1.0
            self.return_rate = base_objective
            aggregation_mode = "option2_portfolio_return_times_win_rate"

        if self.save_artifacts:
            trading_info = {
                "return_rate": self.return_rate,
                "base_objective": base_objective,
                "portfolio_return_rate": portfolio_return_rate,
                "portfolio_capital": portfolio_capital,
                "contract_capital": float(self.initial_wallet_balance),
                "tr": portfolio_return_rate,
                "win_rate": win_rate,
                "portfolio_mdd": portfolio_mdd,
                "mdd_factor": mdd_factor,
                "total_trades": total_trades,
                "trade_guard": trade_guard,
                "equal_weighted_mean_return": float(result_df[MetricColumns.RETURN_RATE].mean()),
                "total_reward_sum": total_reward_sum,
                "aggregation": aggregation_mode,
                "contract_count": len(contract_results),
            }
            trading_info_path = os.path.join(self.test_path, ArtifactNames.TRADING_INFO_NPY)
            np.save(trading_info_path, trading_info)
            logger.info("[Artifacts] Saved aggregated trading info to %s", trading_info_path)

        logger.info(
            "[Test End] Multi-contract test completed | Contracts: %d | Total Reward Sum: %.4f | "
            "Portfolio Return: %.6f | Win Rate: %.4f (%.1f%%) | Portfolio MDD: %.4f (%.2f%%) | "
            "MDD Factor: %.4f | Total Trades: %d | Final Score: %.6f | Output Dir: %s",
            len(contract_results),
            total_reward_sum,
            portfolio_return_rate,
            win_rate,
            win_rate * 100.0,
            portfolio_mdd,
            portfolio_mdd * 100.0,
            mdd_factor,
            total_trades,
            self.return_rate,
            self.test_path,
        )
        return self.return_rate

    def initial_rollout(self, env: Base_Env, s, info):
        done = False
        r = 0
        rollout_window_length = self.initial_rollout_window_length
        for i in range(rollout_window_length):
            step_idx = self.step_idx
            old_position = float(env.position)
            action = rule_based_close(
                info,
                self.zero_position_action,
                self.leverage_choices,
                self.position_list,
                env.position,
                env.leverage,
            )
            target_position, _ = map_action_to_position_leverage(
                action, self.leverage_choices, self.position_list
            )
            self.action_decision_reason_history.append(
                ActionDecisionReasons.WARM_UP
            )
            s, r, done, info = env.step(action)
            self._record_trade_audit(
                env=env,
                step_index=step_idx,
                action=action,
                old_position=old_position,
                target_position=target_position,
                decision_reason=ActionDecisionReasons.WARM_UP,
                prediction_id="",
            )
            self.step_idx += 1
            vae_idx = min(self.step_idx, len(self.vae_slope_array) - 1)
            vae_s_slope = self.vae_slope_array[vae_idx]
            vae_s_vol = self.vae_vol_array[vae_idx]
            self.get_quantiles(vae_s_slope, vae_s_vol)
            if done:
                break
        return env, s, r, done, info

    def _record_trade_audit(
        self,
        env: Base_Env,
        step_index: int,
        action: int,
        old_position: float,
        target_position: float,
        decision_reason: int,
        prediction_id: str,
    ) -> None:
        run_id = str(self.run_id)
        contract = str(self.current_contract_name)
        decision_id = build_decision_id(run_id, contract, step_index)
        row_idx = min(step_index, len(self.df) - 1) if self.df is not None else 0
        event_time = (
            str(self.df["timestamp"].iloc[row_idx])
            if self.df is not None and "timestamp" in self.df.columns
            else ""
        )
        trading_day = (
            str(self.df["trading_day"].iloc[row_idx])
            if self.df is not None and "trading_day" in self.df.columns
            else ""
        )
        decision_reason_str = ActionDecisionReasons(decision_reason).name
        reference_price = float(getattr(env, "last_reference_price", env.current_markprice))
        wc = getattr(env, "last_wallet_change", None)
        executed_position = float(env.position)
        commission = float(env.commission_fee_step)
        realized_pnl = float(env.realized_pnl_step)
        slippage = float(env.slippage_step)

        if step_index == 0:
            pre_snapshot_id = build_account_snapshot_id(run_id, contract, 0, "initial")
        else:
            pre_snapshot_id = build_account_snapshot_id(run_id, contract, step_index - 1, "post")
        post_snapshot_id = build_account_snapshot_id(run_id, contract, step_index, "post")

        # 1. Reversal: old_position * target_position < 0 and executed_position != old_position
        if old_position * target_position < 0 and executed_position != old_position:
            close_qty = float(getattr(wc, "closed_quantity", abs(old_position))) if wc else float(abs(old_position))
            close_val = float(getattr(wc, "closed_value", 0.0)) if wc else 0.0
            close_fee = float(getattr(wc, "closing_fee", commission / 2.0)) if wc else commission / 2.0
            close_fill_p = (close_val / close_qty) if close_qty > 0 and close_val > 0 else reference_price

            open_qty = float(getattr(wc, "opened_quantity", abs(executed_position))) if wc else float(abs(executed_position))
            open_val = float(getattr(wc, "opened_value", 0.0)) if wc else 0.0
            open_fee = float(getattr(wc, "opening_fee", commission / 2.0)) if wc else commission / 2.0
            open_fill_p = (open_val / open_qty) if open_qty > 0 and open_val > 0 else reference_price

            trade_close = TradeRecord(
                trade_id=build_trade_id(run_id, contract, step_index, leg=0),
                decision_id=decision_id,
                prediction_id=prediction_id,
                run_id=run_id,
                contract=contract,
                step_index=step_index,
                event_time=event_time,
                trading_day=trading_day,
                leg_index=0,
                event_type="reversal_close",
                execution_status="filled",
                decision_reason=decision_reason_str,
                requested_position=0.0,
                executed_position=0.0,
                requested_quantity=float(abs(old_position)),
                executed_quantity=close_qty,
                remaining_quantity=float(max(0.0, abs(old_position) - close_qty)),
                reference_price=reference_price,
                fill_price=close_fill_p,
                commission=close_fee,
                realized_pnl=realized_pnl,
                slippage=slippage / 2.0,
                pre_snapshot_id=pre_snapshot_id,
                post_snapshot_id=post_snapshot_id,
            )
            trade_open = TradeRecord(
                trade_id=build_trade_id(run_id, contract, step_index, leg=1),
                decision_id=decision_id,
                prediction_id=prediction_id,
                run_id=run_id,
                contract=contract,
                step_index=step_index,
                event_time=event_time,
                trading_day=trading_day,
                leg_index=1,
                event_type="reversal_open",
                execution_status="filled",
                decision_reason=decision_reason_str,
                requested_position=float(target_position),
                executed_position=float(executed_position),
                requested_quantity=float(abs(target_position)),
                executed_quantity=open_qty,
                remaining_quantity=float(max(0.0, abs(target_position) - open_qty)),
                reference_price=reference_price,
                fill_price=open_fill_p,
                commission=open_fee,
                realized_pnl=0.0,
                slippage=slippage / 2.0,
                pre_snapshot_id=pre_snapshot_id,
                post_snapshot_id=post_snapshot_id,
            )
            self.audit_trades.append(trade_close.to_dict())
            self.audit_trades.append(trade_open.to_dict())
            return

        # 2. Rejection / price limit blocked
        if target_position != old_position and executed_position == old_position:
            req_qty = float(abs(target_position - old_position))
            trade = TradeRecord(
                trade_id=build_trade_id(run_id, contract, step_index, leg=0),
                decision_id=decision_id,
                prediction_id=prediction_id,
                run_id=run_id,
                contract=contract,
                step_index=step_index,
                event_time=event_time,
                trading_day=trading_day,
                leg_index=0,
                event_type="rejection",
                execution_status="rejected",
                decision_reason=decision_reason_str,
                requested_position=float(target_position),
                executed_position=float(executed_position),
                requested_quantity=req_qty,
                executed_quantity=0.0,
                remaining_quantity=req_qty,
                reference_price=reference_price,
                fill_price=0.0,
                commission=0.0,
                realized_pnl=0.0,
                slippage=0.0,
                pre_snapshot_id=pre_snapshot_id,
                post_snapshot_id=post_snapshot_id,
            )
            self.audit_trades.append(trade.to_dict())
            return

        # 3. Hold
        if old_position == executed_position and old_position == target_position:
            trade = TradeRecord(
                trade_id=build_trade_id(run_id, contract, step_index, leg=0),
                decision_id=decision_id,
                prediction_id=prediction_id,
                run_id=run_id,
                contract=contract,
                step_index=step_index,
                event_time=event_time,
                trading_day=trading_day,
                leg_index=0,
                event_type="hold",
                execution_status="filled",
                decision_reason=decision_reason_str,
                requested_position=float(target_position),
                executed_position=float(executed_position),
                requested_quantity=0.0,
                executed_quantity=0.0,
                remaining_quantity=0.0,
                reference_price=reference_price,
                fill_price=reference_price,
                commission=commission,
                realized_pnl=realized_pnl,
                slippage=slippage,
                pre_snapshot_id=pre_snapshot_id,
                post_snapshot_id=post_snapshot_id,
            )
            self.audit_trades.append(trade.to_dict())
            return

        # 4. Partial fill or full fill
        req_qty = float(abs(target_position - old_position))
        exec_qty = float(abs(executed_position - old_position))

        if old_position == 0:
            ev_type = "open"
        elif target_position == 0:
            ev_type = "close"
        elif abs(target_position) > abs(old_position):
            ev_type = "open"
        else:
            ev_type = "close"

        if ev_type == "open":
            exec_qty = float(getattr(wc, "opened_quantity", exec_qty)) if wc else exec_qty
            exec_val = float(getattr(wc, "opened_value", 0.0)) if wc else 0.0
            commission = float(getattr(wc, "opening_fee", commission)) if wc else commission
            fill_p = (exec_val / exec_qty) if exec_qty > 0 and exec_val > 0 else reference_price
        else:
            exec_qty = float(getattr(wc, "closed_quantity", exec_qty)) if wc else exec_qty
            exec_val = float(getattr(wc, "closed_value", 0.0)) if wc else 0.0
            commission = float(getattr(wc, "closing_fee", commission)) if wc else commission
            fill_p = (exec_val / exec_qty) if exec_qty > 0 and exec_val > 0 else reference_price

        rem_qty = float(max(0.0, req_qty - exec_qty))
        is_partial = rem_qty > 1e-6 and exec_qty > 0

        trade = TradeRecord(
            trade_id=build_trade_id(run_id, contract, step_index, leg=0),
            decision_id=decision_id,
            prediction_id=prediction_id,
            run_id=run_id,
            contract=contract,
            step_index=step_index,
            event_time=event_time,
            trading_day=trading_day,
            leg_index=0,
            event_type=ev_type,
            execution_status="partial_fill" if is_partial else "filled",
            decision_reason=decision_reason_str,
            requested_position=float(target_position),
            executed_position=float(executed_position),
            requested_quantity=req_qty,
            executed_quantity=exec_qty,
            remaining_quantity=rem_qty,
            reference_price=reference_price,
            fill_price=fill_p,
            commission=commission,
            realized_pnl=realized_pnl,
            slippage=slippage,
            pre_snapshot_id=pre_snapshot_id,
            post_snapshot_id=post_snapshot_id,
        )
        self.audit_trades.append(trade.to_dict())


if __name__ == "__main__":
    seed_torch(42)
    args = parser.parse_args()
    logger.info("Starting VAE risk-aware routing test with args: %s", args)
    vae_routing = vae_risk_aware_routing(args)
    final_return_rate = vae_routing.test()
    logger.info("Test finished with final return rate: %.6f", final_return_rate)
