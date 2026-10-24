# 专家路由迟滞：冻结实验配置与最终重载

本页说明 `expert_hysteresis.v1` 的配置与证据格式。数值由独立校准确定；下文占位符不是已经验证的参数、模型或分区。现有 `10min_parallel` 资产仍有 TradingDay 分区污染，且 valid 已用于专家选拔，不能直接认证为本功能的独立实验。

## 准备独立来源

先固定拟合、专家选拔、迟滞校准、最终测试四个时间区段。拟合和专家选拔可以互相重叠，但都必须结束于校准开始前；校准必须结束于最终测试开始前。所有区段采用带时区 ISO 时间的 `[start, end)`。源模型、特征、参考分布、Selection Manifest 与切分证据都必须已存在；本入口不会自动重训或修复污染。

运行目录中的 `train/*.feather` 对应拟合，`valid/*.feather` 对应独立迟滞校准，`test/*.feather` 对应最终测试。专家选拔数据来自 Selection Manifest 的 `valid_root`，它应指向较早的选拔数据，不能指回当前校准 valid。每个输入文件须保留带时区的 `timestamp` 与显式 `trading_day`；后者接受 `YYYY-MM-DD` 或 `YYYYMMDD`。程序检查实际时间范围及 TradingDay 跨边界重用。拟合/选拔文件与校准、最终测试文件都纳入哈希身份。

请使用绝对路径书写证据、Selection Manifest 中的模型和选拔数据引用，保证换工作目录重载时含义不变。Python 命令先激活环境：

```bash
source /home/lanceliang/miniconda3/etc/profile.d/conda.sh
conda activate finetf
export PYTHONPATH="$PWD:$PWD/FineFT"
```

## 输入 experiment_manifest

使用 `--experiment_manifest /absolute/path/experiment.json` 提交以下结构。大写占位符需替换为真实事实，布尔值必须得到来源证据支持；把旧模型标成 true 不能使其恢复样本外资格。

```json
{
  "schema_version": 1,
  "experiment_id": "YOUR_IMMUTABLE_EXPERIMENT_ID",
  "strategy_version": "expert_hysteresis.v1",
  "split_version": "YOUR_REBUILT_SPLIT_VERSION",
  "trading_day_atomic": true,
  "models_retrained_for_split": true,
  "periods": {
    "fit": {"start": "FIT_START_WITH_TIMEZONE", "end": "FIT_END_WITH_TIMEZONE", "source_evidence": {"path": "/absolute/path/fit-evidence.json", "sha256": "ACTUAL_SHA256"}},
    "selection": {"start": "SELECTION_START_WITH_TIMEZONE", "end": "SELECTION_END_WITH_TIMEZONE", "source_evidence": {"path": "/absolute/path/selection-evidence.json", "sha256": "ACTUAL_SHA256"}},
    "calibration": {"start": "CALIBRATION_START_WITH_TIMEZONE", "end": "CALIBRATION_END_WITH_TIMEZONE", "source_evidence": {"path": "/absolute/path/calibration-evidence.json", "sha256": "ACTUAL_SHA256"}},
    "final": {"start": "FINAL_START_WITH_TIMEZONE", "end": "FINAL_END_WITH_TIMEZONE", "source_evidence": {"path": "/absolute/path/final-evidence.json", "sha256": "ACTUAL_SHA256"}}
  },
  "scoring": {
    "protocol_version": "seeded_vae_quantiles.v1",
    "seed": "FIXED_NONNEGATIVE_INTEGER",
    "batch_size": "FIXED_POSITIVE_INTEGER",
    "slope_window_length": "FIXED_POSITIVE_INTEGER",
    "volatility_window_length": "FIXED_POSITIVE_INTEGER",
    "slope_gamma": "FIXED_NUMBER_IN_0_TO_1",
    "volatility_gamma": "FIXED_NUMBER_IN_0_TO_1",
    "architecture": {"z_dim": "ACTUAL_INTEGER", "vae_hidden_dims": ["ACTUAL_LAYER_SIZES"], "loss_type": "ACTUAL_LOSS_TYPE"},
    "column_order": ["slope.label_0", "slope.label_1", "slope.label_2", "volatility.label_0", "volatility.label_1", "volatility.label_2"]
  },
  "artifacts": {"EVERY_REQUIRED_LOGICAL_NAME": {"path": "/absolute/path/actual-file", "sha256": "ACTUAL_SHA256"}}
}
```

`artifacts` 必须恰好覆盖实际运行的文件集合：`selection_manifest`、`specialist_model`、`rl_state_features`、`vae_slope_state_features`、`vae_volatility_state_features`、`maintenance_margin_ratio_dict`；每轴每个标签的 `<axis>.<label>.model` 和 `<axis>.<label>.reference`；以及每个数据角色与合约的 `data.<fit|selection|calibration|final>.<contract>`。轴和标签顺序来自 Selection Manifest，示例的三标签并非另加约束。

可通过只读公共函数 `describe_experiment_inputs(args)` 获取实际 `artifacts` 与 `column_order`，通过 `file_sha256(path)` 计算证据哈希。这些函数只清点文件，不生成独立性证明。`args` 是公开基础 parser 解析得到的配置，包含数据、VAE、Selection Manifest 路径。模型版本由实际内容哈希锁定，不能仅用会被覆写的 `model_latest.pth` 文件名充当版本。

## 准备、导出、重载

公开基础 parser 接受 `--gating_strategy expert_hysteresis`、`--strategy_version expert_hysteresis.v1`、两轴 `--*_rule_base_threshold`（E）、`--hysteresis_exit_ratio`、`--expert_switch_lead`（Δ）、`--expert_confirm_steps`（K），以及两轴 `--*_window_length`、`--*_gamma`。新策略要求明确两轴 E 与评分设置。H 由 `exit_ratio × E` 派生并写入配置；合法关系为 `0 <= H < E <= 1`、`0 < exit_ratio < 1`、有限非负 Δ、正整数 K。

公共调参准备入口 `vae_routing_optuna.prepare_base_args(base_args, search_args)` 会调用 `prepare_experiment(args, phase="calibration")`，验证证据、时间、版本及固定评分。`expert_hysteresis` 的公开调参入口只采样五个策略维度：两轴 E、`exit_ratio`、共享 `Δ` 和正整数 `K`。两轴评分窗口和 gamma、模型、参考分布、缓存、账户及执行/风控设置在 trial 间保持不变。每个轻量 trial 将策略版本、固定评分及派生 H 写入 Optuna user attributes；它不会写完整轨迹或日度审计。

新策略必须明确给出五个维度的搜索边界，避免把历史 absolute 的评分搜索误用于迟滞校准。以下命令中的上下界均为待研究者依据独立校准设计提供的占位符，并非推荐或已验证的数值：

```bash
python -u FineFT/RL/DiHFT/high_level/vae_routing_optuna.py \
  --gating_strategy expert_hysteresis \
  --experiment_manifest /absolute/path/experiment.json \
  --selection_manifest /absolute/path/two_dimensional_selection_manifest.json \
  --slope_window_length FIXED_SLOPE_WINDOW \
  --volatility_window_length FIXED_VOLATILITY_WINDOW \
  --slope_gamma FIXED_SLOPE_GAMMA \
  --volatility_gamma FIXED_VOLATILITY_GAMMA \
  --slope_rule_base_threshold FIXED_INITIAL_SLOPE_E \
  --volatility_rule_base_threshold FIXED_INITIAL_VOLATILITY_E \
  --expert_slope_entry_threshold_min CALIBRATION_SLOPE_E_MIN \
  --expert_slope_entry_threshold_max CALIBRATION_SLOPE_E_MAX \
  --expert_volatility_entry_threshold_min CALIBRATION_VOLATILITY_E_MIN \
  --expert_volatility_entry_threshold_max CALIBRATION_VOLATILITY_E_MAX \
  --hysteresis_exit_ratio_min CALIBRATION_RATIO_MIN \
  --hysteresis_exit_ratio_max CALIBRATION_RATIO_MAX \
  --expert_switch_lead_min CALIBRATION_DELTA_MIN \
  --expert_switch_lead_max CALIBRATION_DELTA_MAX \
  --expert_confirm_steps_min CALIBRATION_K_MIN \
  --expert_confirm_steps_max CALIBRATION_K_MAX
```

入口拒绝缺失或非法边界：两轴 E 必须满足 `0 < min <= max <= 1`，ratio 满足 `0 < min <= max < 1`，Δ 满足 `0 <= min <= max`，K 为正整数且 `min <= max`。完成搜索后，入口将最优 trial 的全部五个值重新应用到新路由实例，先写入不可覆盖的 `selected_expert_hysteresis_config.json`，再在校准段执行恰好一次带完整产物的回放。该回放是校准证据，不是最终测试；最终测试仍须用下面的冻结配置重载入口。

选定参数后，先应用最优参数，再调用：

```python
from RL.DiHFT.high_level.frozen_experiment import export_frozen_experiment
export_frozen_experiment(selected_args, "/absolute/path/selected_config.json")
```

导出保存完整策略、版本、两轴 E/H、退出比率、Δ/K、固定评分、模型架构、账户、动作空间、费用与风控参数，以及实验身份和配置身份。相同路径允许重复写入完全相同内容；不能用新参数覆盖已有冻结文件。最终测试不得反向改变该选择。

最终重载使用独立输出目录，避免覆盖历史产物：

```bash
python -u FineFT/RL/DiHFT/high_level/vae_routing_final_result_macro_action.py \
  --frozen_config /absolute/path/selected_config.json \
  --eval_stage test \
  --output_sub_dir YOUR_NEW_FINAL_OUTPUT_DIRECTORY
```

最终入口 `parse_and_prepare_args(...)` 不会为冻结配置自动发现旧 para/Optuna CSV，也拒绝显式混用。显式覆盖已冻结的策略、评分、账户、费用或风控值会报错。最终回放后再次检查配置及来源，保存 `frozen_evaluation_receipt.json`、`frozen_config.json`、`experiment_manifest.json`；receipt 关联最终评分缓存哈希。

## 评分缓存与独立状态

新策略使用 `ensure_precomputed_vae_quantiles(args)` 生成一次固定 seed、批大小和稳定合约/轴/标签顺序的评分，随后复用只读 `(T, labels × 2)` float32 缓存。`quantile_cache_manifest.json` 记录 `schema_version`、实验身份、完整评分协议、评估阶段、各合约缓存形状和内容哈希。既有无来源缓存、部分缓存、文件变化、错维度或非有限/越界分数会阻断正式运行；不要为旧缓存手工补一个“通过”标记。

缓存只有评分，不保存当前专家、候选或确认次数；每个合约 rollout 与重配置重建独立路由状态。VAE 本身在 eval 模式仍有随机采样，因此比较策略时需复用同一冻结缓存，不能把两次不同的随机评分解释成路由收益。

历史 `absolute`、`hierarchical` 的文本参数和 CSV 解析保持原语义。它们可以用于历史诊断对照；独立研究结论仍须遵守 ADR-0060 的修正分区、模型重训、共享执行账户与正式计量要求。本页的证据门禁不宣称这些尚未完成的前提已经得到满足。
