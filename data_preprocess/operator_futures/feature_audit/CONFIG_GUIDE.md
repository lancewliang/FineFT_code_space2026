# 多预测期限特征审计（Multi-Horizon Feature Audit）配置与参数说明文件

本文档详细说明 `data_preprocess/operator_futures/feature_audit/` 独立审计模块使用的配置文件及相关 Manifest 输入文件的参数规范、语义定义与取值要求。

---

## 1. 配置文件整体结构

审计模块采用 YAML / JSON 格式配置，分为两类配置文件：
1. **单次审计运行配置（Audit Config）**：驱动单一频率（如 10min 或 30min）在七个预测期限上的完整特征审计。
2. **跨频率/跨期限对比配置（Comparison Config）**：驱动对两个已完成的审计运行进行共同有效样本的横向比对。

---

## 2. 审计运行配置（Audit Config）参数详解

模板路径：`data_preprocess/operator_futures/feature_audit/configs/audit_fu_10min_template.yaml`

### 2.1 核心元数据与协议版本

| 参数名 | 类型 | 必填 | 默认值 / 推荐值 | 说明 |
|---|---|---|---|---|
| `schema_version` | string | 是 | `multi_horizon_audit.v1` | 配置文件结构版本，用于保障模式兼容性。 |
| `protocol_version` | string | 是 | `multi_horizon_audit.research.v1` | 研究评估协议版本。当统计口径、置信区间算法或期限定义调整时须更新。 |
| `symbol` | string | 是 | `fu` | 期货品种代码（如燃料油 `fu`）。 |
| `frequency` | string | 是 | `10min` / `30min` | 特征数据采样频率。当前支持 `10min` 与 `30min`。 |

### 2.2 预测期限与目标定义

| 参数名 | 类型 | 必填 | 默认值 / 推荐值 | 说明 |
|---|---|---|---|---|
| `horizons` | list[string] | 是 | `[1h, 2h, 4h, today, next_td, next_5td, next_15td]` | 待评估的预测期限列表。包含 3 个有效交易时间小时期限和 4 个交易日终点期限。 |
| `hour_clock` | string | 否 | `active_trading_minutes` | 小时期限计时方式。`active_trading_minutes` 按交易日历累计实际开盘分钟数，自动跳过午休、休市和节假日。 |
| `trading_day_offsets` | dict | 否 | `{"today": 0, "next_td": 1, "next_5td": 5, "next_15td": 15}` | 交易日收盘偏移量。以当前决策时点所属 TradingDay 为 D，目标锚定为 D、D+1、D+5、D+15 的日盘收盘。 |
| `targets` | list[string] | 否 | `[return, abs_return]` | 评估目标类别。`return` 为带符号终点方向收益率；`abs_return` 为终点绝对收益率幅度。 |
| `contract_policy` | string | 否 | `same_contract` | 合约绑定策略。严格限定同一真实合约在期限两端匹配，禁止主力换月跨合约拼接。 |
| `candidate_policy` | string | 否 | `causal_quality_only` | 特征候选准入策略。仅纳入具备因果可用性与数值质量的特征，不受旧三流专属黑名单和统计门禁限制。 |

### 2.3 边界安全与隔离保护参数

| 参数名 | 类型 | 必填 | 默认值 / 推荐值 | 说明 |
|---|---|---|---|---|
| `read_final_test` | bool | 否 | `false` | 是否封存最终 Test 集。为防止前瞻偏差，研发探索期间强制为 `false`。 |
| `purge_by_label_available_at` | bool | 否 | `true` | 是否根据标签真实成熟时点执行样本边界净化（Purge），避免切分边界泄漏未来信息。 |
| `write_selection_outputs` | bool | 否 | `false` | 是否回写特征选择结果。审计模块保持纯只读独立性，此项必须为 `false`。 |
| `result_root` | string | 是 | `analysis_result/DiHFT/feature_horizon_audit/fu/10min` | 审计结果产物持久化根目录。 |
| `legacy_selection_manifest` | string | 否 | `null` | 可选的历史三流特征选择结果路径，仅作为对比注释使用，不参与特征过滤。 |

### 2.4 参考价协议（`reference_price_protocol`）

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `field` | string | 否 | `LastPrice` | 参考价字段名，通常使用细粒度快照行情的最新成交价。 |
| `observation` | string | 否 | `latest_valid_snapshot_at_or_before_target` | 价格观察规则：取目标时点或之前的最后一条有效快照。 |
| `same_contract` | bool | 否 | `true` | 参考价是否必须与特征观察点来自相同合约代码。 |
| `same_session` | bool | 否 | `true` | 参考价是否必须属于同一交易 Session。 |
| `max_quote_age_seconds` | int | 否 | `60` | 快照最大允许新鲜度年龄（秒）。超过 60 秒的快照报价被判定为失效（`STALE_PRICE`），不使用中间价降级。 |
| `fallback` | string | 否 | `none` | 缺价降级策略。`none` 表示绝不使用收盘价、均价或相邻合约伪造价格。 |

### 2.5 统计与不确定性协议（`evidence_protocol`）

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `method` | string | 否 | `joint_trading_day_moving_block_bootstrap` | 不确定性评估算法。按品种 TradingDay 进行联合移动区块重抽样（MBB），保留跨合约横截面相关性与序列相关性。 |
| `confidence_level` | float | 否 | `0.95` | 双侧置信区间置信水平（95%）。 |
| `replicates` | int | 否 | `2000` | Bootstrap 重抽样次数（单元测试中可设为 50 加速）。 |
| `seed` | int | 否 | `42` | 随机数发生器种子，确保结果确定性可复现。 |
| `minimum_block_trading_days` | int | 否 | `20` | 基准重抽样区块长度（交易日数）。 |
| `sensitivity_block_multiplier`| int | 否 | `2` | 区块长度敏感性分析乘数（如 `20 * 2 = 40` 天），用于稳健性检验。 |
| `minimum_effective_blocks` | int | 否 | `20` | 允许计算置信区间的最小有效区块数，不足时输出 `INSUFFICIENT_DATA`。 |
| `minimum_contracts_for_sign_consistency` | int | 否 | `3` | 计算符号一致性所需的最小有效合约数，不足时标记明确原因。 |

---

## 3. 关联 Manifest 输入文件格式与参数要求

配置文件中引用的 4 个 Manifest 文件路径是运行的核心输入依赖，其格式定义如下：

### 3.1 交易日历清单（`calendar_manifest`）
- **文件格式**：JSON 或 YAML
- **核心结构**：
  ```json
  {
    "symbol": "fu",
    "trading_days": [
      {
        "trading_day": "20230103",
        "sessions": [
          {"name": "m1", "start": "2023-01-03 09:00:00", "end": "2023-01-03 10:15:00"},
          {"name": "m2", "start": "2023-01-03 10:30:00", "end": "2023-01-03 11:30:00"},
          {"name": "aft", "start": "2023-01-03 13:30:00", "end": "2023-01-03 15:00:00"},
          {"name": "night", "start": "2023-01-03 21:00:00", "end": "2023-01-03 23:00:00"}
        ]
      }
    ]
  }
  ```
- **字段要求**：
  - `trading_day`: 交易所业务交易日（`YYYYMMDD`）。注意夜盘归属于下一个自然日所在的交易日。
  - `sessions`: 当日所属的全部有效交易小节，包含 `start`、`end` 及小节名称。

### 3.2 候选特征数据清单（`feature_availability_manifest`）
- **文件格式**：JSON 或 YAML
- **核心结构**：
  ```json
  {
    "symbol": "fu",
    "data_path": "PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/10min/fu/train"
  }
  ```
- **字段要求**：
  - `data_path`: 存放各合约特征 Feather / Parquet 文件的目录。每个文件包含 `timestamp`、`contract` 及数值特征列。

### 3.3 参考价源清单（`price_source_manifest`）
- **文件格式**：JSON 或 YAML
- **核心结构**：
  ```json
  {
    "price_file": "path/to/fu_snapshots.feather",
    "price_field": "LastPrice",
    "max_quote_age_seconds": 60
  }
  ```
- **字段要求**：
  - `price_file`: 包含高频报价快照数据的 Feather 文件，包含 `timestamp`、`contract` 及 `LastPrice`。
  - `price_field`: 参考价格所使用的列名。
  - `max_quote_age_seconds`: 允许最大行情回看时间窗。

### 3.4 数据集切分清单（`split_manifest`）
- **文件格式**：JSON 或 YAML
- **核心结构**：
  ```json
  {
    "train_contracts": ["fu2305", "fu2309"],
    "valid_contracts": ["fu2401"],
    "test_contracts": ["fu2405"]
  }
  ```
- **字段要求**：
  - `train_contracts`: 参与训练/开发阶段评估的合约集合。
  - `test_contracts`: 封存合约列表。在 `read_final_test=false` 时严禁读取。

---

## 4. 跨运行比对配置（Comparison Config）参数详解

模板路径：`data_preprocess/operator_futures/feature_audit/configs/comparison_fu_10min_vs_30min_template.yaml`

| 参数名 | 类型 | 必填 | 默认值 / 示例 | 说明 |
|---|---|---|---|---|
| `schema_version` | string | 是 | `multi_horizon_audit.comparison.v1` | 比对配置模式版本。 |
| `protocol_version` | string | 是 | `multi_horizon_audit.research.v1` | 研究协议版本，要求待比对的两个 run 协议版本必须一致。 |
| `comparison_type` | string | 是 | `matched_frequency` / `matched_horizon` | 比对模式：`matched_frequency` 为跨频率（10min vs 30min）；`matched_horizon` 为同频率跨期限。 |
| `run_a` | string | 是 | `analysis_result/DiHFT/feature_horizon_audit/fu/10min` | 基准运行 A 的输出根目录（需含有 `_SUCCESS` 及标签产物）。 |
| `run_b` | string | 是 | `analysis_result/DiHFT/feature_horizon_audit/fu/30min` | 对照运行 B 的输出根目录。 |
| `output_dir` | string | 是 | `analysis_result/DiHFT/feature_horizon_audit/fu/comparison_10_vs_30` | 比对结果报告及清单的输出目录。 |

---

## 5. CLI 命令调用指引

审计引擎通过 `data_preprocess.operator_futures.feature_audit.cli` 执行：

```bash
# 激活工程环境
conda activate finetf

# 1. 静态前置校验（检查配置与 Manifest 文件有效性）
python -m data_preprocess.operator_futures.feature_audit.cli validate-inputs --config data_preprocess/operator_futures/feature_audit/configs/audit_fu_10min_template.yaml

# 2. 运行特征审计
python -m data_preprocess.operator_futures.feature_audit.cli run --config data_preprocess/operator_futures/feature_audit/configs/audit_fu_10min_template.yaml

# 3. 运行配对比较
python -m data_preprocess.operator_futures.feature_audit.cli compare --manifest data_preprocess/operator_futures/feature_audit/configs/comparison_fu_10min_vs_30min_template.yaml
```
