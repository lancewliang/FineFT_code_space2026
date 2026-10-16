# 多预测期限特征审计（Multi-Horizon Feature Audit）配置与参数说明文件

本文档详细说明 `data_preprocess/operator_futures/feature_audit/` 独立审计模块使用的配置文件及相关 Manifest 输入文件的参数规范、语义定义、多合约配置方式与取值要求。

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
| `symbol` | string | 是 | `fu` | 期货品种代码（如燃料油 `fu`）。审计引擎以此识别品种并关联日历与数据池。 |
| `frequency` | string | 是 | `10min` / `30min` | 特征数据采样频率。当前支持 `10min` 与 `30min`。 |

### 2.2 预测期限与目标定义

| 参数名 | 类型 | 必填 | 默认值 / 推荐值 | 说明 |
|---|---|---|---|---|
| `horizons` | list[string] | 是 | `[1h, 2h, 4h, today, next_td, next_5td, next_15td]` | 待评估的预测期限列表。包含 3 个有效交易时间小时期限和 4 个交易日终点期限。 |
| `hour_clock` | string | 否 | `active_trading_minutes` | 小时期限计时方式。`active_trading_minutes` 按交易日历累计实际开盘分钟数，自动跳过午休、休市和节假日。 |
| `trading_day_offsets` | dict | 否 | `{"today": 0, "next_td": 1, "next_5td": 5, "next_15td": 15}` | 交易日收盘偏移量。以当前决策时点所属 TradingDay 为 D，目标锚定为 D、D+1、D+5、D+15 的日盘收盘。 |
| `targets` | list[string] | 否 | `[return, abs_return]` | 评估目标类别。`return` 为带符号终点方向收益率；`abs_return` 为终点绝对收益率幅度。 |
| `contract_policy` | string | 否 | `same_contract` | **合约绑定策略**。严格限定同一真实合约在期限两端匹配，禁止主力换月跨合约拼接伪造收益。 |
| `candidate_policy` | string | 否 | `causal_quality_only` | 特征候选准入策略。仅纳入具备因果可用性与数值质量的特征，不受旧三流专属黑名单和统计门禁限制。 |

### 2.3 边界安全与隔离保护参数

| 参数名 | 类型 | 必填 | 默认值 / 推荐值 | 说明 |
|---|---|---|---|---|
| `read_final_test` | bool | 否 | `false` | 是否封存最终 Test 集。为防止前瞻偏差，研发探索期间强制为 `false`。此时所有属于 `test_contracts` 的合约数据严禁读取。 |
| `purge_by_label_available_at` | bool | 否 | `true` | 是否根据标签真实成熟时点执行样本边界净化（Purge），避免切分边界泄漏未来信息。 |
| `write_selection_outputs` | bool | 否 | `false` | 是否回写特征选择结果。审计模块保持纯只读独立性，此项必须为 `false`。 |
| `result_root` | string | 是 | `analysis_result/DiHFT/feature_horizon_audit/fu/10min` | 审计结果产物持久化根目录。 |
| `legacy_selection_manifest` | string | 否 | `null` | 可选的历史三流特征选择结果路径，仅作为对比注释使用，不参与特征过滤。 |

`legacy_selection_manifest` 支持原选择器的嵌套三流结构：
`vae_slope_stream.selected_features`、`vae_volatility_stream.selected_features`、
`rl_stream.selected_features`，以及旧的顶层 `*_features` 格式。
显式配置的文件缺失或格式不受支持时会报错；省略该配置仍可独立运行审计。
同一特征的多流归属用逗号连接，原始清单不被修改。

运行同时输出 `legacy_selection_comparison.csv`，记录入选状态、多流归属、
按流保留的明确 `* Dropped` 原因及源文件 SHA-256；`Hard Filter` 和
`Composite Score` 是阶段保留名单，不作为剔除原因。没有明确记录的原因标为
`unknown`，不能由“未入选”推断“短期限 IC 不合格”。这项配置不控制 CI 或共同样本比较。

### 2.4 参考价协议（`reference_price_protocol`）

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `field` | string | 否 | `LastPrice` | 参考价字段名，通常使用细粒度快照行情的最新成交价（或 `close`）。 |
| `observation` | string | 否 | `latest_valid_snapshot_at_or_before_target` | 价格观察规则：取目标时点或之前的最后一条有效快照。 |
| `same_contract` | bool | 否 | `true` | **同合约约束**。参考价必须与特征观察点来自完全相同的合约代码，严禁借用相邻合约价格。 |
| `same_session` | bool | 否 | `true` | 参考价是否必须属于同一交易 Session。 |
| `max_quote_age_seconds` | int | 否 | `60` | 快照最大允许新鲜度年龄（秒）。超过阈值的快照报价被判定为失效（`STALE_PRICE`），不使用中间价降级。 |
| `fallback` | string | 否 | `none` | 缺价降级策略。`none` 表示绝不使用收盘价、均价或相邻合约伪造价格。 |

### 2.5 统计与不确定性协议（`evidence_protocol`）

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `method` | string | 否 | `joint_trading_day_moving_block_bootstrap` | 不确定性评估算法。**按品种 TradingDay 进行联合移动区块重抽样（MBB）**，重抽样某一天时同时打包当日所有活跃合约样本，保留跨合约横截面相关性与时序自相关性。 |
| `confidence_level` | float | 否 | `0.95` | 双侧置信区间置信水平（95%）。 |
| `replicates` | int | 否 | `2000` | Bootstrap 重抽样次数（单元测试中可设为 50 加速）。 |
| `seed` | int | 否 | `42` | 随机数发生器种子，确保结果确定性可复现。 |
| `minimum_block_trading_days` | int | 否 | `20` | 基准重抽样区块长度（交易日数）。 |
| `sensitivity_block_multiplier`| int | 否 | `2` | 区块长度敏感性分析乘数（如 `20 * 2 = 40` 天），用于稳健性检验。 |
| `minimum_effective_blocks` | int | 否 | `20` | 允许计算置信区间的最小有效区块数，不足时输出 `INSUFFICIENT_DATA`。 |
| `minimum_contracts_for_sign_consistency` | int | 否 | `3` | **计算符号一致性所需的最小有效合约数**。有效合约不足时标记为 `INSUFFICIENT_CONTRACTS_FOR_SIGN_CONSISTENCY`。 |

---

## 3. 商品期货多合约（Multi-Contract）体系配置与工作机制

商品期货的核心特征是**多合约轮换交割、生命周期有限、跨合约流动性与基差分化**。因此，特征审计模块在底层逻辑与配置架构上均严格以“多合约”为第一公民（First-Class Citizen）设计。

### 3.1 核心设计原则：宏观多合约联合分析 vs 微观单合约收益闭环

1. **宏观多合约联合分析**：
   - 商品期货单一交割合约通常只有几个月至一年的活跃期。如果只审计单个合约，样本时间跨度过短且无法验证特征对不同交割月份（如 01、05、09 月合约）的泛化能力。
   - 模块必须同时载入该品种下的全部历史真实合约，综合评估特征在跨合约上的普遍有效性。
2. **微观单合约收益闭环（`contract_policy: same_contract`）**：
   - 在计算任一预测期限（如 1h、today、next_5td）的收益率标签时，**起点价格与终点价格必须完全来自同一真实合约代码**。
   - **严禁跨月拼接**：当主力合约换月时，绝不能将下一个主力合约的价格作为当前合约的终点来计算收益率。拼接会引入人为的基差跳空与虚假收益，严重扭曲特征评价。
   - **到期退市保护**：若某一合约在目标预测期限到达前已停止交易或缺乏行情，判定为 `CONTRACT_EXPIRED` 或 `MISSING_TARGET_PRICE`，作明确的无效样本隔离，而不是借用相邻月份合约伪造价格。

### 3.2 多合约特征数据目录配置与自动扫描（`feature_availability_manifest`）

用户**不需要**在主配置文件里逐个枚举合约代码列表，而是通过目录自动扫描完成多合约加载：

- **目录结构规范**：
  在 `feature_availability_manifest.json` 中指定 `data_path`，指向存放各合约独立 Feather 文件的目录：
  ```
  PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/10min/fu/train/
  ├── fu2305.feather
  ├── fu2309.feather
  ├── fu2310.feather
  ├── fu2311.feather
  ├── fu2312.feather
  ├── fu2401.feather
  └── ...
  ```
- **合约字段与文件规范**：
  - 引擎会自动发现读取 `data_path` 下全部 `*.feather` 文件。
  - 每个文件内部必须包含：
    - `contract`（string 列，如 `"fu2309"`）：明确指示样本所属的具体交割合约。
    - `timestamp`（Datetime 或 `"YYYY-MM-DD HH:MM:SS"` 字符串）：特征决策时点。
    - 各候选特征列（Float / Int 数值列）。

### 3.3 数据集切分与测试集合约隔离封存（`split_manifest`）

为了防止未来合约数据泄漏，模块提供严格的测试集合约物理封存保护。支持以下两种配置格式：

#### 格式 A：审计专用精简清单（推荐）
直接在 `split_manifest.json` 中列出合约划分列表：
```json
{
  "symbol": "fu",
  "train_contracts": ["fu2305", "fu2309", "fu2310", "fu2311", "fu2312", "fu2401"],
  "valid_contracts": ["fu2402", "fu2403", "fu2404", "fu2405"],
  "test_contracts": ["fu2406", "fu2409", "fu2411", "fu2501"]
}
```

#### 格式 B：标准 Stage 6 工程清单兼容格式（`dataset_split_manifest.json`）
可以直接复用预处理阶段生成的切分清单：
```json
{
  "symbol": "fu",
  "target_freq": "10min",
  "sets": {
    "train": {
      "contracts": [{"contract": "fu2305"}, {"contract": "fu2309"}]
    },
    "valid": {
      "contracts": [{"contract": "fu2402"}, {"contract": "fu2403"}]
    },
    "test": {
      "contracts": [{"contract": "fu2406"}, {"contract": "fu2409"}]
    }
  }
}
```

- **封存执行机制**：
  当主配置中 `read_final_test: false`（研发探索期默认）时，引擎会在文件扫描阶段自动比对文件名（`stem`）与数据内的 `contract` 列，**彻底排除所有命中 `test_contracts` 的数据**，物理杜绝未来测试集前瞻偏差。

### 3.4 多合约参考价源配置与文件内容规范（`price_source_manifest`）

#### 3.4.1 `price_file` 的核心定位与上游现状
- **为什么需要独立的参考价源？**
  - 在七个预测期限（1h、2h、4h、today、next_td 等）评估中，决策时刻是确定的业务时点，收益率标签需要锚定目标终点时刻的真实市场价格。
  - 按照研讨规范（Q11），参考价应独立于特征生成逻辑，采用无偏、可追溯的行情快照，绝不在特征表内部通过 `shift(-n)` 等方式粗糙构造。
- **当前上游为何没有单独产出此文件？**
  - 上游流水线（Stage 1～6）专注于生成包含 1300+ 维特征工程宽表（如 `train.feather` 或 `train/fu*.feather`），并没有专门切分出只含价格的独立行情参考产物。
  - 因此，用户需要根据实际数据条件，选择**方案 A（复用现有 Stage 6 Bar 数据）**或**方案 B（从原始高频快照/Tick 数据抽取）**来提供 `price_file`。

#### 3.4.2 `price_file` 的文件格式与 Schema 规范
`price_file` 支持 Feather (`.feather`)、Parquet (`.parquet`)、CSV (`.csv`) 单文件，或包含各合约 Feather 文件的目录路径。

文件内部必须且仅需包含以下 **3 列核心字段**：

| 列名 | 数据类型 | 允许空值 | 示例值 | 语义与要求 |
|---|---|---|---|---|
| `contract` | String | 否 | `fu2305` | 真实交割合约代码。必须覆盖所有需要审计的交割合约。 |
| `timestamp` | Datetime[μs] 或 String (`YYYY-MM-DD HH:MM:SS`) | 否 | `2023-01-03 09:10:00` | 报价产生的时间戳（无时区标准北京时间）。 |
| `LastPrice` 或 `close` | Float64 | 否 | `2764.0` | 对应时点的参考价格。必须为严格大于 0 的正有限数。列名与配置中的 `price_field` 一致。 |

> **数据样本示例**：
> ```
> ┌──────────┬─────────────────────┬─────────────┐
> │ contract ┆ timestamp           ┆ LastPrice   │
> │ ---      ┆ ---                 ┆ ---         │
> │ str      ┆ datetime[μs]        ┆ f64         │
> ╞══════════╪═════════════════════╪═════════════╡
> │ fu2305   ┆ 2023-01-03 09:00:00 ┆ 2764.0      │
> │ fu2305   ┆ 2023-01-03 09:10:00 ┆ 2759.0      │
> │ fu2309   ┆ 2023-01-03 09:00:00 ┆ 2680.0      │
> └──────────┴─────────────────────┴─────────────┘
> ```

#### 3.4.3 两种构建 `price_file` 的落地实施方案

##### 方案 A：基于现有 Stage 6 产物快速提取（零新依赖，推荐日常使用）
直接复用 Stage 6 已有的 10min 降采样数据（例如 `PREPROCESS_DATASET/.../10min/fu/train/` 下各合约文件），仅提取 `['contract', 'timestamp', 'close']` 3 列，保存为独立的 Feather 文件。

- **快速生成脚本**：
  ```python
  from pathlib import Path
  import polars as pl

  # 1. 指定 Stage 6 特征目录与输出路径
  input_dir = Path("PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/10min/fu/train")
  output_price_file = Path("data_preprocess/operator_futures/feature_audit/data/fu_prices_10min.feather")
  output_price_file.parent.mkdir(parents=True, exist_ok=True)

  # 2. 批量读取各合约并提取价格三列（4.6 万行仅需不到 0.1 秒）
  feather_files = sorted(list(input_dir.glob("*.feather")))
  frames = [pl.read_ipc(f).select(["contract", "timestamp", "close"]) for f in feather_files]
  price_df = pl.concat(frames).sort(["contract", "timestamp"])

  # 3. 持久化为轻量参考价文件（体积通常仅几百 KB）
  price_df.write_ipc(output_price_file)
  print(f"成功物化多合约参考价文件: {output_price_file}, 共 {len(price_df)} 行")
  ```

- **对应的 `price_source_manifest.json` 与参数配置**：
  ```json
  {
    "price_file": "data_preprocess/operator_futures/feature_audit/data/fu_prices_10min.feather",
    "price_field": "close",
    "max_quote_age_seconds": 600
  }
  ```
  > ⚠️ **关键注意**：使用 10min Bar 数据作为参考价时，`max_quote_age_seconds` **必须设置为 600 秒**（即 10 分钟）。若误设为默认的 60 秒，会导致 10min 步长的所有报价被误判为 `STALE_PRICE`（陈旧报价），导致标签大面积失效。

##### 方案 B：基于高频快照/Tick 原始行情抽取（高精度生产基准）
从原始 CTP Level-1 行情源抽取全合约的高频快照记录，包含 `InstrumentID`（重命名为 `contract`）、`UpdateTime`（重命名为 `timestamp`）与 `LastPrice`，保存为 `fu_snapshots.feather`。

- **对应的 `price_source_manifest.json` 配置**：
  ```json
  {
    "price_file": "data/snapshots/fu_snapshots.feather",
    "price_field": "LastPrice",
    "max_quote_age_seconds": 60
  }
  ```
  > 当快照到达频率达到秒级或毫秒级时，`max_quote_age_seconds` 可严格遵循 60 秒约束。

#### 3.4.4 引擎内部索引与同合约二分查找
- 引擎载入 `price_file`（或目录）后，在内存中按 `contract` 自动切分构建微秒级时间戳排序数组：`_contract_data[contract] = (sorted_timestamps, prices)`。
- 对于每个决策时点和预测目标时点，引擎在**相同合约**内执行 $O(\log N)$ 二分查找，定位小于等于目标时点的最新有效快照，并严格检验报价新鲜度与正有限性。

### 3.5 多合约统计指标与聚合策略

在完成各合约独立标签计算后，模块采用多层次跨合约统计聚合：

1. **合约独立评估（Per-Contract Metrics）**：
   对每个有效样本对数 >= 3 的合约分别独立计算全特征与目标收益的 Spearman 秩相关系数矩阵 IC_c。
2. **合约等权平均（`contract_equal_spearman`，主汇总指标）**：
   对所有符合样本量要求的有效合约计算 Spearman 秩相关的未加权算术均值：
   `IC_contract_equal = (1 / C) * sum(IC_c)`
   - **设计目的**：在商品期货中，主力合约的样本量通常数倍于非主力合约。若直接混在一起计算全样本相关性，统计结果会被极少数主力合约完全支配。因此，**合约等权平均是评估特征跨合约普遍有效性的主指标**。
3. **样本量加权平均（`sample_weighted_spearman`，敏感性指标）**：
   以各合约的有效样本对数 N_c 为权重计算加权均值：
   `IC_sample_weighted = sum(N_c * IC_c) / sum(N_c)`
   - 作为敏感性对照，观察在高流动性高密度样本下特征表现的偏离程度。
4. **跨合约符号一致性（`sign_consistency`）**：
   统计各合约独立相关系数方向与主汇总方向相同的合约比例：
   `SignConsistency = (1 / C) * sum(I(sign(IC_c) == sign(IC_contract_equal)))`
   - **门禁参数**：`minimum_contracts_for_sign_consistency: 3`。若有效合约数不足 3 个，该指标安全返回 `null` 并标记原因为 `INSUFFICIENT_CONTRACTS_FOR_SIGN_CONSISTENCY`，防止在极少数合约上虚夸稳定性。
5. **多合约联合交易日移动区块 Bootstrap（`joint_trading_day_moving_block_bootstrap`）**：
   - 商品期货的多合约具有显著的**同日横截面联动性**（同品种多个交割合约受共同基本面供需和宏观情绪驱动，不能当成相互独立的观测点）。
   - 重抽样以**日历交易日（`trading_day`）**为区块单位。每次抽样某一日时，**同时打包该交易日下所有并发活跃合约的所有样本点**。
   - 此机制完整保留了同日多合约横截面相关性与跨日自相关性，给出的 95% 置信区间最符合真实统计有效性。

### 3.6 多合约审计产物结构与解读

审计运行完成后，在 `result_root` 目录下输出以下多合约分析产物：

| 产物文件名 | 说明与合约维度 |
|---|---|
| `feature_horizon_corr_matrix.csv` | 特征 × 预测期限主相关性矩阵（各合约等权平均值 `contract_equal_spearman`）。 |
| `feature_horizon_coverage_matrix.csv` | 特征 × 预测期限有效数据覆盖率矩阵。 |
| `window_strata_metrics.csv` | **细粒度分合约分窗口指标明细**。包含字段：`feature`、`horizon`、`target_kind`、`contract`、`window_id`、`spearman_corr`、`valid_pairs`、`null_reason` 等。可精确定位特定合约在特定窗口的有效性。 |
| `audit_report.md` | 精简阅读入口：最多 50 个不同特征的复核示例、CI/共同样本概况，以及各产物的粒度和阅读方法；全量明细保留在 CSV/Parquet。 |
| `_SUCCESS` | 空的运行完成标记文件；元数据见 `audit_run_manifest.json`。不代表统计充分或协议全面验收。 |

日常逐特征复核优先从 `insights/executive_summary.md` 进入，依次查看
`insights/feature_evidence.csv`（筛特征、期限和目标）、
`insights/legacy_selection_comparison.csv`（展开各流淘汰原因）、
`window_strata_metrics.csv`（排除历史参考期后检查各合约和窗口），
比较期限时使用 `matched_horizon_comparisons.csv`。每个文件的粒度和关键字段在精简报告中列出。

报告中的核心示例用于说明不同复核问题，不是自动排名或入选推荐。
`--top-k` 仍控制候选 CSV 的摘录数量，不会把全部榜单展开进 Markdown。
Markdown 用作摘要和导航；组合筛选、逐流展开、窗口联动应由交互表格承担。

### 3.7 常见多合约配置误区与避坑指南

- **误区 1：试图将多个合约手工拼接为单个连续主力文件**
  - **后果**：破坏了商品期货交割合约的天然到期边界，引入了换月价差跳空的虚假收益率。
  - **正确做法**：每个真实交割合约独立保存为单独的 Feather 文件（如 `fu2305.feather`, `fu2309.feather`），统一存放在 `data_path` 目录下由引擎自动扫描。
- **误区 2：快照参考价源文件只包含主力合约**
  - **后果**：非主力合约在计算预测终点价格时无法匹配到相同合约的快照报价，导致大量样本被判定为 `MISSING_TARGET_PRICE`，拉低覆盖率。
  - **正确做法**：`price_file` 必须包含所审计目录中所有合约的高频报价快照。
- **误区 3：数据行中遗漏 `contract` 字段**
  - **后果**：引擎无法按合约进行收益隔离和分组统计，前置校验或运行时会报错。
  - **正确做法**：确保每份 Feather 文件内均包含显式的 `contract` 字符串列。

---

## 4. 关联 Manifest 输入文件格式与参数要求

配置文件中引用的 4 个 Manifest 文件路径是运行的核心输入依赖，其格式定义如下：

### 4.1 交易日历清单（`calendar_manifest`）
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

### 4.2 候选特征数据清单（`feature_availability_manifest`）
- **文件格式**：JSON 或 YAML
- **核心结构**：
  ```json
  {
    "symbol": "fu",
    "data_path": "PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/10min/fu/train"
  }
  ```
- **字段要求**：
  - `data_path`: 存放各合约特征 Feather / Parquet 文件的目录。每个文件包含 `timestamp`、`contract` 及数值特征列。引擎会自动扫描该目录下所有合约文件。

### 4.3 参考价源清单（`price_source_manifest`）
- **文件格式**：JSON 或 YAML
- **核心结构**：
  ```json
  {
    "price_file": "data_preprocess/operator_futures/feature_audit/data/fu_prices_10min.feather",
    "price_field": "close",
    "max_quote_age_seconds": 600
  }
  ```
- **字段要求**：
  - `price_file`: 包含多合约报价参考数据的 Feather / Parquet / CSV 文件路径，或包含各合约 Feather 文件的目录路径。数据必须包含 `contract`、`timestamp` 以及 `price_field` 指定的价格列。
  - `price_field`: 参考价格字段名，支持 `close`、`LastPrice` 等。
  - `max_quote_age_seconds`: 允许的最大报价陈旧窗口（秒）。若使用 10min Bar 数据作为参考价，必须设为 `600`；若使用秒级快照，设为 `60`。

### 4.4 数据集切分清单（`split_manifest`）
- **文件格式**：JSON 或 YAML
- **核心结构（精简格式）**：
  ```json
  {
    "train_contracts": ["fu2305", "fu2309"],
    "valid_contracts": ["fu2401"],
    "test_contracts": ["fu2405"]
  }
  ```
- **字段要求**：
  - `train_contracts`: 参与训练/开发阶段评估的合约集合。
  - `test_contracts`: 封存合约列表。在 `read_final_test=false` 时严禁读取，文件及行级数据均被跳过。

---

## 5. 跨运行比对配置（Comparison Config）参数详解

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

## 6. 运行脚本与 CLI 调用指引

### 6.1 一键执行脚本（推荐）

工程提供了封装好的端到端执行脚本，包含环境自适应激活、前置静态校验、全量多合约审计、日志自动留存以及最终产物完整性校验：

```bash
# 一键运行 fu 10min 特征审计（默认使用 configs/10min/audit_fu_10min_template.yaml）
bash data_preprocess/operator_futures/feature_audit/run_audit_fu_10min.sh

# 也可显式传入自定义配置文件
bash data_preprocess/operator_futures/feature_audit/run_audit_fu_10min.sh path/to/custom_config.yaml
```

- 执行日志自动保存至：`log/feature_audit/fu/10min/audit_fu_10min_<timestamp>.log`
- 产物自动物化至：`analysis_result/DiHFT/feature_horizon_audit/fu/10min/`

### 6.2 Python 原生 CLI 调用

如果需要分步执行或集成到外部流水线中，可通过原生 Python 模块执行：

```bash
# 激活工程环境
conda activate finetf

# 1. 静态前置校验（检查配置与 Manifest 文件有效性）
python -m data_preprocess.operator_futures.feature_audit.cli validate-inputs --config data_preprocess/operator_futures/feature_audit/configs/10min/audit_fu_10min_template.yaml

# 2. 运行特征审计
python -m data_preprocess.operator_futures.feature_audit.cli run --config data_preprocess/operator_futures/feature_audit/configs/10min/audit_fu_10min_template.yaml

# 3. 运行配对比较
python -m data_preprocess.operator_futures.feature_audit.cli compare --manifest data_preprocess/operator_futures/feature_audit/configs/comparison_fu_10min_vs_30min_template.yaml
```

### 6.3 分析已有结果及三流选择证据

```bash
conda activate finetf
python -m data_preprocess.operator_futures.feature_audit.analyze_audit_results \
  --audit-dir analysis_result/DiHFT/feature_horizon_audit/fu/10min \
  --config data_preprocess/operator_futures/feature_audit/configs/10min/audit_fu_10min_template.yaml
```

`--config` 只读取其中的三流 Manifest，并核对品种和频率；不重新计算标签或 IC。
也可单独传 `--legacy-selection-manifest <path>`，两者互斥。
显式指定来源会标注为 `explicit_retrospective`，不能据此宣称该选择版本在历史运行时已经存在。
不指定时优先读取审计目录中的 `legacy_selection_comparison.csv`，其次按运行 Manifest
记录的路径和 SHA-256 校验原选择文件；两者都没有时保留 `unknown`。

默认分析全部七期限。封装脚本
`data_preprocess/script_preprocess/future_upgraded/commodity/run_analyze_fu_10min.sh`
依次接受审计目录、输出目录、期限列表、审计 YAML 四个可选参数；默认使用 fu/10min 配置。

新增分析输出：

- `feature_evidence.csv`：全部指标及三流归属、剔除证据、CI 状态。
- `legacy_selection_comparison.csv`：每个候选的选择证据及来源。
- `selection_dynamics_summary.csv`：探索性期限分类与入选状态的交叉计数。
- `ci_evidence_summary.csv`：按期限、目标汇总区间证据状态。
- `analysis_manifest.json`：输入文件指纹、注释来源、展示参数及比较证据状态。

CI 为空且上游没有原因字段时标为 `MISSING_CI_REASON_UNRECORDED`。上游 `feature_metrics.csv`
与 `ci_diagnostics.csv` 显式记录 `ci_status`（如 `INSUFFICIENT_EFFECTIVE_BLOCKS`）及实际有效交易日数、
区块长度与有效区块数。三维共同样本比对产物 `matched_horizon_comparisons.csv` 存在且格式完备时标为
`COMPLETED_MATCHED_EVALUATION`，缺失时标为 `NOT_AVAILABLE`。
展示列表按绝对 IC 排序并保留正负号；`ENDPOINT_SIGN_ALIGNED` 仅指两个端点同号，
不能解释为全期限稳定。所有分类和排序均为探索性展示，不输出自动入选结论。
