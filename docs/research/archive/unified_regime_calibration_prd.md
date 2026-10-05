# PRD: 训练集与验证集宏观波段体制切分算法统一及 9 宫格分层回放池 (Unified Segment-Level Regime Calibration PRD)

**文档状态**: Approved (待代码实现)
**创建日期**: 2026-09-12
**关联规范**:
- 架构决策记录: [ADR-0012](/home/lanceliang/opt/aiwork/FineFT_code_space2026/docs/adr/0012-regime-stratified-buffer-and-rotating-curriculum.md:1), [ADR-0013](/home/lanceliang/opt/aiwork/FineFT_code_space2026/docs/adr/0013-unified-segment-regime-calibration.md:1)
- 技术规范: [openspec/specs/fineft-regime-stratified-buffer/spec.md](/home/lanceliang/opt/aiwork/FineFT_code_space2026/openspec/specs/fineft-regime-stratified-buffer/spec.md:1)
- 领域术语表: [CONTEXT.md](/home/lanceliang/opt/aiwork/FineFT_code_space2026/CONTEXT.md:328)

---

## 1. 业务背景与问题定义 (Context & Problem Statement)

### 1.1 背景
在 FineFT 强化学习强化训练第一阶段（Stage I 多样化探索训练）中，并行 Rollout 收集的所有状态转移经验长期统一汇入单一大池（`buffer_diverse`）。由于金融行情天然被常见状态（如低波动震荡）占据大部分时长，极值或稀缺状态（如低波动单边下跌）样本稀少。均匀采样导致网络梯度仅拟合平均行情，方向相反的信号互相抵消，造成各 Agent 快照策略高度同质化。在 Stage II 选拔阶段，极值宫格（如熊市槽位）几乎全部回退为 `empty_model`。

为了打破同质化，项目设计了基于 3×3（波动率 × 斜率）正交网格的体制分层经验回放池（`RegimeStratifiedReplayBuffer`）与轮次体制课程调度。

### 1.2 核心痛点与缺陷
前期实现中存在严重的“双轨体制粒度割裂”问题：
1. **算法定义割裂**：训练集采用 48-bar 因果滚动窗口计算步级标签，容易被局部高频噪音干扰，造成微观体制剧烈跳跃；而下游 Stage II 验证集评估切片是由 `datahandler` 宏观转折波段算法（`slice_and_merge`，基于双向滤波、极值拐点与 DTW 合并）提取的持续性牛熊波段。训练经验的路由坐标系与验证评估的宏观坐标系完全不一致。
2. **代码重复与分化**：RL 模块和 datahandler 各自维护了一套斜率与波动率的计算与定标代码，难以维护与复用。
3. **因果性假设厘清**：`regime_grid_id` 仅存在于交易环境的 `info` 字典中，专用于离线经验分流与课程调度，**绝不进入神经网络观测特征向量** $S_t$。在历史训练集上使用宏观波段打标既不会引起任何前视信息泄漏，又能为经验分层提供准确的真实波段归属。

---

## 2. 需求目标与方案设计 (Objectives & Solution Architecture)

### 2.1 核心目标
1. **算法统一**：完全废弃 48-bar 步级因果滚动计算，以 `datahandler` 的 `Worker(slice_and_merge)` 宏观转折点波段合并与打标算法作为全系统的唯一定标基准。
2. **底层引擎抽象复用**：在 `FineFT/datahandler/` 下抽象无副作用、高内聚的 2D 体制校准引擎 `regime_calibration_engine.py`，负责波段提取、2D 联合分档、跨合约分位数拟合、符号不变性断言与行级标签映射。
3. **独立对称定标**：训练集基于 `train_all_contracts` 独立拟合阈值（输出 `train/regime_thresholds.json`）；验证集基于 `valid_all_contracts` 独立拟合阈值（输出 `valid/slope/slice_manifest.json` 与 `valid/volatility/slice_manifest.json`）。
4. **时序连贯性保障**：训练集保持连续长切片（`chunk_length=3200`），在切片前对完整合约进行波段拟合与打标，行级注入 `regime_grid_id`，保证 12 步贴现与连续持仓回合不被截断。
5. **外部契约零破坏**：验证集物理切片输出保持 `valid/slope` 与 `valid/volatility` 目录组织，100% 兼容现存 Stage II 2D Agent Selector。

---

## 3. 详细功能与数据流契约 (Detailed Specifications)

### 3.1 核心引擎接口契约 (`FineFT/datahandler/regime_calibration_engine.py`)

#### (1) 波段提取与打标 (`extract_contract_segments`)
- **输入**:
  - `df`: 单合约 DataFrame（必须包含 `key_indicator` 列，默认 `"mark_price"`）
  - `key_indicator`: 基准价格列，若不存在则 Fail-fast 抛出 `KeyError` 或 `ValueError`
  - 滤波与合并参数: `filter_strength=1`, `min_length_limit=288`, `merging_threshold=0.0003`, `merging_metric="DTW_distance"`, `merging_dynamic_constraint=1`
- **计算逻辑**:
  - 调用 `util.Worker(slice_and_merge)` 提取波段转折点 `turning_points`
  - 对每个波段提取带符号百分比斜率: $100 \times \text{coef} / P_{\text{start}}$
  - 对每个波段提取对数收益率总体标准差波动率: $100 \times \text{std}(\Delta \log P, \text{ddof}=0)$
- **输出**: `ContractSegments` 对象，包含该合约的转折点索引列表、波段斜率列表、波段波动率列表。

#### (2) 跨合约阈值拟合 (`calibrate_regime_thresholds`)
- **输入**:
  - `pooled_slopes`: 全体参与合约的所有波段斜率一维数组
  - `pooled_vols`: 全体参与合约的所有波段波动率一维数组
  - `dynamic_number`: 每个维度的分档数（默认为 3，即三分位）
- **计算逻辑**:
  - 计算斜率三分位阈值: $q_{\text{slope}} = [1/3, 2/3]$
  - 计算波动率三分位阈值: $q_{\text{vol}} = [1/3, 2/3]$
  - **符号不变性断言**:
    - 强制断言 $T_{\text{slope}}[0] < 0.0$，否则抛出 `ValueError`
    - 强制断言 $T_{\text{slope}}[1] > 0.0$，否则抛出 `ValueError`
    - 强制断言 $T_{\text{vol}}[0] > 0.0$ 且 $T_{\text{vol}}[1] > T_{\text{vol}}[0]$，否则抛出 `ValueError`
- **输出**: 字典包含 `slope_thresholds: [s0, s1]`, `vol_thresholds: [v0, v1]`, `sample_count`, `fit_scope` 等。

#### (3) 行级标签映射 (`apply_regime_labels_to_dataframe`)
- **输入**: `df`, `segments`, `slope_thresholds`, `vol_thresholds`
- **映射逻辑**:
  - 将每个波段映射为 `slope_label \in {0, 1, 2}` 与 `volatility_label \in {0, 1, 2}`
  - 合成 `regime_grid_id = volatility_label * 3 + slope_label \in [0, 8]`
  - 波段内所有行填充该标签
  - **涨跌停掩码覆盖规则**:
    - 涨停行 (`limit_up`): `slope_label = 2`, `volatility_label = 2`, `regime_grid_id = 8`
    - 跌停行 (`limit_down`): `slope_label = 0`, `volatility_label = 2`, `regime_grid_id = 6`
  - 校验全表无未标记行（禁止存在 -1）。
- **输出**: 注入了 `slope_label`, `volatility_label`, `regime_grid_id` 的 DataFrame。

---

### 3.2 数据集生成流水线交互 (`FineFT/datahandler/commodity_contract_dataset.py`)
1. 在 `write_stage_datasets` 将 SCALE_SAVE 特征文件拷贝到 `train/<contract>.feather` 之后：
   - 触发训练集跨合约定标，遍历 `train` 目录下所有合约文件，提取全部波段；
   - 池化所有波段得分，拟合全局三分位阈值（`fit_scope="train_all_contracts"`）；
   - 将定标阈值持久化保存为 `train/regime_thresholds.json`；
   - 调用行级映射函数，将 `regime_grid_id` 直接写回各 `train/<contract>.feather` 文件。
2. 随后执行 `rebuild_train_slice_plan` 与 `write_train_slices`：
   - 从已注入 `regime_grid_id` 的完整合约文件中切出连续 3520 步的 `train/slice/df_*.feather` 切片；
   - 使得切片文件无需二次计算，原生具备无瑕疵的宏观波段 `regime_grid_id`。

---

### 3.3 验证集切片生成流水线交互 (`FineFT/datahandler/valid_cross_contract_label_calibration.py`)
1. 模块内部重构：底层核心提取与打标全部委托给 `regime_calibration_engine`。
2. 独立定标：在验证集合约上独立提取波段并拟合阈值（`fit_scope="valid_all_contracts"`）。
3. 切片导出与向后兼容：
   - 遍历各合约，按连续相同标签切分段落，分别输出 `valid/slope/<contract>/label_*/df_*.feather` 与 `valid/volatility/<contract>/label_*/df_*.feather`；
   - 在生成的切片文件中，完整携带 `slope_label`, `volatility_label`, `regime_grid_id` 列；
   - 写入 `valid/slope/slice_manifest.json` 与 `valid/volatility/slice_manifest.json`。

---

### 3.4 强化学习执行层交互 (`base_env` + `RegimeStratifiedReplayBuffer`)
1. `commodity_env` / `base_env` 初始化时读取切片中的 `regime_grid_id` 数组；
2. 环境在 `reset()` 与 `step()` 的 `info` 字典中返回当前的 `regime_grid_id`；
3. Rollout Worker 在本地沿连续轨迹计算 12 步贴现回报，以决策初始步的 `regime_grid_id` 标记转移数据并传输至主进程；
4. `RegimeStratifiedReplayBuffer` 依据该 ID 精准分流入 9 个隔离的经验队列，配合 3 阶段轮次课程调度驱动策略分化。

---

## 4. 验证接缝与测试策略 (Verification Seams & Testing Strategy)

为确保工程质量与契约健壮性，围绕 4 个高层业务接缝（High-Level Seams）构建端到端回归测试集：

### 接缝 1: 2D 体制校准核心引擎测试 (`test_regime_calibration_engine.py`)
- **测试内容**:
  - 单合约波段提取：准确获取转折点与波段斜率/波动率；
  - 跨合约分位数拟合：多合约波段池化与三分位阈值计算；
  - 符号断言防护：人工构造非负斜率分布，验证抛出 `ValueError` 并阻断运行；
  - 涨跌停硬性覆盖：验证涨跌停行被正确置为极值标签；
  - 行级映射完整性：校验无 NaN、无 -1、输出行数守恒。

### 接缝 2: 训练集切片生成流水线测试 (`test_commodity_contract_dataset.py`)
- **测试内容**:
  - 调用 `run_dataset_generation` 生成模拟商品期货数据集；
  - 验证 `train/regime_thresholds.json` 成功生成且包含合法字段；
  - 验证 `train/<contract>.feather` 包含 `regime_grid_id`；
  - 验证切片目录 `train/slice/df_*.feather` 正确继承 `regime_grid_id`，且在 3200 步边界前后波段标签保持连贯。

### 接缝 3: 验证集跨合约切片导出测试 (`test_valid_cross_contract_label_calibration.py`)
- **测试内容**:
  - 对包含多个合约的验证集目录运行构建；
  - 验证双轨物理目录 `valid/slope` 和 `valid/volatility` 正常生成；
  - 验证切片内部包含 `regime_grid_id` 列，且与对应波段标签数值匹配；
  - 验证生成的 slice manifest 包含完整的定标事实与合约覆盖统计。

### 接缝 4: 环境与 9 宫格分层回放池端到端测试 (`test_regime_stratified_replay_buffer.py`)
- **测试内容**:
  - 模拟真实 `commodity_env` 加载带 `regime_grid_id` 的切片文件；
  - 验证环境步进时 `info["regime_grid_id"]` 正确暴露；
  - 验证 12 步轨迹贴现后的样本被准确分流入对应的队列；
  - 验证轮次体制课程（Downtrend -> Flat -> Uptrend）按 3-epoch block 周期正常交替抽样。

---

## 5. 边界与非目标 (Out of Scope)
1. 不修改 Stage II `FineFT_two_dimensional_agent_selector.py` 的算法逻辑，不破坏验证集既有目录接口；
2. 不修改 Stage III Meta Router 与 VAE 网络的架构；
3. 不修改 Algorithm 2 的偏损失加权公式。
