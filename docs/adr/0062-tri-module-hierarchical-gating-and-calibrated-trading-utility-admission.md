---
status: accepted
---

# 0062. 三模块高层门控与校准交易效用准入

## 背景

`ADR-0031` 的 `HierarchicalDualGating` 把三类不同问题放进了同一个门控决策：

1. 输入是否仍处于训练分布内；
2. 当前市场应路由到哪个二维状态槽位；
3. 槽位策略此刻提出的交易是否值得承担成本和尾部风险。

VAE 似然与模式间距可以描述状态匹配程度，但不能直接证明候选交易具有扣费后正期望。相反，市场状态匹配不清晰也不必然意味着所有候选交易都应被拒绝。因此，高层门控必须按决策职责拆分，不能继续用一个混合阈值同时承担拒识、路由和交易准入。

现有实现还存在以下约束：

- `ensemble_Qnet` 中的成员分别对应不同 regime 槽位，并不是针对同一候选动作训练的同质模型集成；跨槽位 Q 值的均值和方差不能解释为该候选动作的期望收益与认知不确定性；
- 现有 Q 值是带奖励塑形的折扣回报，不等于未来固定视界内的真实扣费净 PnL，也不是胜率；
- Flat 动作编号由动作空间动态计算，当前配置下恰好为 Action 1，不得把数字 `1` 固化为跨配置契约；
- 当前 VAE 推理仍可能经过随机重参数采样；硬 OOD 判定必须改用确定性表征；
- valid 与 test 数据量有限，不能再永久切出一个独立校准集，但仍必须产生严格未见的校准与评价证据。

本 ADR 经 2026-10-21 至 2026-10-22 的质询访谈确认，取代 `ADR-0031` 作为新架构决策。`ADR-0031` 所定义的旧门控代码继续作为独立 legacy 栈保留，何时删除由用户另行决定；本 ADR 不授权自动删除、自动替换或根据 test 结果自动切换默认门控。

## 决策

### 1. 保留两套物理隔离的高层门控栈

系统同时保留且只保留以下两个可选择的高层门控栈：

- `legacy`：保留现有旧门控行为；
- `tri_module`：实现本 ADR 的 OOD 拒识、regime 路由和交易效用准入。

顶层通过单一参数选择，例如：

```text
--high_level_gating_stack legacy|tri_module
```

初期默认值保持 `legacy`，以避免现有实验被静默改变。实际取值必须写入运行配置快照、模型清单和审计产物。

两套栈遵守以下隔离原则：

- 代码位于独立模块，通过公共适配器接入，不在同一次运行中串联或混用；
- `tri_module` 的模型、校准器或清单异常时必须 fail-closed：拒绝启动或禁止新增风险，不得静默回退到 legacy；
- legacy 不依赖新 Utility Model 的资产，缺少新资产不得影响 legacy 运行；
- 可以共享不可变的基础资产，例如已冻结的 VAE 权重和明确版本化的输入数据，但不得共享有决策权的门控状态；
- test 对两套门控的比较只形成审计证据，不触发自动切换、自动删码或默认值变更。

### 2. `tri_module` 严格拆成三个职责模块

三个模块按顺序执行，但决策权彼此隔离。

#### 2.1 Module 1：极端状态 / OOD 与输入有效性拒识门控

Module 1 只回答“当前输入是否可信”，不得决定 regime 槽位，也不得评估交易是否赚钱。

它包含两个显式子检查：

1. **确定性 VAE OOD 检查**：推理时使用 `z = μ`，禁止使用随机重参数采样；对每个 VAE 轴使用 train-ID 经验似然分位数。任一轴的最佳槽位分位数低于 `0.005` 时进入降级；
2. **数据完整性检查**：检测数据跳空、缺失、时间次序异常以及执行所需字段不可用。数据无效与 VAE OOD 使用不同 reason code，不允许混为一个布尔值。

进入降级后：

- 阻断开仓、同向加仓和反手等所有新增风险；
- 取消尚未执行的新增风险订单，并解除会继续扩大风险的动作持续指令；
- 在行情与交易条件可用时，以 best-effort 方式安全减仓或平仓；
- 数据缺失时不得假设能够按理想价格瞬时成交，也不得伪造成交记录。

恢复采用迟滞规则：双轴最佳分位数均达到 `0.01` 以上，数据检查持续干净至少

```text
K = max(VAE rolling window, H)
```

个有效决策步后才退出降级。审计中分别记录 `enter`、`stay`、`recover` 及其具体原因。

VAE 权重、train scaler 和 train-ID 似然基准在两个 valid 外层 Fold 中保持冻结，不进行逐 Fold 重训。需要调整的是确定性评分、有效性检查、阈值统计的作用域，以及把 test 分析从训练入口中拆出；test 不得再被训练后自动分析或用于调参。

#### 2.2 Module 2：Regime Slot Router

Module 2 只回答“当前市场最匹配哪个二维状态槽位”，输出：

- `(volatility_slot, slope_slot)`；
- 路由所需的连续证据，例如每轴似然、最大概率和首二差值；
- 结构化路由 reason code。

它不得使用模式间距执行交易硬拒绝，也不得输出开仓批准。独立的 Candidate Resolver 再把槽位映射为候选策略与候选动作；空槽位映射为 `NullCandidate`。Router 与 Candidate Resolver 不得合并成隐式准入门控。

#### 2.3 Module 3：Trading Utility Admission Gate

Module 3 是交易准入的唯一裁决者。它读取状态、实际候选动作、候选策略信息以及宏微观特征，直接估计真实交易效用。VAE 似然、路由置信度和模式间距可以作为连续特征输入，但不得在模型外重新变成隐藏硬 veto。

准入审查覆盖所有风险增加动作：

- 空仓开仓；
- 同方向加仓；
- 反手换向。

拒绝后的语义为：

- 拒绝开仓：保持 Flat；
- 拒绝同向加仓：保持当前仓位，不增加暴露；
- 拒绝反手：关闭原方向后保持 Flat，不直接建立反向仓位。

Flat 必须通过动作空间的符号映射取得，不能硬编码为 Action 1。单纯维持已有仓位或按既有计划减仓不属于新增风险；Module 1、硬止损与断路器仍保留更高优先级的安全处置权。

### 3. 真实经济标签与反事实候选数据集

Utility Model 使用独立的 shadow-counterfactual 数据集训练。离线回放在每个可评价时点为所有可用的风险增加候选动作生成 episode；线上只评分实际策略提出的候选动作。

标签比较“批准候选状态转换”与“拒绝后的基准动作”，固定视界为：

```text
H = action_persistence
```

fu 10min 当前配置下 `H = 3`。持仓 episode 不得跨 Fold 拆分，尾部不足 H 的样本直接丢弃。反事实退出仅用于标签计算，不得写入真实 `TradeRecord`。

所有目标以 bps 表示，使用真实市场与账户执行量，不使用奖励塑形、limit reward 或折扣 Q 值：

```text
Y_net = gross_PnL - commission - slippage - impact
```

成本只扣一次。如果采用 `1.5×` 压力摩擦，应在标签构造时一次性写入压力成本，准入时不得再减一次相同成本。

交易效用定义为：

```text
U = μ_net - λ_CVaR × L_CVaR - λ_MAE × L_MAE
```

其中：

- `μ_net`：相对拒绝基线的未来 H 步扣费净 PnL 期望；
- `L_CVaR`：下行 5% 条件尾部损失，损失取正数；
- `L_MAE`：H 内最大不利偏移，损失取正数；
- `λ_CVaR`、`λ_MAE`：全局非负且非零的尾部惩罚系数。

两个 λ 只能从预注册集合 `{0.25, 0.5, 1, 2}` 中通过内层 prequential 选择。先满足风险、显著性与单调性约束，再按净效用保守下界排序；选择完成后在 test 前冻结。

### 4. Utility Model 是独立训练的模型包

不采用“跨槽位 Q 均值/方差 + 解析公式”的免训练方案。新门控使用一个按市场/频率共享的轻量监督模型包，slot、方向和状态作为特征，并对合约、槽位与方向进行样本权重平衡。

模型包至少包含：

1. `MeanNetPnLRegressor`：预测 `μ_net`；
2. `LowerTailModel`：预测多个下分位点并近似 5% CVaR；
3. `MAERegressor`：预测最大不利偏移；
4. `WinProbabilityClassifier`：预测扣费后正收益概率。

四个 head 共享特征规范、Fold 定义和 artifact manifest。首选 CatBoost，但必须位于可替换接口之后，模型类型和搜索空间在首个外层评价前冻结。

训练与运行时推理分开放置。训练侧拥有候选标签、模型定义、拟合、prequential 校准/验证、资格评估和制品发布；`high_level` 只拥有模型加载、在线预测和交易准入。模型 bundle 的预测/加载契约由训练侧模型定义提供，在线侧不得导入训练器或训练入口。

实现结构：

```text
FineFT/RL/DiHFT/trading_utility/
  labels.py
  model_bundle.py          # bundle 定义、加载与预测
  model_training.py        # 训练数据与模型拟合
  prequential.py
  eligibility.py
  train.py
  runtime_artifacts.py     # 训练/发布侧生成运行时清单
  frozen_test.py
FineFT/RL/DiHFT/high_level/trading_utility/
  gate.py                  # 在线交易效用准入
  runtime.py               # 在线加载与推理
FineFT/script/train/DiHFT/trading_utility_fu_10.sh
```

运行时只加载显式指定且校验通过的 artifact manifest，禁止寻找隐式 `latest`。因为该模块需要训练独立模型，所以必须提供独立训练入口和独立 shell 脚本；VAE 不因本 ADR 新增重复训练脚本。

### 5. 严格扩展窗口 Prequential，仅两个外层 Fold

valid 使用扩展窗口而不是再切出永久 calibration holdout。以当前 valid 范围为准，固定协议如下：

| 阶段 | 时间范围 | 用途 |
|---|---|---|
| 初始历史 | 2024-07-23 至 2024-10-31 | 只提供首次训练历史，不形成外层评价 |
| 外层 Fold 1 | 2024-11-01 至 2025-02-28 | 第一次未来未见评价 |
| 外层 Fold 2 | 2025-03-01 至 2025-06-27 | 第二次未来未见评价 |

边界按数据中的官方 `TradingDay` 对齐，而不是简单使用自然日或本地 timestamp date。每个边界执行 `embargo ≥ H = 3`；持仓 episode 不得跨边界拆分。因 embargo、数据无效或 episode 不完整而不具备评价资格的样本不强行保留；其余初始期之后的 valid 样本最多进入一次外层未见评价。

每个外层 Fold 严格执行：

1. 只使用该 Fold 之前的历史前缀；
2. 重新生成所有依赖 valid 的统计、切片和 Agent Selection 产物；
3. 在历史前缀内部执行必要的 inner prequential/cross-fit，生成 Utility Model 的 OOF 原始分数；
4. 仅用这些 OOF 预测拟合 Isotonic/Platt 校准器并选择预注册超参数；
5. 冻结 Selector、Utility Model、Calibrator 和阈值；
6. 对紧随其后的外层时间块只预测一次，不回看、不修订。

“两个外层 Fold”只限制正式的外层未见评价次数，不限制每个 Fold 内为了产生 OOF 分数而必须执行的训练次数。

完成两次外层评价后，用全部 valid 重新生成最终 Selector 和 Utility Model。最终校准器只能使用历次严格 OOF/未见预测及其真实结果拟合，禁止使用最终全量模型对自身训练样本产生的 in-sample 分数。此最终生产拟合不是第三个外层评价 Fold。

特征 schema、模型类型及搜索空间、λ 网格、五分位方法、证据不足规则、显著性/单调性门禁、OOD 阈值与恢复规则，必须在 Fold 1 前预注册冻结。之后任何协议变更都会使既有外层结果失效，必须从 Fold 1 完整重跑。

全部资产冻结后，test 只运行一次。test 范围为 2025-06-30 至 2026-02-05；不得用 test 选择模型、校准器、阈值、默认门控或删改 legacy。

### 6. 校准、单调性和准入规则

#### 6.1 双输出校准

- Isotonic Regression 校准连续效用输出；
- Platt scaling 校准扣费后盈利概率。

胜率只用于解释与单调性审计，不设置 `P(win) ≥ 0.52` 或其他在线硬 veto。最终在线准入只依赖经校准且带保守残差修正的效用下界。

#### 6.2 离线 channel eligibility

以 `(slot, direction)` 为最小准入通道。在汇总的外层 OOF/未见证据上，使用以 `contract × TradingDay` 为块的一侧 95% bootstrap，并对多通道检验执行 FDR 控制。只有统计证据和单调性均通过的通道才可上线。

每个通道至少需要：

- 150 个完整候选 episode；
- 每个五分位至少 30 个 episode；
- 至少 3 个合约；
- 至少 20 个有效 `TradingDay` block。

证据不足只禁用该 `(slot, direction)` 通道并降级为不增加风险；不得合并分箱、降低阈值或用全局通过结果替代局部证据。

#### 6.3 单调性审计

将两个外层 Fold 的未见预测按 `(slot, direction)` 汇总，再按预测效用划分为固定方法的五分位。Q1 到 Q5 必须同时满足：

- 扣费后胜率逐档严格上升；
- 盈亏比逐档严格上升；
- 两项指标相对效用分位的 Spearman `ρ ≥ 0.80`；
- bootstrap 总体趋势斜率的单侧置信下界大于 0。

Fold 级表格保留为诊断，不代替汇总门禁。

盈亏比定义为：

```text
mean(positive Y_net) / abs(mean(negative Y_net))
```

每个五分位必须同时具有足量盈利样本和亏损样本。无法计算时记为证据不足/不通过，不能记为无穷大，也不能自动通过。

#### 6.4 在线准入

从严格 OOF 残差估计保守修正：

```text
U_lower = U_hat_calibrated + Q5%(OOF residual)
```

只有同时满足以下条件才允许风险增加动作：

1. 对应 `(slot, direction)` 已通过离线 eligibility；
2. `U_lower > 0`；
3. Module 1 未处于 OOD/数据无效降级；
4. 所有 artifact、schema 和版本校验通过。

由于 `U_hat_calibrated` 已针对扣费标签训练，`U_lower > 0` 表示候选动作的保守净期望显著覆盖标签中已计入的交易摩擦；在线不得再次扣除同一成本。

### 7. 产物与审计契约

Utility artifact manifest 至少记录：

- 门控栈类型及版本；
- 数据范围、官方 TradingDay Fold 边界、embargo 和 H；
- feature schema/version/hash；
- 候选标签与成本模型版本；
- Selector、四个模型 head、Isotonic、Platt 和残差分布的类型、参数、路径及 hash；
- λ、搜索空间、随机种子与训练代码版本；
- 每个 `(slot, direction)` 的样本支持度、eligibility、bootstrap/FDR 和单调性结果；
- train/valid/test 数据与资产血缘。

两套门控通过同一版本化审计 envelope 输出，尽可能保持列名和语义一致。新字段在 legacy 下允许为 null，不得为了填充字段而强制 legacy 执行原本不会发生的候选推理。

公共审计至少覆盖：

- `gate_stack`、Module 1 状态及 reason code；
- 路由槽位与连续证据；
- candidate action、denied baseline、final action；
- 原始/校准效用、`U_lower`、校准胜率；
- channel eligibility、拒绝阶段和拒绝原因；
- artifact/schema/version/hash；
- 对应 Prediction、Trade、Account 的关联标识。

Utility Gate 的拒绝属于决策层拒绝，不得记录为交易所拒单；影子反事实成交不得混入真实 Trade/Account 账本。

## 后果

### 正面影响

- OOD、防御、路由和交易价值的责任边界可单独测试与审计；
- 准入目标与真实扣费 PnL、尾部损失直接一致，不再把 regime 置信度当作盈利能力；
- 两套门控可进行公平对照，旧实验不会因新资产缺失或新逻辑变更而受干扰；
- prequential OOF 校准在不永久牺牲 valid 子集的情况下保留了时间顺序与未见证据；
- channel 级统计门禁避免稀疏槽位借用全局平均表现上线。

### 成本与限制

- 需要新增反事实标签生成、四 head 模型包、校准、残差下界与 channel 审计流水线；
- 两个外层 Fold 只能提供两个较宽市场时期的稳定性证据，不能支持细粒度跨期结论；
- 严格证据要求会使部分稀疏 `(slot, direction)` 长期保持禁用，这是预期的安全结果；
- OOD 时的 best-effort 退出依赖真实可执行行情，系统不能承诺瞬时平仓；
- 本 ADR 不保证消灭所有亏损交易，只要求准入规则拥有可复现、未泄漏且经济含义一致的证据。

## 本轮范围

本 ADR 仅固化架构、训练与审计协议。本轮不实现代码、不训练模型，也不改变运行默认值。后续实现必须以本 ADR 为验收基准，并在独立变更中完成。
