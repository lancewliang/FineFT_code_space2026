## ADDED Requirements

### Requirement: 因果市场状态锚点 State Feature
系统 SHALL 生成七个因果、无绝对价格、可缩放的市场状态锚点 State Feature：`log_price_slope_48`、`log_price_slope_96`、`trend_to_noise_48`、`trend_to_noise_96`、`signed_efficiency_48`、`trend_r2_48` 和 `log_return_vol_quantile_192`。

#### Scenario: 平滑方向行情产生正确锚点
- **WHEN** 正价格序列在完整窗口内平滑上涨或平滑下跌
- **THEN** Log 价格斜率、趋势/噪声比和方向效率 SHALL 保留对应正负方向
- **AND** 趋势拟合优度 SHALL 位于 `[0, 1]`
- **AND** 所有输出 SHALL 为有限值

#### Scenario: 常数价格和窗口不足输出中性值
- **WHEN** 价格窗口为常数或尚未达到特征所需长度
- **THEN** 方向量和趋势拟合优度 SHALL 输出中性值 `0`
- **AND** 系统 SHALL NOT 生成 Null、NaN 或 Inf

#### Scenario: 锚点满足无绝对价格约束
- **WHEN** 同一严格正价格序列整体乘以任意正常数
- **THEN** 七个锚点 SHALL 在数值容差内保持不变
- **WHEN** 计算窗口包含非正价格
- **THEN** 系统 SHALL Fail-fast

#### Scenario: 锚点无未来信息
- **WHEN** 在某个输入前缀之后追加未来 Bar 并重新计算
- **THEN** 原前缀内已成熟锚点值 SHALL 保持不变
- **AND** 滚动窗口 SHALL 在合约边界重置

#### Scenario: 锚点作为普通候选进入下游
- **WHEN** 商品期货候选 State Feature 生成完成
- **THEN** 七个锚点 SHALL 进入 Feature Selection candidate universe
- **AND** 七个锚点 SHALL 接受 NaN Validation 与 Scale Save
- **AND** 七个锚点 SHALL NOT 作为 `Base_Time_feature` 跳过缩放

### Requirement: Feature Selection 输出 4×4 条件市场状态指标
系统 SHALL 使用 train-only 统一阈值将可用步骤划分为四个斜率格与四个波动率格，并对全部 16 个组合输出 candidate State Feature 条件指标。

#### Scenario: 训练集拟合共享二维阈值
- **WHEN** train Feature Selection 处理多个合约
- **THEN** 系统 SHALL 使用因果 48 Bar Log 价格斜率和 48 Bar Log return 总体标准差作为二维统计轴
- **AND** 系统 SHALL 在全部训练合约的可用行上分别拟合 25%、50%、75% 行级四分位阈值
- **AND** 每个合约 SHALL 使用同一组训练阈值
- **AND** 不足 48 Bar 的 warm-up 行 SHALL 从二维统计中排除

#### Scenario: 验证集复用训练阈值
- **WHEN** valid Feature Selection 生成条件指标
- **THEN** 系统 SHALL 读取并应用 train 持久化的二维阈值
- **AND** valid SHALL NOT 重新拟合阈值
- **AND** valid SHALL 保持 report-only 且不得改写 train State Feature

#### Scenario: 全部 16 格显式输出
- **WHEN** 任一 stage 完成二维条件统计
- **THEN** 输出 SHALL 包含斜率格 `0..3` 与波动率格 `0..3` 的全部组合
- **AND** 零样本格 SHALL 以步骤数 `0` 和空指标显式记录
- **AND** 16 格步骤数之和 SHALL 等于排除 warm-up 后的可用步骤数
- **AND** 每格 SHALL 记录步骤数、占比、参与合约数和可评估特征数

#### Scenario: 条件指标覆盖所有候选和预测窗口
- **WHEN** 某候选特征在某个二维格和预测窗口中有足够对齐行
- **THEN** 系统 SHALL 记录逐合约 IC、RankIC 和样本数
- **AND** 聚合结果 SHALL 记录合约等权均值、标准差、中位数、符号一致率和 90% 单侧下置信界
- **AND** 96/192 窗口锚点 SHALL 排除自身尚未成熟的行

#### Scenario: 二维统计格不改变 Dynamic Label 语义
- **WHEN** 二维格编号写入条件指标或 Manifest
- **THEN** 编号 SHALL 仅表示统计区间
- **AND** 系统 SHALL NOT 使用该编号解释现有 Dynamic Label 的方向或风险语义
- **AND** 系统 SHALL NOT 使用该编号执行 Agent 路由或动作约束

### Requirement: 市场状态锚点条件保留
系统 SHALL 在现有 train 全局 Feature Selection 结果之外，对声明的市场状态锚点应用受限的条件保留规则，并以稳定顺序生成最终 State Feature。

#### Scenario: 锚点满足目标格条件门槛
- **WHEN** 锚点在波动率格 `0` 或 `1` 与斜率格 `0` 或 `3` 的任一组合中可评估
- **AND** 每个参与合约至少有 30 个对齐步骤
- **AND** 至少有 3 个合约参与
- **AND** 非零 RankIC 符号一致率不低于 60%
- **AND** 按一致方向对齐后的合约等权 RankIC 90% 单侧下置信界不低于 `min_abs_ic`
- **THEN** 系统 SHALL 将该锚点补充到现有全局选择结果
- **AND** 系统 SHALL 在 Manifest 中记录保留原因与通过格

#### Scenario: 普通候选不得通过条件规则补充保留
- **WHEN** 非锚点 candidate State Feature 的全局筛选失败
- **THEN** 即使其某个条件格指标通过锚点门槛，系统 SHALL NOT 通过条件规则补充保留该特征

#### Scenario: 条件不足不阻断 Feature Selection
- **WHEN** 某目标格为空、样本不足、合约不足、符号一致率不足或下置信界不足
- **THEN** 对应锚点 SHALL NOT 因该格被补充保留
- **AND** 系统 SHALL 继续执行其他格与现有全局 Feature Selection

#### Scenario: 条件选择可关闭以复现基线
- **WHEN** 条件选择开关关闭或读取缺少新字段的旧配置
- **THEN** 系统 SHALL 只使用现有全局 Feature Selection 规则
- **AND** 最终特征顺序与旧行为 SHALL 保持兼容

