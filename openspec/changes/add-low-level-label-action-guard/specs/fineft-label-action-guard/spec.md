## ADDED Requirements

### Requirement: 带守卫低层测试入口 SHALL 显式接收 Label 方向语义

系统 SHALL 为 Stage II 低层 Agent validation 行为测试提供 `FineFT/RL/DiHFT/low_level/test_agent_index_with_guard.py`，通过 CLI 声明当次发现的每个 Label 的方向与强度语义。

#### Scenario: 七类商品期货 Label 映射成功
- **WHEN** 对 fu 的 10min 或 30min validation 切片运行带守卫入口
- **AND** validation 目录发现 `label_0` 到 `label_6`
- **AND** `--label_action_semantics` 为 `label_0:limit_down,label_1:strong_down,label_2:weak_down,label_3:sideways,label_4:weak_up,label_5:strong_up,label_6:limit_up`
- **THEN** 系统 SHALL 将全部七个 Label 映射到显式语义类型
- **AND** 系统 SHALL NOT 读取 `label_semantics.json`

#### Scenario: Label 语义映射不完整时 fail-fast
- **WHEN** validation 目录发现的 Label 在 CLI 映射中缺失、重复或出现额外未知 Label
- **OR** 语义类型不是 `limit_down`、`strong_down`、`weak_down`、`sideways`、`weak_up`、`strong_up`、`limit_up` 之一
- **THEN** 系统 SHALL 在加载模型并执行行为轨迹前报错停止
- **AND** 错误 SHALL 列出缺失、重复或未知项

### Requirement: 逐步 Label 动作守卫 SHALL 位于模型与环境之间

系统 SHALL 先使用 Futures Trading Environment 原始 `info["avaliable_action"]` 计算模型原始动作，再由环境外部守卫产生最终动作并交给 `env.step()`。

#### Scenario: Label 守卫不修改 Q 网络可用动作掩码
- **WHEN** Q 网络在一个 validation 时间步产生原始动作
- **THEN** Q 网络收到的 `avaliable_action` SHALL 与环境 `info` 中的数组一致
- **AND** 系统 SHALL NOT 在调用 Q 网络前合并 Label 配额掩码
- **AND** 守卫 SHALL 在模型原始动作可观测后才应用配额与降级

### Requirement: 守卫 SHALL 使用滚动逆 Label 动作配额

系统 SHALL 对每条行为轨迹独立维护最终动作窗口，默认窗口大小为 10，小幅 Label 逆向比例为 0.40，大幅 Label 为 0.20，涨跌停 Label 为 0。

#### Scenario: 默认窗口容量与轨迹起始配额
- **WHEN** `--label_action_window_size=10`
- **AND** 小幅比例为 0.40，大幅比例为 0.20
- **THEN** `weak_down` / `weak_up` 的容量 SHALL 为 4
- **AND** `strong_down` / `strong_up` 的容量 SHALL 为 2
- **AND** `limit_down` / `limit_up` 的容量 SHALL 为 0
- **AND** 新行为轨迹 SHALL 立即具有该完整容量，不需先填满 10 个时间步

#### Scenario: 可调比例按向下取整生成容量
- **WHEN** 窗口大小为 10 且某类 Label 比例为 0.25
- **THEN** 容量 SHALL 为 `floor(10 * 0.25) = 2`
- **AND** 系统 SHALL NOT 向上取整或四舍五入为 3

#### Scenario: 配额 CLI 非法时 fail-fast
- **WHEN** 小幅或大幅比例小于 0 或大于 1
- **OR** 窗口大小小于等于 0
- **OR** 止损比例小于 0
- **THEN** 系统 SHALL 在 validation 行为轨迹执行前报错停止

### Requirement: 逆 Label 动作统计 SHALL 以最终仓位变化语义判定

系统 SHALL 将逆向开仓、逆向加仓和减仓后仍保持逆向仓位计为逆 Label 动作，并将每个时间步的最终动作记入窗口分母。

#### Scenario: 逆向开仓加仓和减仓占用配额
- **WHEN** Label 方向为下跌
- **AND** 最终动作从空仓开多、增加多仓，或从较大多仓减到仍非零的较小多仓
- **THEN** 每个该动作 SHALL 记为一次逆 Label 动作
- **AND** 上涨 Label 对空头 SHALL 使用对称规则

#### Scenario: 保持逆向仓位与完全平仓不占用分子
- **WHEN** 最终动作保持已有逆向仓位不变
- **OR** 最终动作将逆向仓位完全平到 0
- **THEN** 该时间步 SHALL 进入窗口分母
- **AND** 该时间步 SHALL NOT 进入逆 Label 动作分子

#### Scenario: 震荡 Label 不产生逆向配额拦截
- **WHEN** Label 语义为 `sideways`
- **THEN** 守卫 SHALL 不定义做多或做空为逆 Label 动作
- **AND** 原始动作 SHALL 仅受环境原始成交约束

### Requirement: 守卫 SHALL 将超额原始动作降级为保持或条件平仓

系统 SHALL 只在当前原始动作使“最近 `window_size - 1` 个最终动作 + 当前候选”超过配额时应用降级。

#### Scenario: 空仓或 Label 同向持仓在拦截后保持
- **WHEN** 原始动作是超额逆 Label 动作
- **AND** 当前 position 为 0 或与 Label 方向同向
- **THEN** 最终动作 SHALL 映射到当前 position/leverage
- **AND** 守卫原因 SHALL 为 `quota_hold`

#### Scenario: 逆向多头达到止损阈值时尝试平仓
- **WHEN** 下跌 Label 下的超额原始动作被拦截
- **AND** 当前为多头持仓
- **AND** `(current_mark_price / current_holding_opening_price - 1) <= -0.03`
- **THEN** 最终动作 SHALL 映射到 position 0
- **AND** 守卫原因 SHALL 为 `stop_loss_close`

#### Scenario: 逆向空头达到止损阈值时尝试平仓
- **WHEN** 上涨 Label 下的超额原始动作被拦截
- **AND** 当前为空头持仓
- **AND** `(current_mark_price / current_holding_opening_price - 1) >= 0.03`
- **THEN** 最终动作 SHALL 映射到 position 0
- **AND** 守卫原因 SHALL 为 `stop_loss_close`

#### Scenario: 未达阈值的逆向持仓保持不动
- **WHEN** 超额原始动作被拦截
- **AND** 当前持仓与 Label 方向相反
- **AND** 按开仓价计算的不利变动小于配置止损阈值
- **THEN** 最终动作 SHALL 映射到当前 position/leverage
- **AND** 守卫原因 SHALL 为 `quota_hold`

#### Scenario: 未发生配额拦截时不主动止损
- **WHEN** 当前持仓不利变动达到或超过止损阈值
- **AND** 模型原始动作不超过滚动逆 Label 动作配额
- **THEN** 守卫 SHALL 允许原始动作
- **AND** 守卫原因 SHALL 为 `allowed`

### Requirement: 守卫状态 SHALL 在每条行为轨迹边界重置

系统 SHALL 以 `(label, epoch, bin_index, df_path, initial_action)` 唯一确定的行为轨迹为滚动历史边界。

#### Scenario: 不同 validation 行为轨迹不共享配额
- **WHEN** 系统完成一个 Market Dynamic Segment 的行为轨迹并开始下一个 Label、Agent head、Initial-action 情景或 df_path
- **THEN** 新行为轨迹的滚动历史 SHALL 为空
- **AND** 它 SHALL 具有完整初始配额

### Requirement: 带守卫行为明细 SHALL 区分原始动作、最终动作与实际仓位

系统 SHALL 在带守卫入口的逐步明细中记录模型原始动作、最终动作、守卫原因、配额状态和环境实际执行结果。

#### Scenario: 明细与聚合指标使用已确定口径
- **WHEN** `test_agent_index_with_guard.py` 启用逐步明细并完成一条 validation 行为轨迹
- **THEN** 明细 SHALL 包含 `proposed_action`、`action`、`guard_decision`、当前逆 Label 数、容量、`current_holding_opening_price` 和 `current_holding_average_price`
- **AND** `action` SHALL 表示最终交给 `env.step()` 的动作
- **AND** 换手率、动作变化次数和动作列表 SHALL 使用最终动作
- **AND** 仓位及盈亏类指标 SHALL 使用环境实际执行后状态

#### Scenario: 环境未完成目标时滚动历史仍记录最终动作
- **WHEN** 守卫将最终动作交给 `env.step()`
- **AND** 环境因订单簿深度、保证金或涨跌停成交约束未完成目标仓位
- **THEN** 滚动历史 SHALL 记录交给 `env.step()` 的最终动作 ID
- **AND** 明细 SHALL 另行保留实际 `position_after`

### Requirement: 带守卫入口 SHALL 与无守卫入口隔离

系统 SHALL 仅在 `test_agent_index_with_guard.py` 应用逐步 Label 动作守卫，不改变现有无守卫低层测试入口。

#### Scenario: 重命名不自动迁移调用方
- **WHEN** 原 type-index 守卫实验入口重命名为 `test_agent_index_with_guard.py`
- **THEN** 本变更 SHALL NOT 自动更新 `FineFT/script/`、`.vscode/`、文档或其他调用方中的旧文件名引用
- **AND** 现有 `test_agent_index.py` 的模型决策行为 SHALL 保持不变
