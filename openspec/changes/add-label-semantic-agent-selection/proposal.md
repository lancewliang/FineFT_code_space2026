# add-label-semantic-agent-selection

## 背景与目标

当前低层 agent picker 对每个 `label_i` 的最终选择依据主要是验证集上的 `trans_reward_mean`。这能选出历史回报最高的候选，但没有约束候选 agent 的交易行为必须符合该 label 的市场语义。

这会带来一个关键风险：如果 `label_3` 的语义是“大涨”，当前逻辑仍可能选中一个在该 label 下回报均值最高、但行为上并不偏多的 agent；如果 `label_0` 的语义是“大跌”，也可能选中一个不具备做空盈利能力的 agent。这样的选择会削弱 Stage II 低层 agent 专门化和 Stage III VAE 路由之间的语义一致性。

本次变更目标是把“label 语义”落到低层 agent 选择链路中：每个 label 先匹配其市场方向语义和涨跌停语义，再在语义匹配的候选中按现有收益/风险指标选择 agent。

## 用户场景

### 场景 1：大涨 label 选择偏多盈利 agent

用户将某个 label 定义为“大涨”或“上涨”后，期望最终选中的 agent 在该 label 的验证样本中主要通过做多获得正收益，而不是仅因为总体 `trans_reward_mean` 偶然最高被选中。

### 场景 2：大跌 label 选择偏空盈利 agent

用户将某个 label 定义为“大跌”或“下跌”后，期望最终选中的 agent 在该 label 的验证样本中主要通过做空获得正收益，并具备足够的空头持仓行为证据。

### 场景 3：震荡 label 不强行选择趋势 agent

用户将某个 label 定义为“震荡”后，期望 picker 倾向选择低净敞口、低无效换手或风险更低的 agent，而不是把震荡 label 绑定到强多或强空 agent。

### 场景 4：语义不匹配时失败而不是静默选择

如果某个 label 下没有任何候选 agent 满足语义约束，用户希望 picker 明确失败并输出原因，而不是退回到旧的“只看收益最大”逻辑。

### 场景 5：涨跌停 label 选择符合限价状态的 agent

如果某个 label 表示“涨停”或“接近涨停”，用户期望最终选中的 agent 不只是验证收益高，还应在涨停样本上表现为顺势多头盈利、避免在无 ask 流动性时盲目追多或频繁反向做空。如果某个 label 表示“跌停”或“接近跌停”，最终 agent 应体现顺势空头盈利、避免在无 bid 流动性时盲目追空或频繁反向做多。

## 设计方向

采用“显式 label 语义 + agent 行为指标 + 硬语义过滤 + 现有收益排序”的方案。

1. `label_i` 的语义必须由机器可读配置提供，例如 `label_semantics.json`。配置记录每个 label 的方向语义、涨跌停语义、强度和中文描述。
2. 当前商品期货 label 编号约定 SHALL 明确表达首尾涨跌停语义：`label_0` 表示跌停/接近跌停，`label_{n+1}` 表示涨停/接近涨停，`label_1..label_n` 表示非涨跌停的普通下跌、震荡、上涨动态。
3. `test_agent_index.py` 在验证回测时输出每个候选 agent 的方向行为指标，例如平均仓位、多头步数占比、空头步数占比、多头收益、空头收益和净敞口。
4. `FineFT_single_agent_with_different_position.py` 在最终选择前读取 label 语义，并按语义过滤候选：
   - bullish/up/strong_up label 只能选择偏多且多头收益为正的候选。
   - bearish/down/strong_down label 只能选择偏空且空头收益为正的候选。
   - neutral/sideways label 只能选择净敞口受控的候选。
   - limit_up/near_limit_up label 必须额外满足涨停状态下的多头盈利和低无效追多/反向做空行为。
   - limit_down/near_limit_down label 必须额外满足跌停状态下的空头盈利和低无效追空/反向做多行为。
5. 通过语义过滤后，再沿用当前收益选择逻辑：第一阶段使用 `trans_reward_mean - std_preference * trans_reward_std`，最终阶段使用按 `label + bin_index + epoch_path` 聚合后的 `trans_reward_mean`。
6. 最终 `selection_manifest.json` 记录每个 label 的语义、过滤阈值、候选行为指标和最终选择原因。

## 关键决策

- label 语义必须显式存在；不能只根据 `label_0`、`label_1` 的编号猜测语义。
- label 语义至少包含两个独立维度：趋势方向 `direction` 和涨跌停状态 `limit_state`。
- 当前 label 编号约定中，`label_0` 固定保留给跌停/接近跌停，最后一个 label 固定保留给涨停/接近涨停；普通趋势强弱只在中间 label 中排序。
- 对 `slope` 或 `quantile` label，可以生成默认从低到高的方向语义，但仍必须落盘到 `label_semantics.json` 供审计。
- 对 `DTW` label，cluster id 没有天然方向顺序，必须由用户或分析脚本提供显式语义。
- 语义过滤是硬约束；默认不允许无匹配候选时静默回退到收益最大。
- 收益仍是最终排序指标，但只在语义匹配候选集合内比较。
- 本变更不修改低层训练算法、不修改动作空间、不修改 VAE 路由模型结构。
- 涨跌停语义依赖商品期货数据中的 `LowerLimitPrice`、`UpperLimitPrice` 以及 `limit_up_single_sided_ratio`、`limit_down_single_sided_ratio` 等现有字段；本变更不重新定义商品涨跌停合法性判断。

## 范围边界

**包含：**
- 定义 `label_semantics.json` 的数据契约。
- 扩展低层测试聚合结果，记录候选 agent 的方向行为指标。
- 扩展低层测试聚合结果，记录候选 agent 在涨停/跌停样本上的行为指标。
- 修改低层 picker，使 agent 选择先满足 label 语义，再按收益指标排序。
- 扩展 `selection_manifest.json`，记录语义匹配和最终选择证据。
- 增加 focused tests 覆盖大涨/大跌/震荡 label 的选择行为。

**不包含：**
- 重新训练低层 agent。
- 改变 Stage I selective-update 训练逻辑。
- 改变 Stage III VAE reconstruction loss 路由算法。
- 改变环境撮合、手续费、滑点、保证金或强平风险逻辑。
- 保证某个 label 一定盈利；本变更只约束选择逻辑与验证集行为一致。

## 验收标准

- [ ] picker 必须读取或生成可审计的 `label_semantics.json`。
- [ ] `label_semantics.json` 必须把 `label_0` 标记为 `limit_down` 或 `near_limit_down`，并把最后一个 label 标记为 `limit_up` 或 `near_limit_up`。
- [ ] 大涨/上涨 label 的候选必须满足偏多行为和多头正收益约束。
- [ ] 大跌/下跌 label 的候选必须满足偏空行为和空头正收益约束。
- [ ] 震荡 label 的候选必须满足净敞口受控约束。
- [ ] 涨停/接近涨停 label 的候选必须满足涨停状态下的多头盈利约束，并记录无效追多或反向做空行为指标。
- [ ] 跌停/接近跌停 label 的候选必须满足跌停状态下的空头盈利约束，并记录无效追空或反向做多行为指标。
- [ ] 没有语义匹配候选时 picker 必须失败，并指出 label、语义和被拒绝原因。
- [ ] `selection_manifest.json` 必须记录每个 label 的语义、过滤阈值、候选行为摘要和最终选择依据。
- [ ] 现有跨合约样本等权、label 覆盖校验和 label 顺序组装 `model.pth` 的行为保持不变。
