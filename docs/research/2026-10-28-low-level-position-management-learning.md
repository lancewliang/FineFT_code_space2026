# 低层 Agent 的持仓管理学习机制：现状、缺口与研究路线

日期：2026-10-28

## 先看结论

**可以去掉 low 普通策略的固定持仓步数，让模型每根 K 线自己判断继续持有还是退出。并行探索和经验去重可以保留，但它们不能代替交易成本约束。**

每步判断不等于每步交易。例如，模型连续 20 次都选择“保持当前多头仓位”，实际可以一直持仓，不发生这 20 次调仓的手续费和滑点。需要学习的是：**多交易一次带来的预期收益，是否值得支付这次交易成本。** [目标仓位与执行入口](../../FineFT/env/env_class/base_env.py#L628)

目前已有持仓收益、回撤、时长输入，也已有保持和平仓的动作，第一阶段可以沿用。先把训练、去重和执行规则接好，再判断是否需要增加新特征或新网络。下面六步是建议执行顺序；后面的技术证据可以按需查阅。本次只更新研究报告，尚未执行这些代码修改或重训。

## 补充建议：不用把所有不同后续都保存（待确认）

用户已确认精确区分不同持仓状态，同时提出容量问题。这里建议将“不同后续都视为不同经验”与“每条经验都必须留在池里”分开：前者用来防止错误合并，后者可以通过容量与保留策略控制。

**第一版推荐：精确区分当前持仓，同一种当前情况最多留少量不同后续，满池后允许更新旧经验。** 当前经验池本来就按总容量分为9格，并非无限增长（[容量配置](../../FineFT/RL/util/regime_stratified_replay_buffer.py#L437)）。需要避免的是重复探索同一情况，占掉过多名额。

| 环节 | 推荐做法 | 通俗解释 |
|---|---|---|
| 区分当前情况 | 按模型实际看到的行情、仓位、浮盈、回撤、时长、动作可用性和时间上下文区分，再结合当前动作分组。 | 盈利持有和亏损持有不能混为一题。 |
| 限制不同后续 | 同一组最多先留4条不同后续，作为待验证的初始容量设置。 | 同一题留几种做法即可，不需要留几千份。4是保存数量上限，不是持仓步数。 |
| 处理重复或超额 | 与组内已保存结果完全相同的合并；超额候选采用不依据利润大小的随机选择/轮换。 | 不只挑赚钱最多的结果，也不永远锁住最早几条结果。该有限样本方案不能保证保留每一条罕见尾部路径。 |
| 保住覆盖面 | 保留现有市场体制分格容量，统计盈利、亏损、退出、持有等当前状态的覆盖。 | 先保证各种情况都有机会学习，再谈哪条应多学。 |
| 满池以后 | 允许新探索经验替换旧经验；组索引也随淘汰回收，不建立无限增长的历史去重表。 | 池满意味着空间用完，不意味着模型已经学完。 |

目前缓冲池本身支持FIFO覆盖，但训练编排会在池满后跳过后续探索（[覆盖逻辑](../../FineFT/RL/util/regime_stratified_replay_buffer.py#L588)；[停止探索](../../FineFT/RL/DiHFT/low_level/parallel_diverse_train.py#L1993)）。若采用上述方案，应同步处理这一点，并检查训练耗时与新经验覆盖率。每组上限也不能阻止“不同当前状态”本身很多，因此仍须保留全局容量限制。

**“有价值”不能简单解释为“赚得最多”。** 亏损、必要止损和普通持有同样有学习价值。只保留最高收益，会把同一情况的后续结果筛得偏乐观；只保留最大误差，也可能过度集中于噪声或难以预测的路径。

另一个需要纠正的认识：当前探索记录的 `td_error` 主要是 `abs(teacher_q - chosen_q)`，缺教师值时才退化为即时奖励与Q的差（[记录方式](../../FineFT/RL/DiHFT/low_level/parallel_diverse_train.py#L1142)）。它不是修正后的多步 Bellman 误差，不适合直接作为全池学习价值排名。

后续若要进一步提高样本利用率，可以单独研究优先经验回放：用训练时实际计算并定期刷新的TD误差，增加尚未学好样本的抽取机会，同时保留随机覆盖，并处理优先抽样带来的偏差。[Prioritized Experience Replay 原始论文](https://www.davidsilver.uk/wp-content/uploads/2020/03/prioritized-replay.pdf)区分了随机优先抽样与偏差校正；它不等于“永久删掉所有低误差经验”。

本轮建议先落实学习计算修复及上述有界保留方案，保持现有分格随机采样，暂不同时引入完整优先回放。这样更容易判断修复本身的效果。此节是待确认的设计建议，尚未修改训练实现。

## 你的三个条件，对方案有什么影响

| 你的条件 | 对方案的影响 | 接下来怎么处理 |
|---|---|---|
| 并行探索 | 能在同一段历史中尝试不同交易路径，适合比较“继续持有”和“提前退出”。去掉锁定后，随机探索也可能更频繁打断持仓，推理耗时会增加。 | 保留并行框架，先测耗时和持仓样本覆盖。训练探索时交易多，不代表最终模型一定交易多；最终要关闭随机探索再评估。 |
| 经验重复过滤 | 可以减少重复记录占用经验池，但**去重不会减少真实交易次数**。当前去重还可能把同一时刻的盈利持仓和亏损持仓合并。 | 保留去重，先检查哪些经验被当成重复。盈利、亏损、利润回吐以及不同冷却状态等有决策意义的差异应能留下。 |
| 担心反复交易增加摩擦 | 担心有依据。取消固定保持后，模型有更多调仓机会，也有更多无效交易的机会。 | 用真实手续费、盘口成交成本和清晰的额外调仓惩罚约束它；依据扣费后表现筛选模型。避免把必要止损也罚成“不该交易”。 |

依据：并行探索的[动作选择](../../FineFT/RL/DiHFT/low_level/parallel_diverse_train.py#L1243)、[去重键](../../FineFT/RL/util/regime_stratified_replay_buffer.py#L339)、[调仓惩罚](../../FineFT/env/env_class/base_env.py#L670)。更多 worker 增加的是既有行情上的尝试路径，不会增加独立历史时期；仍需在未参与训练和调参的时段验证。

## 接下来按这六步做

### 第一步：保存旧结果，先把学习计算修正确

保存当前模型、配置、数据范围、净收益、最大回撤和交易成本，作为历史参照。先修正前文发现的“累计 18 步奖励，却只对后续价值折扣 1 步”的计算问题，并检查数据窗口结束时的结算。

通俗地说：模型在判断“现在持有值不值”之前，必须先把未来收益算对。准备修正后的重训基线配置，等第二、三步准备完成后统一运行第五步的对照；新旧方案都用正确计算比较，不能把修复计算带来的变化归功于取消持仓锁定。[累计回报](../../FineFT/RL/util/regime_stratified_replay_buffer.py#L34)／[后续价值计算](../../FineFT/RL/DiHFT/low_level/parallel_diverse_train.py#L2294)

**这一步交付：可复现的旧结果，以及计算修正后的基线配置。**

### 第二步：保留去重，但别把不同的持仓情况合并掉

先检查一个具体例子：同一根 K 线，两条探索路径都持有多头、都选择保持，但一条已经盈利，另一条正在亏损。这两条经验对学习止盈、止损有不同作用，不能仅凭“同一时间、同一动作”就当成重复。

当前键没有包含持仓收益和回撤，持仓时长也只分为几个粗组。应补上与决策有关的区别；收益和回撤的分组用于保存训练经验，不是新增交易触发阈值。新增冷却或利润回吐状态时，也要同步检查去重。[当前键与合并逻辑](../../FineFT/RL/util/regime_stratified_replay_buffer.py#L650)

同时观察经验池何时停止收集：当前实现会在池满，或连续多轮没有新增记录后跳过探索。不能把“没有新记录”当作“已经学会持仓”。先记录盈利持有、亏损持有、主动退出等样本是否充分，再决定是否需要定期补充新策略的探索记录。[停止探索条件](../../FineFT/RL/DiHFT/low_level/parallel_diverse_train.py#L1993)

**这一步交付：去重前后样本统计，确认不同时间的连续持有样本和关键盈亏状态能保留。**

### 第三步：让模型明确知道交易成本，并在训练中遇到同样的风控

把两笔账分清楚：实际手续费和成交成本是交易损耗；额外换手惩罚是人为表达“少做无效交易”的训练偏好。两者分别记录，真实成本在相同对照中保持一致，滑点若已进入成交价格，不能再重复扣除。

当前训练脚本手续费为 `0.002`，低层验证脚本为 `0.0003`。先明确这种差异是否是有意的保守训练设置，不能不加说明地混用。[训练成本](../../FineFT/script/train/train_commodity_fu_10.sh#L25)／[验证成本](../../FineFT/script/test/DiHFT/low_level/test_util_fu_10.sh#L42)

还要单独检查现有 5 倍趋势离场惩罚：它可能同时阻止无效早退和必要退出。取消持仓锁定的第一组对照先保持惩罚不变；再单独比较降低这类离场惩罚的版本，仍保留真实成本。这样才能知道改变来自哪里。[惩罚规则](../../FineFT/env/env_class/futures_util.py#L150)

止损、冷却和熔断继续兜底，并尽量在训练环境中复现同一规则。加入这些状态后，经验去重和预训练教师也要一起对齐，避免老师教一种规则、实际执行另一种规则。训练和评估都区分模型主动退出与硬规则强制退出。

**这一步交付：统一的成本口径、风控执行顺序和独立记录的额外惩罚。**

### 第四步：去掉普通策略的固定动作锁定，重新训练

取消训练探索、低层验证和高层正常专家执行中的“选中一个动作后强制保持 6 步”。让模型每步根据最新行情和持仓状态重新选择；连续选择当前目标仓位，就自然形成持仓保持。

对照实验可先统一设置 `action_persistence=1`，表达每步判断；正式新路径再删除普通策略对该参数和计数器的依赖。高层硬止损、冷却、熔断照常执行；空槽位的有限退出缓冲另行保留，不能因为取消 low 的动作锁定而一起删除。这与 [ADR-0062](../adr/0062-stateful-expert-routing-hysteresis-and-frozen-score-calibration.md) 的普通专家每步决策设计一致。

继续使用并行探索，从新经验池重训，不沿用旧锁定规则下的经验池。先不同时改网络、奖励、随机探索强度和 worker 数量。若发现随机动作使长持仓样本很少，再单独研究加入少量低随机度探索路径；这些路径也不规定必须持有多少步。

**这一步交付：取消动作锁定、保留摩擦成本与硬风控的新模型。**

### 第五步：关闭随机探索，比较最终模型是否反复交易

不要用探索 worker 的交易次数判断模型会不会过度交易。探索本来就在尝试不同动作；评估时应关闭随机动作，让模型按学到的价值选择。[探索入口](../../FineFT/RL/DiHFT/low_level/parallel_diverse_train.py#L1243)／[现有无随机探索评估](../../FineFT/RL/DiHFT/low_level/parallel_diverse_train.py#L421)

按顺序做三组比较，其他条件保持一致。三组共用修正后的计算、去重、真实成本和风控规则；原始旧结果仅作历史参照。每组使用相同的一组随机种子，观察结果是否稳定：

| 版本 | 持仓决策 | 额外换手惩罚 | 要回答的问题 |
|---|---|---|---|
| A：修正后的基线 | 保持原 6 步锁定 | 原设置 | 正确计算下，旧机制表现如何？ |
| B：自主持仓 | 每步自己判断 | 与 A 相同 | 单独取消锁定，有什么影响？ |
| C：自主持仓、调整退出惩罚 | 与 B 相同 | 单独调整趋势离场惩罚，真实成本不变 | 原惩罚是否压制了必要退出？ |

三组都检查实际成交次数、手续费与滑点、扣费后净收益、最大回撤、持仓时长，以及模型主动退出和硬止损接管的次数。训练内评估、低层验证、完整高层回放必须使用各自实验对应的同一执行规则。

判断标准是扣费后的收益与风险表现。交易次数增加未必失败，长时间不交易也未必成功；尤其要排除“永远空仓”或“亏损死扛”造成的表面低换手。实验前固定可接受的成本和风险范围，避免看完结果再改标准。

**这一步交付：一张可以直接判断成本、收益和风险变化的对照表。**

### 第六步：通过后，再验证完整组合和未见时段

把候选放回完整高层路由，检查专家切换、止损后再入场、主次合约限制及合约边界。先在已有验证范围选定方案并冻结，再在未参与选择的时段做最终检查。

若仍频繁交易，先查成本是否正确、关键样本是否被去重合并、随机探索是否污染评估、退出惩罚是否失衡。若出现明显利润回吐，再单独试验增加“当前利润距已观察峰值的回吐”等持仓状态。第一阶段不同时上复杂退出网络或多种奖励。

**这一步交付：证明模型能自主决定持仓与退出、且摩擦和风险可接受的完整评估结果。**硬风控继续保留。

## 技术证据与备选研究

以下保留源码核查及论文依据。执行顺序以前面的六步为准。

### 原始核查结论

职责划分应当是：**low 学习何时继续持仓、兑现利润和主动退出亏损仓位，硬风控守住不可绕过的底线。** 当前低层 Q 网络已经根据净权益变化学习动作价值，动作集合含当前目标仓位和 Flat，并接收 signed exposure、持仓收益率、最大回撤和持仓时长四个过程状态。这提供了隐式学习持仓与退出的基础，但尚不足以证明模型已经学好这些行为。

但当前 rollout 会将非 Flat 动作硬保持若干步，重用旧动作和旧 Q 值；高层 expert-hysteresis 路由也会在保持期间跳过新的 low 推理。止损阈值、止损后方向冷却和断路器是高层确定性抢占。环境有可选涨跌停 shaping，但不是一般化止盈机制。

所以最准确的工程问题是：**低层有学习信号和持仓状态，却在固定保持期间缺少逐步重新估值当前动作的机会；硬止损、冷却和熔断由高层规则负责。**没有独立止盈目标本身并非缺陷：Q(Flat) 与 Q(Hold) 的价值比较已能表达退出，不需要独立 TP/SL head 或奖金。

## 当前实现核查

### 动作、状态与隐式学习已有基础

`Base_Env` 的离散动作是 position/leverage 目标组合；当前训练参数 `position_choices=3` 时，仓位列表仅为空头、Flat、多头三档（[训练脚本](../../FineFT/script/train/train_commodity_fu_10.sh#L22)；[base_env.py:154-175](../../FineFT/env/env_class/base_env.py#L154)）。low 可选择保持或完全退出，但没有多档减仓动作；环境还禁止一步直接反手，反向开仓须先平再开（[futures_util.py:240-250](../../FineFT/env/env_class/futures_util.py#L240)）。保证金/清算可行性和涨跌停规则还会过滤动作（[base_env.py:998-1037](../../FineFT/env/env_class/base_env.py#L998)）。

Q 网络把行情特征、时间、当前 position/leverage 编码（名为 `previous_action`）和 `trading_info` 合并后输出动作 Q 值（[low_level.py:12-46](../../FineFT/model/low_level.py#L12)）。`trading_info` 四维为归一化 signed exposure、当前持仓收益率、当前持仓最大回撤、归一化持仓时长（[base_env.py:46-51](../../FineFT/env/env_class/base_env.py#L46)；[base_env.py:234-247](../../FineFT/env/env_class/base_env.py#L234)；域定义见 [CONTEXT.md:1027-1056](../../CONTEXT.md#L1027)）。低层不是“看不到持仓”；时长归一化后封顶。

每步奖励主项为权益变化（`wallet_balance + unrealized_pnl - previous_margine_balance`），扣除方向性换手惩罚，并可选加涨跌停 shaping（[base_env.py:1041-1046](../../FineFT/env/env_class/base_env.py#L1041)）。浮盈浮亏随持仓标记到市值变化进入每步奖励：持仓期间的 mark-to-market PnL 不必等到平仓才反馈（[base_env.py:891-905](../../FineFT/env/env_class/base_env.py#L891)）。这样 Q 值理论上可以学到“继续持有的未来净回报是否高于退出”，是隐式持仓保持/离场能力的来源。

低层 learner 的 Bellman target 为 `accumulated_reward + (1-done) * gamma * max_a Q_target(s',a)`（[parallel_diverse_train.py:2285-2299](../../FineFT/RL/DiHFT/low_level/parallel_diverse_train.py#L2285)），其中 next state 使用 next action mask 和 next `trading_info`。reward 已是 n-step 累计回报，所以这是 n-step return 加 bootstrap，不是一步 reward target。Q(Flat) 对 Q(当前仓位目标) 的比较可隐式表达退出与继续持有，无须单独 TP/SL 标签。

### 当前缺口与规则边界

训练探索 rollout 对非 Flat action 设置 `remaining_persist = action_persistence - 1`，后续若动作仍可用则无条件执行同一 action 并复用旧 Q 值，直到计数耗尽（[parallel_diverse_train.py:1122-1141](../../FineFT/RL/DiHFT/low_level/parallel_diverse_train.py#L1122)；[另一 rollout 路径](../../FineFT/RL/DiHFT/low_level/parallel_diverse_train.py#L1290)）。该 persistence 不作为网络输入或 transition 字段。行为策略在保持期间不能改动作，而 n-step target 仍在 next state 对可行动作取 max。离策略 Q-learning 允许行为策略和目标策略不同，动作重复本身不自动构成算法错误；这里需要核对的是部署也强制重复动作时，所学价值是否对应部署任务。

当前 10 分钟训练脚本配置 `action_persistence=6`、`n_step=18`、`gamma=0.992`、`transcation_cost=0.002`，换手惩罚基准0.002、逆势比例5（[训练脚本](../../FineFT/script/train/train_commodity_fu_10.sh#L25)）。按真实参数直接调用时，多头减仓 `1→0` 在上涨 regime、空头平仓 `-1→0` 在下跌 regime 的换手惩罚率均为0.01，横盘减仓为0.002（[实现](../../FineFT/env/env_class/futures_util.py#L150)）。这确认规则提高这些趋势退出动作的惩罚；行为与收益因果仍待重训消融。

最高优先级 learner 核查：18-step reward 以 `gamma**k` 累加，target update 对 bootstrap 只乘单次 `gamma`（[accumulator](../../FineFT/RL/util/regime_stratified_replay_buffer.py#L34)；[update](../../FineFT/RL/DiHFT/low_level/parallel_diverse_train.py#L2285)）。[buffer insert](../../FineFT/RL/util/regime_stratified_replay_buffer.py#L731) 与 [sample](../../FineFT/RL/util/regime_stratified_replay_buffer.py#L1130) 路径原样保留累计 reward，未发现后续折扣修正。函数探针使用 25 步、每步 reward=1、`n_step=18,gamma=0.992`，首条累计 reward 为 16.82669，next state 为18且非终止；代码 bootstrap 系数为 0.992，而完整18步应为 `0.992**18=0.865386`。这确认该数据路径存在 n-step bootstrap 系数不匹配；尚未测量重训后的策略/收益影响，应在持仓行为研究前修正并重建基线。

线上/评估高层路由的 expert-hysteresis 路径也服从 persistence：只要当前动作可用且 `remaining_persist > 0`，直接输出 `current_action`，不调用 `agent_act`（[vae_routing_util.py:2045-2066](../../FineFT/RL/DiHFT/high_level/vae_routing_util.py#L2045)）。ADR-0062 已设计普通专家每步重新推断并否决保持上次动作；当前源码仍保留 persistence，故是实现与设计不一致。ADR-0061 的 empty-slot 持仓缓冲是另一语义，仍独立适用。

高层硬止损在低层动作查询前执行：用 `-unrealized_pnl / (abs(position)*markprice)` 与 `stop_loss_return_threshold` 比较，命中后重置 persistence、打断专家状态并强制 `rule_based_close`（[vae_routing_util.py:1929-1972](../../FineFT/RL/DiHFT/high_level/vae_routing_util.py#L1929)）。止损后方向性冷却在 low 候选动作产生后拦截同向再开/加仓（[vae_routing_util.py:2099-2123](../../FineFT/RL/DiHFT/high_level/vae_routing_util.py#L2099)）；此外合约断路器优先。当前 ADR-0047 明确要求这些是确定性安全底线，不进入 Optuna 搜参；ADR-0061 保留高风险时的防御抢占与“空槽不新增风险、已有仓位平滑退出”语义。

环境可选 `enable_limit_reward` 使用涨跌停方向/盘口深度/临近限价信号，对顺势持仓和停留给奖励、反向操作惩罚（[base_env.py](../../FineFT/env/env_class/base_env.py#L250)；[futures_util.py](../../FineFT/env/env_class/futures_util.py#L54)）。它仅适用于涨跌停情境，不是普适止盈监督。没有独立止盈目标本身不是缺陷：Q(s, HOLD=当前目标仓位) 与 Q(s, FLAT=0) 的比较已表达策略退出选择；浮盈已逐步进入权益变化奖励，平盈利仓无需另加实现利润奖金。回放可按浮盈/浮亏给退出分类，但不能据此证明 Q 网络内部意图。ADR-0007 定义的是事后持仓回合分类。

### 训练/运行与时间语义注意

- `Base_Env.step()` 在决策价完成目标仓位调整和交易成本，再推进到下一条行情；收益是该 transition 的净权益变化（[base_env.py:628-686](../../FineFT/env/env_class/base_env.py#L628)；[base_env.py:874-905](../../FineFT/env/env_class/base_env.py#L874)；[base_env.py:1041-1046](../../FineFT/env/env_class/base_env.py#L1041)）。凡新增状态必须遵守动作时可用信息，不能把 transition 后价格泄漏到动作前 observation。
- 环境禁止单步直接反手（[futures_util.py:240-250](../../FineFT/env/env_class/futures_util.py#L240)），因此不将方向反转时持仓时长未重置作为已复现缺陷。
- 当前 `trading_info` 由动作执行后的环境状态更新，下一次 low 选择可用；experience 记录中当前和 next 的 trading info 分开保存（[parallel_diverse_train.py:1154-1167](../../FineFT/RL/DiHFT/low_level/parallel_diverse_train.py#L1154)）。若将止损阈值距离、冷却剩余步数/方向、熔断状态或 no-new-risk mask 暴露给 low，训练、valid、Q-table teacher 和 runtime 必须共享相同的风险状态/转移定义；若某些状态只在 runtime 存在，应继续视为硬安全拦截而非宣称 low 已学习这些边界。
- 单持仓收益率使用 `single_holding_return / required_money`（[base_env.py:1048-1066](../../FineFT/env/env_class/base_env.py#L1048)），它将净盈亏和保证金需求口径映射为收益率；高层止损用的是相对名义货值浮亏，两者单位和风险含义不同，不能共用数值解释。
- reward 已包含实现/浮动盈亏、实际成交滑点/费用以及 turnover penalty；对止盈平仓再重复扣一次手续费、对滑点再次扣费、或在已有 PnL 基础上再加同源收益奖励，会造成重复计数。[CONTEXT.md:1023-1025](../../CONTEXT.md#L1023) 明确指出滑点已经反映在成交价值和 PnL 中，不得再次扣除。新奖励如另加 drawdown 惩罚，应明确这是改变风险偏好的附加成本，并与权益损失分开报告。
- 当前 episode 可因强平或数据窗口结束而终止，窗口结束不等于策略自主退出。开仓前初始化状态、开仓成交后的首个可用观测、持仓中逐步更新，以及数据末端 truncation/强平 terminal 必须区别定义；truncation 时不应把末端未平仓误标成策略主动止盈/止损。若新增平仓结算奖励，须避免与末端权益/PnL重复计账。

## 方法学建议

### 最小先行方案：让 low 每个可决策步重新估值持仓动作

1. 首先审计 n-step target 的实际累计长度与 bootstrap 折扣幂，修正前先冻结基线。确认动作前后的市场快照、terminal 标记与最后一步截断一致。
2. 训练、验证回放和 runtime 必须共享风险转移/状态定义：持仓方向、止损阈值距离、止损冷却剩余步数和方向、熔断剩余状态、no-new-risk/动作 mask。规则含状态时应以相同转移驱动训练和运行；只在 runtime 拦截会造成 policy/environment 不一致。无法在当前数据模拟的风险状态，应明确隔离为 runtime 安全层，不伪装成 low 已学会。
3. 比较当前 `action_persistence=6` 基线与同步重训、valid/runtime 一致设置 `action_persistence=1` 的逐步重新决策对照，训练和 runtime 同步配置；记录每步动作来自 low 重算、旧动作复用、硬规则拦截或 mask。普通专家每步推理落实 ADR-0062 已接受设计并修正当前实现与设计不一致；ADR-0061 的空槽缓冲保持为独立规则，不随普通专家 persistence 改动而移除。既有“只改 runtime 保持期”的回放探针使用旧模型，不是训练重训对照，不能单独证明逐步重新决策无效。
4. 对趋势离场非对称惩罚做独立消融。若要取消/调整 ADR-0050 已接受的 exit penalty，需正式重开该 ADR，并评估训练、DP教师和 runtime 成本的一致性，不将研究建议写成已接受规则。
5. DP Q-table teacher 当前索引为时间、previous action、action（[Q-table 构建](../../FineFT/env/env_class/futures_util.py#L1345)），不包含持仓峰值/冷却等扩展状态。若训练回报或转移依赖这些变量，不能只传 reward 参数就宣称监督对齐；先关闭/弱化该部分教师监督作对照，或实现含相同状态和转移的动态规划/轨迹 teacher。当前训练仍含教师 KL 项（[learner](../../FineFT/RL/DiHFT/low_level/parallel_diverse_train.py#L2315)），所以这也涉及实际训练目标。
6. 不增加手工 hold/TP 奖金；先用现有净权益变化与 Q(Flat) 对 Q(Hold) 评估退出行为。一般奖励塑形可能改变最优策略；若目标只是加快学习而不改变任务，应先核对 potential-based shaping 的适用条件（[Ng 等原始论文](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf)）。风险偏好修改应明确为任务目标变化。

回放还须区分 `policy_proposed_action`、风险批准动作与真实执行后仓位。先决定 Q 的 action 表示“安全集合内动作”还是“经过安全层映射的提议”，再让 transition、奖励与 next state 使用同一语义；不能在拦截后随意替换 replay 中的动作标签。被硬规则强平的样本不能算作模型主动退出的证据。

若新增训练观察或改变动作时间契约，应升级状态/动作兼容版本并重训。当前 valid 已用于 selector 和路由校准（[CONTEXT.md:742](../../CONTEXT.md#L742)），新方案应预先定义比较并用未参与选择的 test/walk-forward 区间最终判断。

### 高级方案：显式风险敏感持仓管理

在基础闭环稳定后，再选择单一研究分支：

- **持仓状态增强**：输入 point-in-time 开仓均价/有效开仓价、持仓内最大有利/不利变动（MFE/MAE）、当前浮盈距已观察峰值的回吐、时长及成本预算。所有字段从成交事件和已观察 mark 更新，动作前 observation 只能使用当时数据。已有最大回撤统计不能直接替代“当前距峰值的回吐”；新增状态仍应通过消融证明价值。
- **风险目标增强**：以独立风险敏感算法的 episodic CVaR/回撤预算优化风险调整价值，不直接硬编码“盈利 X% 就卖”。现有 ensemble Q 子网络不等于轨迹回报分位数估计；CVaR 需要显式目标/算法支持，会增加复杂度和样本需求，适合作独立实验。[CVaR-RL 原始研究](https://proceedings.mlr.press/v202/wang23m.html)讨论了 CVaR 风险目标与其学习代价。
- **持仓回合终止/option**：让策略决定进入/持有/退出并学习终止条件，属于 temporal abstraction/SMDP 方案；必须使用适配持续时间的回报递推。现有三档仓位没有部分减仓档，对照只需开放每步 Flat/保持；Sutton 等的 options 形式将时间扩展动作建模为含策略和终止条件的 SMDP，要求 Bellman 目标匹配持续时间（[论文](https://www.ece.uvic.ca/~bctill/papers/learning/Sutton_etal_1999.pdf)）。

## 安全规则应守住的分层

硬规则应处理不可违背的交易/账户约束：交易所限制、保证金/强平、最大仓位/风险预算、数据陈旧或状态不完整时禁止新增风险、账户/合约熔断，以及明确定义的止损触发条件。它们优先于 low action，模型不能通过学习绕过。

高层仍然有两类不同职责：VAE/专家适用性判断与路由是模型调度；硬止损、熔断、保证金和合约资格/日度主力约束是安全边界。Low 学习策略择时和持仓退出。若未来希望学习部分减仓，需要先扩展当前三档动作集合，再升级动作契约并重训。

普通专家每步重决策与 [ADR-0062](../adr/0062-stateful-expert-routing-hysteresis-and-frozen-score-calibration.md) 已接受设计一致；当前代码仍持续复用动作，属于实现与设计不一致。[ADR-0061](../adr/0061-graceful-hold-exit-and-no-new-risk-semantics-for-flat-degradation-slots.md) 的 empty-slot 持仓缓冲仍应保留。调整/取消趋势离场惩罚会改变已接受的 [ADR-0050](../adr/0050-asymmetric-directional-turnover-penalty.md)，需重开该决策并记录理由。本报告不改变这些规则。

配置的“3%硬止损”是触发条件，不保证实际亏损不超过3%：bar 间跳空、盘口深度不足、涨跌停和执行延迟均可能扩大损失。当前 selector manifest 明确为 `enable_hard_stop_loss_gate=false`（[manifest](../../analysis_result/DiHFT/low_level/fu/10min_parallel/two_dimensional_selection/two_dimensional_selection_manifest.json)），该次候选筛选没有使用硬止损 episode gate；运行时 router 止损仍是独立保护路径。

策略层止盈属于可学习的退出选择；若另有组合利润回撤限制，应单独定义风险单位、口径与触发条件，并像其他安全规则一样评估可执行性。

## 评估与消融

采用时间顺序 walk-forward，按合约/阶段分层，冻结成交成本、费用、数据可用时间和硬风控。主要消融应依次隔离：

1. 先修正/验证 n-step bootstrap 折扣幂，建立所有后续实验共用的重训基线；
2. 当前 persistence 与训练/valid/runtime 同步设置 `action_persistence=1` 后重训的每步重新推理；若继续研究动作重复，再单独比较时长正确的 SMDP/option learner；
3. 对称成本基线与当前 regime-dependent directional turnover penalty，分解开仓、减仓、清仓成本和净优势；如调整惩罚规则，重开 ADR-0050 并同步 DP teacher；
4. 基础 reward 与单一风险增强目标（例如 CVaR），禁止同时调多个止盈 bonus；
5. 当前 `trading_info` 与候选新增状态（一次只加一类）。

共同报告净收益、手续费/滑点、换手、持仓时长分布、回合盈利/亏损、盈利/亏损状态下的策略退出次数、MFE 到已实现利润转化、MAE/持仓最大回撤、最大回撤、尾部损失/CVaR、强平/硬止损次数、冷却拦截次数，以及不同市场阶段的样本量和置信区间。净收益提升不能掩盖左尾风险恶化，胜率或单笔利润不能代替组合净收益与风险结论。单独核查反手、加减仓、合约边界和最后一根 bar 的记账。

当前仓库已有一份专家迟滞策略真实回放报告，显示固定“浮盈达到名义金额1%后回吐一半退出”方案在该冻结配置下提高回合胜率但净收益与MDD更差；该结果支持先不直接增加固定止盈规则，不能证明可学习退出无效：[回放研究](2026-10-28-expert-hysteresis-entry-research.md)。其中动作保持 6→1 的实验只改 runtime 持续参数并沿用旧模型；它不是训练和 runtime 同步改为 `action_persistence=1` 后重训的对照。

## 方案范围

本轮优先研究取消普通策略的固定动作保持。前述 options/SMDP、CVaR 和新增网络均为后续备选，不是开头六步的前置工作；不同动作/状态契约下的旧 checkpoint 和旧经验池需显式检查兼容性（ADR-0002），本轮对照使用新实验目录和新经验池。

## 一手资料

- Mnih et al., *Human-level control through deep reinforcement learning*, Nature (2015)：DQN 通过 experience replay 与独立 target network 稳定学习延迟累计回报；Q-learning 的目标包含即时 reward 与 target Q bootstrap。 [Nature 原文](https://doi.org/10.1038/nature14236)；[作者公开 PDF](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)
- Sutton, Precup & Singh, *Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning* (1999)：option 是含策略及终止条件的时间扩展动作，形成 SMDP；学习目标须表达实际持续时间。 [论文 PDF](https://www.ece.uvic.ca/~bctill/papers/learning/Sutton_etal_1999.pdf)；[期刊页](https://www.sciencedirect.com/science/article/pii/S0004370299000521)
- Ng, Harada & Russell, *Policy Invariance under Reward Transformations* (ICML 1999)：一般 reward shaping 可能改变最优策略；potential-based transition shaping 是保持策略不变的关键形式。因此不宜随意叠加固定持仓/止盈奖金。 [作者公开论文 PDF](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf)
- Wang, Kallus & Sun, *Near-Minimax-Optimal Risk-Sensitive Reinforcement Learning with CVaR* (ICML 2023)：CVaR 是可定义的风险敏感 RL 目标，但其样本/理论界随尾部风险容忍参数而变化，提示它是高级方案而非零成本替换。 [PMLR 原文](https://proceedings.mlr.press/v202/wang23m.html)

以上外部资料提供的是通用 RL 理论，不证明任何特定止盈/止损参数对本策略有效。本次按仓库约定激活 `finetf` 后执行最小函数探针，验证了方向惩罚率与 n-step 累计路径；没有修改训练/交易实现，没有运行策略重训、valid/测试收益对比或线上验证。

## 补充源码证据：并行探索与经验去重

当前 replay buffer 的“去重”是按语义键合并训练记录，不是限制交易次数。紧凑路径的键包含数据片段/时间步、前一动作、当前动作、仓位方向和粗粒度持仓时长桶；重复键只保留 TD error 较高的记录。[键与字段](../../FineFT/RL/util/regime_stratified_replay_buffer.py#L339) [择优保留](../../FineFT/RL/util/regime_stratified_replay_buffer.py#L650) 因而，连续相同 HOLD 若发生在不同 bar/step，会有不同键，不会仅因动作相同而去重；同一 bar/step、同一动作、同方向且落在同一 duration 桶的记录，则可能合并，即使当前浮盈或最大回撤不同，因为这两项不在键中，冷却状态也未纳入该训练键。[旧路径键的持仓字段](../../FineFT/RL/util/regime_stratified_replay_buffer.py#L358)

本次在 `finetf` 中执行最小函数探针：保持市场状态、动作、仓位和时长相同，将 `trading_info` 的持仓收益从 `+0.02` 改为 `-0.02`，同时改变回撤，旧路径生成的去重键仍相同；紧凑路径将 step 从100改为101，键不同。该结果验证键的区分范围，不证明合并已造成多少收益损失。既有测试也明确要求收益/回撤变化不改变键（[test_semantic_prioritized_buffer.py](../../FineFT/tests/rl/test_semantic_prioritized_buffer.py#L12)）；若现在为了学习持仓管理修改语义，需要同步修订测试及 [ADR-0052](../adr/0052-compact-spatio-temporal-trajectory-ipc-and-micro-batched-ingestion.md)，保留紧凑回传和轻量键的性能目的。

n-step 压缩先从起始步累计未来最多 n 步的折扣奖励并确定 next state；但语义去重键没有包含累计回报或未来轨迹。[n-step 构造](../../FineFT/RL/util/regime_stratified_replay_buffer.py#L189) 所以相同起点及键字段下、未来轨迹不同的记录会竞争同一槽位，按 TD error 留一条，而非保留两种结果。[紧凑记录去重](../../FineFT/RL/util/regime_stratified_replay_buffer.py#L674) 这可能压掉同一起点不同随机动作/后续行为带来的目标差异，应作为状态与 replay 设计的评估点，不等于已经证明训练受损。

去重只发生在入池阶段，并不意味着一条经验只用于一次梯度更新；采样器在记录不足配额时还会有放回抽样（[sample](../../FineFT/RL/util/regime_stratified_replay_buffer.py#L1065)）。应分别统计探索生成数、入池数、替换数、训练抽样数，不能将它们都称为“交易次数”。

训练 rollout 中，固定 persistence 期间会复用当前动作；到重新决策步才调用带探索的动作选择，逐步记录 transition 后再做 n-step 压缩。[rollout 与 persistence](../../FineFT/RL/DiHFT/low_level/parallel_diverse_train.py#L1122) 配置脚本启用 32 个 diverse workers、epsilon 探索及 persistence=6。[训练配置](../../FineFT/script/train/train_commodity_fu_10.sh#L20) 探索会增加训练行为中的动作变化，但不能据此推断最终 greedy 策略有同等换手；去重也不直接控制实盘成交次数。多个 worker 并行处理的是既有历史片段/任务，不会增加独立历史时期的数量。[并行任务枚举](../../FineFT/RL/DiHFT/low_level/parallel_diverse_train.py#L391) 因此取消固定持仓步数时，应将摩擦成本与逐步决策作为成对训练/验证对照，并按最终 greedy 执行统计换手、成本和净收益。

当前训练内的周期性 greedy 探针已经每步推理（[评估循环](../../FineFT/RL/DiHFT/low_level/parallel_diverse_train.py#L512)），低层独立验证脚本则仍传入保持6步（[验证配置](../../FineFT/script/test/DiHFT/low_level/test_util_fu_10.sh#L45)）。实验 A/B/C 必须按各自配置统一这几个入口，不能把不同行为约束下的数字直接比较。该训练内探针的 `mean_trades` 由建议动作变化计数（[计数逻辑](../../FineFT/RL/DiHFT/low_level/parallel_diverse_train.py#L540)），不等于实际成交笔数；正式成本评估应读取执行后仓位变化、成交和账户记录。
