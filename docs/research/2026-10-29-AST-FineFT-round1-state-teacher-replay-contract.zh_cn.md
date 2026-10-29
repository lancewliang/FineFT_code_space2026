# AST FineFT 第一轮交易状态、DP 老师与经验去重契约

日期：2026-10-29；更新：2026-10-30（参数沿用 shell/CLI 传递；P=2、止损3%；E_ref 取传入初始资金；不做冷却/熔断；确认 DP 规模预算）

状态：第一轮设计与参数传递方式已确认；尚未实现或训练；本文是后续实现依据，不是收益结论。

关联：[独立训练增强方案第 6 节](2026-10-29-AST-FineFT-LowAgent-independent-training-enhancement-plan.zh_cn.md#6-步骤三补充交易记忆和完整状态)。

## 1. 第一轮决定与源码依据

第一轮补齐当前交易状态、独立多头/空头退出记忆和实际执行约束；DP 老师同步增加原始状态维度；replay 同时区分模型输入精度和精确执行状态。新增字段不直接产生奖励项，本轮不增加新 loss、追踪止损、账户回撤停机或日亏限制。奖励保留交易价格损益及真实成本的经济口径；账户余额不参与奖励归一化或动作约束。与旧实现的具体计算差异必须通过拆账审计确认。

这里的“冻结”指字段、计算和生命周期方案确定，不要求绑定特定实验配置或匹配历史训练。按用户确认，动作持续 P=2、持仓浮亏硬止损阈值3%，仍由 shell 脚本通过现有 CLI 参数传入；初始资金也从现有参数读取，不在代码或本文锁死金额。其余参数继续沿用当前 shell/CLI 方式。用户已明确本轮不做冷却、熔断：相关状态槽位保留并固定填零，不实现或启用对应规则。已启用规则必须由训练、老师和独立回测共享执行器落实，不能只提供状态而不执行。

| 源码事实 | 一手证据 | 本轮含义 |
|---|---|---|
| 现有四维为仓位敞口、持仓收益率、持仓最大回撤、归一化持仓时长；空仓整块为零 | [base_env.py](../../FineFT/env/env_class/base_env.py#L46)、[观测构造](../../FineFT/env/env_class/base_env.py#L233) | 保留四维原语义；空仓不再清除独立退出记忆 |
| Qnet 分别编码市场、时间、previous_action 和 trading_info；mask 目前用于输出屏蔽 | [low_level.py](../../FineFT/model/low_level.py#L12) | 新字段进入交易状态编码器；mask 继续控制合法输出，不计入交易向量宽度 |
| legacy 持仓收益与 MDD 的资金分母依赖历史；不能用单一峰值/MDD 恢复 | [calculate_metric.py](../../FineFT/analysis/calculate_metric/calculate_metric.py#L44) | 按本次资金充足要求改为固定尺度的单笔交易损益摘要，不再引用账户资金历史 |
| DP 表为 time × previous_action × action；每时点重建保证金，前态浮盈亏置零，钱包置 1e5 | [futures_util.py](../../FineFT/env/env_class/futures_util.py#L1258)、[固定账户重建](../../FineFT/env/env_class/futures_util.py#L1291) | 本轮资金充足假设允许移除钱包轴，但真实持仓成本/浮盈亏与路径状态仍须保留 |
| Demo 按时间、previous_action 查老师 | [demo_env.py](../../FineFT/env/env_class/demo_env.py#L146) | 改为完整状态查询；旧 Q 表不能静默充当新老师 |
| 路由持有单方向冷却、止损计数、断路器等状态；当前在止损触发时递增计数 | [vae_routing_util.py](../../FineFT/RL/DiHFT/high_level/vae_routing_util.py#L1890) | 本轮不迁入冷却、计数或熔断规则；统一传入阈值下的止损执行与实际退出事件 |
| replay 已按模型 float32 上下文精确去重、同组最多四种后续随机轮换 | [去重键](../../FineFT/RL/util/regime_stratified_replay_buffer.py#L338)、[保留规则](../../FineFT/RL/util/regime_stratified_replay_buffer.py#L586)、[ADR-0065](../adr/0065-n-step-discount-and-bounded-holding-context-replay.md) | 扩展键而保留容量及随机轮换原则 |

**与既有决策的关系：**本设计拟扩展 [ADR-0065](../adr/0065-n-step-discount-and-bounded-holding-context-replay.md) 的模型可见去重身份，加入精确执行状态，避免归一化隐藏约束差异；保留每组四条、随机轮换和 FIFO。[ADR-0047](../adr/0047-contract-level-hard-stop-loss-cooldown-and-circuit-breaker.md) 描述高层三级风控，本轮独立 Low Agent 明确不迁入其冷却与熔断。硬止损退出的锁存重试仍需共享执行器落实，触发请求不能冒充实际退出；不修改已接受 ADR 或现有高层规则。

## 1.1 资金充足的决策前提

按用户明确要求，本轮模型优先学习开仓、持有、退出、重入的交易动作优劣，**假设资金充足**。模型不输入账户余额、权益、可用资金或保证金；老师、训练动作 mask、终止条件和经验去重也不根据这些账户量改变。否则即便删了输入，老师标签仍会要求学生学习看不见的资金限制。

- 仓位动作采用预设固定策略数量/档位，不随余额复利放大或缩小。资金充足不等于无限仓位；原有仓位档位与数量上限保留。
- 继续计算真实成交价格、手续费、滑点、持仓浮盈亏和交易净损益；资金充足不代表免费交易或抹掉亏损。单步 reward 用交易净损益增量及已冻结塑形项计算，若归一化只使用固定尺度，不使用实际账户权益。
- 教学环境不因保证金不足拒单，不因账户亏空触发强平/终止。仍保留交易时段、盘口/涨跌停、待退出，以及传入参数控制的动作持续和持仓价格亏损止损；本轮不启用冷却或熔断。
- 实际运行账户仍可由外部执行层施加资金与保证金限制；其拒单记录与真实资金回测单独评估，不冒充本轮资金充足老师的轨迹。教学环境与实际账户环境因此存在明确边界。
- 首轮不把完整账户账本放进 `teacher_state_key` 或经验去重键。同一行情、仓位、交易损益与交易记忆，仅账户余额不同，模型输入、老师 Q、动作 mask 和经验身份应一致。

**避免资金从旧特征间接进入：**旧 `single_holding_return_rate` 和 MDD 使用动态所需资金分母，本轮同时改为固定尺度。设当前单笔的累计交易净损益为 C_t（含已发生费用与盯市损益），H_t=max(0,C_0,...,C_t)，D_t=max(D_(t-1),H_t-C_t)。模型分别输入 C_t/E_ref 与 D_t/E_ref。每次新开仓开始新的单笔统计，空仓输入零；成交平仓损益仍完整计入奖励。T09 只保存 C_t/H_t/D_t，不保存账户历史。该变更是主动调整指标定义，不宣称与旧 legacy 指标数值等价。

## 1.2 2026-10-30 访谈已确认决定

| 决定 | 用户确认 | 实施边界 |
|---|---|---|
| 执行规则 | 动作持续 P=2、硬止损3%；不做冷却、熔断 | 动作持续、止损沿现有 shell/CLI 传参；M22–M26 五个模型槽位与 T22–T25 四个原始状态固定为零；不递增连续止损计数、不实现冷却/熔断状态机，不因这些字段限制重入 |
| 损益尺度 | `E_ref` 用传入的初始资金 | 从本次运行的 `initial_wallet_balance` 读取一次严格正、有限的值，固定为参考本金；不随运行时余额、权益、合约或 reset 后余额变化，不从旧 DP 的硬编码钱包值推断 |
| 配置方式 | 不需要特定配置，忽略历史训练；与现有代码风格一致 | shell 脚本 → 现有 CLI 参数 → 主进程/worker/环境/老师/评估；不增加独立配置文件或历史 checkpoint 匹配前置条件 |
| DP 规模试验 | 按推荐门槛 | 依次试验 16、32、64 个有效步；每次至多 250,000 个精确状态、2 GiB 状态存储、60 秒 DP 计算时间，任一触顶即中止并报告失败，不自动改用近似老师 |

配置选择已收敛，不再将指定实验名、历史运行目录、历史 checkpoint 或固定初始资金金额作为前置条件。`E_ref` 是传入初始资金的派生常量，不另设一个可与初始资金冲突的独立金额参数；组合评估本金或旧 DP 钱包常量不自动覆盖它。此处确认的是通用能力及其参数接口，具体每次运行的数值由 shell 脚本传入。

### 1.3 沿用现有 shell/CLI 接口

| 现有参数 | 本轮约定 | 传递要求 |
|---|---|---|
| `--action_persistence` | 本轮取2；实现继续读取参数，不硬编码2 | 同一次运行的训练、老师和独立评估使用同一解析值；worker 不隐藏另一套计数或默认值 |
| `--stop_loss_return_threshold` | 本轮取0.03；实现继续读取参数，不硬编码3% | 共享执行器使用同一阈值与浮亏率公式；规则退出优先于动作持续 |
| `--initial_wallet_balance` | 金额由调用方传入；启动时令 `E_ref` 等于该值 | 原始初始值与运行时余额分开保存；教学模式仅用其初始值确定固定数值尺度，不据余额约束动作或终止 |

对应入口仍采用项目既有 argparse、shell 变量和 worker 参数传递风格，不为此次状态补充另建配置系统。DP 试验门槛沿第4.1节执行；记录本次实际解析参数以便核验一致性，不要求恢复或对齐历史训练。

## 2. 统一时间、归一化与枚举

- `s_t` 是决策前、已应用此前成交和上一步计时更新的状态；`a_t` 为请求目标仓位。`s_(t+1)` 必须依据实际成交更新。有效步指同合约数据上的一个环境转移步，不等于自然十分钟；休市不凭墙钟递减。
- `E_ref = initial_wallet_balance > 0`，从本次运行解析的初始资金参数读取一次并固定，是交易损益数值尺度；不是运行时账户余额，也不约束可交易数量。交易收益与回撤统一使用该尺度；本决定不额外改变 reward 的既有归一化配置。Train/Valid/Test、合约与 reset 共用该运行常量，恢复时核验实际参数一致；改变初始资金视为改变运行参数，不混用经验或老师。策略仓位沿用 [CONTEXT.md](../../CONTEXT.md) 的策略数量口径，不冒称交易所手数。
- `P_t` 为当时可用正 mark price，`tick_size > 0`。冻结 `scale_t = max(tick_size, mean(abs(P_i-P_(i-1)), i=t-17..t))`，即最近 18 个一步绝对价格变化的均值；它不是 ATR。
- 价格变化中的 `long_exit_scale = scale_(long_exit_step-1)`、`short_exit_scale = scale_(short_exit_step-1)` 均由原始 df 回查；无对应退出事件时不求值。它们是临时计算值，不是新增存储字段。
- `b(x)=tanh(x/5)` 用于退出后价格变化；其参数只是首轮固定编码选择，不宣称经验最优。价格尺度需要19个可用价格，预热不足不采样。仓位和时长保留现有编码；单笔收益和回撤改为下述固定尺度口径，金额比率不额外限幅。所有网络数值最终为 float32；精确状态保留执行器原精度。
- 退出原因 `strategy_exit / hard_stop_loss / liquidation / terminal / unknown` 只用于执行归因和审计，本轮不作为历史记忆模型特征。
- 老师保存待退出原因枚举，无待退出时为 none；模型不输入原因 one-hot。不存在的退出记录其余原始字段规范为零，不能保留旧垃圾值。合法输入不允许 NaN/Inf；零仓不计算持仓除法，使用明确零值。

## 2.1 三级评分标准与取舍

评分是**第一轮实施优先级**，依据当前代码语义和本轮目标，不是预测收益、统计显著性或训练后的特征重要性。先检查因果可用性：未来/事后字段直接排除，不以低分保留。

| 分数 | 评定标准 | 本轮处理 |
|---|---|---|
| **3分：必须优先** | 删除会隐藏已启用规则的关键条件或未来转移；或者属于“双方向退出记忆→重入判断”的最小信息：是否退出、距今多久、相对实际退出价的变化。不存在同等直接的已保留字段替代 | 保留；给出原始来源、模型编码和老师对应关系 |
| **2分：有用但可延后** | 有合理用途，但与保留项重叠、主要帮助解释，或可由原始执行状态派生；尚不足以证明必须作为额外网络输入 | 从第一轮模型输入移除；若是执行必需原始状态，按老师侧3分保留 |
| **1分：暂不引入** | 与本轮资金充足目标无关，或依赖新选窗口/新造信号且与既有特征重叠、没有独立增益证据 | 本轮不计算、不输入、不增加对应老师维度 |

评分遵循两点：一是同一概念在模型和老师中的必要性不同，不能因为网络里不需要重复输入就删持仓成本和交易损益；二是本轮的“最小交易记忆”是研究目标选择，不意味着已证实退出价能提高收益。风险规则若配置禁用，对应字段固定零，不凭3分主动启用规则。

原有四个交易槽位、previous_action、市场/时间和动作 mask 是本次基础输入；其中持仓收益与回撤按资金无关口径调整，其余保持，不在“新增特征淘汰”范围内。基础定位字段 df/step/state_key 也不是预测特征，不参与1–3分排名。以下表格保留低分项**仅用于解释删除理由**，只有下一节实施表中的3分新增项进入第一轮。

| 原ID | 新增模型字段 | 分数 | 评定理由 | 本轮处理 |
|---|---|---:|---|---|
| M05 | `price_adverse_return` | 3 | 直接呈现硬止损使用的逆向价格率；原 legacy 收益率不是同一口径 | 保留 |
| M06 | `entry_price_distance` | 2 | 可帮助解释入场成本，但与保留的浮亏率、市场波动特征重叠；精确成本仍由执行器保存 | 删除模型输入 |
| M07 | `wallet_balance_ratio` | 1 | 资金充足前提下不服务交易动作优劣判断；不以账户余额或保证金余量影响决策 | 删除模型输入 |
| M08 | `equity_ratio` | 1 | 资金充足前提下不服务交易动作优劣判断；不以账户余额或保证金余量影响决策 | 删除模型输入 |
| M09 | `initial_margin_ratio` | 1 | 资金充足前提下不服务交易动作优劣判断；不以账户余额或保证金余量影响决策 | 删除模型输入 |
| M10 | `maintenance_margin_ratio` | 1 | 资金充足前提下不服务交易动作优劣判断；不以账户余额或保证金余量影响决策 | 删除模型输入 |
| M11 | `available_balance_ratio` | 1 | 资金充足前提下不服务交易动作优劣判断；不以账户余额或保证金余量影响决策 | 删除模型输入 |
| M12 | `long_exit_exists` | 3 | 区分从无退出历史与退出后恰好零变化，属于记忆的缺失标记 | 保留 |
| M13 | `long_exit_reason` | 2 | 退出背景可能有预测价值，但尚无必须增加该输入的独立证据；本轮不启用基于历史原因的冷却/熔断 | 删除模型输入 |
| M14 | `steps_since_long_exit_norm` | 3 | 退出记忆的新近程度；否则刚退出与很久前退出的价格变化可能相同 | 保留 |
| M15 | `price_change_since_long_exit` | 3 | 本轮重入判断的核心历史参照：当前价格相对实际退出价格是否发生变化 | 保留 |
| M16 | `signal_change_since_long_exit` | 1 | 新定义的6步/18步动量差尚无增益证据，且与既有市场趋势特征重叠，引入额外窗口选择 | 删除模型输入 |
| M17 | `short_exit_exists` | 3 | 空头记忆必须独立存在，不能被多头事件覆盖 | 保留 |
| M18 | `short_exit_reason` | 2 | 与多头退出原因同理，属于可延后的上下文 | 删除模型输入 |
| M19 | `steps_since_short_exit_norm` | 3 | 空头退出新近程度不可由当前市场向量恢复 | 保留 |
| M20 | `price_change_since_short_exit` | 3 | 空头重入的实际退出价参照，独立于多头事件 | 保留 |
| M21 | `signal_change_since_short_exit` | 1 | 与多头自定义动量差同理，先避免引入未经检验的额外信号 | 删除模型输入 |
| M22 | `long_cooldown_remaining_norm` | 3 | 决定未来多头可请求动作；当前 mask 不能表达还要等几步 | 保留 |
| M23 | `short_cooldown_remaining_norm` | 3 | 决定未来空头可请求动作，与多头分开表达 | 保留 |
| M24 | `consecutive_stop_count_norm` | 3 | 下一次实际止损是否触发熔断取决于该计数 | 保留 |
| M25 | `breaker_remaining_norm` | 3 | 有限熔断何时解除决定后续交易机会 | 保留 |
| M26 | `breaker_permanent` | 3 | 永久熔断不能与零剩余步混同 | 保留 |
| M27 | `pending_exit` | 3 | 区别已退出与退出被拒仍持仓，影响下一步强制请求 | 保留 |
| M28 | `pending_exit_reason` | 2 | 对模型而言 pending 已给出持续退出约束；具体原因由执行器维护即可 | 删除模型输入 |
| M29 | `remaining_persist_norm` | 3 | 决定下一次允许重新决策的时机，不能隐含在worker中 | 保留 |

老师删除项也逐项评分（资金充足前提下资金字段调整为1分）：

| 原ID | 老师候选字段 | 分数 | 原因 | 本轮处理 |
|---|---|---:|---|---|
| T04 | wallet_balance | 1 | 资金充足假设下不参与交易动作价值或可行性 | 删除老师状态与去重字段 |
| T06 | initial_margin | 1 | 本轮不依据账户保证金余量筛选动作 | 删除老师状态与去重字段 |
| T11 | long_exit_reason | 2 | 历史原因不服务保留模型输入，且本轮不启用依赖它的冷却/熔断规则 | 不入老师状态键，执行事件可审计 |
| T14 | long_exit_scale | 2 | 已有 long_exit_step，可回查退出决策时 scale | 删除独立存储/维度，数值按固定公式派生 |
| T15 | long_exit_signal | 1 | 对应已删除自定义动量差 | 删除 |
| T17 | short_exit_reason | 2 | 与多头历史原因同理 | 不入老师状态键，执行事件可审计 |
| T20 | short_exit_scale | 2 | 已有 short_exit_step，可回查退出决策时 scale | 删除独立存储/维度，数值按固定公式派生 |
| T21 | short_exit_signal | 1 | 对应已删除自定义动量差 | 删除 |

## 3. 模型状态表（实施集合：原有项＋3分新增项）

每行一个逻辑字段，列出实际张量宽度和对应老师状态。`P` 为动作持续配置。`C/B/K` 的公式仅解释预留冷却/熔断槽位；本轮这些规则禁用，M22–M26 无条件填零，不需要选择 C/B/K 或实现相应计时/计数。表中其3分必要性以规则启用为条件，不表示本轮有有效信号。以下所有输入只作为 TD/老师监督的条件，**不单独转成奖惩**。

| ID | 模型状态字段 | 第一轮定义与编码 | 宽度 | 当前处理 | 对应老师状态及作用 | 经验定位 DP 行的方式 | 优先级评分 |
|---|---|---|---:|---|---|---|---|
| M01 | `position_exposure` | 实际策略数量 / 最大绝对允许数量；零仓为 0 | 1 | 保留原输入 | T02 实际数量；不是请求目标数量 | 不直接用模型编码定位；对应原始 T 字段纳入 `teacher_state_key`，见老师表 | 原有基础项 |
| M02 | `single_holding_return_rate` | 当前单笔从开仓起累计交易净损益 C_t / E_ref；零仓为 0 | 1 | 保留槽位，改为资金无关口径 | T09 单笔交易损益摘要 | 派生值不入行键；用 T09 原始摘要定位 | 原有槽位调整 |
| M03 | `single_holding_max_drawdown` | 单笔累计损益历史最大回落 D_t / E_ref；零仓为 0 | 1 | 保留槽位，改为固定尺度回撤 | T09 单笔交易损益摘要；不是账户回撤 | 派生值不入行键；用 T09 原始摘要定位 | 原有槽位调整 |
| M04 | `holding_duration_norm` | 保留 `min(duration / holding_duration_norm_steps, 1)`；零仓为 0 | 1 | 保留 | T08 原始持仓步数 | 不直接用模型编码定位；对应原始 T 字段纳入 `teacher_state_key`，见老师表 | 原有基础项 |
| M05 | `price_adverse_return` | 持仓时 `max(0,-unrealized_pnl)/(abs(q)*P_t)`；零仓为 0 | 1 | 新增 | T02/T05 和当前行情派生；对应硬止损使用的逆向价格率 | 派生/归一化摘要不入行键；用对应原始 T 状态定位，禁止从该 float32 值反推原态 | 3 |
| M12 | `long_exit_exists` | 最近一次实际多头完整退出是否存在 | 1 | 新增 | T10 存在位 | 不直接用模型编码定位；对应原始 T 字段纳入 `teacher_state_key`，见老师表 | 3 |
| M14 | `steps_since_long_exit_norm` | `min(t-long_exit_step,256)/256`；无记录为 0 | 1 | 新增 | T12 原始退出步 | 派生/归一化摘要不入行键；用对应原始 T 状态定位，禁止从该 float32 值反推原态 | 3 |
| M15 | `price_change_since_long_exit` | `b((P_t-long_exit_price)/long_exit_scale)`；无记录为 0 | 1 | 新增 | T13 退出价；尺度由 T12 退出步回查行情派生 | 派生/归一化摘要不入行键；用对应原始 T 状态定位，禁止从该 float32 值反推原态 | 3 |
| M17 | `short_exit_exists` | 最近一次实际空头完整退出是否存在 | 1 | 新增 | T16 存在位 | 不直接用模型编码定位；对应原始 T 字段纳入 `teacher_state_key`，见老师表 | 3 |
| M19 | `steps_since_short_exit_norm` | `min(t-short_exit_step,256)/256`；无记录为 0 | 1 | 新增 | T18 原始退出步 | 派生/归一化摘要不入行键；用对应原始 T 状态定位，禁止从该 float32 值反推原态 | 3 |
| M20 | `price_change_since_short_exit` | `b(-(P_t-short_exit_price)/short_exit_scale)`；无记录为 0 | 1 | 新增 | T19 退出价；尺度由 T18 退出步回查行情派生 | 派生/归一化摘要不入行键；用对应原始 T 状态定位，禁止从该 float32 值反推原态 | 3 |
| M22 | `long_cooldown_remaining_norm` | long_remaining / max(C,1)，无冷却为 0 | 1 | 新增 | T22 精确剩余步；仅表示现有单方向规则 | 不直接用模型编码定位；对应原始 T 字段纳入 `teacher_state_key`，见老师表 | 3 |
| M23 | `short_cooldown_remaining_norm` | short_remaining / max(C,1)，无冷却为 0 | 1 | 新增 | T23 精确剩余步 | 不直接用模型编码定位；对应原始 T 字段纳入 `teacher_state_key`，见老师表 | 3 |
| M24 | `consecutive_stop_count_norm` | `min(stop_count,K)/max(K,1)`；规则禁用填 0 | 1 | 新增 | T24 精确计数；老师/去重不截顶 | 不直接用模型编码定位；对应原始 T 字段纳入 `teacher_state_key`，见老师表 | 3 |
| M25 | `breaker_remaining_norm` | 有限剩余步 / max(B,1)；永久或不生效均填 0，靠 M26 区分 | 1 | 新增 | T25 原始计时器 | 不直接用模型编码定位；对应原始 T 字段纳入 `teacher_state_key`，见老师表 | 3 |
| M26 | `breaker_permanent` | 断路器原始计时器等于 -1 时为 1，否则为 0 | 1 | 新增 | T25 派生；永久不能误编码成无冷却 | 派生/归一化摘要不入行键；用对应原始 T 状态定位，禁止从该 float32 值反推原态 | 3 |
| M27 | `pending_exit` | 已锁存退出请求但实际仓位未归零，取 0/1 | 1 | 新增 | T26；覆盖止损请求后被拒绝的情形 | 不直接用模型编码定位；对应原始 T 字段纳入 `teacher_state_key`，见老师表 | 3 |
| M29 | `remaining_persist_norm` | 精确剩余动作持续步 / max(P-1,1) | 1 | 新增 | T28；不允许 worker 保有隐形计数 | 不直接用模型编码定位；对应原始 T 字段纳入 `teacher_state_key`，见老师表 | 3 |
| M30 | `previous_action` | 沿既有独立网络入口编码；以实际仓位/杠杆映射的动作语义为准 | 1（独立入口） | 保留并审计拒单后的更新 | T02/T03 派生，不能记成未成交请求 | `previous_action` 作仓位映射一致性校验；实际 T02/T03 入状态键，不再单独决定行 | 原有基础项 |
| M31 | `action_mask_flat` | 当前 Flat 请求是否合法，0/1 | 1（mask） | 保留并合入共享约束 | T33 派生；合法不保证成交 | 不入行键；查到行后校验合法动作列/label-valid | 原有基础项 |
| M32 | `action_mask_long` | 当前 Long 请求是否合法，0/1 | 1（mask） | 同上 | T34 派生 | 不入行键；查到行后校验合法动作列/label-valid | 原有基础项 |
| M33 | `action_mask_short` | 当前 Short 请求是否合法，0/1 | 1（mask） | 同上 | T35 派生 | 不入行键；查到行后校验合法动作列/label-valid | 原有基础项 |

**筛选后宽度：**保留四个基础槽位（其中收益/回撤口径调整），新增 14 个原评分3分标量，`TRADING_INFO_DIM=18`。其中 M22–M26 为五个固定零预留槽位，实际非恒定输入为原有4项＋其余9个新增项；保留槽位不代表实现冷却、熔断。原方案41维不再实施。previous_action、市场/时间入口与三个 mask 仍保持独立。表中保留原 M 编号以便追踪，编号空缺代表本轮删除，实际编码按保留行顺序连续拼接。

M07/M08/M09/M10/M11 均不输入模型，资金充足教学环境也不以这些值筛选动作或产生强平。实际账户资金限制由独立执行层处理，不反向混入本轮教学目标。以上精简是首轮设计优先级判断，不是经训练验证的特征重要性排名。

## 4. 老师状态表与新增维度

“新增维度”指相对于旧 `(time, previous_action)` 的状态查询键增加逻辑分量，不意味着必须增加 NumPy 稠密轴。派生字段不重复增加独立轴。老师首轮采用**完整精确状态的稀疏 DP**，保留3分执行状态以及恢复资金无关模型观测所需的原始信息。删去退出原因历史、自定义动量历史和可由退出步回查的价格尺度独立字段。本轮不以固定后续策略配对回放替换 DP。

| ID | 老师状态字段 | 保存内容 | 是否新增独立维度 | 原因及模型对应 | 经验携带字段/恢复来源 | DP 行定位用途 | 老师侧评分 |
|---|---|---|---|---|---|---|---|
| T01 | `time_index` | 当前环境步 t，完整市场路径由数据身份引用 | 已有轴 | 市场/时间输入来源 | `step_index`（当前 df 的绝对行位置） | 行键：行情位置，不是采样窗口内偏移 | 基础/派生/定位项 |
| T02 | `actual_position_qty` | 已成交精确策略数量 | 已有仓位轴的精确化 | 旧 previous_action 可编码档位，但新查询不能假定请求已成交；对应 M01/M30 | `teacher_state_snapshot.actual_position_qty`；经验携带快照引用及 `teacher_state_key` | 行键：原精度字段纳入 `teacher_state_key` | 基础/派生/定位项 |
| T03 | `actual_leverage` | 当前杠杆 | 已有动作映射信息；本轮固定参数 | 保留恢复信息；对应 M30，非新增可变轴 | `teacher_state_snapshot.actual_leverage`；经验携带快照引用及 `teacher_state_key` | 行键：原精度字段纳入 `teacher_state_key` | 基础/派生/定位项 |
| T05 | `unrealized_pnl` | 精确浮动盈亏 | **新增** | 交易价格损益与硬止损仍需真实浮盈亏；对应 M05，不代表资金可用性 | `teacher_state_snapshot.unrealized_pnl`；经验携带快照引用及 `teacher_state_key` | 行键：原精度字段纳入 `teacher_state_key` | 3：恢复交易转移或保留观测必需 |
| T07 | `entry_price` | 当前方向有效开仓成本，零仓规范为 0 | **新增** | 执行器需要精确成本；模型不再单独输入成本偏离 | `teacher_state_snapshot.entry_price`；经验携带快照引用及 `teacher_state_key` | 行键：原精度字段纳入 `teacher_state_key` | 3：恢复交易转移或保留观测必需 |
| T08 | `holding_duration_steps` | 未裁剪整数持仓步数 | **新增** | 恢复 M04，不因归一化饱和丢失历史 | `teacher_state_snapshot.holding_duration_steps`；经验携带快照引用及 `teacher_state_key` | 行键：原精度字段纳入 `teacher_state_key` | 3：恢复交易转移或保留观测必需 |
| T09 | `holding_pnl_summary` | 当前单笔累计交易净损益 C_t、历史峰值 H_t、最大回落 D_t | **新增复合状态** | 恢复资金无关 M02/M03；不含钱包、保证金或账户权益 | `teacher_state_snapshot.holding_pnl_summary`；经验携带引用及状态键 | C_t/H_t/D_t 原精度纳入状态键 | 3：恢复保留观测必需 |
| T10 | `long_exit_exists` | bool | **新增** | M12；不存在时该方向其他字段规范零 | `teacher_state_snapshot.long_exit_exists`；经验携带快照引用及 `teacher_state_key` | 行键：原精度字段纳入 `teacher_state_key` | 3：恢复交易转移或保留观测必需 |
| T12 | `long_exit_step` | 原始退出步 | **新增** | M14，由 t 与该值派生时长 | `teacher_state_snapshot.long_exit_step`；经验携带快照引用及 `teacher_state_key` | 行键：原精度字段纳入 `teacher_state_key` | 3：恢复交易转移或保留观测必需 |
| T13 | `long_exit_price` | 完整退出的成交加权价 | **新增** | M15 | `teacher_state_snapshot.long_exit_price`；经验携带快照引用及 `teacher_state_key` | 行键：原精度字段纳入 `teacher_state_key` | 3：恢复交易转移或保留观测必需 |
| T16 | `short_exit_exists` | bool | **新增** | M17 | `teacher_state_snapshot.short_exit_exists`；经验携带快照引用及 `teacher_state_key` | 行键：原精度字段纳入 `teacher_state_key` | 3：恢复交易转移或保留观测必需 |
| T18 | `short_exit_step` | 原始退出步 | **新增** | M19 | `teacher_state_snapshot.short_exit_step`；经验携带快照引用及 `teacher_state_key` | 行键：原精度字段纳入 `teacher_state_key` | 3：恢复交易转移或保留观测必需 |
| T19 | `short_exit_price` | 完整退出的成交加权价 | **新增** | M20 | `teacher_state_snapshot.short_exit_price`；经验携带快照引用及 `teacher_state_key` | 行键：原精度字段纳入 `teacher_state_key` | 3：恢复交易转移或保留观测必需 |
| T22 | `long_cooldown_remaining` | 原始非负整数 | **新增** | 影响未来动作集合；M22 | `teacher_state_snapshot.long_cooldown_remaining`；经验携带快照引用及 `teacher_state_key` | 行键：原精度字段纳入 `teacher_state_key` | 3：恢复交易转移或保留观测必需 |
| T23 | `short_cooldown_remaining` | 原始非负整数 | **新增** | 同上；M23；任一时刻至多一个方向为正 | `teacher_state_snapshot.short_cooldown_remaining`；经验携带快照引用及 `teacher_state_key` | 行键：原精度字段纳入 `teacher_state_key` | 3：恢复交易转移或保留观测必需 |
| T24 | `consecutive_stop_count` | 原始整数，实际止损退出次数 | **新增** | 决定下一次是否熔断；M24 | `teacher_state_snapshot.consecutive_stop_count`；经验携带快照引用及 `teacher_state_key` | 行键：原精度字段纳入 `teacher_state_key` | 3：恢复交易转移或保留观测必需 |
| T25 | `breaker_remaining` | -1 永久、0 不生效、正数有限剩余步 | **新增** | M25/M26 | `teacher_state_snapshot.breaker_remaining`；经验携带快照引用及 `teacher_state_key` | 行键：原精度字段纳入 `teacher_state_key` | 3：恢复交易转移或保留观测必需 |
| T26 | `pending_exit` | bool，未完成退出请求 | **新增** | M27；退出被拒不能变空仓 | `teacher_state_snapshot.pending_exit`；经验携带快照引用及 `teacher_state_key` | 行键：原精度字段纳入 `teacher_state_key` | 3：恢复交易转移或保留观测必需 |
| T27 | `pending_exit_reason` | 锁存原因枚举或 none | **新增** | 模型只输入 M27；老师保留精确原因以恢复 pending 锁存、原因优先级和退出事件；本轮不更新止损计数 | `teacher_state_snapshot.pending_exit_reason`；经验携带快照引用及 `teacher_state_key` | 行键：原精度字段纳入 `teacher_state_key` | 3：恢复交易转移或保留观测必需 |
| T28 | `remaining_persist` | 原始剩余步数 | **新增** | M29；未来可重新决策的时机依赖该值 | `teacher_state_snapshot.remaining_persist`；经验携带快照引用及 `teacher_state_key` | 行键：原精度字段纳入 `teacher_state_key` | 3：恢复交易转移或保留观测必需 |
| T29 | `persisted_target_action` | 被持续执行的目标动作，计数为零时规范为空 | **新增** | 当目标未完全成交时不能仅用实际仓位推断；经当前 mask 体现为唯一可请求目标，执行器仍处理拒单；不新增自由策略动作 | `teacher_state_snapshot.persisted_target_action`；经验携带快照引用及 `teacher_state_key` | 行键：原精度字段纳入 `teacher_state_key` | 3：恢复交易转移或保留观测必需 |
| T33 | `mask_flat` | 当步 Flat 合法性 | 派生，不加轴 | M31 | 从快照和当前 df 行重算 | 不新增行键；用于状态/合法动作校验 | 基础/派生/定位项 |
| T34 | `mask_long` | 当步 Long 合法性 | 派生，不加轴 | M32 | 从快照和当前 df 行重算 | 不新增行键；用于状态/合法动作校验 | 基础/派生/定位项 |
| T35 | `mask_short` | 当步 Short 合法性 | 派生，不加轴 | M33 | 从快照和当前 df 行重算 | 不新增行键；用于状态/合法动作校验 | 基础/派生/定位项 |
| T36 | `contract_id` | 合约身份 | 行情归属，不增加交易状态轴 | 合约规则、tick 和数量口径来源 | 从 `df_id` 对应行情读取 | 由 df 定位并校验合约归属 | 基础/派生/定位项 |
| T37 | `candidate_action` | 当前评估动作 Flat/Long/Short | 原有输出动作轴，不是状态轴 | 不能写成 replay 已执行行为动作 | 老师输出动作列；读取单动作时使用 `request_action` | 仅选列，不选行；完整 KL 读取整行 | 基础/派生/定位项 |

老师侧本轮新增 **18 个3分逻辑状态分量**：原 T04–T29 删除 T04/T06/T11/T14/T15/T17/T20/T21 后的保留项。T09 改为三个交易损益统计量组成的复合状态。老师表保留原 T 编号，空缺字段不进入快照键。

按第1.2节，T22/T23/T24/T25 本轮固定为整数零（T25 不使用 -1），只保留快照格式，不产生可变状态分支。表中冷却、连续止损和断路器用途是条件说明，不是本轮实施任务。T27 仍保留精确 pending 原因以恢复执行语义和审计，不据此实现止损计数或熔断。

老师与模型分开评分：`entry_price`、`unrealized_pnl`、`pending_exit_reason` 对交易转移是3分，即使其额外模型编码只有2分，也必须保留在老师原始状态。`long_exit_scale/short_exit_scale` 的数值仍用于价格变化计算，但由退出步回查当时行情得到，不作为独立状态存储。

原始执行状态必须可以恢复所有用于 transition 和下一 observation 的信息。T09 按本轮固定尺度定义保存 C_t/H_t/D_t，不再重建旧账户分母指标；新旧指标不得混用。价格/金额遵循现执行器精度与单位规范，不先转 float32 再作为精确状态。digest 是内容索引，须保留可恢复对象和规范序列化约定；随机订单/worker/episode ID 不是状态等价依据。

`retained_teacher_fields` 明确为：`T02, T03, T05, T07, T08, T09, T10, T12, T13, T16, T18, T19, T22, T23, T24, T25, T26, T27, T28, T29`。它不包含已删除的退出历史原因、锁存尺度和动量值。当前/下一状态的精确去重键使用同一集合。

### 4.1 DP 递推与计算边界

```text
state_key = (df_id, time_index, exact(retained_teacher_fields))
teacher[state_key, candidate_action] -> Q_T, valid
```

对每个合法候选动作，由共享执行器计算 `s_next, r, terminated`，进行：

`Q_T(s,a) = r(s,a) + gamma * (1-terminated) * max_b Q_T(s_next,b)`。

下一步只能在同一执行器给出的合法请求集合中取 max。持续动作和 pending_exit 会收窄集合；硬风险优先于持续动作。拒绝成交是实际转移，不能假造平仓成功。强制处置状态应提供执行器的唯一安全请求，不允许空 mask 后仍做任意 argmax。终点使用与训练相同结算，禁止老师额外免费平仓。

老师读取离线未来路径计算 Q，学生观测只读取当前及过去。该 Q 是指定历史路径、执行规则及有限 episode 下的全知行情目标，不是学生在线可实现收益保证。新增退出历史未必改变全知 DP 最优值；加入它们是首轮完整状态契约，不能假定每个历史差异都应产生不同标签。

稀疏存储不保证计算可行。精确持仓损益与交易记忆仍可能使可达状态数接近动作树规模。实现首先做小窗口规模试验，并登记最大状态数、内存与耗时预算；超预算明确失败，不静默丢维、分桶或换成固定策略 rollout。资源不足时必须另立近似老师方案及误差实验，不能把未算出的状态填成旧表值或零值。

**已确认的首次资源门槛：**使用冻结配置依次运行 16、32、64 个有效转移步的短 episode；每次至多 250,000 个唯一精确状态、2 GiB（2,147,483,648 字节）状态存储、60 秒 DP 计算墙钟时间。状态存储包括可恢复快照、内容索引、Q 行及递推缓存，另报告进程峰值 RSS；计时覆盖状态展开和回溯，不计数据读取与预热，并单列这些开销。任一预算触顶即中止该次试验，停止自动扩大窗口并报告触顶维度、已展开状态数、内存和耗时；未完成的根状态/动作不发布为有效标签。每档试验从同一已声明初态开始并记录数据坐标、配置与状态计数口径；短窗口通过仅证明该窗口可计算，不证明正式训练 episode 可计算，也不自行改变终点结算语义。

### 4.2 经验如何定位 DP 的一行

旧查询是 `q_table_for_df[step_index][previous_action]`，返回该状态的全部动作 Q 值。扩维后，同一 df、时间位置和 previous_action 可能对应不同持仓浮盈亏、入场成本、退出记忆和冷却，必须增加精确状态键。**本轮冻结行定位为 `(df_id, step_index, teacher_state_key)`；请求动作只选择该行中的列。**

下表明确经验需要保存的定位字段。定位与去重限定在本次训练使用的同一组行情和固定执行、奖励、折扣、动作及终止配置内；这些配置是计算前提，不再作为经验字段或 DP 行键。df 映射和状态引用必须随 replay 导出可恢复。

| 经验定位字段 | 保存位置 | 是否必需 | 定位哪一部分 | 具体约定 |
|---|---|---|---|---|
| `df_id` | 样本或轨迹头 | 必需 | 选择行情文件/分片 | 本次训练中行情文件/分片的唯一标识，固定其内容与行顺序；同合约不同 df 不混淆 |
| `df_index` | compact 样本/轨迹头 | 现有紧凑路径保留 | 本次进程内的 df 查找编号 | 经本次训练的 `df_index -> df_id` 映射解析；重新加载时恢复这份映射 |
| `step_index` | 样本 | 必需 | df 内当前行情行 | 零起始绝对位置，与环境 day/time_index 明确映射；不是 minibatch 下标、自然时间或截取窗口内偏移 |
| `previous_action` | 样本，原字段保留 | 必需校验 | 校验实际持仓档位 | 与快照 T02/T03 在 action spec 下的映射一致；不能代替新增状态键 |
| `teacher_state_key` | 样本 | **新增必需** | 区分同 df 同时间的交易路径状态 | 规范原精度序列化老师表保留的 T02–T29 字段 后计算内容 digest；与第6节 `exact_state_digest` 使用同一序列化和同一值，不另造第二种键 |
| `teacher_state_snapshot_ref` | 样本/共享状态库 | **新增必需** | 恢复状态与核验键 | 指向老师表保留的 T02–T29 原始对象；T09 交易损益摘要也必须可恢复。只保存不可逆 hash 不足以补算老师 |
| `request_action` | 样本 | 读取单动作值必需 | 选择 Q 列 | 行查找不依赖它；按冻结 action spec 映射列序，不能假定 Flat/Long/Short 的整数顺序 |
| `next_step_index` | n-step 样本 | 后续状态定位必需 | 定位 n-step 终点行情 | 配套 next_df_id，不默认当前 t+1，也不只靠 n_actual 猜行号 |
| `next_df_id` | 样本或轨迹头 | 后续状态定位必需 | 终点 df | 本轮同合约连续 episode 通常与当前相同，但仍要能明确解析 |
| `next_teacher_state_key` | n-step 样本 | 后续状态定位必需 | 终点精确状态 | 与 `next_exact_state_digest` 相同；不能用当前状态键查询下一行 |
| `next_teacher_state_snapshot_ref` | 样本/共享状态库 | 后续恢复必需 | 恢复终点状态 | 与 next key 一致；终止样本保存审计状态，但无需为无 bootstrap 的目标查下一 Q |

```text
# 当前状态老师整行（KL 等监督使用整行，而不是仅行为动作一列）
df_id = df_lookup[df_index]
assert previous_action == action_spec.encode(snapshot.actual_position_qty,
                                             snapshot.actual_leverage)
assert teacher_state_key == exact_state_digest(snapshot.retained_teacher_fields)
row_key = (df_id, step_index, teacher_state_key)
q_row, label_valid_row = teacher.lookup(row_key)

# 如需当前请求动作对应的老师标量，再选列
q_for_request = q_row[action_spec.column(request_action)]
```

这与4.1的展开键等价：`df_id + step_index` 定位外生行情；`teacher_state_key` 压缩精确仓位及全部新增交易状态。保留模型字段的18维编码不用于重建该键，因为归一化、截顶和 float32 可能丢失信息。老师表中的“经验携带字段/恢复来源”“DP 行定位用途”逐项说明了各 T 字段如何进入这个键。

**缺行处理：**离线构建器必须从经验完整快照展开合法后续状态、完成 DP 后原子发布有效行；线上训练取样不得退回旧 `(df,time,previous_action)` 行，也不得默认填零。当前样本没有有效老师行时标记 `teacher_label_available=false` 并屏蔽该样本教师 loss，TD 仍按自身目标计算；记录覆盖率。该屏蔽是缺标签处理，不代表已完成老师扩维验收；声称全样本教师训练前须验证计划监督集合的行全部可定位。超预算按4.1失败报告，不悄然更换目标。

**与去重的区别：**同一个 DP 行可对应三个请求动作，三个动作属于不同经验起始组 D；相同起点和动作的不同 n-step 结果 O 仍共用同一个当前老师行。未来动作路径、累计奖励、n_actual 和 next state 不决定当前老师行。

## 5. 状态生命周期与执行规则

1. **初始化与隔离：**episode/reset、合约切换、Train/Valid/Test 独立账户起点均清除交易记忆、pending 与风险计数；纯历史行情预热不继承交易记忆。连续运行恢复必须恢复完整快照，不能仅载入模型。
2. **实际退出：**非零仓位归零才写退出记录；请求 Flat、触发止损或拒单都不是退出。首轮保留现有禁止直接反向开仓语义，必须先完成退出再开反向仓位。部分减仓不写完整退出事件；同步执行中的部分成交依据实际仓位记账，残余退出请求继续保持。第一轮不引入异步挂单撮合系统。
3. **退出时间：**在 `t -> t+1` 成交完成的退出事件记为 `exit_step=t+1`，首次返回空仓观测时距退出为 0，下一有效步为 1。退出价格取本次完整退出转移实际成交加权价；价格尺度使用退出步前一行 `s_t` 可得的 scale；由退出步回查该行，既不改用当前波动率，也不额外存一个尺度状态。跨步部分退出不累计前期减仓成交价混成另一口径。
4. **双向记忆：**多头退出只覆盖多头记录，空头退出只覆盖空头记录。新开仓不清除上次退出；无 TTL 自动失效，模型步数编码到 256 后饱和，原始退出步保留用于老师/去重。
5. **待退出：**策略主动平仓或硬风险处置若未完成，锁存 pending。首轮采用“持续请求完成退出”语义，pending 期间不能撤回并重新加仓；硬止损原因可覆盖策略退出原因，直到实际退出。强平优先、硬止损次之、终端处置再次、策略退出最后；执行器记录唯一实际原因。清零仓位的同一 transition 写事件、消费 pending，重复调用不再次计数。
6. **冷却禁用：**本轮多空冷却剩余步始终为零，不创建计时器，不因历史止损禁止同向重入。pending 仍要求先完成实际退出；退出完成后的合法动作沿当前执行规则处理。
7. **连续止损和熔断禁用：**本轮连续止损计数和断路器剩余步始终为零，不递增计数、不触发有限或永久熔断。实际止损事件仍按成交事实记录且重试不得重复产生退出事件；这不是冷却/熔断状态机。
8. **持续动作：**读取传入的 `action_persistence`，本轮 P=2；新决策后剩余 P-1，后续有效步递减。保存目标动作 T29，不能只存计数。硬风险/pending 立即打断持续动作，清零计数及目标。限制的是可请求动作，成交依然可能被拒。M31–M33 体现合成后的请求限制，实际可成交性另由执行器检查。
9. **有效配置：**共享执行器迁移后同一配置的训练、老师、回测必须同构。第一轮不启用路由专家选择、OOD 门控或空槽策略；上述状态均归属独立 Low Agent 的执行环境。

## 6. 经验去重表

本次经验池内，去重分为起始决策组 D 与后续结果 O；同一分格内 `(D,O)` 全部相同才是重复经验。老师标签按当前 DP 行与候选动作单独定位，表中 L 表示标签定位层，不用 teacher Q 数值区分环境事实。

| 去重维度 | 属于哪层 | 精度/比较方式 | 第一轮口径 | 是否参与 DP 行定位 |
|---|---|---|---|---|
| regime grid | 池分区 | 整数 | 沿原分格容量策略；事后分层标签不进入网络输入 | 否：模型输入校验、结果去重或标签/诊断内容 |
| 合约身份 | D | 精确 ID | 不合并不同合约的同值向量 | 是：由 df 定位并校验合约归属 |
| 当前市场坐标 | D | 不可变 `(df_index,step_index)` | compact 与完整路径均携带坐标；完整路径同时验证市场内容 | 是：`df_id + step_index` |
| 当前市场向量 | D | 模型 float32 规范字节，或坐标引用同一向量 | 不再跨不同时间仅因向量相同而合并 | 否：模型输入校验、结果去重或标签/诊断内容 |
| 当前时间输入 | D | 模型实际 dtype/编码 | funding hour/minute 等现有输入全部保留 | 否：模型输入校验、结果去重或标签/诊断内容 |
| previous_action | D | 离散值精确比较 | 使用实际仓位映射语义 | 只校验 T02/T03 映射，不独立选行 |
| 当前保留交易向量 | D | 18维最终 float32 字节 | 包括保留的双方向记忆、存在位和执行约束；不另做粗桶 | 否：模型输入校验、结果去重或标签/诊断内容 |
| 当前动作 mask M31–M33 | D | 三个位精确比较 | 相同观测但请求集合不同，不同组 | 否：模型输入校验、结果去重或标签/诊断内容 |
| 当前原始执行状态 digest | D | 保留的老师原始字段规范原精度序列化 | 即便 float32/裁剪值相同，精确交易损益、历史或约束不同仍不同组 | 是：直接复用为 `teacher_state_key` |
| 请求目标动作 a_t | D | 离散整数 | TD 学习请求动作经执行器映射后的结果；不偷换成成交后的仓位码 | 仅选 Q 值列，不影响当前 DP 行 |
| n-step 累计折扣奖励 R_n | O | learner float32，正负零统一 | 相同起点不同累计回报是不同后续 | 否：模型输入校验、结果去重或标签/诊断内容 |
| 实际累积步数 n_actual | O | 整数 | 显式区分短窗和满窗 | 否：模型输入校验、结果去重或标签/诊断内容 |
| effective bootstrap_discount | O | learner 实际系数 | 本轮沿 ADR-0065：满窗非终止用 gamma^n_actual，真实终止或既有短窗关闭bootstrap时为0 | 否：模型输入校验、结果去重或标签/诊断内容 |
| terminated | O | bool | 环境真实终止与采样截断区分保存 | 否：模型输入校验、结果去重或标签/诊断内容 |
| truncated | O | bool | 只增可观测元数据，本轮不改旧短窗禁bootstrap语义 | 否：模型输入校验、结果去重或标签/诊断内容 |
| 下一市场坐标 | O | 不可变坐标 | 指 n-step 终点，不是固定 t+1 | 用于下一状态行定位，不定位当前行 |
| 下一市场向量 | O | float32或坐标引用 | 与下一坐标一致 | 否：模型输入校验、结果去重或标签/诊断内容 |
| 下一时间输入 | O | 最终输入编码 | 后续 Q 条件的一部分 | 否：模型输入校验、结果去重或标签/诊断内容 |
| 下一 previous_action | O | 离散精确 | 实际终点仓位语义 | 否：模型输入校验、结果去重或标签/诊断内容 |
| 下一交易向量 | O | 18维 float32 | 新记忆/约束差异不得被覆盖 | 否：模型输入校验、结果去重或标签/诊断内容 |
| 下一动作 mask | O | 位精确 | bootstrap 可行动作不同即不同 outcome | 否：模型输入校验、结果去重或标签/诊断内容 |
| 下一原始执行状态 digest | O | n-step 终点保留老师字段的原精度内容 | 不仅比较下一模型摘要 | 用于下一状态行定位，不定位当前行 |
| teacher candidate action | L | 候选动作轴 | 不是 D 中的 behavior/request action | 仅选 Q 值列，不影响当前 DP 行 |
| teacher label-valid | L | 每动作有效位 | 缺标签不得伪造0价值；不得以标签存在性重定义环境事实 | 否：模型输入校验、结果去重或标签/诊断内容 |
| teacher Q/优势值 | 不入身份 | 标签内容 | 同一 L 更新须可追溯，不能用数值差增加重复经验 | 否：模型输入校验、结果去重或标签/诊断内容 |
| TD error/采集误差/优先级 | 不入身份 | 诊断元数据 | 不用其比较不同 outcome 的保留价值 | 否：模型输入校验、结果去重或标签/诊断内容 |
| worker/随机episode/订单ID | 不入身份 | 只作来源审计 | 同一物理内容由不同worker采到仍可去重 | 否：模型输入校验、结果去重或标签/诊断内容 |

```text
D = (contract, market_coordinate, model_input_digest, exact_state_digest, request_action)
O = (R_n, n_actual, effective_bootstrap_discount, terminated, truncated,
     next_market_coordinate, next_model_input_digest, next_exact_state_digest)
record_identity = (grid_id, D, O)
label_identity = (df_id, step_index, exact_state_digest, candidate_action)
```

模型浮点先转最终 float32，规范 -0/+0，拒绝 NaN/Inf；精确状态保持执行器原精度并规范 -0/+0，整数不转低精度浮点，存在位为 false 的空事件规范化。序列化固定字段顺序、dtype、shape 和字节序。digest 使用稳定内容 hash，并保留原对象以核验冲突和恢复；不能用 Python 进程随机 hash 作为持久身份。

**保留政策不变：**每个 D 最多驻留 4 个不同 O；完全相同驻留经验合并；第 5 个不同 O 用独立 seed RNG 随机替换已有 O。新组满格沿现有 FIFO，索引随淘汰回收。不是收益择优，也不是全历史等概率蓄水池。精确状态扩维会减少重复率、增加组数，必须报告组数、重复率、FIFO 和组内替换次数，不通过临时粗桶“修复”容量压力。

## 7. 后续实施与验收入口

1. 按资金充足假设实现状态对象、教学执行更新器及18维编码器；按表固定字段顺序、数据类型和编码宽度。先对单个精确快照验证可恢复与逐步一致，再改训练批次。
2. 同步 Qnet、rollout、next observation、CPU/GPU replay、compact IPC、预训练、多样化训练、独立回测；旧 checkpoint/replay 形状不兼容时拒绝静默补零。
3. 教师先按第4.1节的16/32/64步、25万状态、2 GiB、60秒门槛做可达状态规模测试，再接训练；保存峰值状态数、内存、耗时和失败边界。未通过资源门槛不能声称完整老师已可训练。
4. 必测：普通空仓与刚退出不同；多空记忆不覆盖；请求未成交仍持仓；重试不重复生成实际退出事件；M22–M26、T22–T25 在 reset/止损/退出/恢复后均为零且不限制动作；持续目标拒单后仍可恢复；跨合约/分区不串状态。
5. 必测：相同模型float32但不同精确持仓损益/退出步不能合并；相同事实不同worker仍合并；不同n_actual/终点/折扣不合并；同组最多4条、随机轮换和索引回收；compact与完整路径身份一致。
6. 必测：只改变外部账户余额，观测、mask、teacher Q及去重结果不变；交易费用和价格亏损仍改变奖励。奖励沿[执行拆账规范](../specs/low-agent-reward-value-diagnostics.md)对账；本轮不发明退出奖励、亏损持仓惩罚或重入惩罚。新状态对收益是否有效仍需匹配消融，不由状态表推断。

### 7.1 用户补充的独立评估与高层路由范围

- **按索引独立评估：**`FineFT/RL/DiHFT/low_level/test_agent_index.py` 必须随状态迁移，覆盖 ensemble 构造、新 checkpoint 加载、按 context 的18维真实前向、环境初始化、共享 pending/动作持续、实际执行和指标/明细输出。其 shell 入口和受影响的平均评估等共享调用者也须检查，不能只改训练后留下不可运行的评估入口。
- **高层 VAE 路由：**`FineFT/RL/DiHFT/high_level/vae_routing_util.py` 必须随状态迁移，覆盖专家模型加载、实时18维观测、专家切换时交易记忆保留、共享执行状态同步和 reset 隔离；不重训 VAE、不改变既有专家选择算法。这里扩展的是模型消费与执行一致性范围，第5.9节“不启用路由专家选择”的边界仍指独立 Low Agent 教学环境本身。
- **本轮路由参数：**经现有 shell/CLI 链路显式传入 P=2、止损0.03、初始资金，并关闭冷却与熔断；不能继承路由旧默认值重新启用禁用规则。既有其他高层运行保持其可配置风控行为，不全局删除或改写历史默认配置。
- **检验与测试：**更新 `FineFT/tests/rl/test_test_agent_index.py`、`test_qnet_trading_info.py`、`test_evaluation_stop_loss.py` 及交易过程状态、路由动作持续/防御/风控等受影响测试中的4维假设、stub和fixture；检验字段值、真实状态演化、checkpoint兼容性与输出指标，不以改长度或全mock前向充当验收。用确定性新维度权重做短程真实前向与执行验证，无需等待正式重训。

这两个入口分别形成可与经验池、DP老师工作并行的迁移工单，最终集成验收必须同时覆盖训练、独立评估与高层路由；独立评估和路由不查询未来老师Q。

本文件只形成设计与源码依据。本轮未修改训练代码、执行规则、已接受 ADR 或启动训练。
