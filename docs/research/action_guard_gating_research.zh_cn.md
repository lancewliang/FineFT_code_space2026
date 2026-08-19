# FineFT 动作守卫与未来门控风险规则研究

**研究日期**：2026-08-18  
**范围**：当前 `RollingLabelActionGuard`、商品期货环境可用动作/成交约束，以及它们未来并入智能体门控时应具备的规则。  
**结论适用边界**：研究和回测系统设计，不构成实盘交易或法律意见；交易所、期货公司和账户规则必须在运行时从正式接口/公告加载，不能把本文数值写死。

## 1. 执行摘要

当前“动作守卫”作为 **Stage II validation 的标签方向行为约束实验**是合理的：它保留模型原始动作，在模型与环境之间限制逆 Label 调仓，并由环境继续处理保证金、订单簿深度与涨跌停。测试也完整覆盖了当前规格；本次在 `finetf` 环境运行三个定向测试文件，结果为 **52 passed**。

但它目前**不能被视为实盘风险保底层**，也不应原样升级成未来门控的硬规则。最重要的原因有七个：

1. 3% “止损”只在某次动作先触发逆向配额超限时才检查；亏损已越线但动作未超配额时仍会 `allowed`。它是配额降级分支，不是独立止损。
2. 守卫按交给环境的最终动作 ID 记历史，而不是按实际成交仓位变化记历史。涨跌停、深度不足或保证金拒绝时，动作可能没有成交却消耗配额；日志也可能写 `stop_loss_close`，实际持仓却未平。
3. 当前标签切片由整段序列的双向 `filtfilt`、未来转折点和整段斜率生成。把这些 validation Label 当成在线门控已知状态会产生未来信息泄漏；未来门控必须使用时点 `t` 当时可得的因果预测、置信度和标签年龄。
4. 当前环境仍使用 `mark_price × position` 计算名义价值、保证金和盈亏，而 FU 官方合约单位是 10 吨/手；商品配置虽记录 `contract_unit=10`，环境路径没有消费它。未统一“手、吨、价格乘数、手续费口径”前，任何保证金/止损阈值都不能作为实盘硬约束。
5. 当前硬约束主要覆盖单步保证金、盘口深度、反手开关和涨跌停，尚缺交易时段/合约状态、动态保证金、持仓与日内开仓限额、交割月规则、账户组合敞口、日损/回撤熔断、未成交委托、数据陈旧、报撤单与自成交控制、订单回报对账等关键规则。
6. 如果未来在训练中启用滚动门控，窗口历史、标签置信度、干预状态若不进入 observation/replay，智能体看到的状态将不是充分状态；训练必须明确记录“建议动作、批准动作、订单动作、实际成交动作”，并用实际执行结果更新状态。
7. 当前执行层的涨跌停布尔值不是决策时点的独立交易状态，而是由整根 bar 内 `limit_*_single_sided_ratio > 0` 派生；10/30min 窗口内只要曾经锁板，bar 末即使已经开板也可能被整步禁买/禁卖。统计特征不能直接充当硬执行信号。

**总建议**：保留当前 Label 配额为策略软约束；将未来门控拆成四层，优先级依次为：

1. 交易所/经纪接口硬约束；
2. 账户风险硬约束和独立熔断；
3. 执行完整性与对账；
4. Label 语义、换手、冷却期等策略软约束。

任何软约束都不得阻止合法的减仓/平仓；当平仓因涨跌停或流动性不可执行时，系统应进入 `reduce_only_pending`，禁止新增风险并持续重试，而不是宣称已经止损。

---

## 2. 研究方法与证据边界

项目事实优先通过本地 codebase-memory-mcp CLI 的 `search_graph` / `get_code_snippet`（project=`home-lanceliang-opt-aiwork-FineFT_code_space2026`）定位；当图结果不足、需要查字符串或对象是 Markdown/配置/测试等非图代码时，再按 AGENTS.md 回退到 `rg` 和定点逐段读取。外部事实只采用交易所、监管机构、官方产品资料和原始论文。

代码事实均以当前工作树为准。当前工作树另有用户修改，本报告只新增本文件，没有修改任何代码或已有文档。

---

## 3. 当前系统到底在约束什么

### 3.1 动作空间与环境状态

`Base_Env` 的离散动作不是“买/卖/持有”，而是目标 `(position, leverage)`。动作数为 `(position_choices - 1) × leverage_count + 1`，仓位网格覆盖 `[-max_holding_number, +max_holding_number]`；零仓位只保留一个动作。[`FineFT/env/env_class/base_env.py` L54-L149](../../FineFT/env/env_class/base_env.py#L54-L149) 与 [`futures_util.py` L1299-L1336](../../FineFT/env/env_class/futures_util.py#L1299-L1336)

Q 网络消费环境给出的 `avaliable_action` 掩码，并用大惩罚屏蔽不可用动作。[`FineFT/model/low_level.py` L267-L294](../../FineFT/model/low_level.py#L267-L294) 带守卫测试入口仍先把环境原始掩码交给 Q 网络，再对模型建议动作执行守卫。[`test_agent_index_with_guard.py` L685-L721](../../FineFT/RL/DiHFT/low_level/test_agent_index_with_guard.py#L685-L721)

环境向模型提供的交易过程状态目前为四维：仓位暴露、单次持仓收益率、单次持仓最大回撤、归一化持仓时长。[`base_env.py` L46-L51](../../FineFT/env/env_class/base_env.py#L46-L51) 与 [`base_env.py` L204-L223](../../FineFT/env/env_class/base_env.py#L204-L223)

### 3.2 当前 Label 动作守卫规则清单

`RollingLabelActionGuard` 的规则是：

| 规则 | 当前实现 |
|---|---|
| Label 方向 | `limit/strong/weak_down=-1`，`sideways=0`，对应 up 类为 `+1` |
| 逆向定义 | 候选目标仓位方向与 Label 相反，目标非零，且目标仓位不同于当前仓位 |
| 滚动窗口 | 最近 `window_size - 1` 个记录加当前候选；默认窗口 10 |
| 默认容量 | weak 40%=4 次，strong 20%=2 次，limit 0 次，sideways 不定义逆向 |
| 未超额 | 原始动作原样放行，原因 `allowed` |
| 超额且不触发条件止损 | 改为环境当前动作，原因 `quota_hold` |
| 超额且逆 Label 持仓相对开仓价不利 3% | 改为零仓位动作，原因 `stop_loss_close` |
| 状态边界 | 每条 `(label, epoch, bin, df_path, initial_action)` 轨迹新建守卫 |

实现证据见 [`action_guard.py` L10-L139](../../FineFT/RL/DiHFT/harness/action_guard.py#L10-L139)，容量计算和默认值见 [`test_agent_index_with_guard.py` L194-L202](../../FineFT/RL/DiHFT/low_level/test_agent_index_with_guard.py#L194-L202) 与 [`L249-L276`](../../FineFT/RL/DiHFT/low_level/test_agent_index_with_guard.py#L249-L276)。

### 3.3 当前环境硬约束清单

1. **保证金和盘口深度可用动作**：`calculate_avaiable_action` 依据当前钱包、未实现盈亏、杠杆、盘口总深度和开仓损失生成可用目标；反手开启时先模拟平旧仓再检查新仓保证金。[`futures_util.py` L1103-L1230](../../FineFT/env/env_class/futures_util.py#L1103-L1230)
2. **反手开关**：默认禁止一步反手；开启后采用先平后开的 best-effort 语义。[`futures_util.py` L143-L250](../../FineFT/env/env_class/futures_util.py#L143-L250)；ADR 见 [`docs/adr/0001-reverse-position-semantics.md`](../adr/0001-reverse-position-semantics.md)
3. **涨跌停双重防护**：环境收到的布尔状态为真时，跌停禁卖、涨停禁买；既过滤可用动作，也在 `step` 内拒绝绕过掩码的动作。[`base_env.py` L343-L365](../../FineFT/env/env_class/base_env.py#L343-L365) 与 [`L477-L513`](../../FineFT/env/env_class/base_env.py#L477-L513)。但是当前入口的布尔状态来自 bar 统计量而非独立的决策时点锁板状态，见 4.2.H。
4. **强平判定**：保证金余额不高于维持保证金时终止 episode。[`futures_util.py` L1079-L1100](../../FineFT/env/env_class/futures_util.py#L1079-L1100) 与 [`base_env.py` L552-L580](../../FineFT/env/env_class/base_env.py#L552-L580)
5. **真实成交口径成本状态**：开仓价/均价用开仓腿实际成交额、数量和费用维护；部分成交可使目标与实际仓位不同。[`base_env.py` L286-L331](../../FineFT/env/env_class/base_env.py#L286-L331)

项目设计明确将 Label 守卫定位为 Stage II validation 研究功能，不修改训练、Stage III 或无守卫入口，也明确其不是主动止损系统。[`openspec/.../design.md` L12-L34](../../openspec/changes/add-low-level-label-action-guard/design.md#L12-L34)

---

## 4. 当前规则逐条评估

### 4.1 合理并建议保留的部分

#### A. 模型原始动作、守卫最终动作、环境实际仓位分开记录

这是正确的边界。安全屏蔽研究区分 pre-shield（先给安全动作集）和 post-shield（模型之后替换动作）；当前系统同时保留环境硬掩码和 Label post-shield，便于审计模型意图。原始 shielding 论文也明确讨论了这两种集成方式及其对学习收敛的要求。[Alshiekh 等，Safe Reinforcement Learning via Shielding](https://arxiv.org/abs/1708.08611)

#### B. 环境硬约束在 `step` 再检查

不能信任调用方一定遵守掩码。涨跌停既在动作集过滤，又在执行入口复核，是合理的纵深防御。当前测试还验证了守卫请求平仓但涨跌停使其无法成交的情形。[`test_test_agent_index_with_guard.py` L674-L720](../../FineFT/tests/rl/test_test_agent_index_with_guard.py#L674-L720)

#### C. 平仓优先于反向开仓，反手不是原子事务

真实的先平后开存在部分成交和第二腿失败。当前 best-effort 语义比假设原子反手更可信；但未来实盘必须把两腿建模成订单状态机，而不是一个同步函数返回值。

#### D. 强弱 Label 使用不同逆向预算

作为“策略偏好”而非“安全事实”，弱趋势允许更多逆向探索、强趋势更少、涨跌停标签为零，逻辑上自洽。它适合 validation 对比实验，但比例 40%/20%/0 目前是人工参数，不是经过跨合约、跨时期置信区间证明的风险阈值。

#### E. 不把 near-limit 当作硬涨跌停

硬成交限制只依赖显式涨跌停状态，near-limit 仅适合特征或软规则，避免把预测信号误当交易所状态。规格对此边界定义清楚。[`fineft-price-limit-execution/spec.md` L3-L21](../../openspec/changes/add-low-level-label-action-guard/specs/fineft-price-limit-execution/spec.md#L3-L21)

### 4.2 需要修正或重新归类的部分

#### A. `opposed_holding_stop_loss_ratio=3%` 不是保底止损

代码只有在 `exceeds_quota=True` 后才调用 `_should_stop_loss`。[`action_guard.py` L87-L110](../../FineFT/RL/DiHFT/harness/action_guard.py#L87-L110) 测试甚至明确要求“已不利 3%，但候选仍在配额内时继续放行”。[`test_test_agent_index_with_guard.py` L611-L645](../../FineFT/tests/rl/test_test_agent_index_with_guard.py#L611-L645)

**判断**：作为原规格的“被拦截动作降级”合理；作为风险保底不合理。未来应拆成：

- 独立账户/持仓止损或熔断，优先级高于 Label 语义；
- Label 配额超限后的普通降级；
- 若市场不可平，进入 `reduce_only_pending`，而不是把意图当成交。

此外，价格变化 3% 没有反映杠杆、合约乘数、费用、滑点、保证金占用和组合风险，只能是策略软阈值。

#### B. 配额按“最终动作 ID”而非实际执行记账

设计有意规定，即使环境没完成目标，历史仍记录交给 `step` 的动作 ID。[`design.md` L86-L90](../../openspec/changes/add-low-level-label-action-guard/design.md#L86-L90) 这适合测量“守卫批准了多少逆向意图”，却不等于实际风险敞口。当前代码在调用环境前就把动作加入 deque。[`action_guard.py` L112-L116](../../FineFT/RL/DiHFT/harness/action_guard.py#L112-L116)

**风险**：

- 无成交也消耗额度，误伤后续本来可执行的动作；
- 部分成交和完全成交记相同一次，额度与风险增量无关；
- `stop_loss_close` 可能实际未平仓，理由码过度承诺；
- 模型可通过反复提交被拒动作操纵窗口，使旧记录滑出。

**建议**：同时维护三类指标：`attempted_opposed`、`approved_opposed`、`executed_opposed_delta`。风险预算以实际成交后的绝对逆向敞口增量或风险单位记账，反滥用/策略诊断才按尝试次数记账。

#### C. “减仓后仍为逆向仓位”也计为逆向动作，可能惩罚降风险

当前定义只要目标仍与 Label 反向且不同于当前仓位，就算逆向；从多仓 4 减至多仓 2，在下跌 Label 下也占额度。[`action_guard.py` L27-L34](../../FineFT/RL/DiHFT/harness/action_guard.py#L27-L34) 规格也显式如此要求。[`fineft-label-action-guard/spec.md` L53-L67](../../openspec/changes/add-low-level-label-action-guard/specs/fineft-label-action-guard/spec.md#L53-L67)

**判断**：若统计目标是“仍坚持逆势观点”，可以理解；若目标是风险保底，则方向错了，因为减仓降低绝对敞口。硬风险层必须满足单调性：合法减仓/平仓不应被策略配额阻止。可将“逆势但减仓”记录为行为诊断，却不占新增风险预算。

#### D. Label 是整段事后标签，不能直接作为在线门控事实

当前标签器使用 SciPy `filtfilt` 做双向滤波，再根据整段过滤序列的未来转折点分段，并对完整段拟合斜率。[`label_util.py` L414-L425](../../FineFT/datahandler/label_util.py#L414-L425)、[`L451-L460`](../../FineFT/datahandler/label_util.py#L451-L460)、[`L510-L567`](../../FineFT/datahandler/label_util.py#L510-L567)；随后整段被写入对应 `label_*` 目录供 validation 回放。[`slice_model.py` L450-L504](../../FineFT/datahandler/slice_model.py#L450-L504)

**判断**：它适合离线行为诊断和监督标签，不适合作为“当时已知”的实盘信号。未来门控须输入在线模型的 `P(label | information_<=t)`，并至少使用：置信度下限、滞回、最短驻留时间、OOD/未知标签和标签切换冷却。必须做 walk-forward 回测，禁止读取切片终点或未来转折点。

#### E. 滚动门控状态未进入模型观察

守卫窗口只存在于外部 deque，Q 网络输入只有行情状态、时间、上一步环境动作、可用动作和四维交易信息。若未来训练时开启守卫，同一 observation/action 在不同配额余量下会得到不同执行结果。Sutton 与 Barto 强调，RL 决策和值函数依赖当前状态，状态应足以预测下一状态和奖励。[Reinforcement Learning: An Introduction, 2nd ed.](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)

**建议**：训练时把配额余量、窗口占用、标签概率/年龄、最近干预原因、kill-switch 状态、未成交风险加入 observation；replay 中保存 proposal、approval、execution 和干预原因。若只在部署后置守卫，则必须单独报告干预率和性能偏移，不能假设未守卫训练策略天然适配。

#### F. 商品合约经济口径尚未闭环

FU 官方合约为 10 吨/手、最小变动价位 1 元/吨、最低交易保证金为合约价值 8%。[上海期货交易所燃料油期货合约](https://www.shfe.com.cn/products/futures/energyandchemical/fu_f/standard_fu/202312/t20231205_327331.html) 项目商品配置也记录 `contract_unit=10`，但 `use_contract_multiplier=False`。[`commodity/config.py` L46-L65](../../data_preprocess/operator_futures/commodity/config.py#L46-L65) 当前环境保证金与维持保证金仍按 `markprice × position` 计算。[`futures_util.py` L1090-L1098](../../FineFT/env/env_class/futures_util.py#L1090-L1098)

**判断**：当前可用于相对策略研究，但不能声称是交易所/账户真实风险模型。应先定义统一的 `ContractSpec`，明确 position 是“手”还是标的数量，并在名义价值、PnL、保证金、滑点、手续费、持仓限额和下单数量中一致使用。

#### G. 默认维护保证金参数仍带有永续合约遗留语义

`Base_Env` 和商品入口默认维护保证金表注释仍指向 BTCUSDT perpetual，`Commodity_Env` 默认值也沿用同一分级表。[`base_env.py` L72-L81](../../FineFT/env/env_class/base_env.py#L72-L81)、[`commodity_env.py` L24-L39](../../FineFT/env/env_class/commodity_env.py#L24-L39) 商品配置的 `maintenance_margin_rate` 没有进入该构造路径。

**建议**：交易所保证金、期货公司加收比例、特殊时段/临近交割/连续涨跌停调整都应按合约和交易日动态加载；缺失或过期时 fail-closed（禁止新增风险），只允许可验证的减仓。

#### H. 硬涨跌停状态由 bar 内统计比例推断，可能过度封锁

预处理把窗口内 `_limit_up/down_single_sided` 取均值或计数占比，生成 `limit_up/down_single_sided_ratio`。[`downscale.py` L620-L643](../../data_preprocess/operator_futures/commodity/downscale.py#L620-L643) 与 [`L1307-L1323`](../../data_preprocess/operator_futures/commodity/downscale.py#L1307-L1323) 环境入口随后以 `ratio > 0` 生成 `is_limit_up/down_array`。[`base_initiate.py` L78-L83](../../FineFT/env/env_initiate/base_initiate.py#L78-L83)

因此，对 10/30min bar，只要窗口内任一快照曾呈单边锁板，整根 bar 的决策点都会被视为硬锁板；bar 末实际已经打开时也可能拒绝本可执行的减仓。反过来，统计比例为 0 也不等价于经纪/交易所明确确认可交易。

**建议**：保留 ratio 作为模型特征和软流动性强度；执行层另传决策时点最新快照/交易状态的显式 `is_limit_locked_up/down`、数据时间戳与可执行侧，并在 `step` 前校验。硬规则不应从窗口统计特征推断。

#### I. 只换杠杆可以绕过逆向预算

逆向判断只比较目标仓位和当前仓位；当仓位不变、只提高杠杆时，`target_position == current_position`，不会计为逆向动作。[`action_guard.py` L27-L34](../../FineFT/RL/DiHFT/harness/action_guard.py#L27-L34) 当前常见配置只有一个杠杆选择，因此暂时影响有限；一旦未来开放多杠杆，逆 Label 持仓可在不消耗配额的情况下提高杠杆和风险。

**建议**：硬门控基于成交后风险/保证金/压力损失增量，而不是仓位方向或 action id；任何提高杠杆导致的风险增加都应消耗账户风险预算。

---

## 5. 外部硬约束基线

中国《期货和衍生品法》明确规定保证金、持仓限额、当日无负债结算、保证金不足后的追加/自行平仓与强行平仓制度；交易所异常情况下还可调整保证金、涨跌停、交易/持仓限额、限制开仓、强平或暂停交易。[全国人大《期货和衍生品法》](https://www.npc.gov.cn/npc/c2/c30834/202204/t20220420_317569.html)

上期所风险控制体系包括保证金、涨跌停、持仓限额、交易限额、大户报告、强行平仓和风险警示；限额随品种、月份和交割阶段变化，同一客户多个交易编码需合并计算。[上期所 Risk Management 官方页](https://www.shfe.com.cn/eng/services/investor/Investor_risk_control/)

产品合约页上的 FU 5% 涨跌停与 8% 最低保证金是合约基础参数，不代表任意交易日的实际参数。例如，上期所在 2026-06-23 公告中把 FU2608 等合约及后续新合约调整为 14% 涨跌停、一般持仓保证金 16%，并说明连续涨跌停等情况下还会继续调整。[上期所 FU 参数调整通知](https://www.shfe.com.cn/publicnotice/notice/202606/t20260623_832251.html) 这进一步说明门控必须按合约和交易日加载正式结算/交易参数，不能写死 5% 或 8%。

截至本报告日期，上期所自 2026-07-06 起对期货限价单规定 1–500 手、市价单 1–60 手；这些是会调整的场所参数，应由运行时元数据加载。[上期所关于市价指令和下单数量限制的官方通知](https://www.shfe.com.cn/eng/CircularNews/Circular/202606/t20260618_832181.html)

程序化交易方面，证监会规定自 2025-10-09 起实施报告、系统接入、交易监测和风险管理要求。[证监会《期货市场程序化交易管理规定（试行）》发布说明](https://www.csrc.gov.cn/csrc/c100028/c7564353/content.shtml) 上期所/能源中心的官方细则要求技术系统具备连接与报撤单异常监测、阈值预警、错误防范、暂停交易/撤单应急能力和完整日志，并重点监控短时/日内报撤单、报撤成交比和密集大额申报。[上海国际能源交易中心程序化交易管理细则](https://www.shfe.com.cn/regulation/ineregulation/businessmethods/trade/202508/t20250814_828670.html) 上期所异常交易范围还包括自成交、实际控制账户间成交、频繁报撤、大额报撤和超过日内开仓量。[上期所异常交易行为管理办法](https://www.shfe.com.cn/regulation/marketregulation/regulationfile/202508/t20250813_828648.html)

作为跨市场工程基线而非中国法域要求，CFTC Rule 1.73 官方文本要求清算会员按持仓、订单量、保证金等建立账户风险限额，以自动方式做交易前筛查，并持续监控和压力测试；这支持把前置风险筛查与事后监控分开。[CFTC Final Rule 2012-7477](https://www.cftc.gov/LawRegulation/FederalRegister/FinalRules/2012-7477.html) CME 的官方 GC2 也以实时 exposure/max-quantity limit、block/cancel/alert 展示了场所信用控制的常见分层。[CME Globex Credit Controls](https://www.cmegroup.com/tools-information/webhelp/globex-credit-controls/Content/CME-Globex-Credit-Controls-Management.html)

因此，“门控”不能只是一个方向过滤器；它至少还是一个前置交易风控、订单生命周期控制器和可审计的运行时安全层。

---

## 6. 建议新增规则与优先级

### 6.1 P0：交易所/经纪接口硬约束

这些规则违反时必须拒单或只允许减仓，不能用 reward 惩罚替代。

| 规则 | 决策口径 | 当前状态 |
|---|---|---|
| 合约元数据合法性 | 合约存在、可交易、未停牌/到期；tick、手数、乘数、币种、开平/平今属性正确 | 缺失统一运行时 `ContractSpec` |
| 交易时段与状态 | 集合竞价/连续交易/休市/临停/夜盘跨日；只在允许阶段发对应指令 | 配置有静态时段，守卫未使用 |
| 价格合法性 | 价格在当日上下限内并对齐 tick；市价单使用交易所支持与保护范围 | 当前由 bar 内 ratio 推断锁板，不是决策时点显式状态 |
| 决策时点锁板状态 | 最新快照/场所状态明确是否单边锁板；含时间戳和可执行侧 | 缺失，当前 `ratio > 0` 可能整 bar 过度封锁 |
| 单笔数量与日内开仓量 | 最小/最大下单手数、交易所临时交易限额 | 未实现 |
| 持仓限额 | 客户、实际控制账户组、品种/合约、一般/套保、月份阶段合并校验 | 未实现 |
| 交割与临近交割 | 最后交易日、交割资格、交割单位整倍数、限仓收紧、强制换月 | 未实现 |
| 动态保证金 | 交易所标准 + 期货公司加收 + 临时调整，按最不利成交后状态计算 | 当前是静态通用表 |
| 开平语义 | 开仓、平今、平昨、平仓优先级及可平数量，避免平仓单变开仓 | 当前目标仓位抽象未表达 |
| 账户权限 | 投机/套保、品种权限、只平权限、经纪端风控返回 | 未实现 |

### 6.2 P0：账户风险硬约束

1. **风险单位统一**：以 `price × contract_multiplier × lots` 得到名义价值；用真实费用、滑点和保证金率做 pre-trade worst-case。
2. **持仓/敞口上限**：每合约、品种、方向、跨合约总 gross/net、单边集中度；跨月相关仓不能简单净额抵消。
3. **保证金缓冲**：不仅要求“当前够开仓”，还要保留配置的可用资金和维持保证金缓冲；压力测试至少覆盖一个或多个涨跌停/跳空情景。
4. **独立账户熔断**：当日亏损、峰值回撤、净值下限、保证金使用率、距强平缓冲越线时进入 `reduce_only` 或 `halted`。重启不能自动清除，恢复需要明确的会话/人工策略。
5. **单次持仓风险**：价格止损可保留为软规则；真正硬规则应看含费用的持仓 PnL、风险资金比例、最大持有时长和可实现流动性。止损越线必须独立评估，不依赖 Label 配额。
6. **只减不增单调性**：任何风控/语义规则均不得把降低绝对仓位或保证金占用的合法动作替换成更高风险动作。
7. **未成交风险计入**：持仓上限和保证金检查必须同时考虑当前持仓、活动委托、部分成交余量和正在反手的第二腿。
8. **多策略共享账户预算**：按策略/Agent 分配子预算，但总账户风控以经纪/交易所真实账户为最终权威。

### 6.3 P0：执行完整性规则

1. **四阶段事实模型**：`proposed → approved → submitted/acknowledged → filled`；再派生 `executed_position`。不得把 `approved` 写成 `closed`。
2. **幂等和去重**：每个决策/订单有唯一 client order id；超时重试前查询状态，避免网络抖动重复下单。
3. **订单状态机**：处理 ack、reject、partial fill、cancel pending、cancel reject、过期、乱序回报；反手两腿需要显式依赖关系。
4. **周期性对账**：账户持仓、可用资金、活动委托以经纪端为权威；本地漂移立即禁止开仓并报警。
5. **行情健康**：时间戳单调、最新价/盘口非 NaN/Inf、bid≤ask、价格与 tick/涨跌停一致、快照新鲜；数据陈旧或断流时取消活动开仓单并进入只减/暂停。
6. **价格/滑点保护**：限制单笔吃单深度、成交均价偏离、参与率；不能只相信历史快照里的总深度。
7. **报撤单和自成交控制**：订单速率、撤单率、报撤成交比、同账户/实控账户潜在自成交、异常密集大单均需前置限流。
8. **紧急停止**：心跳、连接、时钟、风控服务、日志/持久化异常时可一键停止新单、批量撤单；降级行为必须预先定义。

### 6.4 P1：Label 门控规则

1. **因果在线标签**：只用 `t` 时刻可见信息，输出分布而非单标签。
2. **置信度与 OOD**：低置信度、标签分布高熵、输入越界时降低风险预算或归零，不强行套用多/空语义。
3. **切换滞回**：进入/退出阈值不同，设置最短驻留步数，避免 Label 抖动导致频繁反手。
4. **基于风险增量的配额**：逆向新增/加仓消耗预算；逆向减仓不消耗新增风险预算。预算可按 `|Δnotional|`、保证金或预估损失，而不是动作次数。
5. **实际成交回写**：窗口风险状态以 fill 后仓位更新；另存 attempted/approved 计数用于诊断和防滥用。
6. **确定性安全投影**：被拒动作投影到“风险最小且最接近目标”的可行动作，优先顺序为平仓/减仓、保持、同向低风险动作；原因码记录触发规则。
7. **连续超限升级**：同一策略反复建议被拒动作时，降低该 Agent 权重或切换安全策略，而不是只靠窗口自然滑出。

### 6.5 P2：策略软约束

以下应通过回测/走样验证调参，不宜冒充合规硬规则：弱/强 Label 的 40%/20% 配额、3% 价格止损、最短持仓期、换手/冷却、最大单步仓位变化、滑点偏好、收益/回撤型 Agent 选择、CVaR 或风险调整奖励。

无效动作 masking 有理论和实验依据，尤其比单纯给无效动作负奖励更适合大离散动作空间；但它只说明如何处理确定无效动作，不证明某个 Label 方向就是安全事实。[Huang 与 Ontañón，Invalid Action Masking](https://arxiv.org/abs/2006.14171) 对需要累计成本约束的学习目标，可把软风险建模为 CMDP；CPO 原始论文讨论了在策略更新中约束期望成本。[Achiam 等，Constrained Policy Optimization](https://proceedings.mlr.press/v70/achiam17a.html) 尾部风险可作为补充评估维度，而不能替代硬限制；CVaR MDP 原始工作说明了风险敏感目标与模型扰动的关系。[Chow 等，CVaR Optimization](https://papers.neurips.cc/paper_files/paper/2015/hash/64223ccf70bbb65a3a4aceac37e21016-Abstract.html)

---

## 7. 推荐门控架构与决策顺序

```text
行情/账户/订单回报
       │
       ▼
[0 健康与时钟] ─失败→ cancel-open + reduce_only/halt
       │
       ▼
[1 合约/交易所合法性] ─过滤时段、价格、手数、限仓、交割、权限
       │
       ▼
[2 账户真实状态对账] ─漂移→ 禁止新增风险
       │
       ▼
[3 独立账户熔断] ─越线→ reduce_only；可成交时分步减仓
       │
       ▼
[4 生成硬可行动作集] ─含持仓+活动委托的最不利成交后风险
       │
       ▼
[5 在线 Label/策略软门控] ─置信度、滞回、逆向风险预算、冷却
       │
       ▼
[6 确定性安全投影] proposed → approved，保证减仓不被软规则阻止
       │
       ▼
[7 订单状态机] submit/ack/partial/cancel/reject
       │
       ▼
[8 fill 对账与回写] 更新 executed position、预算、RL transition、审计日志
```

关键不变量：

- 硬规则优先于软规则；交易所/经纪端是最终权威。
- `reduce_only` 下不得增加任何合约或组合层风险。
- “请求平仓”不等于“平仓成功”。
- 软 Label 门控不得阻止合法减仓。
- 同一输入、同一规则版本和同一账户快照应产生确定性相同决策。
- 无合法动作时必须有定义清楚的结果：`hold`、`reduce_only_pending` 或 `halted`，不能 fail-open。

---

## 8. 威胁与失效模型

| 失效/规避方式 | 当前可能表现 | 推荐防线 |
|---|---|---|
| 事后 Label 泄漏 | validation 结果异常乐观 | 因果在线标签、walk-forward、记录 label as-of time |
| Label 抖动/错分 | 多空门控频繁翻转或错误封锁 | 概率、滞回、最短驻留、OOD 降险 |
| 重复提交被拒动作 | 窗口滑动后重新获得额度 | attempt 限流 + executed 风险预算 + 连续干预升级 |
| 部分成交 | 最终动作与实际仓位不一致 | fill-based 状态机和预算回写 |
| 涨跌停无法止损 | 日志显示 close，持仓仍在 | `reduce_only_pending`、禁止加仓、风险告警 |
| 盘口快照陈旧/虚假深度 | 低估滑点与可成交量 | freshness、价格保护、参与率、真实 fill 校准 |
| 保证金临时提高 | 旧参数仍允许开仓 | 动态元数据、版本/有效期、缺失时只减仓 |
| 合约乘数/手续费错误 | PnL、保证金和止损尺度全错 | 单一 ContractSpec + 单位属性测试 |
| 活动委托未计入 | 多次决策叠加超仓 | pending exposure 纳入 pre-trade 风险 |
| 网络超时重试 | 重复订单 | client id 幂等、查询后重试 |
| 本地/经纪持仓漂移 | 守卫基于错误 current_action | 周期/事件驱动对账，漂移即禁开 |
| 多账户规避限额 | 单账户看来合规、实控组超限 | 账户组聚合限额 |
| 报撤/自成交异常 | 合规风险与限制交易 | 速率、撤单率、自成交预防、kill switch |
| 守卫服务自身故障 | fail-open 继续交易 | 独立健康检查、默认禁止新增风险 |
| 规则配置漂移 | 回测与实盘口径不一致 | 规则版本、签名、有效期、审计与变更审批 |

---

## 9. 可观测性与审计要求

每个决策至少记录：

- `decision_id`、规则版本、模型/checkpoint、数据 as-of 时间；
- label 概率、置信度、OOD 分数、标签年龄/切换次数；
- proposed/approved/submitted/filled action 与目标/实际仓位；
- 当前持仓、活动委托、可用资金、权益、保证金、距强平缓冲；
- 每条约束的阈值、实测值、pass/fail 和主原因/全部原因；
- order id、ack/reject、部分成交、撤单、成交均价、费用、滑点；
- attempted/approved/executed 的逆向计数和风险量；
- kill-switch/reduce-only 状态及进入、恢复主体和时间。

核心监控指标：

- 干预率、按原因拒绝率、连续干预长度；
- approved-to-fill 比率、目标/实际仓位偏差、部分成交与拒单率；
- 行情/账户/订单回报延迟与数据陈旧时间；
- 账户对账漂移次数、重复 client id、乱序回报；
- 保证金使用率、距强平缓冲、日损、回撤、gross/net exposure；
- 报单/撤单速率、撤单率、报撤成交比、潜在自成交次数；
- `stop_loss_requested` 与 `stop_loss_filled` 分开统计，未成交持续告警。

当前明细已经记录 proposed/final/position_after、费用、滑点和守卫原因，是很好的基础。[`test_agent_index_with_guard.py` L493-L550](../../FineFT/RL/DiHFT/low_level/test_agent_index_with_guard.py#L493-L550) 但原因码需要从单值升级为分层多原因，并明确 `requested` 与 `executed`。

---

## 10. 测试矩阵

### 10.1 单元与属性测试

| 类别 | 关键场景 | 必须断言 |
|---|---|---|
| 单位口径 | FU 1 手、10 吨、价格变动 1 元 | 名义价值/PnL/保证金/费用一致 |
| 动作映射 | 全部 position×leverage、零仓、非网格实际仓位 | 双向映射或显式不可映射，不静默错位 |
| 风险单调性 | 任意持仓下减仓/平仓 | 软门控不替换为更高风险动作 |
| 保证金 | 费前够、费后不够；保证金临时上调 | 拒绝新增风险，减仓仍可用 |
| 涨跌停 | 跌停多仓平仓、涨停空仓平仓 | 请求记录但不虚构成交，进入 pending reduce-only |
| 锁板信号时点 | bar 内曾锁板但末端已开板；末端刚锁板 | 硬执行使用末端/决策时点状态，不使用窗口 ratio 推断 |
| 深度 | 零深度、部分深度、深度在 ack 前消失 | 实际 fill 回写，未成交量仍计风险 |
| 配额 | 开/加/减/平、只换杠杆、无成交/部分成交/全成交 | attempted/approved/executed 三套计数正确；减仓不消耗新增风险预算，提高杠杆会消耗 |
| 止损 | 配额内/外均越线 | 独立硬止损不依赖配额；软价止损口径明确 |
| Label | 低置信度、OOD、快速翻转 | 降险、滞回与冷却符合配置 |
| 订单状态 | 超时重试、重复 ack、乱序 fill、cancel reject | 幂等且最终状态可对账 |
| 数据健康 | NaN/Inf、时间倒退、stale、crossed book | 禁止新增风险并报警 |
| 熔断 | 日损/回撤/权益/保证金越线 | 只减仓；重启不自动解除 |
| 限额 | 多合约、多账户、活动订单叠加 | 聚合后不越限 |
| 交割 | 临近交割、手数非整倍、最后交易日 | 按运行时规则拒绝/减仓/移仓 |

推荐属性不变量：

```text
filled_position == broker_reconciled_position
approved_risk_after <= hard_limit
reduce_only => risk_after <= risk_before
halted => no new order except cancel or explicitly authorized reduce order
executed_budget_update == function(actual_fills), not function(requested_action)
```

### 10.2 场景与回测测试

1. 连续涨跌停、开板后流动性瞬时恢复；
2. 保证金率盘中/隔夜上调和经纪端额外加收；
3. 主力换月、临近交割限仓收紧；
4. 夜盘跨交易日、节假日前后和午间休市；
5. 盘口断流、延迟尖峰、重复/乱序行情；
6. 大幅跳空、部分成交、反手第二腿失败；
7. 标签置信度骤降、错误方向、快速切换；
8. 多 Agent 同时共享账户预算；
9. 模型持续攻击守卫边界（反复被拒、微量加仓、动作 ID 抖动）；
10. shadow mode：门控只记录不下单，对比 proposed/approved/executed 和收益/风险偏移。

### 10.3 评估口径

除收益、Sharpe、回撤外，必须报告：硬规则违规数（目标为 0）、熔断次数、干预率、错误干预率、未完成平仓持续时间、最大保证金使用率、最小强平缓冲、目标/实际仓位偏差、拒单/部分成交/重复单、标签校准误差，以及 guard on/off 的策略性能差异。

当前 52 个定向测试证明实现符合现有规格，不证明规格足以覆盖上述实盘风险；新增规则应先写失败测试，再实现。

---

## 11. 推荐实施顺序

### Phase 0：先把研究口径做对

1. 统一 FU 合约乘数、手数、手续费、保证金和 PnL 单位。
2. 将 validation 的事后 Label 与未来在线 gate Label 明确分离；建立 causal walk-forward 基线。
3. 把现有 `stop_loss_close` 改名/语义拆成 `stop_loss_requested` 与执行结果（后续代码变更任务）。

### Phase 1：建立最小可信硬门控

1. `ContractSpec` 与运行时交易所/经纪元数据；
2. 数据健康、账户对账、持仓/活动委托聚合；
3. 保证金缓冲、敞口上限、日损/回撤 kill switch；
4. reduce-only、订单幂等状态机、fill-based 风险回写；
5. 全量原因码和审计日志。

### Phase 2：重构 Label 守卫

1. 在线 label distribution + confidence/OOD/hysteresis；
2. attempted/approved/executed 三套状态；
3. 动作次数配额改为“新增风险预算”；
4. 逆势减仓不消耗新增风险预算；
5. 将门控状态纳入训练 observation/replay，或明确只做部署 shield 并评估偏移。

### Phase 3：压力验证后再小流量部署

历史压力场景 → 仿真撮合 → shadow → 极小限额 canary → 逐级扩大。任何阶段出现账户漂移、重复订单、硬规则违规或无法解释的干预，自动回退到只减仓/停止状态。

---

## 12. 待确认问题

这些答案会直接改变规则，而不宜由实现者猜测：

1. 未来门控用于训练、validation、模拟盘还是实盘？若用于训练，是否允许修改 observation/replay 契约并重训 checkpoint？
2. `position` 的权威单位要定义为“手”还是“吨/标的数量”？FU 的 10 吨/手何时进入环境结算？
3. 实盘接口/期货公司是哪一家，能否提供合约元数据、动态保证金、可平今/昨、活动委托和实际控制账户组信息？
4. 风险预算的账户层级：单 Agent、单策略、单合约、品种、账户还是账户组？是否允许跨月对冲净额抵消，抵消比例是多少？
5. 日损、回撤、保证金使用率、最大敞口、单笔参与率的业务阈值由谁批准？触发后是自动恢复、次日恢复还是人工恢复？
6. “止损”是价格阈值、含费用持仓 PnL、账户权益回撤，还是多条件组合？涨跌停无法平仓时希望采用何种告警和重试策略？
7. Label gate 的在线输入来自哪个模型？能否输出概率、OOD 分数与 as-of 时间？现有事后切片仅作为评估标签是否可以接受？
8. 反手在实盘是否必须先确认旧仓完全平掉再允许第二腿？是否允许部分平仓后按剩余额度反向开仓？
9. 是否存在多 Agent/多进程共享同一交易账户？若存在，风险检查和订单状态必须集中化，不能由每个 Agent 本地维护。

---

## 13. 最终判断

当前动作守卫的**研究逻辑是合理的，风险定位不够准确**：它擅长回答“在已知 Label 的 validation 轨迹中，如何限制模型逆方向行为并保留审计证据”，不擅长回答“真实账户此刻是否允许下这笔单”。

未来最稳妥的做法不是继续往 `RollingLabelActionGuard` 里堆 if/else，而是把它保留为门控中的 **Label Policy Layer**，在其前面建立交易所/经纪硬约束、账户硬风控和执行状态机。完成单位口径、因果标签、独立熔断和 fill-based 回写之前，不应把 40%/20%/0 配额或 3% 价格阈值称为实盘“保底规则”。
