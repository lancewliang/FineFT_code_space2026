# Agent 形态分析：双分类器窗口模型

> Target triage label: `ready-for-agent`

## Problem Statement

当前低层 Agent 选择链路主要根据 reward 类聚合指标对 `(label, epoch, bin_index)` Agent triple 排序。策略研究员只能看到粗粒度平均回报，无法回答“哪个 Agent 在什么 K 线形态下，采取了什么策略二阶形态，并产生了多少盈亏”。

同样 reward 水平的 Agent 可能分别表现为突破即时重仓、回调加仓或金字塔递增，风险、滑点敏感性和适用行情截然不同。缺少窗口级“K 线形态 × 策略二阶形态 × 盈亏”事实后，研究员无法分析 Agent 能力边界，也无法为未来的 Stage III 路由准备比 Label 更细的诊断信号。

## Solution

在现有 Detail CSV 的行为轨迹上建立两个正交的离线分类器：K 线形态分类器只描述行情侧的二阶形态，策略形态分类器描述 Agent 已执行仓位变化与行情事件的关系。K 线形态每窗口单选，策略二阶形态每窗口可多选，两轴共享同一形态识别窗口和行情事件语义。

系统覆盖所有具有 Detail CSV 的 Agent Pattern Candidate Universe，Selection Manifest 只标记当前已选 Agent，不过滤候选。输出窗口级明细、形态组合展开表、情景级和 triple 级汇总、Detail Coverage 诊断以及可验证的分析 manifest。每个窗口的 PnL 在展开前只记录一次，避免多选策略放大账户盈亏。

## User Stories

1. As a 策略研究员, I want 查看任意 Agent triple 在不同 K 线形态下的盈亏分布，so that 我能判断它擅长的行情。
2. As a 策略研究员, I want 查看任意 Agent triple 在各策略二阶形态下的盈亏分布，so that 我能区分突破即时、回调加仓、金字塔递增等行为。
3. As a 策略研究员, I want 查看 7×6 K 线形态与策略形态交叉表，so that 我能发现特定行情与特定 Agent 行为组合的绩效。
4. As a 策略研究员, I want 一个窗口同时命中多个兼容的策略形态，so that 复合行为不会被强制压缩成单一标签。
5. As a 策略研究员, I want 按 Label 过滤形态明细，so that 我能比较同 Label 内不同 Agent 的策略差异。
6. As a 策略研究员, I want Label 1–5 使用长度 20、步长 20 的完整非重叠窗口，so that 盈亏不会因窗口重叠而重复归因。
7. As a 策略研究员, I want Label 0/6 的整条行为轨迹作为单一涨跌停事件窗口，so that 极短轨迹不会被不合理地切成标准窗口。
8. As a 策略研究员, I want 短事件窗口仅运行满足最小样本数的策略规则，so that 系统不会补齐或外推不存在的行为。
9. As a 策略研究员, I want K 线形态按已确定的优先级互斥单选，so that 回调和背离不会被通用突破规则吞没。
10. As a 策略研究员, I want 策略形态各规则独立判定，so that 一个窗口可以保留所有兼容的形态。
11. As a 策略研究员, I want 无法分类的窗口显式保留诊断哨兵，so that 我能观测分类覆盖率而不是丢失样本。
12. As a 策略研究员, I want 窗口同时输出手续费前 gross PnL 和手续费后 net PnL，so that 我能分别诊断策略本身与真实账户绩效。
13. As a 策略研究员, I want 每个窗口只保存一行和一组 PnL，so that 策略多选不会复制账户收益。
14. As a 策略研究员, I want 展开表只用于单形态和交叉形态分析，so that 展开后的 PnL 不会被误解为账户总盈亏。
15. As a 策略研究员, I want 先查看 Initial-action Scenario 级指标，再查看命中情景等权平均与情景覆盖率，so that 系统不会伪造零 PnL 情景或直接累加反事实账户。
16. As an Agent 调度设计者, I want 分类键与部署 Agent triple 对齐，so that 形态产物能与当前已选 Agent 直接关联。
17. As an Agent 调度设计者, I want 保留全部 Agent Pattern Candidate Universe 并标记已选子集，so that 我能同时分析当前部署 Agent 和备选 Agent。
18. As an Agent 调度设计者, I want Initial-action Scenario 保留为诊断切片轴而不进入 Agent triple，so that 反事实起始仓位不会改变部署 Agent 身份。
19. As an Agent 调度设计者, I want Market Dynamic Segment 只作为数据来源而不进入分类维度，so that 文件切片方式不会污染 Agent 身份。
20. As an Agent 调度设计者, I want 任意窗口能追溯到合约、数据文件、Initial-action Scenario 和时间步，so that 我能还原原始行为轨迹。
21. As a 策略研究员, I want 在真实 Detail CSV 上检查命中分布、未分类率和盈亏区分度，so that 阈值调整基于数据而不是固定类别配额。
22. As a 策略研究员, I want 查看每个判别特征的真实分布，so that 我能判断提议阈值是否合理。
23. As a 策略研究员, I want 标定后的阈值由回归测试锁定，so that 后续重构不会静默改变分类语义。
24. As a 后续维护者, I want 两个分类器是不依赖 I/O 的纯函数，so that 形态语义可以被独立测试和复用。
25. As a 后续维护者, I want 明细生成器只编排读取、分组、切窗、调用和写出，so that 形态判别逻辑不会散落在 I/O 脚本中。
26. As a 后续维护者, I want 形态标签使用稳定的 KT/KM/KX 和 ST/SM 前缀，so that 机器产物简短、明确且不混淆。
27. As a 后续维护者, I want 形态分析使用已执行仓位而不是仅使用目标动作，so that 成交深度、保证金或限制造成的实际执行差异被正确反映。
28. As a 数据工程维护者, I want Detail 行为轨迹按权威 `timestep` 排序并校验连续性，so that CSV 全局行序不会改变窗口、分类或 PnL。
29. As a 数据工程维护者, I want 缺少期望 Initial-action Detail 行为轨迹时立即失败，so that 不完整输入不会生成偏置汇总。
30. As a 数据工程维护者, I want Selection Manifest 在标记已选 Agent 前通过数据集、实验、Label 和 epoch 归属校验，so that 另一实验中恰好同名的 triple 不会被误标。
31. As a 审计者, I want 分析 manifest 记录输入、输出和配置的可验证身份，so that 我能证明某次分析使用了哪些数据并复现结果。
32. As a 审计者, I want `window_id` 在阈值、Selection Manifest 和数据根目录变化时保持稳定，so that 同一逻辑窗口可以跨运行比较。

## Implementation Decisions

- 采用已记录的双轴窗口模型：K 线形态与策略二阶形态正交，不合并成单一 Agent 类别。
- Agent triple 固定为 `(label, epoch, bin_index)`。Initial-action Scenario 和 Market Dynamic Segment 不进入 Agent 身份。
- 分类范围是所有具有 Detail CSV 的 Agent Pattern Candidate Universe。Selection Manifest 只生成 `is_selected` 标记，不缩小候选范围。
- Selection Manifest 匹配前必须校验逻辑数据集、实验、Label 完整性、epoch 一致性和 checkpoint 存在性。逻辑归属不依赖机器绝对路径前缀。
- 普通 Label 的行为轨迹使用长度 20、步长 20 的完整非重叠窗口。不足 20 步的尾部不生成形态行，但必须报告丢弃步数与对应 gross/net PnL。
- 涨跌停 Label 的整条行为轨迹构成一个 KX1 事件窗口。各策略规则仅在达到自身最小样本数时运行。
- K 线形态包含 KT1 突破即时、KT2 回调、KT3 加速、KM1 V/倒V反转、KM2 箱体、KM3 背离和 KX1 涨跌停。它们按 `KX1 → KM1 → KT2 → KM3 → KT1 → KT3 → KM2 → 未分类` 互斥单选。
- 所有依赖 price Z-score 的规则使用完整形态识别窗口的 mark price 均值和总体标准差（`ddof=0`）。标准差为零时，这些规则均不命中。
- K 线形态规则固定为：
  - KM1：极值点 `|z_price| ≥ 2.0`，极值前后各至少 3 步，两条反向腿的单边收益均至少 0.5%，多空对称。
  - KT1：第 1–5 步建立基准区间，第 6–10 步突破基准高/低至少 0.3%；窗口终点沿突破方向相对边界延伸至少 0.5%，且突破后至少 80% 的观测点保持在突破侧。
  - KT2：先按 KT1 时序突破，再形成同向新极值，随后反向回撤至少 0.3% 并返回基准边界 ±0.5% 范围，最后在窗口结束时重新沿突破方向延伸至少 0.5%。突破触发点不得同时充当回踩点。
  - KT3：分别对前 10 步和后 10 步 log price 线性拟合，两段斜率同号，后半段绝对斜率至少是前半段的 1.5 倍，且整窗 `|cum_ret| ≥ 0.5%`。
  - KM2：`std(log_return) ≤ 0.3%`，全部价格位于均价 ±0.5%；上下触边区为均价 ±0.25%，忽略中间区并压缩连续同区状态后，上下沿至少转换 4 次。
  - KM3：第 1–5 步建立价格/成交量基准，第 6 步后价格突破基准高/低至少 0.3%；突破点及前两步的 volume 中位数比基准期下降至少 20%，且价格相对基准均价移动至少 0.5%。
  - KX1：仅由涨跌停 Label 决定，不进入普通窗口分类。
- 策略二阶形态包含 ST1 突破即时、ST2 回调加仓、ST3 金字塔递增、SM1 硬边界反转、SM2 离散网格调仓和 SM3 背离过滤。各规则独立判定并允许兼容多选。
- 策略二阶形态规则固定为：
  - ST1：价格按 KT1 时序突破，Agent 在突破当步或下一步从空仓直接开到至少 80% 最大绝对仓位的同向近满仓；开仓当步计入，至少 10 个连续观测的执行后仓位与突破同号。同号减仓允许，平仓或变号中断。
  - ST2：价格先按 KT2 事件链突破并回踩，Agent 在回踩当步或下一步沿原突破方向同向加仓，且窗口结束前价格再延续。ST2 不要求加仓前盈利。
  - ST3：至少出现一次执行前后仓位同号、绝对仓位增加，且执行前浮动 PnL 大于零的同向加仓。开仓、减仓和反手不计入。
  - SM1：普通窗口在 `z_price ≤ -2.0` 时逆向开/加多，在 `z_price ≥ 2.0` 时逆向开/加空；涨跌停窗口按极端 Label 反向开/加仓。动作后仓位至少达最大绝对仓位的 80% 且绝对仓位增加；仅维持、顺势、减仓、平仓或反手不命中。
  - SM2：`z_price ≥ 0.5` 时仓位向空头方向移动一个相邻档位，`z_price ≤ -0.5` 时向多头方向移动一个相邻档位；时间上必须存在一对方向相反的高低侧调仓事件。允许经过零档，不要求最终回到初始档，跨多档或直接反手不计入。
  - SM3：窗口存在价格创新高/低但 volume 趋势下降的背离段，且该段 `|cum_ret_price| ≥ 0.5%`。背离段不得沿突破方向开仓或加仓；非背离时段至少有一次有效仓位变化，排除全程不交易。
- 普通 20 步窗口运行全部策略规则。涨跌停事件窗口中，ST1/ST2/SM3 至少需要 20 步，ST3 与涨跌停语义下的 SM1 至少需要 1 条执行记录，SM2 至少需要 2 条记录且价格方差非零。所有可运行规则均不命中时保留策略未分类哨兵。
- ST1 和 SM3 不得归因于同一背离突破事件；只有窗口内存在两个不同合格事件时，才可同窗命中两者。
- K 线分类器使用 mark price 和原始 volume。策略分类器使用已执行的 position before/after、浮动 PnL、mark price 和原始 volume，不仅分析目标动作。
- 商品期货数据链必须将原始 volume 作为非模型列原样传递，不将它加入 State Feature，不改变模型输入。
- 重新生成的 Detail CSV 必须包含合约、原始 volume、执行仓位、行情、PnL 和追溯字段，并由 epoch sidecar 记录权威动作空间与完整有序仓位档位。
- 输入允许现有中文表头或对应英文机器名，读取后统一为英文内部 schema。同义双列值冲突时立即失败。
- 行为轨迹以 `timestep` 为唯一权威顺序。每组 timestep 必须是从 0 开始、唯一且连续的整数序列；负值、重复或缺口均立即失败。CSV 全局行序不构成契约。
- 每个 epoch 的期望 Initial-action 集合由 sidecar 中的动作空间参数按环境公式生成，不从已观测行反推。任一期望 Initial-action 的 Detail 行为轨迹缺失时立即失败。
- 窗口表每个 `window_id` 恰好一行。K 线形态用单元素 JSON 数组保存，策略形态用多选 JSON 数组保存。
- 窗口表至少包含 `label`, `epoch`, `bin_index`, `is_selected`, `contract`, `df_path`, `initial_action`, `window_index`, `start_timestep`, `end_timestep`, `window_id`, `kline_patterns`, `strategy_patterns`, `gross_pnl`, `net_pnl`。
- `window_id` 由 Label、epoch、bin index、合约、相对数据路径、Initial-action、窗口索引和起止 timestep 的规范 JSON 计算 SHA-256。Selection 状态、形态结果、PnL、阈值和绝对目录不进入哈希。
- 展开表对每个 `(window_id, kline_pattern, strategy_pattern)` 保存一行，仅用于条件形态分析，不用于账户总 PnL 求和。
- 固定输出窗口表、展开表、Detail Coverage 报告、分析 manifest，以及 K 线/策略/交叉三类视图各自的 Scenario 级和 triple 级 summary，共六个 summary 文件。
- 窗口 gross PnL 等于窗口内已实现 PnL 之和加窗口边界的浮动 PnL 变化；net PnL 再减去窗口手续费。滑点已反映在实际成交价值中，不重复扣除。
- 聚合首先生成 Initial-action Scenario 级 `total_net_pnl`, `window_count`, `pnl_p25`, `pnl_p50`, `pnl_p75`。triple 级每个形态组只对至少命中一个该形态窗口的情景等权平均。
- triple 级必须输出已命中情景数、期望情景数和 Initial-action 覆盖率。未命中该形态的情景不伪造为零 PnL。
- 未分类哨兵保留在窗口表和展开表，但不进入任何正式 K 线、策略或交叉 summary。标定诊断单独报告其窗口数、比率和 PnL 分布。
- 分析 manifest 必须记录阈值、窗口配置、缺少 Detail 的 checkpoint epoch，并为所有实际读取的 Detail CSV、sidecar、Selection Manifest 和生成的 CSV/JSON 输出记录逻辑相对路径、字节数和 SHA-256。
- 分类器、明细生成器和聚合器为新的分析关注点。明细生成器保持为薄 orchestrator，不包含形态判别逻辑。
- 实施需修改商品期货 Scale Save 的非模型列传递，修改低层评估 Detail 行与 epoch sidecar 产生，并新增分类、窗口生成、展开、聚合和标定诊断模块。Agent 选择逻辑、模型动作空间和训练链路保持不变。
- 提议阈值必须通过真实 Detail 数据标定。类别零命中、未分类率偏高或 PnL 区分度弱只生成诊断告警，不作为自动调参目标或硬验收失败。

## Testing Decisions

- 主测试 seam 是两个纯分类函数的外部行为：给定窗口序列和参数，断言输出 K 线形态或策略形态集合。这是新 seam，但是该关注点可用的最高、最稳定 seam。
- 测试只验证外部语义，不断言中间特征的内部计算步骤。
- 合成输入覆盖全部 7 类 K 线形态、6 类策略形态、未分类、多选、事件冲突、多空对称、严格事件时序和阈值上下边界。
- K 线分类测试覆盖优先级，特别是回调与背离优先于通用突破。
- 策略分类测试覆盖 ST2+ST3 兼容多选，以及 ST1/SM3 同事件冲突与不同事件共存。
- 仓位相关规则使用至少两组 sidecar 仓位档位参数化测试，验证近满仓、相邻档位和同向加仓不依赖写死档位数。
- 明细生成器使用一个端到端 smoke test，覆盖多 epoch 候选、Selection Manifest 标记、窗口行数、PnL 守恒、乱序 CSV 稳定性、timestep 异常、epoch 冲突、Detail Coverage 告警和失败边界。
- 聚合器使用一个 smoke test，覆盖窗口唯一、JSON 数组展开、多形态 PnL 不放大、Initial-action 不相加、命中情景等权平均、情景覆盖率、不伪造零 PnL 以及期望行为轨迹缺失失败。
- 数据链测试必须证明原始 volume 传递前后行数、行序和值不变，且 volume 不进入 State Feature。
- 硬验收仅针对可证明不变量：schema、唯一键、必需字段、候选与 Initial-action 覆盖、PnL 守恒、文件指纹、确定性和合成边界测试。真实数据类别占比和 PnL 区分度只用于标定诊断。
- 既有分析纯转换测试是纯函数 seam 的 prior art；既有数据处理测试是上游 volume 传递与切片验收的 prior art。

## Out of Scope

- 不建立本次 6 类策略二阶形态与既有 12 类 Agent Archetype 之间的映射。
- 不计算 per-triple Sharpe、Calmar、MDD 或 win rate。
- 不重新启用 per-step NPY 轨迹转储；本功能直接消费 Detail CSV。
- 不将形态结果接入 Meta Router 或修改 Stage III 路由行为。
- 不进行其他品种或频率的泛化标定；本次范围仅是已确定的商品期货数据集和频率。
- 不修改模型 State Feature、Q 网络输入、训练权重或动作空间。
- 不重新训练模型。上游数据与 Detail 产物需要重生成，既有评估 epoch 需要重跑。
- 不修改 Labeling Method；既有 slope 决策保持不变。
- 不将未分类哨兵纳入正式形态 summary。

## Further Notes

- 该 spec 遵循已有的 Agent 形态双轴窗口架构决策和 FineFT 领域词汇。
- Label 0/6 的实测轨迹很短，KX1 是对涨跌停事件的显式语义，不是为补齐类别数量而构造的普通形态。
- 7×6 组合空间在单个 Label 内天然稀疏。稀疏是行情与 Agent 行为约束的信息，不是需要补齐的缺陷。
- Initial-action Scenario 是反事实起始条件，不是独立账户。情景存在与否取决于行为轨迹是否完成，与是否命中某形态无关。
- 所有数值阈值仍是待真实数据标定的提议值。阈值人工确认是实施流程中已知的检查点，不是未解决的产品设计问题。
- 因当前环境没有 GitHub API 或 `gh` CLI，本文以 issue-ready 本地 spec 交付，未在远程 issue tracker 创建 issue。
