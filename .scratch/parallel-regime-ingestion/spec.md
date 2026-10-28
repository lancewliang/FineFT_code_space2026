# 体制分片并行入池与同环境训练复现实施规格

Status: ready-for-agent
Date: 2026-10-28
Decision: ADR-0066（accepted）
Related decisions: ADR-0012、ADR-0039、ADR-0040、ADR-0044、ADR-0052、ADR-0065

## Problem Statement

研究人员希望缩短当前并行训练的完整轮次耗时。探索虽然已经由多个进程执行，主进程仍逐条构造经验身份、查询去重索引、决定组内替换与 FIFO 槽位，再整理数组并写入 GPU。只增加探索进程无法消除这段串行工作，可能使完成的探索结果继续排队等待入池。

现有系统已经具备体制分层经验回放池、GPU 驻留张量、共享市场数据包、共享推断模型、常驻探索进程池与紧凑轨迹回传。本次应在这些能力上分担 CPU 入池工作，而非重新引入完整行情 IPC、每进程数据副本或每轮重新启动进程。

研究人员接受新方案保留的具体经验与旧实现不同，但要求在同一机器、软件版本、数据、代码、配置及种子下，重复训练获得相同的逐轮经验池内容和最终模型权重。当前探索任务缺少任务级随机流，结果按完成时序切批入池，多个用途共享随机状态，仅给所有子进程设置相同种子不能满足该要求。

成功标准是稳态完整训练轮次的中位耗时至少下降 20%，且复现检查通过。局部去重函数加速、CPU 占用率提高或单次空池写入加速不能代替该标准。

## Solution

为每个体制入池分片配置一个独立、常驻的 CPU 入池进程；当前 3×3 体制格对应 9 个进程。每个进程独占所属格的经验组索引、后续经验变体、随机轮换状态及 FIFO 游标。训练主进程保留唯一 GPU 写入者和现有训练采样接口，不再逐条参与去重与保留决策。

探索结果按稳定任务身份及轨迹步序，经有界重排后流式分发到各格，使探索与入池重叠。分片通过有界队列返回紧凑的槽位写入批次，主进程批量还原行情状态并写 GPU。所有任务、分片与 GPU 写入完成后，才能进行本轮训练更新。

总种子按任务、体制格和采样用途派生独立随机流；预训练、探索、入池、采样和 GPU 更新共同满足同环境复现要求。子进程异常时本次运行明确失败并清理资源。保持 ADR-0065 的精确经验身份、同组最多 4 种后续、随机轮换、格内 FIFO 及分格采样规则。

## User Stories

1. As a training researcher, I want complete training epochs to finish faster, so that I can evaluate experiments sooner without reducing the amount of learning.
2. As a training researcher, I want one ingestion process per regime grid, so that the current 3×3 topology uses nine independent ingestion processes.
3. As a training operator, I want ingestion processes to remain alive across epochs, so that repeated process startup does not consume the expected speedup.
4. As a training researcher, I want deduplication and retention decisions to run outside the training coordinator, so that completed rollouts do not wait behind one per-record Python loop.
5. As a training operator, I want one process to own GPU replay writes, so that training continues to use one coherent GPU-resident replay buffer.
6. As a training operator, I want market data to remain shared and read-only, so that ingestion processes do not duplicate the entire dataset.
7. As a training operator, I want compact numerical batches transferred between processes, so that repeated wide market states do not inflate IPC traffic.
8. As a training researcher, I want exploration and ingestion to overlap, so that their costs do not simply add together at every epoch.
9. As a training researcher, I want experiences routed by their starting decision regime, so that a rollout crossing several grids remains correctly stratified.
10. As a training researcher, I want multi-step returns accumulated along the original trajectory before routing, so that transitions from different regime visits are never joined into a false return.
11. As a training researcher, I want all model-visible holding inputs included in experience identity, so that different positions, profits, drawdowns and holding durations remain distinguishable.
12. As a training researcher, I want identity comparisons to use the learner's float32 precision, so that deduplication matches the information consumed by training.
13. As a training researcher, I want signed zero normalized and non-finite identity fields rejected, so that malformed or equivalent numerical values receive consistent handling.
14. As a training researcher, I want up to four distinct outcomes retained for one holding decision experience group, so that alternative subsequent paths are not reduced to a single teacher-preferred result.
15. As a training researcher, I want identical resident transitions merged with the existing metadata-refresh rule, so that duplicate records do not occupy additional outcome slots.
16. As a training researcher, I want a fifth distinct outcome to trigger seeded random rotation, so that retention does not rank paths by profit or teacher disagreement.
17. As a training researcher, I want each regime grid to retain its own capacity and FIFO policy, so that common regimes cannot evict scarce-regime experiences.
18. As a training operator, I want identity indexes reclaimed when records are evicted, so that metadata remains bounded by resident replay capacity.
19. As a training researcher, I want an evicted outcome to become eligible again, so that historical deduplication does not permanently suppress later exploration.
20. As a training researcher, I want exploration to continue after replay reaches capacity, so that new model behavior can refresh the resident experience set.
21. As a training researcher, I want rollout randomness bound to stable task identity, so that worker scheduling does not change a task's exploration decisions.
22. As a training researcher, I want each regime grid to own a derived retention random stream, so that process scheduling does not change which outcomes are rotated.
23. As a training researcher, I want replay sampling to own an independent random stream, so that unrelated random operations do not shift training batches.
24. As a training researcher, I want fixed logical input order and batch boundaries, so that queue timing does not change FIFO, retention or replay contents.
25. As a training researcher, I want pretraining collection and sampling to be reproducible, so that repeated runs enter Diverse Training with the same model state.
26. As a training researcher, I want GPU updates checked under strict deterministic settings, so that identical replay batches produce identical model weights in the supported environment.
27. As a training researcher, I want data, configuration, code and runtime identities recorded, so that a reproducibility result has an explicit environment scope.
28. As a training operator, I want in-flight tasks, reorder storage and transfer queues bounded, so that long or uneven rollouts cannot consume unbounded memory.
29. As a training operator, I want the coordinator to service all ready queues fairly, so that backpressure on one grid does not deadlock the whole pipeline.
30. As a training researcher, I want tasks with no experiences for a particular grid to advance that grid's completion watermark, so that empty contributions do not block an epoch forever.
31. As a training researcher, I want per-grid write batches applied in sequence, so that delayed old writes cannot overwrite a slot that has been reused.
32. As a training researcher, I want training updates to start only after every accepted write is committed, so that one epoch never samples a partially ingested replay state.
33. As a training operator, I want child-process failures and protocol errors to fail the run explicitly, so that incomplete experiences are never silently treated as a valid epoch.
34. As a training operator, I want failure and normal shutdown to release all owned processes and queues, so that subsequent experiments do not inherit blocked or orphaned resources.
35. As a training researcher, I want a deterministic serial reference using the same retention rules, so that parallel correctness can be checked against an unambiguous baseline.
36. As a training researcher, I want repeated short training runs compared at epoch boundaries and final weights, so that local deduplication tests do not create false confidence in full-run reproducibility.
37. As a training researcher, I want representative full-buffer epochs benchmarked with identical workloads, so that speedup measurements include replacement and actual learning costs.
38. As a training operator, I want per-stage timings, per-grid load, queue backlog and peak memory reported, so that remaining bottlenecks can be identified from evidence.
39. As a training researcher, I want at least a 20% reduction in median steady-state epoch duration with reproducibility preserved, so that added multiprocessing complexity earns its cost.
40. As a training researcher, I want historical artifacts preserved and new experimental identities used for the changed retention protocol, so that old results are not overwritten or falsely presented as equivalent runs.

## Implementation Decisions

1. **分片与进程对应。** 使用领域术语“体制入池分片”表示单个体制格的逻辑入池单元；每格由一个 CPU 进程独占管理。入池进程数跟随体制格总数，当前为 9；不改成 4 个进程承载 9 格，也不根据 CPU 使用率自动合并分片。现有探索进程数是独立配置，不因增加入池进程而擅自修改。

2. **常驻生命周期。** 分片进程池与 Diverse Training 阶段绑定，跨轮次保留去重、保留和随机状态；正常退出或失败时统一回收。预训练仍按其既有阶段运行，本次不为其复制一套体制分片池。

3. **单一状态所有者。** 分片进程持有所属格的经验组、已驻留后续变体、槽位映射、占用计数、FIFO 游标、保留统计及保留随机状态。主进程持有最终 GPU 张量、训练网络、优化器及训练采样器。主进程不得再次逐条构键、查重或重做分片保留决策。

4. **保留 GPU 经验池。** 沿用现有 GPU 驻留张量布局与训练采样契约，分片进程不创建 GPU 上下文、不各自复制整个经验池。共享市场数据包继续只读共享；共享内存暂存区不是首版要求。

5. **接口职责集中。** 在现有 Stage I 训练编排入口后接入一个内部并行入池模块，由其统一封装分片生命周期、流量控制、确定性排序、完成协议及错误传播。入口沿用现有数据、训练参数和结果产物约定，不要求其他调用者了解 9 套队列或内部元数据结构。现有经验池公开契约及训练采样接口继续使用，避免跨代码库新增多个公开控制接口。

6. **任务与路由身份。** 为轮次内完整探索任务表赋予稳定任务编号；探索结果标识轮次、任务、原始行情切片、子策略、初始动作和轨迹步序。一次任务可以跨多个体制格，应根据每条经验起始决策步的体制标签批量分流。沿用轨迹优先多步累加，禁止先分格再跨不连续记录累加回报。

7. **稳定任务随机流。** 使用版本化且稳定的种子派生约定，将总种子与随机用途、轮次和任务身份组合，派生任务独立的探索随机流；不依赖 PID、worker 编号、任务实际完成顺序或进程随机化的 Python hash。避免所有子进程反复消费相同随机序列。

8. **独立保留与采样随机流。** 每个体制格持有按总种子和体制身份派生的独立保留随机流；训练采样器持有设备适配、与网络初始化及评测隔离的随机流。相同逻辑输入下，其他格运行快慢不得改变本格的抽签序列。新协议不要求复现旧共享随机流的具体槽位结果。

9. **确定性流式提交。** 为每格定义按轮次、稳定任务编号及轨迹步序排列的输入序列。乱序完成的结果在有界窗口内暂存，只有满足规定序列的部分可提交。微批次边界由固定配置和逻辑记录序号决定，禁止以队列暂空、到达时刻或定时刷新作为会改变经验保留的切批依据。

10. **有界资源与公平推进。** 任务派发窗口、乱序缓存、各格输入队列和输出队列均有明确上限；预算应同时考虑记录数、消息字节量和长轨迹差异。协调器交错派发、收集和排空输出，不能在填满有界输入队列后才读取结果，也不能阻塞等待一个格而停止服务其他格。配置及观测方式需进入运行记录；具体容量由实现时的内存测量确定，不把此前未讨论的数字伪装成用户决定。

11. **完成与水位协议。** 使用显式任务完成、每格连续任务水位、批次序号和轮末完成确认。某任务没有某格经验时，该格仍须能够确认已越过该任务。队列暂空、最后一次读取成功或收到部分格的结束消息均不代表本轮完成。

12. **槽位写入契约。** 分片输出标识轮次、分片、单调提交序号、目标槽位和已接纳经验的行情坐标及动态数值字段。主进程验证协议归属及顺序，通过共享行情批量还原状态并写 GPU；同一格旧批次不能覆盖已复用槽位的新数据。不同格的提交交错不能影响去重、保留随机流或后续采样。

13. **轮次完成条件。** 全部预期探索任务已完成，各格已确认最终水位及输出批次数，所有已接纳批次均已提交 GPU，才允许保存一致的经验池快照并进入本轮参数更新。保持完整探索后完整更新的轮次界限；不引入模型训练与下一轮探索交错的异步 Actor-Learner 语义。

14. **经验身份保持。** 沿用 ADR-0065 的 float32 模型可见上下文、正负零规范化及非有限值拒绝规则。当前持仓收益、回撤、时长、敞口、时间和可用动作掩码不能粗化或删除；教师意见与收集误差不成为经验身份。紧凑路径使用本次运行内不可变的行情坐标，完整转移契约仍遵循同一等价关系。

15. **保留规则保持。** 同一持仓决策经验组最多保留 4 种后续经验变体；完全相同的已驻留转移按现有严格比较规则刷新允许的元数据或跳过。第 5 种后续随机选择现有槽位替换，新候选进入；不按利润、亏损或旧误差筛选不同后续，不宣称这种轮换是对所有历史结果的均匀蓄水池抽样。保持每格容量、FIFO、索引同步回收、淘汰结果重新入选及现有分格采样行为。

16. **保持学习任务。** 经验池满后仍收集完整任务集。沿用现有轮次课程、参数调度、动作保持、动作空间、奖励与费用口径、DP 教师以及已修正的多步后续价值折扣。本次不改变训练轮数、更新次数、网络结构或风控语义来获得加速。

17. **完整训练复现。** 预训练收集及提交顺序、其采样随机流、网络初始状态、探索、经验保留、训练采样及 GPU 更新共同纳入同环境复现要求。核查并验证严格 GPU 确定性设置；遇到不支持的非确定性算子应明确报错，不静默回退或改为仅比较近似指标。相同条件指同硬件、软件版本、数据、代码、配置及总种子，不承诺跨环境逐位一致。

18. **错误与清理。** 分片异常退出、worker 错误、数据身份异常、错误轮次/分片/序号、重复或缺失完成协议应显式失败。必要的等待须具备存活检查及明确的超时失败路径，不能永久卡在缺口。首版采用 Fail-fast，停止本次运行并清理拥有的进程、队列及临时资源；不丢弃坏格继续训练，不重置随机状态后假装恢复。

19. **串行参考语义。** 建立使用相同稳定输入顺序、每格随机流、保留规则和 GPU 确定性设置的串行参考方式，用于正确性与性能对照。该方式仅为参考及验证需要，不改变用户选定的正常运行拓扑。旧实现受完成顺序影响，其历史保留结果不作为新协议必须逐槽位复制的答案。

20. **可观测性和结果身份。** 报告分阶段时间、每格接收/保留/替换计数、队列积压、峰值内存、启动与回收成本，以及数据、代码、配置、随机协议和运行环境身份。比较规范化数值内容，不比较日志时间戳、运行耗时或序列化附带差异。新协议使用独立实验身份和产物空间，保留历史数据与模型；不承诺旧张量快照能恢复新的完整去重索引及随机状态。

## Testing Decisions

1. **主测试 seam 使用现有完整 Stage I 训练编排入口。** 通过同一现有入口装配串行参考方式和 9 进程方式，以短但完整的训练运行验证公开行为；覆盖必要预训练、多个 Diverse Training 轮次、实际入池、采样、更新和模型产物。不为每个队列、私有函数或字典新增单独的公开测试接口。这个选择落实已确认的完整训练复现与完整轮次性能要求，不新增一轮访谈。

2. **优先比较可观察结果。** 好的测试固定输入、初始状态、种子和环境，检查按格导出的经验内容、公开保留统计、可重放采样结果、逐轮损失、最终权重及明确的失败/退出行为。避免只断言私有函数调用次数、内部容器形状或以 mock 替代真实多进程路径；已有内部不变量回归可以保留作为诊断辅助，不能代替主接口验证。

3. **复用已有经验池契约测试。** 现有有界持仓经验、紧凑轨迹与完整转移一致性、语义去重、组内随机轮换、FIFO 索引回收、分层采样及多步学习目标测试是先例。继续覆盖精确模型上下文、正负零、非有限字段、同组不同后续、元数据刷新、容量上限和淘汰后再入选，确保并行化未改变 ADR-0065。

4. **复用已有轮次编排与生命周期测试思路。** 当前课程轮换、经验池快照后开始更新、探索任务派发/收集、CPU 单线程 worker、异常传播与停机清理测试提供回归基础。更新其中把队列到达或暂空切批当成正确行为的旧断言，使其服从确定顺序的新契约，而不是保留旧时序偶然性。

5. **真实多进程协议测试。** 经主接口运行真实的分片进程，主动扰动任务完成顺序，混入跨格轨迹、没有某格经验的任务、空贡献格、短长任务和多次同槽复用。验证结果与确定性参考一致，本轮数据全部提交后才开始更新，所有进程在多个轮次中正确复用并在结束后回收。

6. **背压与失败测试。** 使用较小但有效的容量预算放大队列背压，验证较慢格、乱序结果及满输入/输出队列不会死锁或造成无界缓存。注入分片报错和非正常退出，验证主入口明确失败、阻塞能解除、尚未完成的轮次不会更新网络或冒充成功检查点。测试应有总截止时间，避免失败场景无限挂起。

7. **完整复现对照。** 同环境重复运行短完整训练，并在串行参考与 9 进程实现之间比较逐轮经验张量、保留统计、后续采样、损失及最终权重。至少跨越一次满池替换，覆盖预训练进入 Diverse Training 的起始状态。CPU 测试用于快速功能反馈；要求 GPU 更新的逐位复现必须在实际 GPU 上验证，不以 CPU 成功或 GPU 跳过代替。

8. **GPU 确定性失败也属于结果。** 若当前运行环境中的更新路径无法满足严格确定性，应保留可诊断错误并报告尚未通过复现验收；不能悄悄放宽设置、忽略失败或把数值接近当作逐位相同。环境版本及确定性设置必须随验证结果保存。

9. **完整轮次性能对照。** 使用代表性训练数据、固定相同任务、更新次数和学习参数，在两种执行方式下采用一致预热和测量方式，统计多个连续的满池稳态轮次。轮次费用包括探索、重排与协调、分片入池、GPU 提交、快照、全部参数更新、权重同步及模型保存；后台评测使用相同配置，并在比较窗口结束时计入未完成评测收尾。既记录每轮分布，也记录窗口总墙钟时间，避免隐藏流水线尾部成本。

10. **性能通过条件。** 9 进程方式的稳态完整轮次中位耗时必须不高于串行参考方式的 80%，同时通过语义和复现验证；报告轮次数、耗时分布、峰值内存、队列积压、每格负载及一次性启动/回收成本。测试中仍执行相同训练工作量。若未达标，明确标记性能验收未完成并按证据继续归因，不以局部微基准、减少任务或放宽复现要求补足目标。

11. **当前规格发布的验证范围。** 本轮检查规格结构、状态标签、与 ADR 的一致性及测试契约覆盖；不运行正式训练或声称已经完成上述代码、GPU、复现或性能验证。后续 Python 验证遵守项目既定的 finetf Conda 环境要求。

## Out of Scope

- 改变经验身份精度、把持仓上下文分粗桶、以利润或教师误差挑选不同后续、取消精确去重，或者永久维护全历史去重表。
- GPU 全局排序去重、GPU 端并发哈希表、每个分片各建 GPU 写入者、将最终经验池搬回 CPU，或首版直接建设共享内存传输环。
- 以 4 个进程承载 9 格、自动合并体制分片、自动调节入池进程数、改变体制拓扑或体制标签来源。
- 优先经验回放、新的损失加权、改变分层采样分布、奖励、手续费、换手惩罚、多步目标、动作保持、网络结构、课程表或训练工作量。
- 让下一轮探索与本轮梯度更新混跑的异步 Actor-Learner，或让未完成入池的数据被训练消费。
- 自动重启入池进程、恢复半轮状态、从旧张量快照重建完整索引/RNG、分布式多机训练及跨硬件/软件版本逐位复现。
- 与本次入池或复现控制无关的环境历史统计加速、预训练算法/吞吐改造、高层路由、风控、实盘执行或策略收益优化。
- 本轮规格发布之外的源代码实现、正式长程重训、历史产物清理与部署。规格完成不代表这些动作已经执行。

## Further Notes

- ADR-0066 是本规格的已接受架构决定；ADR-0065 的经验身份与有界保留规则优先于旧文档中粗时长桶、误差择优筛掉不同后续等历史表述。ADR-0012 的体制隔离和轨迹优先多步累加继续保留；不借本次优化重新选择课程表或体制标签算法。
- 已观察到的历史训练中，探索与入池合计是主要轮次费用。已有一次批量构键临时对照，在同一批 38,512 条真实经验上将入池中位耗时由约 2.09 秒降至 1.12 秒，并通过该批次的内容一致性检查；这只是局部原型证据，不是 9 进程方案、完整训练复现或整轮 20% 加速的通过证明。
- 代码现有注释、历史性能文案或“纳秒级”等描述不能当作性能事实。应以本规格约定的当前代码、固定工作量和端到端测量为准。
- 首版选择有界队列是为了集中验证 CPU 去重并行收益。只有测量证明 IPC 已成为完整轮次瓶颈，才单独讨论共享内存暂存替代方案；该替代不属于本规格的默认实施范围。
- 具体队列容量、批次大小、在途窗口、等待超时和测量轮次数是需由实现与实测固定并记录的工程参数，当前没有用户批准的最优数字。不得任意改变这些参数而仍宣称重跑的是同一配置。
- 本规格已综合完整设计访谈，不存在待用户补答的架构问题；ready-for-agent 表示可供后续实现任务直接消费，不表示实现或验收已完成。本轮不拆分额外实施票据。
