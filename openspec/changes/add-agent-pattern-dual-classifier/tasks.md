# Tasks — add-agent-pattern-dual-classifier

> Spec 见 [issue-spec.md](./issue-spec.md)，概要见 [proposal.md](./proposal.md)，双轴语义见 [ADR-0006](../../../docs/adr/0006-agent-pattern-dual-classifier.md)，旁路边界见 [ADR-0007](../../../docs/adr/0007-isolate-agent-pattern-data-collection.md)。
>
> 状态：**ready-for-agent**。

## 0. 不变性护栏

- [ ] 为既有 Scale Save、单 Agent 测试、Agent 选择和 Selection Manifest 相关文件建立变更前基线，实施 diff 必须证明这些 Python 文件未被修改
- [ ] 确认新入口不导入既有单 Agent 测试入口的专用函数，不读取其 Detail/Aggregate 文件，也不读取 Selection Manifest
- [ ] 确认全部新文件只写入显式 `--output_dir`，不写 checkpoint 或既有选择结果目录
- [ ] 在发现任何目标输出文件已存在时于评估开始前失败，避免部分执行和混合运行
- **验证**：既有代码与产物契约零变更，新功能可在独立空输出目录运行

## 1. 单文件入口与运行配置

- [ ] 新增唯一入口 `test_agents_indexs.py`；评估、纯分类函数、窗口、PnL、展开、统计、聚合和 manifest 逻辑全部位于该文件
- [ ] 提供单个模型参数目录、验证数据根目录、独立输出目录、数据集/实验身份、动作空间和分类阈值所需 CLI 参数
- [ ] 限制模型目录为单一参数目录，只扫描直接 `epoch_<N>` 子目录并按 epoch 数值排序
- [ ] 对缺少模型文件的 epoch 生成 coverage 记录并跳过；模型存在但加载失败时立即失败
- [ ] 使用同一套 CLI 动作空间配置执行全部 epoch，按环境公式生成 Position Level 并在环境创建后校验一致性
- [ ] 发现 contract/原始 volume 缺失、空值或非有限值时立即失败，不使用处理后 volume 特征替代
- **验证**：CLI 配置可重现；多参数目录、输出碰撞、动作空间不一致和必需字段异常均在写正式产物前失败

## 2. 完整评估与按 epoch Detail

- [ ] 对每个可用 epoch 执行全部 `bin_index × label × df_path × initial_action` 组合，不使用 Agent 选择结果过滤
- [ ] 全部 Initial-action 从共享动作空间生成，不从实际观测轨迹反推
- [ ] 每条行为轨迹按 timestep 记录市场、动作、执行前后仓位、奖励、手续费、滑点、已实现/浮动 PnL 和追溯字段
- [ ] 校验行为轨迹 timestep 从 0 开始、唯一、连续且非负
- [ ] 按 epoch 写 `step_detail/agent_pattern_step_detail_epoch_<N>.csv`，所有表头为英文并包含显式 epoch/contract/volume
- [ ] 以流式或分 epoch 方式控制内存，但不得改变输出顺序和确定性
- **验证**：期望笛卡尔积全部有完整轨迹；Detail 文件数与已分析 epoch 数一致，唯一键和值可追溯

## 3. K 线形态纯函数

- [ ] 在单入口文件内实现无 I/O 的 K 线分类函数，输入窗口 price/volume 序列和阈值，输出单一形态
- [ ] 保留 KX1、KM1、KT2、KM3、KT1、KT3、KM2 和未分类语义
- [ ] 普通窗口按 `KM1 → KT2 → KM3 → KT1 → KT3 → KM2 → 未分类` 单选；KX1 仅由 Label 0/6 事件窗口决定
- [ ] 保留突破、回调、加速、V/倒 V、箱体和量价背离的严格时序、对称性与阈值参数化规则
- [ ] 统一完整窗口 z-price 和零方差规则
- **验证**：合成输入覆盖所有正式形态、未分类、优先级、多空对称和阈值边界

## 4. 策略二阶形态纯函数

- [ ] 在单入口文件内实现无 I/O 的策略分类函数，显式接收完整 Position Level、窗口行为与阈值，输出形态集合
- [ ] 保留 ST1、ST2、ST3、SM1、SM2、SM3 多选语义和策略未分类哨兵
- [ ] 保留突破开仓、回踩加仓、盈利后加仓、硬边界反转、离散网格调仓和背离过滤规则
- [ ] 保留 ST2+ST3 兼容、ST1/SM3 同事件冲突、多空对称和每条规则最小样本数
- [ ] 仓位相邻、近满仓和同向加仓全部由共享 Position Level 推导，不写死档位
- **验证**：至少两套动作空间参数化覆盖全部正式形态、多选、冲突、未分类和阈值边界

## 5. 窗口、身份与 PnL

- [ ] Label 1~5 对每条行为轨迹生成连续、不重叠的完整 20 步窗口
- [ ] Label 0/6 对每条完整行为轨迹只生成一个涨跌停事件窗口
- [ ] 尾部不足 20 步不生成窗口，并记录 dropped tail steps/gross PnL/net PnL
- [ ] window id 使用已确认的九个身份/边界字段规范 JSON 的 SHA-256，必须区分 Initial-action
- [ ] 计算 realized PnL sum、浮动 PnL 起止、手续费、滑点诊断、gross PnL 和 net PnL；不得重复扣除滑点
- [ ] 生成固定英文 schema 的 `agent_pattern_window_table.csv`，每个 window id 恰好一行
- **验证**：窗口数、窗口身份、首/后续窗口 PnL 边界、尾部 PnL 和全轨迹 PnL 守恒全部通过

## 6. 展开、Coverage 与 Diagnostics

- [ ] 从 Window table 生成固定 schema 的 `agent_pattern_expanded_table.csv`
- [ ] 保证 Expanded 唯一键为 `(window_id, kline_pattern, strategy_pattern)`，并明确其 PnL 不可用于账户总额求和
- [ ] 生成固定 schema 的 `agent_pattern_coverage_report.csv`，同时覆盖 epoch 与 trajectory 记录
- [ ] Coverage 验证完整的 epoch/bin/Label/df path/Initial-action 组合及 dropped tail
- [ ] 生成固定长表 schema 的 `agent_pattern_classifier_diagnostics.csv`
- [ ] Diagnostics 覆盖 overall/label/epoch/triple 和 kline/strategy/cross，报告类别数量、比率、净 PnL 分位数、median range 及告警
- [ ] 未分类率高、正式类别零命中或区分度弱仅写告警，不使运行失败
- **验证**：策略多选比率允许合计超过 1；未分类哨兵保留；相同输入和配置输出确定

## 7. 六个固定聚合视图

- [ ] 生成 `agent_pattern_{kline,strategy,cross}_scenario_summary.csv` 三类 Scenario summary，键中包含 contract、df path 和 Initial-action
- [ ] Scenario summary 固定输出 total net PnL、window count 和窗口 net PnL 的 p25/p50/p75
- [ ] 生成 `agent_pattern_{kline,strategy,cross}_triple_summary.csv` 三类 triple summary
- [ ] Triple summary 只对实际命中目标形态的 Scenario 统计做等权平均，不为未命中情景补零
- [ ] 输出 observed/expected Initial-action count 和 coverage ratio
- [ ] 不跨 Initial-action 或策略形态生成账户总 PnL
- [ ] 未分类哨兵不进入六个正式 summary
- **验证**：六个文件 schema 固定，多选展开和情景聚合均不放大账户盈亏

## 8. Analysis Manifest

- [ ] 生成固定顶层键的 `analysis_manifest.json`
- [ ] 记录数据集/实验身份、模型和数据逻辑根、评估配置、动作空间、窗口配置、分类阈值、发现/分析/缺失模型的 epoch、候选全集和告警
- [ ] 为所有实际模型、验证数据和生成输出记录逻辑相对路径、字节数与 SHA-256
- [ ] 不记录 Selection Manifest，不生成 is selected，不记录 manifest 自身指纹
- [ ] 绝对输出目录不进入窗口或数据身份
- **验证**：修改任一输入或输出都会改变对应指纹；同一输入配置的逻辑身份稳定

## 9. 端到端与回归验收

- [ ] 使用多个小 epoch、多个 bin、多个 Label、多个数据文件和全部 Initial-action 构建端到端 CLI smoke test
- [ ] 验证逐步 Detail 按 epoch 分区，其余分析文件跨 epoch 汇总，固定文件清单全部生成
- [ ] 验证所有 CSV 英文表头及固定列顺序
- [ ] 验证候选与行为轨迹全覆盖、窗口和展开唯一键、PnL 守恒、聚合不放大以及 manifest 指纹
- [ ] 验证输出碰撞、缺字段、非有限值、非法 timestep、动作空间错误、模型加载错误均失败
- [ ] 用 `conda activate finetf` 环境运行目标测试与相关低层回归测试
- **硬验收**：schema、唯一键、候选全集、全部 Initial-action、PnL、确定性、指纹和合成分类边界全部通过；真实类别占比仅作诊断
