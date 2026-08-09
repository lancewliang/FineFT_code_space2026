# fineft-agent-pattern-analysis Delta

## ADDED Requirements

### Requirement: Commodity evaluation data SHALL preserve raw volume without changing model input

系统 SHALL 将 split Feather 中的原始 `volume` 作为非模型列保留到 Scale Save 产物和 label 切片，不得将它加入 `state_features.npy`。

#### Scenario: Scale Save directly passes through raw volume

- **GIVEN** commodity Scale Save 正在处理一份包含 `timestamp`, `contract`, `volume` 的 split DataFrame
- **WHEN** Scale Save 写出该合约的 train/valid/test Feather
- **THEN** 输出 SHALL 保留与输入逐行完全相同的原始 `volume`
- **AND** 行数和行序 SHALL 不变
- **AND** 实现 SHALL 从当前 DataFrame 直接选取 `volume`，不得为此执行二次 join
- **AND** `volume` SHALL NOT 出现在 `state_features.npy`

#### Scenario: Label slicing retains evaluation identity and volume

- **WHEN** `slice_model.py` 从 Scale Save 产物生成 `valid/<contract>/label_*/df_*.feather`
- **THEN** 每行 SHALL 保留非空 `contract` 和原始 `volume`
- **AND** 切片逻辑和模型 State Feature 集合 SHALL 不变

### Requirement: Low-level Detail artifacts SHALL carry classifier inputs and action-space provenance

系统 SHALL 在逐步 Detail CSV 中输出形态分类所需的实际执行字段，并用 epoch sidecar 记录评估时的动作空间。

#### Scenario: Detail rows include contract and raw volume

- **WHEN** `test_agent_index.py` 为一个 validation step 生成 Detail row
- **THEN** row SHALL 包含非空 `contract` 和原始 `volume`
- **AND** `volume` SHALL 等于当前 label Feather 同一 timestep 的值
- **AND** row SHALL 包含 `position_before`, `position_after`, `mark_price`, `unrealized_pnl`, `realized_pnl_step`, `commission_fee_step`

#### Scenario: Detail sidecar records exact action space

- **WHEN** epoch `N` 的 Detail CSV 写出
- **THEN** 同目录 SHALL 写出 `trading_action_detail_epoch_<N>.manifest.json`
- **AND** sidecar SHALL 包含 `epoch`, `max_holding_number`, `position_choices`, `leverage_choices`, `position_levels`
- **AND** `position_levels` SHALL 与评估环境的完整有序 signed position 档位一致

#### Scenario: Classifier rejects stale or mismatched Detail artifacts

- **GIVEN** Detail CSV 缺少 `contract` 或原始 `volume`，或 sidecar 缺失
- **WHEN** 明细表生成器读取该 epoch
- **THEN** 系统 SHALL fail fast 并提示重新生成 Detail 产物
- **AND** 系统 SHALL NOT 使用 `log_volume_origin` 替代原始 `volume`
- **AND** 任何 CLI 动作空间 override 与 sidecar 不一致时 SHALL fail fast

### Requirement: Pattern analysis SHALL classify the full available Agent candidate universe

系统 SHALL 分类 `--model_root` 下所有存在 Detail 的 `(label, epoch, bin_index)` Agent triple，`selection_manifest.json` 只用于标记已选项。

#### Scenario: Selection manifest marks but does not filter candidates

- **GIVEN** Detail CSV 中同时存在已选与未选 Agent triple
- **WHEN** 系统生成形态明细
- **THEN** 所有 triple SHALL 出现在输出中
- **AND** 只有与 manifest 的 `(label, epoch, bin_index)` 精确匹配项 SHALL 设置 `is_selected=true`

#### Scenario: Selection manifest belongs to the analyzed experiment

- **GIVEN** 系统收到独立的 `model_root` 和 `selection_manifest`
- **WHEN** 系统解析已选 Agent
- **THEN** manifest 的 `dataset_name` 和 `experiment_name` SHALL 与 `model_root` 的逻辑数据集和实验一致
- **AND** manifest SHALL 对 `label_0` 至 `label_6` 各包含恰好一条记录
- **AND** 每条 `epoch_path` 和 `model_path` 解析出的 epoch SHALL 一致
- **AND** `model_path` 的逻辑末尾 SHALL 为该 epoch 下的 `trained_model.pkl`
- **AND** 对应 `model_root/epoch_<N>` SHALL 存在
- **AND** 校验 SHALL 只比较逻辑数据集、实验、epoch 和文件末尾，不依赖机器绝对路径前缀
- **AND** 任何错配、重复、缺失或格式错误 SHALL fail fast

#### Scenario: Detail coverage failures are distinguished

- **WHEN** checkpoint epoch 缺少 Detail 但不包含 manifest 已选 Agent
- **THEN** 系统 SHALL 在覆盖率报告告警并继续
- **WHEN** manifest 的任一已选 triple 在 Detail rows 中不存在
- **THEN** 系统 SHALL fail fast
- **AND** Detail 文件名、父目录或 sidecar 的 epoch 不一致，或同 epoch 存在重复 Detail 时 SHALL fail fast
- **AND** 覆盖与尾部丢弃诊断 SHALL 写入 `agent_pattern_coverage_report.csv`

### Requirement: Kline Pattern classifier SHALL return one deterministic market pattern per window

系统 SHALL 对 `mark_price` 和原始 `volume` 序列返回单一 K 线形态，并按 `KX1→KM1→KT2→KM3→KT1→KT3→KM2→未分类` 解决多规则命中。

#### Scenario: Ordinary labels use complete non-overlapping windows

- **GIVEN** label 属于 `label_1` 至 `label_5`
- **WHEN** 系统切分一条行为轨迹
- **THEN** 系统 SHALL 只产生长度 20、步长 20 的完整窗口
- **AND** 尾部不足 20 步 SHALL NOT 产生形态行
- **AND** 覆盖率报告 SHALL 按轨迹记录 `dropped_tail_steps`, `dropped_tail_gross_pnl`, `dropped_tail_net_pnl`

#### Scenario: Limit labels use one whole-trajectory event window

- **GIVEN** label 为 `label_0` 或 `label_6`
- **WHEN** 系统分类该行为轨迹
- **THEN** 整条轨迹 SHALL 产生恰好一个事件窗口
- **AND** `kline_patterns` SHALL 等于 `["KX1"]`
- **AND** 轨迹长度达到某策略规则最小样本数时，该规则 SHALL 扫描整条轨迹而非只看前 20 步

#### Scenario: Z-score rules use one defined convention

- **WHEN** KM1、SM1 或 SM2 计算 `z_price`
- **THEN** 它 SHALL 使用完整形态识别窗口的均值和 `ddof=0` 标准差
- **AND** 标准差为零时所有依赖 `z_price` 的规则 SHALL NOT 命中

#### Scenario: Kline priority prevents generic breakout from swallowing specific patterns

- **GIVEN** 同一窗口同时满足 KT1 与 KT2 或 KM3
- **WHEN** K 线分类器返回结果
- **THEN** KT2 SHALL 优先于 KM3 和 KT1
- **AND** KM3 SHALL 优先于 KT1
- **AND** 输出 SHALL 始终只包含一个 K 线形态

### Requirement: Strategy Pattern classifier SHALL use executed positions and allow compatible multi-label results

系统 SHALL 使用 Detail sidecar 仓位档位和实际 `position_before/position_after` 识别 ST1、ST2、ST3、SM1、SM2、SM3，一个窗口可命中多个兼容形态。

#### Scenario: ST1 holding count includes the opening observation

- **GIVEN** Agent 在合格突破当步或下一步从空仓开到近满仓且方向与突破一致
- **WHEN** 系统计算 ST1 持仓长度
- **THEN** 开仓当步 SHALL 计为第一个持仓观测
- **AND** 至少 10 个连续观测的 `position_after` SHALL 与突破同号
- **AND** 同号减仓 SHALL 仍算持仓，平仓或变号 SHALL 中断计数

#### Scenario: ST2 and ST3 can co-exist

- **GIVEN** Agent 在突破回踩时同向加仓
- **AND** 加仓前 `unrealized_pnl_before > 0`
- **WHEN** 策略分类器返回结果
- **THEN** `strategy_patterns` SHALL 同时包含 `ST2` 和 `ST3`
- **AND** 窗口 PnL SHALL 仍只保存一次

#### Scenario: SM2 requires an ordered opposite adjustment pair

- **GIVEN** 窗口内有一次高位向空头相邻档位移动和一次低位向多头相邻档位移动
- **WHEN** 两个事件在时间上形成一对方向相反的调仓
- **THEN** SM2 SHALL 命中
- **AND** 最终仓位 SHALL NOT 被要求回到窗口初始档位
- **AND** 跨多档跳仓或直接反手 SHALL NOT 计为 SM2 事件

#### Scenario: Unmatched short event remains observable

- **GIVEN** 涨跌停事件窗口的所有可运行策略规则均未命中
- **WHEN** 系统生成窗口行
- **THEN** `strategy_patterns` SHALL 等于 `["策略未分类"]`
- **AND** 窗口行 SHALL NOT 被丢弃

### Requirement: Window PnL SHALL be conserved once before any pattern expansion

系统 SHALL 用已实现 PnL、窗口边界浮动 PnL 和手续费计算每个窗口唯一的毛盈亏与净盈亏。

#### Scenario: Window PnL follows the execution ledger

- **WHEN** 系统计算 `[start, end]` 窗口
- **THEN** `gross_pnl` SHALL 等于 `sum(realized_pnl_step[start:end+1]) + unrealized_pnl[end] - unrealized_pnl_before_start`
- **AND** `net_pnl` SHALL 等于 `gross_pnl - sum(commission_fee_step[start:end+1])`
- **AND** 非首窗口的 `unrealized_pnl_before_start` SHALL 取同轨迹前一行，首窗口 SHALL 取 0
- **AND** `slippage_step` SHALL NOT 被再次扣除

#### Scenario: Expanded patterns do not multiply account PnL

- **GIVEN** 一个窗口同时命中多个策略形态
- **WHEN** 系统生成窗口表和展开表
- **THEN** 窗口表 SHALL 对该 `window_id` 只保存一行和一组 PnL
- **AND** 账户总盈亏 SHALL 只能从窗口表的唯一 `window_id` 计算
- **AND** 展开表 SHALL NOT 被跨 `strategy_pattern` 求和为账户总盈亏

### Requirement: Pattern artifacts SHALL use stable machine schemas

系统 SHALL 规范化输入表头，并生成可重现、具有明确唯一键的窗口级与展开级 CSV。

#### Scenario: Bilingual Detail headers normalize without ambiguity

- **GIVEN** Detail CSV 使用现有中文表头或对应英文机器名
- **WHEN** 系统读取 Detail
- **THEN** 它 SHALL 规范化为唯一英文内部 schema
- **AND** 同一语义的中英文列同时存在且值冲突时 SHALL fail fast

#### Scenario: Trajectory timestep defines deterministic row order

- **GIVEN** Detail CSV 的不同行为轨迹任意交错，或同一轨迹的行未按时间排列
- **WHEN** 系统按 `(label, epoch, bin_index, contract, df_path, initial_action)` 构建行为轨迹
- **THEN** 系统 SHALL 以 `timestep` 升序排序该轨迹
- **AND** `timestep` SHALL 是从 0 开始、唯一且连续的整数序列
- **AND** 任一负值、重复或缺口 SHALL fail fast，不得静默重新编号
- **AND** CSV 全局行序 SHALL NOT 影响切窗、分类或 PnL 结果

#### Scenario: Window table has one stable row per window

- **WHEN** 系统写出 `agent_pattern_window_table.csv`
- **THEN** 每行 SHALL 包含 `label, epoch, bin_index, is_selected, contract, df_path, initial_action, window_index, start_timestep, end_timestep, window_id, kline_patterns, strategy_patterns, gross_pnl, net_pnl`
- **AND** `kline_patterns` 和 `strategy_patterns` SHALL 是可由 `json.loads` 解析的 JSON 数组
- **AND** `window_id` SHALL 由 `(label, epoch, bin_index, contract, df_path, initial_action, window_index, start_timestep, end_timestep)` 的规范 JSON 表示计算 SHA-256
- **AND** `df_path` SHALL 规范化为相对 `valid/` 的 POSIX 路径
- **AND** `is_selected`、形态结果、PnL、阈值配置和绝对目录 SHALL NOT 进入 `window_id` 哈希
- **AND** `window_id` SHALL 唯一

#### Scenario: Expanded table has one row per pattern combination

- **WHEN** 系统写出 `agent_pattern_expanded_table.csv`
- **THEN** 形态列 SHALL 命名为 `kline_pattern` 和 `strategy_pattern`
- **AND** `(window_id, kline_pattern, strategy_pattern)` SHALL 唯一
- **AND** 展开行 SHALL 继承 `is_selected` 和全部追溯字段

### Requirement: Pattern summaries SHALL preserve Initial-action scenario semantics

系统 SHALL 先计算 Initial-action 情景级指标，再按形态组对实际命中该形态的情景等权平均为 Agent triple 指标，并显式报告情景覆盖率。

#### Scenario: Summary files and statistics are fixed

- **WHEN** 聚合脚本运行
- **THEN** 它 SHALL 写出 `agent_pattern_{kline,strategy,cross}_{scenario,triple}_summary.csv` 六个文件
- **AND** 情景级文件 SHALL 包含 `total_net_pnl, window_count, pnl_p25, pnl_p50, pnl_p75`
- **AND** triple 级文件 SHALL 对至少有一个窗口命中该形态的 Initial-action 情景等权平均，并输出 `mean_initial_action_*` 字段
- **AND** triple 级文件 SHALL 输出 `observed_initial_action_count`, `expected_initial_action_count`, `initial_action_coverage_ratio`
- **AND** 未命中该形态的情景 SHALL NOT 被伪造为零 PnL 情景

#### Scenario: Missing expected Initial-action Detail trajectory fails aggregation

- **GIVEN** 某 epoch sidecar 的 `position_choices` 和 `leverage_choices` 可按环境公式生成期望 Initial-action 集合
- **AND** 某 `(epoch, label, bin_index, contract, df_path)` 缺少其中任一 Initial-action 的 Detail 行为轨迹
- **WHEN** 系统生成 triple summary
- **THEN** 系统 SHALL fail fast 并报告缺少的情景
- **AND** 期望集合 SHALL NOT 从已观测 Detail rows 推断
- **AND** 系统 SHALL NOT 直接相加不同 Initial-action 情景的盈亏

#### Scenario: Pattern absence is coverage, not a fabricated zero-PnL scenario

- **GIVEN** 所有期望 Initial-action 的 Detail 行为轨迹均存在
- **AND** 某形态只在部分 Initial-action 情景中命中
- **WHEN** 系统生成该形态的 triple summary
- **THEN** `mean_initial_action_*` SHALL 只对命中该形态的情景计算算术平均
- **AND** `observed_initial_action_count` SHALL 等于命中该形态的不同 Initial-action 数
- **AND** `expected_initial_action_count` SHALL 等于 sidecar 推导的完整 Initial-action 数
- **AND** `initial_action_coverage_ratio` SHALL 等于两者之比

#### Scenario: Unclassified sentinels remain diagnostic only

- **WHEN** 明细中存在 `未分类` 或 `策略未分类`
- **THEN** 它们 SHALL 保留在窗口表和展开表
- **AND** 它们 SHALL NOT 进入任何正式 kline、strategy 或 cross summary
- **AND** 六个 summary 文件 SHALL 只包含正式的 7 类 K 线形态和 6 类策略形态
- **AND** 标定报告 SHALL 单独输出未分类窗口数、比率和 PnL 分布

### Requirement: Threshold calibration SHALL be diagnostic and reproducible

系统 SHALL 将阈值集、动作空间和输入身份写入分析 manifest，并将真实数据类别分布作为诊断而非验收目标。

#### Scenario: Analysis manifest makes a run reproducible

- **WHEN** 形态分析完成
- **THEN** `analysis_manifest.json` SHALL 记录阈值配置和窗口配置
- **AND** 每个实际读取的 Detail CSV、epoch sidecar 和 `selection_manifest.json` SHALL 记录逻辑相对路径、字节数和 SHA-256
- **AND** 每个生成的 CSV/JSON 输出（`analysis_manifest.json` 自身除外） SHALL 记录逻辑相对路径、字节数和 SHA-256
- **AND** Detail/sidecar 路径 SHALL 相对 `model_root`，输出路径 SHALL 相对 `output_dir`，selection manifest 路径 SHALL 相对其父目录
- **AND** manifest SHALL 记录扫描到但缺少 Detail 的 checkpoint epoch 清单
- **AND** 同一输入与配置 SHALL 产生同一窗口身份和分类结果

#### Scenario: Calibration warnings do not become quota targets

- **WHEN** 未分类率至少 30%、某类零命中或盈亏区分度弱
- **THEN** 系统 SHALL 产生诊断告警而非失败
- **AND** 盈亏区分度 SHALL 定义为同一视图各形态 `pnl_p50` 的最大值减最小值
- **AND** 系统 SHALL NOT 为满足固定类别占比而自动调整阈值
