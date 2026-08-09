# Tasks — add-agent-pattern-dual-classifier

> Spec 见 [proposal.md](./proposal.md)。架构决策见 [ADR-0006](../../../docs/adr/0006-agent-pattern-dual-classifier.md)。
>
> 状态：**ready-for-agent**（测试 seam、动作空间来源、边界语义和聚合产物已确认；阈值待标定）。

## 0. 上游 Detail 字段补全

- [ ] 修改 commodity Scale Save：从当前正在处理的 split DataFrame 直接 passthrough 原始 `volume` 到最终 train/valid/test Feather；不做二次 join，不把 `volume` 加入模型 `state_features`
- [ ] 验证 `slice_model.py` 产出的 label Feather 保留原始 `volume`，无需改变 label 切分逻辑
- [ ] 修改 `test_agent_index.py` 的 Detail 输出以包含 `contract`；确认现有市场字段输出同步写出原始 `volume`
- [ ] 为每个 Detail CSV 写出 `trading_action_detail_epoch_<N>.manifest.json`，记录 `epoch, max_holding_number, position_choices, leverage_choices, position_levels`，并校验档位与环境构造结果一致
- [ ] 为 Scale Save passthrough 与 Detail row builder 添加测试：原始 `volume` 逐行一致、行数/行序不变、`contract`/`volume` 非空，且 `volume` 不进入 `state_features.npy`
- [ ] 重新生成 30min/fu 数据集与切片，并重跑已有评估 epoch 生成新版 Detail CSV；不重新训练模型
- **验证**：新版 Detail CSV schema 包含 `合约`、`成交量`，抽样值与 split/label Feather 完全一致，sidecar 可重建评估时的完整仓位档位

## 1. K 线形态分类器（纯函数）

- [ ] 新建分类器模块包（建议 `FineFT/analysis/classify_agent/`）
- [ ] 实现 K 线形态分类器纯函数：输入 N=20 步窗口的 `mark_price`/`volume` 序列，输出单一 K 线形态标签
- [ ] 实现 7 类判别（KX1 由 label 决定，不入纯函数；纯函数负责 KM1/KT2/KT1/KT3/KM2/KM3/未分类）
- [ ] 实现命中顺序 `KM1→KT2→KM3→KT1→KT3→KM2→未分类`，确保回调和低量背离优先于普通突破
- [ ] 实现突破时序：第 1~5 步建立基准区间，第 6~10 步检测突破；KT1 要求最终价格从基准边界同向延伸 ≥0.5% 且突破后 ≥80% 观测点保持在突破侧；KT2 要求先形成同向新极值，再反向回撤 ≥0.3% 并进入基准边界 ±0.5%，最后重新同向延伸 ≥0.5%；突破触发点不能计作回踩
- [ ] KT3 使用前后各 10 步的 log-price 线性斜率判定加速：方向一致、后半绝对斜率达到参数化倍数、累计收益达到参数化阈值；上涨/下跌对称
- [ ] KM1 使用参数化 Z-score、双腿最小收益和最小边长识别 V/倒V反转；极值点不得靠近窗口端部，上下方向对称
- [ ] KM2 使用参数化 log-return 波动率、均价外沿/触沿和上下沿最小转换次数识别箱体；忽略中间区域并压缩连续同区状态后计数
- [ ] 实现 KM3/SM3 共用的价格—成交量背离检测纯函数：前 5 步基准、第 6 步后价格突破、突破附近成交量中位数下降、价格移动达标；新高/新低对称且阈值参数化
- [ ] 阈值参数化（fixture 可注入，便于标定后只改 fixture）
- [ ] 统一 `z_price`：使用完整形态识别窗口的均值和 `ddof=0` 标准差；零方差时所有 z-score 规则不命中
- **验证**：`FineFT/tests/analysis/test_kline_pattern_classifier.py` 全部通过（合成输入覆盖 7 类 + 命中顺序 + 未分类 + 阈值边界）

## 2. 策略形态分类器（纯函数）

- [ ] 明细表生成器读取 epoch sidecar 中的完整仓位档位并传给纯策略分类器；可选 CLI 动作空间 override 与 sidecar 不一致时失败
- [ ] 从 sidecar 仓位档位计算 `max_abs_position`、近满仓档位、相邻档位和同向加仓；不得从单条行为轨迹推断动作空间
- [ ] 实现 6 类判别（ST1/ST3/ST2/SM1/SM2/SM3）
- [ ] ST1 仅识别第 6~10 步价格突破当步或下一步发生的空仓→近满仓同向开仓；从开仓当步起至少 10 个观测保持执行后仓位同号，允许减仓但不允许平仓或变号
- [ ] ST2 识别突破→回踩突破位 ±0.5%→回踩当步或下一步同向加仓→再延续事件链；不要求加仓前盈利，允许与 ST3 同时命中
- [ ] ST3 按盈利后同向加仓判定：执行前后仓位同号、绝对仓位增加且加仓前浮动盈亏大于 0；多空对称，排除开仓、减仓和反手
- [ ] SM1 按硬边界反转判定：普通窗口在 `|z_price|≥2.0` 时逆向开仓/加仓，涨跌停事件窗口按 label 极端方向逆向开仓/加仓；动作后近满仓，排除仅持有、顺势、减仓、平仓和反手
- [ ] SM2 按离散网格调仓判定：时间顺序上存在一对方向相反的高位向空头和低位向多头相邻档位移动；不要求最终回到初始档位，排除跨多档跳仓和直接反手
- [ ] SM3 按背离过滤判定：背离段内禁止顺突破方向开仓/加仓，允许空仓、减平顺势仓位或逆向持仓；非背离时段须存在有效调仓以排除全程不交易
- [ ] 多选语义：各类独立判定，命中几个算几个；同一突破事件不得同时命中 ST1 与 SM3，只有不同合格事件才允许二者同窗出现
- [ ] 实现策略规则最小样本约束：普通 N=20 窗口运行全部规则；ST1/ST2/SM3 最少 20 步，ST3 与涨跌停语义的 SM1 最少 1 条执行记录，SM2 最少 2 条执行记录且价格方差非零；样本不足时不命中，全部规则未命中时返回“策略未分类”诊断值
- [ ] 阈值参数化
- [ ] 用至少两组 sidecar `position_levels` 参数化测试，验证所有仓位相关判定随权威档位集合正确变化
- **验证**：`FineFT/tests/analysis/test_strategy_pattern_classifier.py` 全部通过（合成输入覆盖 6 类 + 多选组合 ST2+ST3 + ST1/SM3 同事件冲突 + 未分类 + 阈值边界）

## 3. 明细表生成脚本（薄 orchestrator）

- [ ] 为 orchestrator 提供 `--model_root`、`--selection_manifest`、`--output_dir` 三个独立路径参数；`--model_root` 递归扫描 `epoch_*/trading_action_detail_epoch_*.csv`，`--selection_manifest` 只负责已选标记，所有输出写入 `--output_dir`
- [ ] 输入 schema 强制要求原始 `volume` 与 `contract`；遇到旧版 Detail 缺列时失败并提示先运行上游重新生成流程，不得用 `log_volume_origin` 替代
- [ ] 支持现有中文 Detail 表头和对应英文机器名，读取后规范化为英文内部 schema；同语义双列值冲突时失败
- [ ] 行为轨迹分组后按 `timestep` 升序排序，并校验每组是从 0 开始、唯一、连续的整数序列；负值、重复或缺口时失败，全局 CSV 行序不影响结果
- [ ] 生成 `agent_pattern_coverage_report.csv`：候选全集取实际存在的 Detail CSV；缺少未选 epoch 只告警，manifest 已选 agent 缺少对应 Detail rows 时失败
- [ ] 校验 Detail 文件 epoch：文件名、父目录和解析结果必须一致；重复或冲突文件必须失败
- [ ] 实现明细表生成：扫描 `--model_root` 下全部 Detail CSV，按所有 `(label, epoch, bin_index)` 候选 triple 遍历 → label_0/6 以整条短轨迹建立涨跌停事件窗口并标 KX1 / label_1~5 切 N=20 不重叠窗口 → 调两个分类器 → 产出明细表行；不得用 `selection_manifest.json` 过滤候选
- [ ] 读取 `selection_manifest.json`，按 `(label, epoch, bin_index)` 精确匹配并写入布尔列 `is_selected`；展开级文件继承该字段
- [ ] 在匹配前校验 selection manifest：`dataset_name/experiment_name` 与 `model_root` 逻辑归属一致，`label_0..label_6` 各恰好一条，`epoch_path/model_path` 的 epoch 一致，model 逻辑末尾为 `epoch_<N>/trained_model.pkl`，对应 model_root epoch 存在；忽略机器绝对路径前缀，错配/重复/缺失/格式错误时失败
- [ ] label_1~5 只处理完整的 N=20 窗口；尾部不足 20 步不产出明细，但覆盖率报告按轨迹记录 `dropped_tail_steps/gross_pnl/net_pnl`
- [ ] label_0/6 整条轨迹始终只产生一个 KX1 事件窗口；长度达标的策略规则在整条轨迹扫描
- [ ] 涨跌停事件窗口即使没有策略规则命中，也必须输出 `kline_patterns=["KX1"]`、`strategy_patterns=["策略未分类"]` 的窗口明细行
- [ ] 盈亏归因：`gross_pnl = sum(realized_pnl_step[start:end+1]) + unrealized_pnl[end] - unrealized_pnl_before_start`，`net_pnl = gross_pnl - sum(commission_fee_step[start:end+1])`；非首窗口浮盈基线取同一行为轨迹前一行，首窗口取初始值 0；不得重复扣除已反映在实际成交价值中的滑点
- [ ] 每个窗口恰好产出一行：K 线形态单选结果包装为单元素数组，策略形态多选结果保存为数组，一组 `gross_pnl`/`net_pnl` 只保存一次
- [ ] 输出窗口级 `agent_pattern_window_table.csv`，必含 `label, epoch, bin_index, is_selected, contract, df_path, initial_action, window_index, start_timestep, end_timestep, window_id, kline_patterns, strategy_patterns, gross_pnl, net_pnl`
- [ ] `window_id` 固定由 `(label, epoch, bin_index, contract, df_path, initial_action, window_index, start_timestep, end_timestep)` 的规范 JSON 计算 SHA-256；`df_path` 使用相对 `valid/` 的规范 POSIX 路径，不将 selection、形态、PnL、阈值或绝对目录纳入哈希；形态列使用合法 JSON 数组编码
- [ ] 输出 `agent_pattern_expanded_table.csv`，标量列为 `kline_pattern, strategy_pattern`，唯一键为 `(window_id, kline_pattern, strategy_pattern)`
- [ ] 输出 `analysis_manifest.json`，记录阈值/窗口配置、缺少 Detail 的 checkpoint epoch，并为所有实际读取的 Detail CSV、epoch sidecar、selection manifest 及全部生成的 CSV/JSON 输出（manifest 自身除外）记录逻辑相对路径、字节数和 SHA-256；Detail/sidecar 相对 `model_root`，输出相对 `output_dir`，selection manifest 相对其父目录
- **验证**：一个 smoke test 跑通端到端（多个 epoch 的小样本 Detail CSV → 全部候选均被保留、只有 manifest 精确匹配项 `is_selected=true`、表行数符合预期、两列 PnL 非空且净值关系成立），并覆盖乱序输入结果不变、`timestep` 负值/重复/缺口失败、selection manifest 逻辑错配/重复/缺失失败、未选 epoch 缺失告警、已选 epoch 缺失失败及 epoch 标识冲突失败

## 4. 聚合视图脚本

- [ ] K 线、策略和 7×6 视图默认使用 `net_pnl`，按情景生成 `total_net_pnl, window_count, pnl_p25, pnl_p50, pnl_p75`；`gross_pnl` 仅作诊断口径
- [ ] triple 级视图由情景级结果按 `(label, epoch, bin, 形态维度)` 只对实际命中该形态的 initial-action 情景做算术平均，输出 `mean_initial_action_*`、`observed_initial_action_count`、`expected_initial_action_count`、`initial_action_coverage_ratio`；不得伪造零 PnL 情景或直接相加不同 initial-action 的盈亏
- [ ] 每个 epoch 的期望 initial-action 集合由 sidecar 的 `position_choices` 和 `leverage_choices` 按环境动作数公式生成；不从已观测 rows 推断，某 `(epoch, label, bin_index, contract, df_path)` 的任一期望 Initial-action Detail 行为轨迹缺失时失败
- [ ] 不跨策略形态或 initial-action 情景求和得到所谓账户总盈亏；账户总盈亏只能在单个 initial-action 情景内直接按唯一 `window_id` 汇总原始明细行
- [ ] 固定输出 `agent_pattern_{kline,strategy,cross}_{scenario,triple}_summary.csv`；未分类哨兵值不进入任何正式 kline、strategy 或 cross summary，标定报告单独输出其窗口数、比率和 PnL 分布
- **验证**：一个 smoke test 验证每窗口一行、数组展开、不同策略形态和 initial-action 均不放大账户盈亏，并覆盖命中情景等权平均、覆盖率计算、未命中情景不伪造零 PnL，以及期望 Initial-action Detail 行为轨迹缺失报错

## 5. 阈值标定（goal-driven 循环）

- [ ] 用提议阈值跑真实 Detail CSV（如 `trading_action_detail_epoch_58.csv`，n≈366k 行）
- [ ] 输出未分类窗口数/比率/PnL 分布、7 类 K 线形态分布、6 类策略形态分布和盈亏区分度；区分度定义为视图内正式形态 `pnl_p50` 的 max-min；未分类率 ≥30%、某类零命中或区分度弱只告警
- [ ] 看每个判别特征在真实窗口上的分布（直方图）
- [ ] 经人工确认后调阈值 → 重跑并比较诊断报告；不得以满足固定类别占比为调参目标
- [ ] 锁定阈值 → 更新 fixture → 回归测试覆盖
- **硬性验证**：schema、唯一键、必需字段无非法空值、候选与 initial-action 情景覆盖、窗口盈亏守恒、结果确定性及合成边界测试全部通过；类别占比和盈亏区分度只作诊断
