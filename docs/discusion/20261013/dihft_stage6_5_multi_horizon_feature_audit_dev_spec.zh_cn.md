# FineFT / DiHFT：10min / 30min 多预测期限特征审计开发规范

> 版本：v1.0.0 开发基线 · 2026-10-13。状态：需求与研究协议已确认并冻结，功能尚未实施。
> 确认依据：用户三轮接受全部推荐方案，Q1—Q16 已闭合；第 13 节为正式研究协议。
> 本稿替代原 v0.1.0 三期限草案。旧稿中 Today/Next 专属配置和可执行净盈利 P0 要求不再适用。
> 访谈证据：[设计访谈记录](./multi_horizon_feature_audit_design_interview.zh_cn.md)。架构决定：[ADR-0053](../../adr/0053-multi-horizon-audit-time-semantics.md)、[ADR-0054](../../adr/0054-independent-audit-without-selection-feedback.md)。

## 1. 目标、范围与独立边界

新增可独立运行的多预测期限特征审计模块：对原生 10min 和 30min 特征数据，在 1h、2h、4h、Today、NextTD、Next5TD、Next15TD 七类期限上评价终点方向收益及终点绝对收益关联。范围是全部审计合格候选，并关联现有三流入选状态，发现原短期限门禁可能遗漏的长期关联。

“Stage 6.5”表示读取 Stage 6 候选数据的逻辑位置；其运行结果不是 Stage 7 的前置条件。架构如下：

~~~text
Stage 1—5 → Stage 6 → 原 Stage 7 → 原 Stage 8 → 原训练与回测
                 │
                 └─只读→ 独立 feature_audit → 独立审计产物
                              ↑
                  日历、参考价、特征因果契约
                  原选择结果只作可选对照
~~~

必须保持原三流选择阈值、黑名单、特征顺序、输出文件、缩放参数、checkpoint 和训练输入不变。不得自动回写、追加审计标签到训练表、导入会创建原输出目录的选择器编排，或因审计失败阻塞原流水线。运行入口和数据准备入口分离。

首期不训练 RL/VAE/监督预测器，不计算可执行净盈亏、胜率或概率校准，不评价路径实现波动率、最大有利/不利变化或特征利润贡献率。旧指标可用于对照，但 IC 不等于利润，绝对终点收益不等于期间波动。

## 2. 已核对的项目事实及输入缺口

| 事实 | 对本模块的约束 |
|---|---|
| 原收益使用 mark_price.shift(-window)，默认决策窗口为 6 | 固定行偏移不能直接视作有效交易分钟 |
| 10min/30min 使用相同整数回看窗口 2、6、12、16、24、48、96、192 | 同名特征的物理回看跨度不同；本期保留原生窗口 |
| 本机 10min Stage 6 数据存在，预期 30min Stage 6 目录不存在 | 30min 的真实集成验收需要独立准备输入 |
| 抽查 Stage 6 数据含 contract、无时区 timestamp、多种价格及重复列；无显式 TradingDay、feature_available_at | 必须建立有证据的输入适配契约，不能伪造字段 |
| 原始快照含 InstrumentID、TradingDay、ActionDay、UpdateTime、LastPrice、BidPrice1/AskPrice1 | 可以核验身份、交易日与细粒度价格；完整覆盖尚未确认 |
| UpdateTime 自带小数秒；文件名日期可能不同于 TradingDay | 解析 ActionDay+UpdateTime，核验数据字段，不能依赖文件名 |
| 旧 Bar 存在休市期间标记，部分价格时间字段与当前源码语义未对齐 | 核验产物版本、实际观察范围和可用时刻后才纳入审计 |
| 尚未找到版本化交易所历史日历；现有 Session 表是静态配置 | 日历必须显式补齐并校验，不能用观测日期或自然日推算 |
| 30min 全流程无 Stage 6 截止开关，随后无条件选择和缩放 | 不得把调用现有全流程视作隔离的数据准备 |
| 旧相关实现有双重 argsort 与非有限值置零 | 新模块独立实现标准 Spearman，不修改旧算法 |

核查路径见访谈记录。原历史报告的 566 个候选、165 个并集以及三流 15/12/148 等数值是指定版本事实，不是新审计常量；另有 13 维 Vol 清单快照，必须披露实际来源、顺序、指纹及差异。

当前数据切分元数据为 Train [2023-01-03, 2024-07-23)、Valid [2024-07-23, 2025-06-30)、Test [2025-06-30, 2026-02-06)。这仅是当前快照；实际运行读取版本化边界，不能将上述日期写死为通用常量。日期边界应转换成 TradingDay 归属的 Session 边界，不能以自然日午夜代替夜盘归属。

## 3. 时间语义与七类期限（已确认）

设 t 为审计决策时点，D 为该时点所属 TradingDay。所有期限从 t 开始，时区显式为 Asia/Shanghai。

| horizon_id | 显示名称 | 目标终点 |
|---|---|---|
| 1h | 1 小时 | 从 t 累计 60 分钟有效交易时间 |
| 2h | 2 小时 | 从 t 累计 120 分钟有效交易时间 |
| 4h | 4 小时 | 从 t 累计 240 分钟有效交易时间 |
| today | Today | D 日盘收盘 |
| next_td | Next TradingDay | D+1 日盘收盘 |
| next_5td | Next 5 TradingDays | D+5 日盘收盘 |
| next_15td | Next 15 TradingDays | D+15 日盘收盘 |

有效交易时间由品种与日期对应的 Session 累计，跳过午休、周末、节假日及休市，允许跨 Session、跨 TradingDay。行情缺失不暂停交易时钟。日级偏移使用交易所日历；D+5/D+15 不包含今天计数，不能改为 D+4/D+14。

目标端点与取价时点分别记录。Session 结束恰好累计足够时长时，终点为该 Session 结束，不多跳一个 Session。不能以固定 Bar 数、下一条有数据的日期、24/48 小时或 date()+n 替代以上定义。

如记录旧收益对照，单独命名 legacy_6bar，记录原算法版本；它不属于七期限主矩阵，不能称为新的 1h。两个频率使用相同预测期限，但保留各自特征历史窗口。

## 4. 标签、参考价与合约身份

### 4.1 已确认标签

对同一真实 contract_id，计算：

~~~text
return(t,h)     = reference_price(label_end) / reference_price(t) - 1
abs_return(t,h) = abs(return(t,h))
~~~

主力身份切换不更换标签观察合约。不得拼接其他月份合约、连续主力序列或用换月价差填补收益。同一合约跨输入文件分区时可保持身份连接，但仍受第 6 节边界限制。

标签必须分别记录目标时点、实际价格观察时点、数据可用时间、价格来源及无效原因。标签成熟至少需要终点价格及其数据已可用；即使使用终点之前的 as-of 价格，也不能将成熟时间提前到观察终点之前。

### 4.2 已确认的有效性边界

- 到期、无终点行情、数据尾部不足、无法核验 TradingDay：输出明确原因与覆盖率。
- 不把无效收益填零；不事后缩短期限；不借其他合约续接。
- 目标端点按真实日历确定，再从原始或经核验的细粒度行情取价；不能用下一根粗频 Bar 延长期限。
- 因果可用性不明的特征阻断其统计，不能默认其在行时间戳已经可用。
- 缺少必需日历、身份契约或全局 split 契约属于运行阻断，不是普通行情缺失。

参考价字段、新鲜度约定和休市时点映射按第 13 节 Q11/Q12 的冻结协议执行。

## 5. 候选资格与频率比较（已确认）

从 Stage 6 构建候选目录，不能把所有数值列直接当特征，也不能只读取原三流入选列表。

1. 排除身份字段、未来标签、奖励执行字段、重复合并列和非数值字段；保存排除原因。
2. 通过生成规则、依赖数据时间及版本核验因果可用性。没有足够证据时标记 FEATURE_AVAILABILITY_UNVERIFIED。
3. 对非有限值按特征及期限建立有效性掩码；常数与不足样本输出状态，不用零 IC 代替。
4. 不套用旧 IC/PSI 门禁、方向性偏好或流专属黑名单。已证明包含未来信息等因果性限制仍必须执行，不能因“不套旧黑名单”放行。
5. 强制上下文列可单列审计，但审计不取消其原训练契约。
6. 保存 is_selected、selected_streams 和可溯源的旧剔除原因；缺少证据时记 unknown，不能仅凭未入选推断被哪个门禁淘汰。

10min/30min 分别使用原生特征。特征目录记录窗口根数、时间口径和真实回看范围，名义分钟数只是辅助字段。首期不重建等物理窗口特征。

每个期限输出全部有效样本结果。期限之间比较另输出对应期限的共同有效样本结果，不强制所有期限都缩减到 15TD 的有效样本交集。

跨频率比较按同一合约、真实决策时点、评价日期对齐，并核验特征族及变换定义；不使用最近邻时间匹配或插值制造同一时点。对不上时披露覆盖率或无可比样本，不强行给出“哪个频率更好”。共同有效样本上的差异也不能单独归因于频率，因为历史回看跨度不同。

## 6. 开发评价与最终 Test 隔离（已确认）

首期仅使用当前 Train/Valid 开发范围。Test 的行情、标签不得用于补齐开发期 15TD 标签、拟合分箱、确定阈值、选期限或优化统计规则。协议冻结后的最终确认评价属于另外一次显式运行。

所有合约、频率使用同一业务时间边界。以真实 label_end / label_available_at 控制净化：

~~~text
拟合或历史规则开发样本：label_available_at < 评价窗口起点
评价窗口样本：decision_timestamp 位于该窗口内
独立窗口报告：label_available_at < 该窗口结束
所有开发标签：label_available_at < 最终 Test 起点
~~~

边界比较使用对应 TradingDay 的真实 Session 时刻；夜盘开始可能在前一自然日期。对未来行情缺失或是否最终入选等信息不能进行事前特征筛选。

P0 无预测模型；时间前推报告评价的是关联的跨期稳定性，不能称为已训练模型的预测成绩。原选择器已使用整个 Train 以及部分 Valid 分布信息；基于其入选清单的回顾性分组不能宣称为完全未见的外样本选特征实验。

Valid 特征可以因果回看 Train 历史；禁止的是训练标签跨入评价未来以及使用未允许的数据拟合。分箱、排序方向和任何解释规则若从数据估计，必须记录 fit_scope；初始回看期及后续窗口安排见第 13 节 Q13。

## 7. 统计与证据表达

### 7.1 数学与数值契约

分别计算 feature 与 return、abs_return 的 Spearman 相关，按每个分组实际成对有效样本重新计算秩，使用并列值平均秩。不得将旧双重 argsort 结果直接冒充标准 Spearman，也不能先全样本排名后在不同掩码里直接当标准相关。

分组至少包括 frequency × contract × feature × horizon × target_kind × evaluation_window。常数特征、常数目标、少于 3 对有效观察或不可计算分母均输出 null 及原因；3 对只是计算下限，不是统计证据充分标准。

原生样本与匹配样本分别统计，不混用各自的分母。跨期限、跨频率差异应在匹配样本上计算，并共享重采样，以保留配对关系。

### 7.2 冻结的推断与展示协议

按 Q13/Q14 执行合约等权汇总、时间窗口稳定性、共同 TradingDay 连续区块 bootstrap 的 95% 区间、样本充足性标识，以及完整候选矩阵。区块重采样时同日的所有合约、两个频率及比较期限一起抽取，避免伪造独立样本。

不要把每根 10min Bar 或同一天多个合约当独立试验。区块长度必须覆盖标签重叠的日跨度，并考虑更长依赖；记录长度确定方法与敏感性结果。以日为抽样索引并不意味着把日均特征与日均标签的相关替换成原 Bar 级统计。

所有尝试必须登记。未经多重比较控制的逐项区间仅是探索性不确定性，不能从数千项中挑出“通过”的特征。新审计不产生自动选择名单，也不把相关系数符号直接当交易方向。

## 8. 输入与输出契约

### 8.1 输入

| 输入 | 必须具备的证据 | 缺失处理 |
|---|---|---|
| Stage 6 数据及候选目录 | 真实合约、源行、生成版本、列语义 | 契约失效则阻断 |
| 特征可用性契约 | 观察范围、依赖时间、时区、可用时间推导依据 | 受影响特征/样本排除并报告 |
| 版本化日历与 Session | 品种、交易日、真实开收盘、历史例外 | 缺失或冲突则阻断 |
| 独立细粒度参考价源 | 同合约、事件时间、可用时间依据、价格来源 | 局部缺失使对应标签无效 |
| Split Manifest | 统一业务日期边界及转换后的时间边界 | 缺失则阻断 |
| 原三流清单与 Manifest | 文件指纹、顺序、来源频率 | 可缺省，对照状态 unknown |
| 配置与运行版本 | git/config/data/calendar fingerprints | 缺失不能声称可复现 |

原始价格只按需要投影和读取，限制在开发允许范围内。审计缓存也放在独立输出或指定临时目录，不能修改源表。

### 8.2 行级标签

核心字段：run_id、schema_version、frequency、contract_id、source_row_id、source_bar_label、event_timestamp、feature_available_at、decision_timestamp、trading_day、horizon_id、label_end、label_available_at、reference_price_rule、start_price、end_price、start_price_observed_at、end_price_observed_at、quote_age_seconds、return、abs_return、split、window_id、label_status、invalid_reason。

特征可用时间存在列间差异时，保存特征或特征组可用性表并在统计阶段施加掩码；不能仅靠行级单一时间给所有列放行。重复键必须有可审计的消解规则，不能无条件 drop_duplicates。

### 8.3 独立输出

~~~text
analysis_result/DiHFT/feature_horizon_audit/<symbol>/<frequency>/<run_id>/
    audit_manifest.json
    input_contract_report.json
    candidate_catalog.parquet
    horizon_labels.parquet
    feature_horizon_metrics.csv
    metrics_by_contract.csv
    metrics_by_window.csv
    metrics_by_time_to_close.csv
    matched_horizon_comparisons.csv
    invalid_samples_by_reason.csv
    legacy_selection_comparison.csv
    feature_horizon_audit_report.md

analysis_result/DiHFT/feature_horizon_audit/<symbol>/comparisons/<comparison_id>/
    comparison_manifest.json
    matched_frequency_comparisons.csv
    frequency_comparison_report.md
~~~

统一对照 Manifest 关联两个频率的 run_id 和完整协议，频率或规则不一致时拒绝直接比较。输出目录不得覆盖历史运行；先暂存，完成契约检查后发布完成标记。失败时保留诊断且不伪装完成报告。

指标必须带：样本口径、原始分母、有效行数、TradingDay 数、合约数、缺失率、有效期限分布、正负相关、null 原因、窗口、价格规则、统计版本和比较伙伴。样本不足与数学上不可定义分开表示。

Manifest 还需记录所有排除及规则版本、旧入选来源、历史产物与当前代码差异、读取范围、封存 Test 边界、随机种子、计算设备、资源峰值及输入输出指纹。

## 9. 模块结构与运行契约

建议新建 data_preprocess/operator_futures/feature_audit，包含 config、contracts、input_adapter、calendar_adapter、reference_prices、horizons、labels、eligibility、split_guard、statistics、comparisons、report、cli。

采用可测试的纯时间/数值核心与读写编排分离。身份、日历和时间契约出错应在计算前失败；局部缺价及受影响特征排除作为明确业务结果汇总。具体目录名可以按编码标准调整，独立边界不可改变。

仅复用经验证的纯函数或只读接口；不得调用初始化时即创建原选择结果目录的 PipelineIOManager。新算法必须单独版本化；GPU 可作为优化，但不能以牺牲并列秩、掩码正确性或确定性来复用旧实现。

建议入口（尚未实现）：

~~~bash
conda activate finetf
python -m data_preprocess.operator_futures.feature_audit.cli validate-inputs --config <audit-config>
python -m data_preprocess.operator_futures.feature_audit.cli run --config <audit-config>
python -m data_preprocess.operator_futures.feature_audit.cli compare --manifest <comparison-config>
~~~

run 不隐式生成特征，不调用 Stage 7，不训练模型；compare 只接受协议兼容且已完成的运行。输入准备是另一个显式任务，不能把不存在的 30min 数据当成空的成功报告。

## 10. 配置契约

以下为冻结的配置契约与默认值，不代表对应配置文件或工具已经实现。输入 Manifest 路径由具体运行绑定。研究参数变更必须生成新 protocol_version 和配置指纹，不能覆盖已完成的历史运行。

~~~yaml
schema_version: multi_horizon_audit.v1
protocol_version: multi_horizon_audit.research.v1
symbol: fu
frequency: 10min  # 另一个运行使用 30min
horizons: [1h, 2h, 4h, today, next_td, next_5td, next_15td]
hour_clock: active_trading_minutes
trading_day_offsets: {today: 0, next_td: 1, next_5td: 5, next_15td: 15}
contract_policy: same_contract
candidate_policy: causal_quality_only
targets: [return, abs_return]
calendar_manifest: <required>
feature_availability_manifest: <required>
price_source_manifest: <required>
split_manifest: <required>
read_final_test: false
purge_by_label_available_at: true
legacy_selection_manifest: <optional>
write_selection_outputs: false
result_root: analysis_result/DiHFT/feature_horizon_audit
reference_price_protocol:
  field: LastPrice
  observation: latest_valid_snapshot_at_or_before_target
  same_contract: true
  same_session: true
  max_quote_age_seconds: 60
  fallback: none
decision_time_protocol:
  require_verified_availability: true
  lower_bound: max_verified_availability_and_source_bar_label
  outside_session: next_session_open
  duplicate_policy: latest_available_source_snapshot
  ambiguous_duplicate: reject_key
  zero_length_today: invalid
evaluation_protocol:
  initial_history_calendar_months: 6
  evaluation_calendar_months: 3
  align_boundaries: next_trading_day_session_start
  split_at_train_valid_boundary: true
  retain_partial_final_window: true
  aggregate: equal_contract_weight
  sensitivity_aggregate: sample_weighted
evidence_protocol:
  method: joint_trading_day_moving_block_bootstrap
  confidence_level: 0.95
  replicates: 2000
  seed: 42
  minimum_block_trading_days: 20
  respect_max_label_trading_day_span: true
  sensitivity_block_multiplier: 2
  minimum_effective_blocks: 20
  minimum_contracts_for_sign_consistency: 3
  multiple_comparison_adjustment: none_exploratory_only
  automatic_pass_list: false
report_protocol:
  scope: all_candidates
  matched_horizon_comparisons: true
  matched_frequency_comparisons: true
  default_order: feature_family_then_name
  cross_horizon_total_score: false
  automatic_top_n: false
acceptance_profile:
  symbol: fu
  required_real_data_frequencies: [10min, 30min]
  prepare_missing_inputs: separate_isolated_entry
  run_legacy_stage7_or_stage8: false
  modify_shared_summary: false
~~~

不接受任意字符串静默落回默认期限，不允许配置输入覆盖输出或指向原选择器/训练产物路径。配置含未知字段、负时长、缺少必需协议、日历范围不足应给出清晰错误。

## 11. 验收与测试要求

| 类别 | 必须验证 |
|---|---|
| 期限 | 七期限在两个频率语义一致；Today/D+1/D+5/D+15 无偏移一日错误 |
| Session | 午休、夜盘跨自然日期、周末、节假日、临时变化、恰好收盘 |
| 缺失 | 少 Bar 不改变目标时点；终点缺价、到期和数据尾部不足有原因 |
| 因果 | 不可用特征不能参与；部分 Bar 与休市标签不倒填到更早决策 |
| 价格 | 同合约 as-of，无未来报价；过旧、非法、乱序、重复价格按协议处理 |
| 切分 | 长标签成熟跨边界剔除；更改封存 Test 不改变开发产物 |
| 统计 | 并列秩与标准 Spearman 参考值一致；不同掩码重新排名；常数/缺失输出 null |
| 比较 | 原生/共同样本分母分开；跨频率配对；没有共同样本时不能生成虚假差异 |
| 不确定性 | 区块同时保留同日多合约；相同种子重现；短序列标证据不足 |
| 隔离 | 审计运行前后原选择文件、配置、训练输入指纹不变；原入口无需审计即可运行 |
| 故障 | 必需契约错误阻断；局部缺失可追踪；无静默跳过、无伪完成报告 |
| 真实集成 | 使用 fu 两个频率真实开发样本；覆盖全部七期限、匹配比较和长标签无效案例 |

测试应验证边界与业务不变量，不用大量镜像实现的测试凑覆盖率。代码实施时遵循仓库编码标准；Python 脚本在 conda activate finetf 后执行。

性能采用投影、按合约/特征块处理、参考价索引复用和批量统计；不将全量原始快照同时载入内存。记录参考机器、数据规模、耗时、峰值内存/显存及重采样开销，基准测量前不虚构运行分钟数或吞吐承诺。

## 12. 实施顺序与完成条件

1. 验证并准备独立输入：日历、特征可用性、参考价与 split 契约；明确旧物化产物版本差异。
2. 实现纯时间目标与同合约参考价解析，先通过小型确定性边界样例。
3. 实现候选目录、标签、无效原因和净化，确保不接触 Test。
4. 实现标准 Spearman、分组统计和共同样本比较，再实现不确定性协议。
5. 关联旧清单、生成独立产物和报告；运行原产物指纹检查。
6. 执行两个频率真实集成与资源基准，填写验收表。

本轮交付的是开发规范、术语及 ADR。代码实现尚未启动。实施完成必须有真实运行证据，不能以 CLI 存在、合成测试通过或 30min 可配置代替两个频率的真实数据验收。输入证据不足时，应给出具体阻断项与可完成部分，不能伪造补齐。

## 13. 冻结研究协议（Q11—Q16，已确认）

用户于 2026-10-13 回复“六项全按推荐来”确认本节。以下数值是冻结的项目研究约定，不是交易所规则或统计保证；对它们的后续修改必须保留版本和比较记录。

### Q11 参考价与新鲜度

统一使用“目标时点之前或恰好该时点的最新有效原始快照携带的 LastPrice”，要求同合约、同一 Session，快照距目标不超过 60 秒，价格为正且有限。不自动切换 midprice 或插值，不能往未来取价。记录 quote_age_seconds；快照新鲜不证明最后成交也新鲜，因而只称参考收益，不称可执行收益。

同一协议用于起点与终点。Session 开始时若没有符合条件的新快照，样本无效；不能从前一 Session 搬价格。端点 Session 的开闭归属须在适配器中统一；收盘端点允许取该 Session 的最后快照。需按新鲜度区间披露覆盖，60 秒修改必须产生新协议版本。

同一合约同一事件时刻存在冲突报价且无可靠序号时，不可凭读取顺序选择；标记价格源冲突。起点和终点分别记录快照年龄，不能只保留一个年龄值掩盖另一端的缺陷。

### Q12 决策时点及零长度标签

只接受能够从生成规则及数据版本证明可用时刻的特征。以已核验的可用时刻和保守的源 Bar 标记为依据，映射到不早于两者的首个交易时点；休市期间可用的特征顺延到下一开市时点。不能仅因发现最后行情较早就把特征倒填到过去。

多个源 Bar 映射到同一合约、频率、决策时点时，保留在该时刻已可用的最新源快照，记录被合并来源；无法确定唯一顺序则阻断该键。特征列可用时间仍逐列核验，不允许因行有效而放行尚未可用的列。

日盘收盘时 Today 若 label_end <= decision_timestamp，标记 ZERO_LENGTH_HORIZON，不能补零收益。其他期限可继续独立判定。所有被顺延样本披露延迟分布；因果证据无法恢复时明确阻断，不改变原特征文件。

### Q13 时间窗口与统计聚合

使用最早 6 个自然月作为初始历史参考区间，之后每 3 个自然月形成一个按时间前推的评价窗口，窗口统一映射到品种 TradingDay 开始时点；末尾不足 3 个月仍保留并标记。窗口在 Train/Valid 原边界处分开，标签不跨评价窗口取未来；不生成随机切分。

历史参考区间不计入后续评价汇总；分箱、阈值等需要拟合时只用已成熟历史样本并记录 fit_scope。逐窗口报告全候选方向/绝对收益 Spearman、样本数和覆盖；汇总以合约等权为主，另披露样本加权敏感性。时间窗口指标保留全序列，不仅展示最优季度。Today 按剩余有效交易时间分层；小时与日级标签均记录实际经过时间。

窗口起点以最早允许开发 TradingDay 为锚点，按自然月偏移生成，再映射到不早于目标业务日期的首个 TradingDay；日历月末按标准月偏移截至有效日。保存最终窗口表并供两个频率共用。主汇总先对每个合约汇集所有合法评价窗口行并重新计算 Spearman，再对可定义的合约指标等权平均；窗口级结果另列，不能默默替换为各窗口 IC 的简单平均。

### Q14 不确定性、少样本与多重比较

主汇总使用共同 TradingDay 上的连续移动区块 bootstrap，95% 逐项区间，2,000 次重采样，固定并记录种子（默认 42）。各合约、期限、频率在同次重采样中共享日区块，重算原 Bar 级目标统计；跨频率/期限差异使用配对统计。

初始区块长度取 max(20 个 TradingDay, 当前比较涉及标签的最大交易日覆盖数)，并报告两倍区块长度的敏感性；这只是预先声明的保守起点，不能宣称保证独立。用覆盖区间日数除以区块长度报告近似有效区块数，并同时报告有效日期/缺失率；不足 20 个有效区块时不作可信区间或稳定性通过结论，仅给描述性指标和 INSUFFICIENT_DATA。逐季度点估计可以展示，不能因为短窗自动被当成充分证据。

实现时同时披露名义覆盖日数和有有效观察的唯一 TradingDay 数，充足性门禁以后一者除以区块长度向下取整，不能把完全缺失日期计为有效证据；该数也不是真实独立样本量的估计。区块沿完整交易日历滑动，保留缺失掩码，不压缩缺失日期来伪造连续性。95% 区间采用有效重采样统计的 2.5%/97.5% 分位数，记录不可定义的重采样次数；无法得到完整协议要求的有效统计或双倍长度证据不足时，对相应区间输出 null 与原因，不静默更换方法或缩短区块。

合约符号一致性至少有 3 个有效合约才汇总；变异不足的比值指标为 UNDEFINED。P0 不给“显著有效”判定、不输出自动通过名单；逐项区间未作多重比较调整，全部候选与所有尝试保留。CI 跨零、样本不足、价格覆盖偏低分别表述，不能统称特征无效。

### Q15 报告重点与候选展示

交付可检索全量表与 Markdown 总报告，包含“特征 × 七期限”方向及绝对收益热图、原三流入选/未入选标记、三个月窗口稳定性、Today 剩余时长分层、覆盖率与无效原因、匹配样本的期限及频率比较。

候选默认按特征族/名称组织，允许按指定指标排序但明确是开发期探索。没有跨期限单一总分、最佳特征 Top-N 或“短期无效/长期必选”硬分类；保留正负相关，避免仅展示绝对值最好的结果。报告将原生频率差异与匹配样本差异分开解释。

### Q16 首期验收对象与输入准备范围

首期以 fu 为真实验收品种，两种频率均完成真实开发数据集成；接口和日历按品种参数化，不承诺其他品种已验收。支持 30min 不是只提供一个枚举值。

单独准备缺少的 30min Stage 1—6 输入及日历、参考价适配，写入明确的审计准备目录，复用原逻辑的受控只读/纯计算部分；不运行原全流程、不写共享 Summary、不运行 Stage 7/8。审计入口自身不生成数据。必要上游错误单独报告修复需求，不能悄悄改变已有 10min 历史产物。

本轮开发规范及默认研究协议已冻结；该状态不表示功能代码或数据准备已经实施。

## 14. 依据

- [项目架构](../../research/2026-10-11-fineft-project-architecture-v2.zh_cn.md)。
- [fu/10min 特征选择归因](../../research/2026-10-12-fu-10min-feature-selection-manifest-attribution.zh_cn.md)。
- [多尺度验证方法与一手来源](../../research/2026-10-12-multiscale-trading-validation-sources.zh_cn.md)。
- CONTEXT.md、ADR-0041、ADR-0042，以及本次只读代码/数据核查。
- 编码规范：.codex/python_commodity_futures_rl_engineering_standard_v1.0.md。

参考文档中的实施指令、历史成绩和建议阈值均不能替代本次用户确认或真实验收证据。
