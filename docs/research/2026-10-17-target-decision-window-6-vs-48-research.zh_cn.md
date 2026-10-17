# `target_decision_window=6 → 48` 与策略收益下降：证据、解释和下一步实验

研究日期：2026-10-17（Asia/Shanghai）。范围：当前工作区代码、现存实验产物及公开一手资料；没有修改训练代码、运行新训练或更改特征选择结果。

## 结论与证据边界

**TR 从约 0.06 降到约 0.04，直接说明这次完整策略的已报告收益较低；不能单凭这个数字推出“48 步特征 IC 比 6 步差”，更不能推出“长预测期限普遍不可用”。** 必须分别检查相同特征在两个预测期限的样本外预测关联、选择出的三流特征变化，以及执行策略和训练随机性。IC 是特征/预测与标签的关联，TR 是交易执行后的组合结果；Qlib 官方也将 IC/RankIC 分析与含交易成本的回测分析分成不同产物。[Qlib 官方文档](https://qlib.readthedocs.io/en/stable/component/recorder.html#record-template)

本次进一步重算发现：当前 RL 特征在 Valid 上的平均绝对 RankIC 从 6 步约 0.0426 增至 48 步约 0.0688，但按 Train 确定方向后的 Valid 平均 IC 从 +0.0114 变为 −0.0117。**首要排查方向应是长窗口关联在 Train/Valid 之间能否保持方向，而非假定绝对预测强度普遍下降。** 该诊断的口径和限制见下文实算表。

以下使用四种证据标记：**已验证**指已读取当前代码或产物；**用户报告**指本次提问提供但未取得完整对照快照的信息；**假设**指解释机制、尚未证实属于本次实验；**建议**指待实施实验。

- **用户报告**：`target_decision_window=6` 时最终 TR 约 0.06，改为 48 后约 0.04。
- **已验证**：当前 [best_result.csv](/home/lanceliang/opt/aiwork/FineFT_code_space2026/analysis_result/DiHFT/high_level_heurstic/fu/10min_parallel/best_result.csv) 中 trial 358 的 `tr` 与 `portfolio_tr` 均为 `0.041789513888889`，12 个合约、总初始本金 72,000。用户已确认旧实验没有保存，因此无法恢复严格的两次实验因果比较。
- **评价边界**：当前高层评估代码默认 `eval_stage=valid`，配套 Optuna 脚本未覆盖，且搜索预算为 400 trials。按此配置，`best_result` 是验证集择优结果，不能称为独立最终 Test 结果；运行时是否另有覆盖仍以原始命令为准。[评估入口](/home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/RL/DiHFT/high_level/vae_routing_util.py:283)、[Optuna 脚本](/home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/script/test/DiHFT/high_level/vae_optuna_fu_10.sh)
- **数值解释**：按用户给出的四舍五入数值，6% → 4% 是下降 2 个百分点，或相对下降约 33%；它不是 IC 下降 0.02 的证据。

## 参数真正改变了什么

### 1. 这是特征选择标签的行偏移参数

当前 `calculate_future_return` 先按 `timestamp` 排序，再以 `mark_price.shift(-window)` 计算

\[
R_{t,h}=\frac{P_{t+h}-P_t}{P_t}.
\]

因此 6 与 48 指未来 6 行和 48 行。对完整、等间隔的 10min Bar，可分别理解为名义 60 与 480 分钟的 Bar 跨度；交易 Session 休市、缺失 Bar 和边界会改变实际经过时间。不能直接将 48 解释为 8 个自然小时或 NextTD。[源码：metrics.py:53](/home/lanceliang/opt/aiwork/FineFT_code_space2026/data_preprocess/operator_futures/feature_selection/muti_contract/metrics.py:53)

仓库独立的多预测期限特征审计另有按**有效交易时间**与显式 TradingDay 偏移构造标签的协议；该协议没有改变原选择器的行偏移行为。现成 `1h / NextTD / Next5TD` 报告可辅助判断特征用途，但不能把 NextTD 当成 `h=48` 的同义词。[ADR-0053](/home/lanceliang/opt/aiwork/FineFT_code_space2026/docs/adr/0053-multi-horizon-audit-time-semantics.md:5)

### 2. 当前选择器不是“所有 IC 统一切换到 48”

| 代码环节 | 改 6 → 48 的实际影响 | 解释边界 |
| --- | --- | --- |
| 跨合约符号一致性 | `SignConsistency`、`VolSignConsistency` 提取目标窗口对应值 | 可能改变三流门禁通过的特征 |
| CatBoost 非线性评分 | 用目标窗口有向未来收益训练，净化隔离带也跟随窗口 | 可能改变重要性排名及最终保留特征 |
| `RankIC_Mean/Std`、`VolRankIC_Mean/Std` | 当前按 feature 聚合全部合约 × windows 的指标 | 主 IC/IR 门禁仍混合多个期限，不是纯目标窗口 IC |
| RL 学习与高层路由 | 经新特征列表间接影响模型和路由 | 参数本身不代表策略改成每 48 步决策或必须持仓 48 步 |

上述实现分别见 [predictive_audit.py:79](/home/lanceliang/opt/aiwork/FineFT_code_space2026/data_preprocess/operator_futures/feature_selection/muti_contract/predictive_audit.py:79)、[nonlinear_scoring.py:43](/home/lanceliang/opt/aiwork/FineFT_code_space2026/data_preprocess/operator_futures/feature_selection/muti_contract/nonlinear_scoring.py:43)、[metrics.py:255](/home/lanceliang/opt/aiwork/FineFT_code_space2026/data_preprocess/operator_futures/feature_selection/muti_contract/metrics.py:255)、[pipeline.py:273](/home/lanceliang/opt/aiwork/FineFT_code_space2026/data_preprocess/operator_futures/feature_selection/muti_contract/pipeline.py:273)。ADR-0036 的历史设计也明确保留多期限诊断，但当前代码才是本次解释的直接依据。[ADR-0036](/home/lanceliang/opt/aiwork/FineFT_code_space2026/docs/adr/0036-modular-three-stage-funnel-feature-selection-and-statistical-gates.md:24)

另一个已验证的耦合是：当前共享 CatBoost 评分使用**有向未来收益**，波动率 VAE 流也使用这份重要性。因而这次变化可以同时影响斜率 VAE、波动率 VAE、RL 决策流，不能把最终收益变化全部归到低层 RL 特征 IC。[nonlinear_scoring.py:46](/home/lanceliang/opt/aiwork/FineFT_code_space2026/data_preprocess/operator_futures/feature_selection/muti_contract/nonlinear_scoring.py:46)、[三流架构 ADR-0041](/home/lanceliang/opt/aiwork/FineFT_code_space2026/docs/adr/0041-triple-stream-feature-decoupling-architecture-without-backward-compatibility.md:29)

### 3. 首先固定可复现的实验条件

**已验证的当前工作区状态**包括 CatBoost 默认迭代数改动、波动率流容量/VIF 配置改动和预处理脚本跳过数据切分等未提交改动。**这些不是已经证明的两次实验差异**，也不能据此断言是谁导致收益下降；它们说明仅记住一个 CLI 参数不足以建立对照。当前 feature-selection manifest 未记录完整的 `target_decision_window` 与 `catboost_iterations`，需要补充运行日志/命令和配置快照才能追溯。当前脚本的 windows 列表还包括 96、384、768，解读聚合 IC 时尤其应确认真实列表。[当前配置入口](/home/lanceliang/opt/aiwork/FineFT_code_space2026/data_preprocess/operator_futures/feature_selection/muti_contract/pipeline.py:194)、[预处理脚本](/home/lanceliang/opt/aiwork/FineFT_code_space2026/data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh:670)

建议每个实验独立保存：代码版本及未提交差异、输入文件摘要、Train/Valid/Test 日期与合约、全部窗口/门禁/CatBoost 参数、三流特征列表及其顺序、缩放产物、训练 seed 与预算、RL/DP 折扣参数、路由与风控参数、选模指标、`best_result.csv` 和逐合约/逐日交易记录。若输入列表改变，重新生成对应下游数据并训练相关模型，避免相同路径下的旧新产物混用。Qlib 的实验记录接口同样将参数、指标和模型/数据产物分开保存，便于恢复单次运行。[Qlib Recorder](https://qlib.readthedocs.io/en/stable/component/recorder.html)

## 本次实算：绝对关联没有普遍减弱，方向稳定性更值得检查

冻结[当前 manifest](/home/lanceliang/opt/aiwork/FineFT_code_space2026/PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/10min/fu/train/feature_selection_manifest.json) 的三流特征，排除 `mandatory_state_features`，在原始 Train 14 个合约、Valid 12 个合约上作描述性重算。当前 dataset 三流 NPY 与特征选择输出逐项相同；这证明当前输入列表一致，仍不能替代旧实验快照或完整 run 配置绑定。

每个合约统一取前 `N−48` 个决策点，再对每个特征使用两期限共同有限值掩码。方向流标签为未来收益，波动流标签为**终点绝对收益**。用平均秩处理并列值计算 Spearman；每个特征、每个期限的方向只由 Train 跨合约平均 IC 决定，然后固定方向评价 Valid：`signed_valid_IC = sign(mean_train_IC) × valid_IC`。下表对特征 × 合约等权平均。

| 流（去除 mandatory 后数量） | Valid 平均绝对 IC，6 → 48 | Train 定向后 Valid 平均 IC，6 → 48 | Train 定向后 Valid 正 IC 占比，6 → 48 |
| --- | --- | --- | --- |
| RL 决策流（126） | 0.04262 → **0.06880** | 0.01144 → **−0.01167** | 56.94% → 46.43% |
| 斜率 VAE 流（15） | 0.06996 → 0.04667 | 0.04788 → **−0.01162** | 81.67% → 36.67% |
| 波动率 VAE 流（8，VolRankIC） | 0.05759 → **0.07324** | 0.04718 → **−0.00285** | 83.33% → 57.29% |

**支持的初步判断**：对当前已入选集合，48 步绝对关联在 RL 与波动流上反而更高，但按 Train 学到的方向在 Valid 上平均不保持；斜率流同时出现绝对关联和方向保持的下降。因此现在比“长窗口 IC 都变小”更有证据的排查方向，是**预测关联跨 Train/Valid 的方向稳定性**。波动流正 IC 占比仍过半而平均值略负，说明少数较大的负关联也可能影响均值；须继续分特征、分合约看，不能用单一占比代替强度分布。

这些数字不是 policy 预测分数的 IC，也不是已证明的 TR 下降原因。它们条件化于当前特征选择，Valid 已用于调参，未提供区块 Bootstrap CI，亦未重新执行所有 Train/Valid 边界净化。原选择器使用 Torch 双 `argsort` 排秩和缺失值处理，本次平均并列秩 + 共同 finite mask 不追求逐位复现原 CSV。故它们适合生成诊断假设，不足以宣布 48 步策略统计显著失败。[原指标实现](/home/lanceliang/opt/aiwork/FineFT_code_space2026/data_preprocess/operator_futures/feature_selection/muti_contract/metrics.py:175)

当前 `predictive_audit.py` 以合约行数总和近似样本量计算相关显著性，未在这一计算中校正标签重叠与合约相关，因此现存 FDR 通过情况也不能补上上述 CI 缺口。[源码](/home/lanceliang/opt/aiwork/FineFT_code_space2026/data_preprocess/operator_futures/feature_selection/muti_contract/predictive_audit.py:250)

## 为什么长窗口可能差，也可能不差

### 假设 A：原有信号短期有效，后续价格变化稀释了关联

例如订单流、短期价格偏离可能只对较短未来区间有用。这里给一个**说明机制的数学例子，不是本次数据的测量结果**：假设特征 \(X\) 只预测前 6 步累计收益 \(Y_6\)，后续 42 步收益 \(Z\) 与 \(X,Y_6\) 均不相关；采用可加收益，各步收益互不相关且方差相同，使 \(\operatorname{Var}(Y_6):\operatorname{Var}(Z)=6:42\)，则

\[
\operatorname{Corr}(X,Y_6+Z)
=\operatorname{Corr}(X,Y_6)\sqrt{6/48}.
\]

因此一个短期有用的信号可在长目标下表现为较低 Pearson IC；但若特征预测的是持续趋势，后续区间也可提供信号，结论可能相反。该例不适用于直接计算真实 RankIC，也不是“期限越长越差”的一般规律。是否属于本次情况，应在冻结的同一组特征和**共同有效样本**上实际比较。

### 假设 B：选出的特征适合长目标，但当前策略没有有效利用

即使长目标的关联更强，也不保证进出场时机、仓位、手续费、滑点、门控或止损的组合更好。高层识别趋势与低层选择入场时机可能需要不同预测期限；三流架构本就区分了用途。[ADR-0041](/home/lanceliang/opt/aiwork/FineFT_code_space2026/docs/adr/0041-triple-stream-feature-decoupling-architecture-without-backward-compatibility.md:29)

检查实际持仓步数分布、动作持续性、风控退出原因，以及 RL/DP 的折扣权重。折扣回报中远期奖励按 \(\gamma^k\) 加权；改变特征选择的标签期限不会自动改变这个目标。\(1/(1-\gamma)\) 是常用的折扣质量尺度，不是强制持仓长度，更不是“只看这么多步”。不要为了匹配 48 就直接增大 gamma；先排除输入变化、训练方差与执行成本。[OpenAI Spinning Up：Reward and Return](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#reward-and-return)

### 假设 C：长窗口统计更不稳定，或只是一次训练波动

相邻 48 步标签共享大量价格增量，Bar 行数不等于独立证据数量；长窗口也会减少合约尾部可用标签。原始研究表明，重叠收益与预测变量持续性会导致不同 horizon 的估计高度相关，不能把多条期限曲线当成独立重复证据；该研究不是本次期货 IC 必然衰减的证明。[Boudoukh、Richardson、Whitelaw 原论文](https://www.nber.org/papers/w11841)

神经网络/RL 的单次训练结果还会受随机初始化与采样影响。NeurIPS 2021 的评估研究明确指出少量 runs 的点估计可能改变算法优劣结论，因此需要重复训练与区间估计；它没有给出适用于本项目的固定“足够 seed 数”。[Agarwal 等，NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html)

## 下一步按这个顺序做

### 第一步：先定位收益变化发生在哪里

旧实验未保存，因此先保存当前 48 步产物，再在冻结配置上重建 6 步基线；不要把历史文档中相近的 TR 数值当成已匹配的旧运行。新对照须使用相同合约、日期、本金、选模流程和交易成本。先按合约计算配对净收益差，再看回撤、持仓/空仓比例、持仓时长、换手、费用、风控触发、VAE OOD/路由变化。总收益降低可能由少数合约造成，均值也可能掩盖大多数合约方向一致的退化。

对毛利与净利分解，只使用账务口径一致且能闭合的数据；若诊断记录标记 `NEEDS_AUDIT` 或存在负成本残差，先核对记录完整性与定义，不把残差全部称为滑点。当前汇总中的逐合约平均 MDD/Sharpe 也不能当成组合时间序列的 MDD/Sharpe。此处是会计与聚合口径要求，而非对收益下降原因的断言。[交易诊断 ADR-0048](/home/lanceliang/opt/aiwork/FineFT_code_space2026/docs/adr/0048-comprehensive-trading-diagnostics-and-friction-decomposition.md:5)

### 第二步：用固定特征、共同样本回答“是不是 IC 衰减”

建议先比较 `h ∈ {6,12,24,48}`；这是降低实验成本的研究设计建议，不是最优窗口声明。对 6 与 48 原入选特征的并集及符合因果/质量条件的候选池，固定：相同真实合约、相同决策时点、相同评价日期和价格定义。统计方向流的 signed RankIC、方向一致性及波动流对终点绝对收益的 VolRankIC，并报告共同样本数量、保留率、按合约/时间分段表现。负 IC 不等于无预测力；方向应在训练期确定，再检查未来是否保持，不能事后逐段翻号使结果变好。[领域定义](/home/lanceliang/opt/aiwork/FineFT_code_space2026/CONTEXT.md:85)、[独立审计边界 ADR-0054](/home/lanceliang/opt/aiwork/FineFT_code_space2026/docs/adr/0054-independent-audit-without-selection-feedback.md:5)

分别回答两个问题：① 同一特征的 \(IC_6\) 与 \(IC_{48}\) 是否不同；② 用 6/48 选择出的两个输入集合，哪个更支持当前策略。改变特征集合后再比较各自最优 IC，会把期限效果与选择效果混在一起。

边界净化应按标签实际终点排除跨 Train/Valid/Test 的观察；时间序列不能随机打散后验证。scikit-learn 的 `TimeSeriesSplit` 明确区分按时间切分和训练末端 gap，但 generic gap 本身不保证本项目所有交易日/同合约标签边界都已正确净化。[TimeSeriesSplit 官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)

置信区间应保持时间依赖，使用已冻结的区块协议；不能把重叠的逐 Bar 观察当成独立样本重采样。仓库 ADR-0056 明确要求在有效区块不足时报告缺失原因，不能为了得到 CI 降低门槛。需区分训练 seed 不确定性和市场时间样本不确定性，两者不能互相替代。[ADR-0056](/home/lanceliang/opt/aiwork/FineFT_code_space2026/docs/adr/0056-ci-diagnostics-and-three-way-matched-horizon-comparisons.md:13)、[arch 作者官方时间序列 Bootstrap 文档](https://bashtage.github.io/arch/doc/bootstrap/timeseries-bootstraps.html)

### 第三步：保留 6 作基线，小范围消融后才扩大搜索

| 优先级 | 待做实验 | 回答的问题 |
| --- | --- | --- |
| P0 | 保存当前 48 产物与完整配置，冻结其他参数后重建 6 基线 | 建立可比较的新实验 |
| P1 | 固定输入与行情，计算 6/12/24/48 的共同样本 IC 曲线 | 哪些特征在何种期限有效？ |
| P2 | 6 与 48 使用相同 seed 列表、训练预算、Optuna 预算和验证选模规则；预算有限先做 3–5 个配对 seed | 下降是否重复出现，是否超过训练波动？ |
| P3 | 根据 P1 结果选择 12 或 24 中一个候选，与 6 配对验证 | 是否存在更合适的中间期限？ |
| P4 | 仅在长期关联稳定时，设计“RL 仍短期限、斜率/波动流分别验证长期限”的消融 | 长信号是否适合作为体制输入，而非替换全部短期输入？ |

P2 中 3–5 个 seed 只用于初筛，不代表统计充分；报告每个 seed 的配对差、均值/中位数与离散程度，分合约结果也一并保留。整个 6/12/24/48 单因素比较固定 `windows_list`、三流特征容量预算、其他选择器参数、训练预算和 Optuna 搜索预算；实际最终入选数量仍需报告。候选在 Valid 上比较，Test 用于冻结方案后的确认；若旧 Test 已反复用于调参，它应被视为已使用的研究数据，后续需要保留新的未触碰时段。[Agarwal 等](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html)

P4 属于未来工程设计：当前统一窗口参数不会自动提供三流独立期限配置，不能误认为已有现成 CLI。也不要根据独立审计报告直接回写三流特征列表，这会违反 ADR-0054；需通过显式、可追溯的选择器实验实施。[ADR-0054](/home/lanceliang/opt/aiwork/FineFT_code_space2026/docs/adr/0054-independent-audit-without-selection-feedback.md:5)

## 根据结果选择调整方向

| 观察到的证据 | 支持的解释 | 下一步方向 |
| --- | --- | --- |
| 共同样本上大部分 RL 特征的 48 步关联减弱，且 6 的配对净收益稳定占优 | 当前特征/策略更适合短期目标 | 优先保留 6，研究 12/24；长期目标需要不同特征证据 |
| 48 步关联稳定，但换手、费用、提前退出或路由异常恶化 | 信号到交易的转换有问题 | 固定特征，逐项检查持仓时间、门控与成本；每次只改一个环节 |
| 高层特征/路由变化大，低层特征变化小 | 可能主要是体制识别受影响 | 单独做高层/低层消融，不笼统归因于 RL IC |
| 多 seed 差异方向不一致，收益差处于训练波动范围 | 尚无稳定的 6/48 优劣证据 | 增加独立重复或更长样本外观察，避免继续大范围调参 |
| 长期 IC 只在少数合约、训练期或本就入选的少数特征上突出 | 可能是样本/选择效应 | 扩大共同样本和未来时段验证，检查方向稳定性 |

最有信息增益的动作是**重建受控基线 + 深入检查已发现的长窗口方向不稳定**。先定位具体哪些特征、哪些合约和哪些时段翻转，再决定是否保留短期限输入或分流使用长期特征。

## 附录：描述性实算的复现方法

在仓库根目录执行；输入文件发生变化时数字也会变化。此片段已于本次研究中在 `finetf` 环境执行并核对表格数字。它只读取 Train/Valid，不写文件、不训练模型；不替代正式的边界净化与区块推断。

```bash
conda activate finetf
python - <<'PY'
import json
from pathlib import Path
import numpy as np
import polars as pl
from scipy.stats import rankdata

root = Path.cwd() / 'PREPROCESS_DATASET/commodity-futures'
manifest = json.loads((root / 'FEATURE_SELECTION/10min/fu/train/feature_selection_manifest.json').read_text())
mandatory = set(manifest['mandatory_state_features'])
for stream in ['rl_stream', 'vae_slope_stream', 'vae_volatility_stream']:
    features = [f for f in manifest[stream]['selected_features'] if f not in mandatory]
    scores = {}
    for split in ['train', 'valid']:
        rows = []
        for path in sorted((root / 'SPLIT-TRAIN-VALID-TEST/10min/fu' / split).glob('*.feather')):
            df = pl.read_ipc(path).sort('timestamp')
            p = df['mark_price'].to_numpy().astype(float)
            n = len(p) - 48
            if n < 3:
                continue
            with np.errstate(divide='ignore', invalid='ignore'):
                ys = np.column_stack([(p[h:h+n] - p[:n]) / p[:n] for h in [6, 48]])
            if stream == 'vae_volatility_stream':
                ys = np.abs(ys)
            values = np.full((len(features), 2), np.nan)
            for j, f in enumerate(features):
                x = df[f].to_numpy()[:n].astype(float)
                mask = np.isfinite(x) & np.isfinite(ys).all(axis=1)
                if mask.sum() < 3:
                    continue
                rx = rankdata(x[mask], method='average')
                if np.std(rx) == 0:
                    continue
                for k in range(2):
                    ry = rankdata(ys[mask, k], method='average')
                    if np.std(ry) > 0:
                        values[j, k] = np.corrcoef(rx, ry)[0, 1]
            rows.append(values)
        scores[split] = np.stack(rows)
    direction = np.sign(np.nanmean(scores['train'], axis=0))
    oriented = scores['valid'] * direction[None, :, :]
    valid_mask = np.isfinite(oriented)
    print(stream, 'features=', len(features))
    print('valid_abs_IC_6_48=', np.nanmean(np.abs(scores['valid']), axis=(0, 1)))
    print('train_oriented_valid_IC_6_48=', np.nanmean(oriented, axis=(0, 1)))
    print('positive_fraction_6_48=', ((oriented > 0) & valid_mask).sum(axis=(0, 1)) / valid_mask.sum(axis=(0, 1)))
PY
```
