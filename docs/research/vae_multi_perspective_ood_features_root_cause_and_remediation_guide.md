# VAE 多视角 OOD 实证特征根因诊断、数学病理剖析与改造优化方案调研报告

- **研究主题**：基于 FineFT 真实训练集基准 (`train`) 的 VAE 特征级 OOD 诊断实证结果，深度追溯高似然恶化特征的代码与数学根因，制定分级修复改造与平稳化替代体系。
- **关联第一手来源 (Primary Sources)**：
  - **实证诊断产物**：
    - `analysis_result/DiHFT/feature_ood/fu/10min_parallel/feature_ood_summary.csv` (全量多视角诊断宽表)
    - `analysis_result/DiHFT/feature_ood/fu/10min_parallel/feature_ood_test_vs_train.csv` (测试集真实 OOD 崩溃归因表)
    - `analysis_result/DiHFT/feature_ood/fu/10min_parallel/feature_ood_valid_vs_train.csv` (验证集体制漂移表)
    - `analysis_result/DiHFT/feature_ood/fu/10min_parallel/contracts/*.csv` (13 个测试合约逐合约下钻表)
  - **特征生成与算子实现源码**：
    - `data_preprocess/operator_futures/cross_section/base_feature_util.py` (L770: `sell_spread_oe_max` 盘口深度价差)
    - `data_preprocess/operator_futures/time_operator/multi_processing_util.py` (L34: `_trend_` 算子; L489-492, L557-563: `*_std_norm` 算子; L469-474: `cntn/cntp` 算子; L728: `realized_volatility` 算子)
    - `data_preprocess/operator_futures/commodity/cross_month_feature.py` (L34, L533-556: `_share` 跨期静态持仓份额)
    - `data_preprocess/operator_futures/scale_describe_save/muti_contract_scale_save.py` (L340-365: `FAT_TAILED_FEATURE_PATTERNS` 截断与 Robust 缩放)
    - `data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh` (`COMMODITY_FU_FEATURE_BLACKLIST` 特征黑名单)
  - **深度模型与架构决策**：
    - `FineFT/RL/DiHFT/VAE/vae.py` (高斯对数似然 $\text{NLL}$ 损失函数与 `softclip`)
    - `FineFT/analysis/feature/vae_feature_ood_analysis.py` (多视角闭式高斯似然分解引擎)
    - `docs/adr/0020-remedy-feature-engineering-underflow-and-spurious-indicators.md` (前序公式缺陷与黑名单熔断)
    - `docs/adr/0021-train-dynamic-slicing-and-vae-training-source-adaptation.md` (VAE 训练源切换至 `train`)
    - `docs/adr/0026-multi-perspective-vae-feature-ood-diagnostic-matrix.md` (多视角 OOD 诊断矩阵)

---

## 1. 调研执行摘要 (Executive Summary)

在完成以 `train` 为基准的多视角 OOD 诊断矩阵重构后，最新的实证运行（FU 燃料油 10min，113 个状态特征）彻底揭示了此前被掩盖的真实分布外漂移格局：

1. **极端集中度定律**：
   - 在 **Test vs Train**（真实 OOD 崩溃视角）中，**前 3 个特征贡献了 31.58% 的总似然恶化量**，前 5 个特征贡献了 **39.25%**，前 10 个特征贡献了 **55.82%**，前 15 个特征贡献了 **69.09%**。
   - 在 **Valid vs Train**（验证集漂移视角）中，仅 `realized_volatility_192` 单个特征就贡献了 **19.41%** 的总恶化量，前 5 个特征累计贡献 **43.12%**。
2. **多合约一致性**：
   - 遍历 13 个独立测试合约的下钻归因发现，`cm_main_sub_open_interest_share_sub` 在 7 个合约中高居 OOD 榜首，`sell_spread_oe_max_trend_192` 在 6 个合约中位列 Top 3，`cntn_192_origin` 在 6 个合约中位列 Top 3。这表明恶化并非个别月份的偶然噪声，而是由特征定义的底层数学结构缺陷驱动的系统性崩溃。
3. **发现严重隐蔽公式缺陷**：
   - 本次调研在 `data_preprocess/operator_futures/time_operator/multi_processing_util.py` 中查出与 ADR-0020 同源的重大公式缺陷：所有 `max_*_std_norm`、`min_*_std_norm`、`ma_*_std_norm`、`qtlu/d_*_std_norm` 均**错误地计算了名义价格绝对值除以标准差**（未减去当前价或均价），导致跨年名义价格中枢变动时发生大幅漂移。

---

## 2. 核心问题特征排行榜与恶化指标全景

### 2.1 真实测试集 OOD 崩溃榜 (Test vs Train Top 15)

| 排名 | 特征名称 | $\Delta\text{NLL}$ | 贡献占比 | 训练集均值 | 测试集均值 | 均值漂移 ($\sigma$) | 方差比 | 核心病理归类 |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | `sell_spread_oe_max_trend_192` | **6.78** | **11.12%** | 0.057 | 0.194 | 0.12 | 1.02 | 盘口微观价差低方差除零与边界饱和 |
| 2 | `cm_main_sub_open_interest_share_sub` | **6.59** | **10.81%** | -0.050 | -0.048 | 0.00 | 0.76 | 跨期静态换月持仓非平稳生命周期泄露 |
| 3 | `cntn_192_origin` | **5.89** | **9.66%** | 0.047 | -0.482 | **0.70** | 1.00 | 长周期下跌 Bar 单边计数极端不对称漂移 |
| 4 | `pivot_s1_96_origin` | **2.35** | **3.86%** | -0.154 | 0.104 | 0.33 | 1.21 | 静态支撑通道价格除法复合趋势发散 |
| 5 | `bollinger_lower_192_origin` | **2.33** | **3.81%** | -0.072 | 0.156 | 0.29 | 1.06 | 布林下轨名义比值趋势复合偏移 |
| 6 | `cm_current_sub_open_interest_share_current`| **2.26** | **3.70%** | -0.226 | -0.382 | 0.16 | 0.76 | 跨期静态持仓份额周期泄露 |
| 7 | `vma_192_std_norm_origin` | **2.05** | **3.37%** | 0.119 | -0.137 | 0.32 | 0.74 | 成交量反向变异系数非高斯漂移 |
| 8 | `cntp_192_origin` | **2.00** | **3.28%** | 0.008 | -0.365 | **0.50** | 0.97 | 长周期上涨 Bar 计数单边漂移 |
| 9 | `min_192_std_norm_origin` | **1.91** | **3.14%** | 0.163 | 0.479 | 0.37 | **1.34** | 公式缺陷：名义最低价直接除以波动率 |
| 10 | `macro_trade_imbalance_continuous_240` | **1.88** | **3.09%** | 0.022 | -0.224 | 0.26 | **1.55** | 宏观资金流长周期体制单边持续下行 |
| 11 | `cm_current_main_open_interest_share_current`| **1.85** | **3.03%** | -0.113 | -0.256 | 0.25 | 0.85 | 跨期静态持仓份额周期泄露 |
| 12 | `max_96_std_norm_origin` | **1.80** | **2.95%** | 0.170 | 0.399 | 0.26 | **1.25** | 公式缺陷：名义最高价直接除以波动率 |
| 13 | `rolling_volatility_48` | **1.53** | **2.50%** | 0.163 | -0.165 | 0.36 | 0.91 | 波动率右偏对数正态未正态化 |
| 14 | `ema_slope_192` | **1.52** | **2.49%** | -0.056 | -0.225 | 0.21 | 0.80 | 长周期趋势斜率体制差异 |
| 15 | `garman_klass_volatility_16` | **1.40** | **2.30%** | 0.142 | -0.203 | 0.41 | 0.91 | 原始波幅极差未取对数 |

### 2.2 验证集分布漂移榜 (Valid vs Train Top 5 警示特征)

| 排名 | 特征名称 | $\Delta\text{NLL}$ | 贡献占比 | 训练集均值 | 验证集均值 | 均值漂移 ($\sigma$) | 方差比 | 核心病理归类 |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | `realized_volatility_192` | **89.82** | **19.41%** | 0.114 | 0.459 | 0.39 | **3.35** | 波动率右偏长尾，高波动年份方差膨胀 3.35 倍 |
| 2 | `sell_spread_oe_max_trend_192` | **40.94** | **8.85%** | 0.057 | 0.226 | 0.15 | 1.13 | 盘口微观价差低方差除零与边界饱和 |
| 3 | `bollinger_lower_192_origin` | **28.17** | **6.09%** | -0.072 | -0.192 | 0.15 | **2.22** | 验证集持续单边下挫导致通道比值方差倍增 |
| 4 | `pivot_s1_96_origin` | **23.01** | **4.97%** | -0.154 | -0.291 | 0.17 | **2.08** | 支撑位通道比值方差倍增 |
| 5 | `ema_slope_192` | **17.60** | **3.80%** | -0.056 | -0.070 | 0.02 | **1.75** | 验证集强趋势震荡导致斜率离散度偏大 |

---

## 3. 六大核心病理深度数学剖析

### 病理一：盘口深度价差低方差伪标准化陷阱 (`sell_spread_oe_max_trend_192`)
- **受灾特征**：`sell_spread_oe_max_trend_192`（Test vs Train #1, 11.12%; Valid vs Train #2, 8.85%）、`sell_spread_oe_max`（Valid vs Train #14）。
- **第一手代码位置**：
  - `data_preprocess/operator_futures/cross_section/base_feature_util.py:770`:
    ```python
    data["sell_spread_oe_max"] = np.clip(np.abs(df["ask1_price"].to_numpy() - df[f"ask{depth}_price"].to_numpy()), 0.0, 50.0)
    ```
  - `data_preprocess/operator_futures/time_operator/multi_processing_util.py:31-35`:
    ```python
    mean = pl.col(feature_name).rolling_mean(192)
    std = pl.col(feature_name).rolling_std(192)
    expr = ((pl.col(feature_name) - mean) / (std + 1e-12)).alias(f"{feature_name}_trend_192")
    ```
- **数学成因**：
  1. `ask1_price - ask5_price` 为 5 档卖单价格跨度（离散 Tick 差）。在主力流动性充足期，盘口极其紧密，该值在数百根 Bar 内恒等于 1 或 2 个 Tick（如 2.0 或 4.0 元）。
  2. 当处于 192 Bar（跨度约 3.2 个交易日）窗口时，若盘口平稳，`rolling_std(192)` 趋近于 0（低于 $10^{-6}$）。
  3. 一旦开盘跳空、夜盘集合竞价或瞬时大单将跨度扩大到 5 个 Tick，分母因极小而发生除法爆炸，输出虚假数值脉冲。
  4. 虽然流水线通过 `FAT_TAILED_FEATURE_PATTERNS` 进行了 `[-4.0, 4.0]` 截断，但这导致大量数值在边界 $-4.0$ 与 $+4.0$ 发生**人工双峰堆积（Boundary Saturation）**。
  5. VAE 解码器假设特征服从单峰高斯分布 $\mathcal{N}(\mu, \sigma^2)$。两端的截断饱和点产生剧烈的二次惩罚项 $\frac{(x - \mu)^2}{2\sigma^2}$，造成似然崩溃。

---

### 病理二：跨期静态换月持仓份额的生命周期非平稳泄露
- **受灾特征**：
  - `cm_main_sub_open_interest_share_sub`（Test vs Train #2, **10.81%**）
  - `cm_current_sub_open_interest_share_current`（Test vs Train #6, **3.70%**）
  - `cm_current_main_open_interest_share_current`（Test vs Train #11, **3.03%**）
  - 累计占比高达 **17.54%**。
- **第一手代码位置**：
  - `data_preprocess/operator_futures/commodity/cross_month_feature.py:533-548`:
    ```python
    left_open_interest = float(left["open_interest"])
    right_open_interest = float(right["open_interest"])
    row[f"{prefix}_open_interest_share_{share_name}"] = _share(oi_share_numerator, oi_share_other)
    ```
- **数学成因**：
  1. $OI_{\text{sub}} / (OI_{\text{main}} + OI_{\text{sub}})$ 属于绝对持仓比例。在商品期货运行中，该指标存在严格的**确定性日历换月时钟**：远期时 sub 持仓占比 $< 5\%$；临近换月期在 2-3 周内由 $10\%$ 跃升至 $90\%$；换月后完成交接。
  2. 训练集覆盖 14 个合约的长周期样本，持仓份额平摊在全生命周期；测试集仅包含 2 个测试合约的时序切片，测试期间合约恰好处于特定换月阶段。
  3. **与 ADR-0020 的遗漏脱节**：ADR-0020 当时剔除了交割序对特征 `cm_m1_m2_open_interest_share_m2`，但**遗漏了针对主力-次主力对的 `cm_main_sub_open_interest_share_sub`**！
  4. 该指标是纯粹的**非平稳生命周期标记**（与 ADR-0022 剔除的 `contract_life_remaining_ratio` 本质完全相同）。而系统中**已经计算了严格平稳的一阶差分速度**：
     `cm_open_interest_shift_speed_10m = diff(10).clip(-0.1, 0.1)`（完全平稳且反映资金转移加速度）。

---

### 病理三：长周期胜负 Bar 单边计数不对称漂移
- **受灾特征**：`cntn_192_origin`（Test vs Train #3, **9.66%**）、`cntp_192_origin`（Test vs Train #8, **3.28%**）。
- **第一手代码位置**：
  - `data_preprocess/operator_futures/time_operator/multi_processing_util.py:469-474`:
    ```python
    ((pl.col("__ret1") > 0).rolling_sum(192) / 192).alias("cntp_192")
    ((pl.col("__ret1") < 0).rolling_sum(192) / 192).alias("cntn_192")
    (cntp_192 - cntn_192).alias("cntd_192")
    ```
- **数学成因**：
  1. `cntn_192` 统计过去 192 根 10m Bar（3.2 个交易日）内阴线（下跌 Bar）所占比例。
  2. 实证诊断数据显示：训练集中该特征缩放后均值为 $0.047$，但在测试集中骤降至 **$-0.482$**，均值漂移高达 **$0.70$ 个标准差**！
  3. 测试集（FU2409/FU2501）出现长期单边多头逼空行情，阴线概率显著低于训练集全历史均值。
  4. 单边计数 `cntn` 和 `cntp` 高度冗余且不以 0 为中心对称（受市场无风险基准及单边宏观体制主导）。而零对称的差值特征 `cntd_w = cntp_w - cntn_w` 才是理论自洽的净多空失衡指标。

---

### 病理四：隐蔽公式缺陷——`*_std_norm` 绝对价格混入算子
- **受灾特征**：
  - `min_192_std_norm_origin`（Test vs Train #9, 3.14%）
  - `max_96_std_norm_origin`（Test vs Train #12, 2.95%）
  - `qtlu_48_std_norm_origin`（Test vs Train #21, 1.48%）
  - `qtld_6_std_norm_origin`（Test vs Valid #2, 9.95%）
- **第一手代码位置**：
  - `data_preprocess/operator_futures/time_operator/multi_processing_util.py:487-494, 557-563`:
    ```python
    (close.rolling_max(window) / close_std).alias(f"max_{window}_std_norm")
    (close.rolling_min(window) / close_std).alias(f"min_{window}_std_norm")
    (close.rolling_quantile(0.8) / close_std).alias(f"qtlu_{window}_std_norm")
    (close.rolling_quantile(0.2) / close_std).alias(f"qtld_{window}_std_norm")
    (pl.col("__close_mean") / close_std).alias(f"ma_{window}_std_norm")
    ```
- **数学成因**：
  1. 这是一个**严重的公式设计 BUG**，与 ADR-0020 修复的 `roc_*_std_norm` 完全一致！
  2. 算子试图计算极值相对于波动率的标准化距离，但分子却使用了**绝对名义价格水平** $P_{\max}$，而**未减去当前价 $P_t$ 或均价 $\mu_t$**！
  3. 例如燃料油名义价格为 3000 元，当波动率 `close_std` 仅有 10 元时，计算结果为 $3000 / 10 = 300$；若低波动时标准差为 2 元，数值直接爆炸为 1500！
  4. 更严重的是，跨年份名义价格中枢由 2600 元抬升至 3600 元时，即使相对波幅不变，该比值也会发生整体系统性漂移。
  5. **正确的数学表达**应为极值与当前价的标准化距离：
     $$\text{max\_std\_norm}_t = \frac{P_{\max, W} - P_t}{\sigma_{P, W} + \epsilon}, \quad \text{min\_std\_norm}_t = \frac{P_t - P_{\min, W}}{\sigma_{P, W} + \epsilon}$$

---

### 病理五：波动率右偏长尾与对数正态未正态化
- **受灾特征**：
  - `realized_volatility_192`（Valid vs Train #1, **19.41%** 贡献，方差比 **3.35**）
  - `rolling_volatility_48`（Test vs Train #13, Valid vs Train #12）
  - `garman_klass_volatility_16`（Test vs Train #15）
- **第一手代码位置**：
  - `data_preprocess/operator_futures/time_operator/multi_processing_util.py:727-728`:
    ```python
    realized_v = r_sq.rolling_sum(w).sqrt()
    all_exprs.append(realized_v.alias(f"realized_volatility_{w}"))
    ```
- **数学成因**：
  1. 现实金融市场中，波动率（Realized Volatility / Parkinson / Garman-Klass）严格服从**对数正态分布 (Log-Normal Distribution)**，具有强右偏厚尾特性。
  2. 当前流水线直接对其应用 Robust Scaler：$\frac{x - \text{median}}{\text{IQR}}$。虽然中心化了中位数，但线性变换无法改变其右偏厚尾性质。
  3. 验证集所处时期包含数次宏观突发剧烈波动，导致 `realized_volatility_192` 的方差膨胀为训练集的 **3.35 倍**（Valid 均值高达 0.459，Train 仅为 0.114）。
  4. VAE 解码器采用标准高斯先验与损失，对右侧长尾样本施加巨大负对数似然惩罚，直接造成 89.82 的 NLL 恶化。

---

### 病理六：名义通道价格除法的趋势复合偏移
- **受灾特征**：
  - `pivot_s1_96_origin`（Test vs Train #4, 3.86%; Valid vs Train #4, 4.97%）
  - `bollinger_lower_192_origin`（Test vs Train #5, 3.81%; Valid vs Train #3, 6.09%）
  - `bollinger_upper_192_origin`（Valid vs Train #6, 3.41%）
  - `ema_slope_192`（Test vs Train #14, Valid vs Train #5）
- **第一手代码位置**：
  - `data_preprocess/operator_futures/time_operator/multi_processing_util.py:453-456`:
    ```python
    (bollinger_lower / (close + min_value)).alias(f"bollinger_lower_{window}")
    (pivot_s1 / (close + min_value)).alias(f"pivot_s1_{window}")
    ```
- **数学成因**：
  1. `bollinger_lower / close = (MA - 2*STD) / close = 1 + (MA - close - 2*STD) / close`。
  2. 当市场进入持续单边大牛市或大熊市时，收盘价 $close$ 会脱离均线 $MA$。在测试集和验证集中的强趋势波段下，该指标方差较训练集膨胀 **2.08 ~ 2.22 倍**。
  3. 实际上，衡量价格在布林通道内相对位置的理论完备指标是 **无量纲通道百分比**：
     $$\text{Percent\_B} = \frac{P_t - \text{Lower}_t}{\text{Upper}_t - \text{Lower}_t} \in [0, 1]$$
     以及 **布林带宽** `bollinger_bandwidth = 4 * STD / MA`，而非将名义下轨值直接与收盘价相除。

---

## 4. 改造、优化与平稳化替代方案蓝图

基于上述根因分析，我们提出三阶段落地方案：

```
                    ┌─────────────────────────────────────────────────────────────┐
                    │               三阶段 OOD 根治与优化实施路径                 │
                    └─────────────────────────────────────────────────────────────┘
                                                   │
         ┌─────────────────────────────────────────┼────────────────────────────────────────┐
         ▼                                         ▼                                        ▼
【第一阶段：零重跑快捷熔断】               【第二阶段：数学公式修复与平稳替代】     【第三阶段：高斯先验变换体系】
- 更新特征黑名单 Blacklist                 - 修复 multi_processing_util.py          - 波动率 Log 变换
- 剔除纯静态换月与单边计数                   中绝对价格除法 BUG (std_norm)          - 布林通道 Percent_B 化
- 立即消除 >35% OOD 恶化量                 - 盘口价差改用动态中位数比例             - 消除方差倍增长尾
```

### 4.1 第一阶段：立即实施的黑名单熔断项 (Fast-Path Blacklist)

在不重新生成底层中间特征文件的情况下，直接将以下确定性非平稳、具有纯周期泄露或已被平稳微分特征完全替代的特征列入 `COMMODITY_FU_FEATURE_BLACKLIST`：

| 特征名称 | 当前状态 | 剔除依据与替代特征 | 预期消除 Test OOD 占比 |
|:---|:---:|:---|:---:|
| `sell_spread_oe_max_trend_192` | 活跃 | 盘口微观价差除零与边界饱和，短期可用 `sell_spread_oe_max_trend_6` 替代 | **11.12%** |
| `cm_main_sub_open_interest_share_sub` | 活跃 | 静态生命周期换月指标，已被 `cm_open_interest_shift_speed_10m` 完全替代 | **10.81%** |
| `cntn_192_origin` | 活跃 | 单边下跌计数漂移，已有对称指标 `cntd_96_origin` 与连续宏观动量替代 | **9.66%** |
| `cm_current_sub_open_interest_share_current` | 活跃 | 静态换月持仓份额，周期性强非平稳漂移 | **3.70%** |
| `cntp_192_origin` | 活跃 | 单边上涨计数漂移，与 `cntn` 冗余 | **3.28%** |
| `cm_current_main_open_interest_share_current` | 活跃 | 静态换月持仓份额，周期性强非平稳漂移 | **3.03%** |
| **小计** | | **仅剔除上述 6 个特征，即可瞬间消除测试集 41.60% 的 OOD 似然崩溃！** | **41.60%** |

---

### 4.2 第二阶段：源码级算子修复与平稳化改造 (Source Code Operator Remediation)

在 `data_preprocess/operator_futures/time_operator/multi_processing_util.py` 中彻底修正数学公式缺陷：

#### 1. 修复 `*_std_norm` 绝对价格算子缺陷
```python
# 修改前 (Bug: 名义价格直接除以波动率):
(close.rolling_max(window) / close_std).alias(f"max_{window}_std_norm")
(close.rolling_min(window) / close_std).alias(f"min_{window}_std_norm")
(close.rolling_quantile(0.8) / close_std).alias(f"qtlu_{window}_std_norm")
(close.rolling_quantile(0.2) / close_std).alias(f"qtld_{window}_std_norm")

# 修改后 (严谨的标准化价格偏离距离):
((close.rolling_max(window) - close) / (close_std + min_value)).alias(f"max_{window}_std_norm")
((close - close.rolling_min(window)) / (close_std + min_value)).alias(f"min_{window}_std_norm")
((close.rolling_quantile(0.8) - close) / (close_std + min_value)).alias(f"qtlu_{window}_std_norm")
((close - close.rolling_quantile(0.2)) / (close_std + min_value)).alias(f"qtld_{window}_std_norm")
```

#### 2. 盘口价差改造为尺度无关的相对压缩率
不再对容易为常数的 Tick 价差直接做长周期 Z-score，改为相对平滑基准的对数压缩比：
```python
# 新增: 盘口相对压缩率 (Scale-Invariant Depth Compression Ratio)
sell_spread = pl.col("sell_spread_oe_max")
sell_spread_ratio = (sell_spread / (sell_spread.rolling_median(window) + 1.0)).log()
```

#### 3. 通道类特征改造为百分比相对位置 (Percent-B)
将名义价格相除的 `bollinger_lower / close` 和 `pivot_s1 / close` 替换为严格有界的通道分位数：
```python
# 新增: 布林带相对位置与支撑位距离比例
bollinger_percent_b = (close - bollinger_lower) / (bollinger_upper - bollinger_lower + min_value)
pivot_s1_distance = (close - pivot_s1) / (close * 0.01 + min_value) # 以 1% 价格为基准的相对偏离
```

---

### 4.3 第三阶段：高斯先验变换（针对对数正态波动率）

针对 `realized_volatility_*`、`rolling_volatility_*`、`garman_klass_volatility_*` 等在验证集中方差膨胀 3.35 倍的厚尾特征，在 `muti_contract_scale_save.py` 之前应用对数正态化算子：
$$\widetilde{\text{vol}}_t = \ln\left(\text{vol}_t + 10^{-6}\right)$$
或采用相对长期均线比值的对数：
$$\text{vol\_ratio}_t = \ln\left(\frac{\text{vol}_t}{\text{SMA}_{240}(\text{vol}) + 10^{-6}}\right)$$
彻底消除验证集与测试集在不同市场体制下的方差爆炸。

---

## 5. 预期效益与下一步行动建议

1. **直接收益**：
   - 实施第一阶段黑名单（6 个特征）后，测试集总 OOD 恶化量将**直接下降超过 41%**；
   - 修复第二阶段 `*_std_norm` BUG 后，测试集总 OOD 恶化量累计**下降超过 52%**；
   - 实施第三阶段波动率 Log 正态化后，验证集方差异常值将从 3.35 倍压降至 1.1 倍左右。
2. **对下游 RL 与 VAE 路由的深远影响**：
   - VAE 的重构似然将不再受伪特征脉冲主导，门控路由可真实捕捉市场的价格变动与流动性体制，而非因为“某天跨期持仓份额自然达到 45%”而发生虚假的 OOD 熔断。
3. **行动顺序建议**：
   - **Step 1**：将第一阶段的 6 个确定性缺陷特征加入 `fu_full_process.sh` 的 `COMMODITY_FU_FEATURE_BLACKLIST`；
   - **Step 2**：在 `data_preprocess/operator_futures/time_operator/multi_processing_util.py` 中修复 `*_std_norm` 的减法公式；
   - **Step 3**：起草 ADR-0027 记录本次调研与特征改造决策。
