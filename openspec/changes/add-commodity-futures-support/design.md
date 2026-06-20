## Context

FineFT 当前数据链路围绕 Tardis 下载的加密货币期货数据构建。reward/execution 数据来自
25 档订单簿快照和 derivative ticker；市场状态特征来自原始 trades 和 quotes。环境侧读取
固定 depth=25 的盘口数组、`mark_price`、`funding_rate` 和 `funding_timestamp`。

本变更的商品期货数据源不同：它是一类五档行情流，字段定义见
`docs/上海商品交易所/商品期货数据结构.md`，样例见 `docs/上海商品交易所/fu2302.csv`。
该数据包含 `LastPrice`、累计 `Volume`、累计 `Turnover`、持仓量、结算价、涨跌停价和
Bid/Ask 1-5 档；不包含真实 trades、真实 quotes、funding、index price 或 mark price。

商品期货原始数据目录 SHALL 以中文品种名组织，根目录为
`data/原始下载/{品种中文名}/{YYYY}`，例如 `data/原始下载/燃料油/2026`。实现 SHALL 在该
年份目录下继续兼容现有规划的月/日/合约层级：`{MM}/{YYYYMMDD}/{合约}.csv`。如果后续实
际数据直接放在年份目录或月份目录下，实现应通过配置或扫描函数显式支持，但默认契约以
`data/原始下载/燃料油/2026/<MM>/<YYYYMMDD>/<contract>.csv` 为准。

这是一个跨模块变更，会影响数据预处理、特征契约、scale/save 输出和环境初始化。由于现有
depth=25 与 funding 假设分散在预处理和环境代码中，本变更需要明确技术设计。

## Goals / Non-Goals

**Goals:**

- 保留现有加密货币期货行为，同时新增独立商品期货分支。
- 为燃料油生成可用于 FineFT 的商品期货数据集，dataset 名称为 `fu`。
- 使用真实 depth=5 盘口列，不补齐伪 25 档。
- 将成交方向特征明确标记为 estimated。
- 在商品期货环境中关闭 funding 行为。
- 使用配置化交易费率：燃料油 `buy_fee_rate=0.0001`、`sell_fee_rate=0.0003`。
- 保持 feature selection target 为 `mark_price.shift(-window) - mark_price`。

**Non-Goals:**

- 不实现商品数据下载。
- 不对主力合约切换做复权或价差调整。
- 秒均价、PnL、保证金或手续费计算不引入合约乘数。
- 不实现五档流动性不足时的部分成交、拒单或第五档外推模型。
- 不恢复真实 trades、真实 quotes、真实 funding、真实 index price 或真实 mark price。
- 不把合约代码作为 `state_features` 输入模型。

## Decisions

### Decision 1: 新增商品期货分支，而不是把 5 档补成 25 档

商品期货 SHALL 使用显式 `orderbook_depth=5` 契约。备选方案是补齐 6-25 档以减少代码改动，
但这会制造虚假的深度特征和虚假的成交能力。当前选择需要更新下游代码的 depth-awareness，
但能保持数据语义干净。

### Decision 2: 按 TradingDay 选择主力且不复权

主力合约拼接 SHALL 对每个 `TradingDay` 只选择一个合约。选择逻辑使用前一交易日候选合约
成交量；如果该合约在目标日缺失或无成交，则回退到目标日成交量最大的可用主力月份合约。
连续序列 SHALL 不做价格复权，因为训练数据应保留真实可交易序列中的换月跳价。

主力拼接输入 SHALL 从 `data/原始下载/燃料油/{YYYY}` 开始扫描，按年份、月份、交易日目录
排序处理，以便跨月、跨年传递“前一交易日”引用。

### Decision 3: 事件时间使用 ActionDay，日归属使用 TradingDay

规范化事件时间戳 SHALL 由 `ActionDay + UpdateTime` 构造，使夜盘事件保留真实时间顺序。
日文件归属和主力合约选择 SHALL 使用 `TradingDay`，与交易所结算日归属一致。

### Decision 4: 使用秒频标准层且不 forward fill

原始快照 SHALL 先标准化为一秒一条的中间层，再聚合到目标频率。同一秒内有多条快照时，秒
频 quote 状态和累计 `Volume`/`Turnover` 使用该秒最后一条。缺失秒 SHALL 保持缺失。这可
避免产生并不存在的 quote 变化或成交。

### Decision 5: 从 Volume/Turnover 差分估计成交

秒级成交估计 SHALL 使用：

- `second_volume = Volume.diff()`
- `second_tradeval = Turnover.diff()`
- `second_avg_price = second_tradeval / second_volume`

公式 SHALL NOT 除以合约乘数。如果 `second_volume > 0` 且 `second_tradeval` 为 0、缺失、
负数或其他无效值，预处理 SHALL fail-fast。

tick rule 方向 SHALL 比较当前有效 `second_avg_price` 和上一秒有效均价。价格上涨为
`buy_estimated`/`up`，价格下跌为 `sell_estimated`/`down`，价格不变为 `flat`，且不归入
buy 或 sell。

### Decision 6: 从秒频五档快照派生 quote 特征

商品期货 quote 特征 SHALL 不依赖原始 `quotes` 文件。流程 SHALL 从秒频五档快照状态计算
quote 变化计数和 quote OHLC/TWAP/AWAP。目标频率 bar SHALL 使用右闭右标，例如
`(09:00:00, 09:05:00]` 标记为 `09:05:00`。

如果目标频率窗口没有 quote 快照，预处理 SHALL fail-fast 并输出日期、窗口和合约上下文。
如果窗口有 quote 但没有成交量，trade OHLC/VWAP/TWAP/AWAP SHALL 使用上一笔有效
`LastPrice`；交易日开头没有上一笔成交价时使用当前快照 `LastPrice`。

### Decision 7: 只保留不暗示真实行为的兼容列

商品期货 derivative-ticker 输出 SHALL 包含 `funding_rate`、`funding_timestamp` 等环境所
需兼容列，但商品期货环境 SHALL 使用 `funding_enabled=false`，MUST NOT 扣 funding，也
MUST NOT 向状态输入暴露 funding countdown。`mark_price` 和 `index_price` SHALL 优先使用
`LastPrice`；仅当 `LastPrice` 缺失、为 0 或超出涨跌停范围时回退 midprice。

### Decision 8: 用 manifest 替代固定 reward 列假设

现有 `scale_save.py` 假设前 106 列是 reward/execution 列。商品期货 depth=5 使该假设失效。
新增分支 SHALL 使用 reward/execution column manifest 或等价显式列列表，避免 depth-specific
列数影响 state-feature selection。

## Risks / Trade-offs

- 估计成交方向可能不同于真实主动买卖方向。缓解方式：所有方向派生字段必须命名为
  `_estimated` 或在元数据中标记。
- 秒频层不 forward fill 会暴露数据缺口。缓解方式：目标频率 quote 窗口为空时 fail-fast。
- depth=5 对大仓位可能不够。缓解方式：本变更假设最大仓位较小且主力月份流动性足够；深度
  不足时 fail-fast，不静默外推。
- 不引入合约乘数会使结果是归一化研究回测，而不是交易所现金口径。缓解方式：在配置和测试
  中记录该假设。
- 商品期货右闭右标 bar 不同于 pandas 默认左标。缓解方式：downscale、merge、cross-section、
  time feature 和 feature selection 全链路统一该约定。

## Migration Plan

1. 新增商品期货专用模块和脚本入口，不修改默认加密货币期货路径。
2. 基于 `fu2302.csv` 样例先补充测试。
3. 加入 depth-aware manifest 和商品环境 initializer。
4. 运行商品期货 smoke test 与加密货币回归 smoke test。
5. 如果商品处理阻塞训练，可回退到现有 crypto 脚本；既有 crypto 文件不受影响。

## Open Questions

当前变更没有阻塞性开放问题。后续可继续扩展合约乘数会计、部分成交模型、复权选择和更多商品
品种。
