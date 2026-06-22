# 商品期货预处理说明

本文档说明上海期货交易所商品期货数据接入 FineFT 的本地目录、字段约定和验证命令。当前实现以燃料油 `fu` 为首个品种。

## 输入目录

本地原始五档行情 CSV 放在：

```text
data/原始下载/{品种中文名}/{YYYY}/{MM}/{YYYYMMDD}/{合约}.csv
```

燃料油示例：

```text
data/原始下载/燃料油/2026/01/20260105/fu2605.csv
```

样例文件为：

```text
docs/上海商品交易所/fu2302.csv
```

## 输出命名

燃料油最终 dataset 使用交易所品种代码 `fu`，不使用中文名。商品期货盘口深度为真实五档，输出只包含 `ask1-ask5` 和 `bid1-bid5`，不会填充假的 6-25 档。

## 时间与主力合约

- 训练时间戳使用真实 `ActionDay + UpdateTime`。
- 日文件归属和主力识别使用 `TradingDay`。
- 主力合约按每个 `TradingDay` 固定选择，不做日内切换，不做复权。
- 主力选择优先使用上一交易日成交量，若选中合约当日缺失或无成交量，则回退到当前交易日成交量最高的合格主力月份合约。

## 商品期货缺失数据

商品期货没有加密货币永续合约中的真实 funding、index price、mark price、trades 和 quotes 数据：

- funding 关闭，环境中不产生 funding fee，也不暴露 funding countdown 状态。
- `index_price` 和 `mark_price` 使用 `LastPrice` 生成；`LastPrice` 缺失、为 0 或越过涨跌停边界时回退到一档 midprice。
- trades 特征由每秒累计 `Volume` 与 `Turnover` 差分估计，`second_avg_price = Turnover.diff() / Volume.diff() / contract_unit`；`tradeval` 保留原始成交额差分。
- 成交方向使用 tick rule：秒均价相对前值上涨为 `buy_estimated`，下跌为 `sell_estimated`，持平为 `flat`，不做方向继承。
- quotes 特征由原始五档快照先整理为秒频最后一条，再聚合到目标频率；秒频层不 forward fill，目标窗口没有 quote 时 fail-fast。

## 手续费

商品手续费按品种配置。燃料油默认：

- 买入费率：`0.0001`
- 卖出费率：`0.0003`
- 合约交易单位：`10`
- `contract_unit` 仅用于商品成交均价和 `vwap` 的价格口径修正，不用于 PnL、保证金或手续费。

## 验证命令

完整燃料油流程入口：

```bash
source data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh
run_commodity_full_process "$(pwd)" 2026-01-01 2026-02-01 5min fu 燃料油 4
```

也可以直接使用 main script，并通过环境变量覆盖默认参数：

```bash
YEAR=2026 START_DATE=2026-01-01 END_DATE=2026-02-01 TARGET_FREQ=5min \
  bash data_preprocess/script_preprocess/future_upgraded/commodity/main.sh
```

该入口会先从日期范围覆盖的 `data/原始下载/燃料油/{YYYY}` 目录扫描所有合约 CSV，按 `TradingDay` 选择每日主力合约，生成连续主力原始日文件：

```text
PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/fu/2026-01-05.csv
PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/fu/2026-01-06.csv
```

然后继续执行商品期货下采样、cross-section、merge/concat、time feature、merge clean、IC feature selection 和 scale/save。最终训练入口数据写入：

```text
PREPROCESS_DATASET/commodity-futures/SCALE_SAVE/fu/5min/2026-01-01-2026-02-01/
```

直接运行连续主力拼接和下采样 CLI：

```bash
PYTHONPATH=data_preprocess python -m operator_futures.commodity.stitch_main_contract \
  --raw_root data/原始下载 \
  --commodity_name 燃料油 \
  --start_date 2026-01-01 \
  --end_date 2026-02-01 \
  --symbol fu \
  --output_dir PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/fu

PYTHONPATH=data_preprocess python -m operator_futures.commodity.downscale_continuous_by_trading_day \
  --input_dir PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/fu \
  --start_date 2026-01-01 \
  --end_date 2026-02-01 \
  --output_root PREPROCESS_DATASET/commodity-futures \
  --symbol fu \
  --target_freq 5min \
  --depth 5
```

运行单日样例 smoke test：

```bash
source data_preprocess/script_preprocess/future_upgraded/commodity/commodity_process.sh
run_commodity_smoke_fu "$(pwd)" 5min
```

直接运行 Python CLI：

```bash
PYTHONPATH=data_preprocess python -m operator_futures.commodity.downscale_single_day \
  --input docs/上海商品交易所/fu2302.csv \
  --output_dir /tmp/fu_downscale_smoke \
  --symbol fu \
  --target_freq 5min
```

输出目录应包含：

```text
derivative_reference.feather
orderbook_5.feather
base_feature.feather
quote_feature.feather
```
