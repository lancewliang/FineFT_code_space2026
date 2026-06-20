# 商品期货支持实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 为燃料油 `fu` 新增原生商品期货数据预处理与 FineFT 环境支持，使用真实五档盘口数据完成从本地 CSV 到可训练环境的闭环。

**架构：** 保留现有 Binance 加密货币期货路径不变，在 `data_preprocess/operator_futures/commodity/` 下新增商品期货模块。下游特征与环境通过配置和 manifest 支持不同盘口深度，商品期货使用 depth=5，加密货币期货继续可使用 depth=25。

**技术栈：** Python 3.10、pandas、numpy、pyarrow/feather、pytest、shell scripts、现有 FineFT gym 环境代码。

**Traceability (sddflow):**
- plan-ready: `openspec/changes/add-commodity-futures-support/plan-ready.md`
- tasks: `openspec/changes/add-commodity-futures-support/tasks.md`
- plan: `docs/superpowers/plans/2026-06-20-add-commodity-futures-support.md`

---

### Task 1: 商品期货数据契约与配置

> **trace:** plan-ready.md → `### Task 1: 商品期货数据契约与配置` | tasks.md → ``- [ ] 1.1 新增燃料油 `fu` 商品期货配置，包含 depth=5、关闭 funding、买入费率 `0.0001`、卖出费率 `0.0003`、主力月份规则和 dataset 命名。``
> **sync:** tasks.md → ``- [ ] 1.1 新增燃料油 `fu` 商品期货配置，包含 depth=5、关闭 funding、买入费率 `0.0001`、卖出费率 `0.0003`、主力月份规则和 dataset 命名。`` | plan-ready.md → `### Task 1: 商品期货数据契约与配置`

**文件：**
- 新增：`data_preprocess/operator_futures/commodity/__init__.py`
- 新增：`data_preprocess/operator_futures/commodity/config.py`
- 新增：`data_preprocess/operator_futures/commodity/schema.py`
- 新增测试：`data_preprocess/tests/test_commodity_config_schema.py`

- [ ] **Step 1: 编写失败的配置与 schema 测试**

创建 `data_preprocess/tests/test_commodity_config_schema.py`：

```python
import pandas as pd

from operator_futures.commodity.config import get_commodity_config
from operator_futures.commodity.schema import (
    build_orderbook_columns,
    get_reward_execution_columns,
    resample_kwargs,
)


def test_fu_config_contract():
    config = get_commodity_config("fu")

    assert config.symbol == "fu"
    assert config.dataset_name == "fu"
    assert config.orderbook_depth == 5
    assert config.funding_enabled is False
    assert config.buy_fee_rate == 0.0001
    assert config.sell_fee_rate == 0.0003
    assert config.main_contract_months == tuple(range(1, 13))
    assert config.use_contract_multiplier is False


def test_depth_five_orderbook_columns_have_no_synthetic_levels():
    columns = build_orderbook_columns(5)

    assert "ask1_price" in columns
    assert "ask5_size" in columns
    assert "bid5_price" in columns
    assert "ask6_price" not in columns
    assert "bid25_price" not in columns
    assert len(columns) == 20


def test_reward_execution_manifest_for_depth_five():
    columns = get_reward_execution_columns(depth=5)

    assert columns[0] == "timestamp"
    assert "mark_price" in columns
    assert "funding_rate" in columns
    assert "ask5_price" in columns
    assert "bid5_size" in columns
    assert "ask6_price" not in columns
    assert len(columns) == 1 + 20 + 5


def test_resample_kwargs_are_right_closed_and_right_labeled():
    kwargs = resample_kwargs()

    assert kwargs == {"closed": "right", "label": "right"}
    series = pd.Series(
        [1, 2],
        index=pd.to_datetime(["2023-01-03 09:00:00", "2023-01-03 09:05:00"]),
    )
    result = series.resample("5min", **kwargs).sum()

    assert result.loc[pd.Timestamp("2023-01-03 09:00:00")] == 1
    assert result.loc[pd.Timestamp("2023-01-03 09:05:00")] == 2
```

- [ ] **Step 2: 运行测试并确认失败**

运行：`PYTHONPATH=data_preprocess python -m pytest data_preprocess/tests/test_commodity_config_schema.py -q`

预期：失败，错误包含 `ModuleNotFoundError: No module named 'operator_futures.commodity'`。

- [ ] **Step 3: 新增商品期货包导出**

创建 `data_preprocess/operator_futures/commodity/__init__.py`：

```python
from .config import CommodityConfig, get_commodity_config
from .schema import build_orderbook_columns, get_reward_execution_columns, resample_kwargs

__all__ = [
    "CommodityConfig",
    "get_commodity_config",
    "build_orderbook_columns",
    "get_reward_execution_columns",
    "resample_kwargs",
]
```

- [ ] **Step 4: 新增商品期货配置模块**

创建 `data_preprocess/operator_futures/commodity/config.py`：

```python
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class CommodityConfig:
    symbol: str
    display_name: str
    dataset_name: str
    orderbook_depth: int
    funding_enabled: bool
    buy_fee_rate: float
    sell_fee_rate: float
    main_contract_months: Tuple[int, ...]
    use_contract_multiplier: bool


COMMODITY_CONFIGS: Dict[str, CommodityConfig] = {
    "fu": CommodityConfig(
        symbol="fu",
        display_name="燃料油",
        dataset_name="fu",
        orderbook_depth=5,
        funding_enabled=False,
        buy_fee_rate=0.0001,
        sell_fee_rate=0.0003,
        main_contract_months=tuple(range(1, 13)),
        use_contract_multiplier=False,
    )
}


def get_commodity_config(symbol: str) -> CommodityConfig:
    normalized = symbol.lower()
    if normalized not in COMMODITY_CONFIGS:
        supported = ", ".join(sorted(COMMODITY_CONFIGS))
        raise ValueError(f"Unsupported commodity symbol {symbol!r}; supported: {supported}")
    return COMMODITY_CONFIGS[normalized]
```

- [ ] **Step 5: 新增商品期货 schema 工具**

创建 `data_preprocess/operator_futures/commodity/schema.py`：

```python
from typing import Dict, List


DERIVATIVE_REFERENCE_COLUMNS = [
    "symbol",
    "funding_timestamp",
    "funding_rate",
    "index_price",
    "mark_price",
]


def resample_kwargs() -> Dict[str, str]:
    return {"closed": "right", "label": "right"}


def build_orderbook_columns(depth: int) -> List[str]:
    if depth < 1:
        raise ValueError("orderbook depth must be positive")
    columns: List[str] = []
    for side in ("ask", "bid"):
        for level in range(1, depth + 1):
            columns.append(f"{side}{level}_price")
            columns.append(f"{side}{level}_size")
    return columns


def get_reward_execution_columns(depth: int) -> List[str]:
    return ["timestamp", *build_orderbook_columns(depth), *DERIVATIVE_REFERENCE_COLUMNS]
```

- [ ] **Step 6: 运行配置与 schema 测试并确认通过**

运行：`PYTHONPATH=data_preprocess python -m pytest data_preprocess/tests/test_commodity_config_schema.py -q`

预期：通过，4 个测试通过。

- [ ] **Step 7: 提交 Task 1**

运行：

```bash
git add data_preprocess/operator_futures/commodity/__init__.py data_preprocess/operator_futures/commodity/config.py data_preprocess/operator_futures/commodity/schema.py data_preprocess/tests/test_commodity_config_schema.py
git commit -m "feat: add commodity futures config contract"
```

预期：提交成功。如用户要求暂不提交，则保留文件变更并继续由用户决定。

- [ ] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 2: 主力合约拼接

> **trace:** plan-ready.md → `### Task 2: 主力合约拼接` | tasks.md → ``- [ ] 2.1 实现商品期货主力合约拼接，从 `data/原始下载/{品种中文名}/{YYYY}` 扫描本地五档 CSV，默认支持 `{MM}/{YYYYMMDD}/{合约}.csv` 层级，使用前一 `TradingDay` 成交量选择主力，并支持当前日 fallback。``
> **sync:** tasks.md → ``- [ ] 2.1 实现商品期货主力合约拼接，从 `data/原始下载/{品种中文名}/{YYYY}` 扫描本地五档 CSV，默认支持 `{MM}/{YYYYMMDD}/{合约}.csv` 层级，使用前一 `TradingDay` 成交量选择主力，并支持当前日 fallback。`` | plan-ready.md → `### Task 2: 主力合约拼接`

**文件：**
- 新增：`data_preprocess/operator_futures/commodity/main_contract.py`
- 新增测试：`data_preprocess/tests/test_commodity_main_contract.py`

- [ ] **Step 1: 编写失败的主力合约测试**

创建 `data_preprocess/tests/test_commodity_main_contract.py`：

```python
from pathlib import Path

import pandas as pd

from operator_futures.commodity.main_contract import (
    calculate_contract_volume,
    iter_contract_files,
    normalize_timestamp,
    select_main_contract_for_day,
    stitch_main_contract_frames,
)


def _frame(contract: str, trading_day: str, action_day: str, volumes):
    rows = []
    for idx, volume in enumerate(volumes):
        rows.append(
            {
                "InstrumentID": contract,
                "TradingDay": trading_day,
                "ActionDay": action_day,
                "UpdateTime": f"21:00:0{idx}.500",
                "LastPrice": 2600 + idx,
                "Volume": volume,
                "Turnover": volume * (2600 + idx),
                "BidPrice1": 2599,
                "BidVolume1": 1,
                "AskPrice1": 2601,
                "AskVolume1": 1,
            }
        )
    return pd.DataFrame(rows)


def test_normalize_timestamp_uses_action_day():
    row = pd.Series({"ActionDay": "20230103", "UpdateTime": "21:00:00.500"})
    assert normalize_timestamp(row) == pd.Timestamp("2023-01-03 21:00:00.500")


def test_calculate_contract_volume_uses_cumulative_volume_delta():
    df = _frame("fu2302", "20230104", "20230103", [10, 12, 18])
    assert calculate_contract_volume(df) == 8


def test_iter_contract_files_scans_raw_download_layout(tmp_path):
    contract_file = tmp_path / "data" / "原始下载" / "燃料油" / "2026" / "01" / "20260105" / "fu2602.csv"
    contract_file.parent.mkdir(parents=True)
    contract_file.write_text("InstrumentID,TradingDay\nfu2602,20260105\n", encoding="utf-8")

    files = list(iter_contract_files(tmp_path / "data" / "原始下载", "燃料油", "2026"))

    assert files == [contract_file]


def test_select_main_contract_uses_previous_day_volume():
    previous = {
        "fu2302": _frame("fu2302", "20230103", "20230102", [10, 12]),
        "fu2303": _frame("fu2303", "20230103", "20230102", [7, 40]),
    }
    current = {
        "fu2302": _frame("fu2302", "20230104", "20230103", [0, 1]),
        "fu2303": _frame("fu2303", "20230104", "20230103", [0, 2]),
    }
    selected, reason = select_main_contract_for_day(previous, current, "fu")
    assert selected == "fu2303"
    assert reason == "previous_trading_day_volume"


def test_select_main_contract_falls_back_to_current_day_volume():
    previous = {"fu2302": _frame("fu2302", "20230103", "20230102", [10, 30])}
    current = {
        "fu2303": _frame("fu2303", "20230104", "20230103", [0, 11]),
        "fu2304": _frame("fu2304", "20230104", "20230103", [0, 5]),
    }
    selected, reason = select_main_contract_for_day(previous, current, "fu")
    assert selected == "fu2303"
    assert reason == "current_trading_day_fallback"


def test_stitch_main_contract_frames_keeps_metadata_and_no_back_adjustment():
    day1 = _frame("fu2302", "20230104", "20230103", [0, 1])
    day2 = _frame("fu2303", "20230105", "20230104", [0, 2])
    stitched = stitch_main_contract_frames(
        [
            ("20230104", "fu2302", day1, Path("fu2302.csv")),
            ("20230105", "fu2303", day2, Path("fu2303.csv")),
        ]
    )
    assert stitched["main_contract"].tolist() == ["fu2302", "fu2302", "fu2303", "fu2303"]
    assert stitched["source_contract"].tolist() == ["fu2302", "fu2302", "fu2303", "fu2303"]
    assert stitched["source_file"].str.endswith(".csv").all()
    assert stitched.loc[1, "LastPrice"] == 2601
    assert stitched.loc[2, "LastPrice"] == 2600
```

- [ ] **Step 2: 运行测试并确认失败**

运行：`PYTHONPATH=data_preprocess python -m pytest data_preprocess/tests/test_commodity_main_contract.py -q`

预期：失败，缺少 `operator_futures.commodity.main_contract`。

- [ ] **Step 3: 实现主力合约模块**

创建 `data_preprocess/operator_futures/commodity/main_contract.py`：

```python
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .config import get_commodity_config


def normalize_timestamp(row: pd.Series) -> pd.Timestamp:
    action_day = str(row["ActionDay"])
    update_time = str(row["UpdateTime"])
    return pd.to_datetime(f"{action_day} {update_time}", format="%Y%m%d %H:%M:%S.%f")


def calculate_contract_volume(df: pd.DataFrame) -> float:
    if "Volume" not in df.columns or df.empty:
        return 0.0
    volume = pd.to_numeric(df["Volume"], errors="coerce")
    if volume.dropna().empty:
        return 0.0
    return float(volume.max() - volume.min())


def iter_contract_files(raw_root: Path, commodity_name: str, year: str) -> Iterable[Path]:
    year_dir = raw_root / commodity_name / year
    if not year_dir.exists():
        raise FileNotFoundError(f"Commodity raw year directory does not exist: {year_dir}")
    return iter(sorted(year_dir.glob("*/*/*.csv")))


def _eligible_contracts(frames: Dict[str, pd.DataFrame], symbol: str) -> Dict[str, pd.DataFrame]:
    config = get_commodity_config(symbol)
    eligible: Dict[str, pd.DataFrame] = {}
    for contract, frame in frames.items():
        normalized = contract.lower()
        if not normalized.startswith(config.symbol):
            continue
        month_text = normalized[-2:]
        if not month_text.isdigit():
            continue
        if int(month_text) in config.main_contract_months:
            eligible[contract] = frame
    return eligible


def _largest_volume_contract(frames: Dict[str, pd.DataFrame]) -> Optional[str]:
    volumes = {contract: calculate_contract_volume(frame) for contract, frame in frames.items()}
    positive = {contract: volume for contract, volume in volumes.items() if volume > 0}
    if not positive:
        return None
    return max(positive, key=positive.get)


def select_main_contract_for_day(
    previous_day_frames: Dict[str, pd.DataFrame],
    current_day_frames: Dict[str, pd.DataFrame],
    symbol: str,
) -> Tuple[str, str]:
    previous_eligible = _eligible_contracts(previous_day_frames, symbol)
    current_eligible = _eligible_contracts(current_day_frames, symbol)
    previous_choice = _largest_volume_contract(previous_eligible)
    if previous_choice in current_eligible and calculate_contract_volume(current_eligible[previous_choice]) > 0:
        return previous_choice, "previous_trading_day_volume"
    fallback = _largest_volume_contract(current_eligible)
    if fallback is None:
        raise ValueError(f"No tradable eligible contract found for symbol {symbol!r}")
    return fallback, "current_trading_day_fallback"


def stitch_main_contract_frames(
    selected_frames: Iterable[Tuple[str, str, pd.DataFrame, Path]]
) -> pd.DataFrame:
    output: List[pd.DataFrame] = []
    for trading_day, contract, frame, source_file in selected_frames:
        copied = frame.copy()
        copied["timestamp"] = copied.apply(normalize_timestamp, axis=1)
        copied["main_contract"] = contract
        copied["source_contract"] = copied.get("InstrumentID", contract)
        copied["source_file"] = str(source_file)
        copied["main_contract_trading_day"] = trading_day
        output.append(copied)
    if not output:
        raise ValueError("No selected main-contract frames to stitch")
    stitched = pd.concat(output, ignore_index=True)
    return stitched.sort_values("timestamp").reset_index(drop=True)
```

- [ ] **Step 4: 运行主力合约测试并确认通过**

运行：`PYTHONPATH=data_preprocess python -m pytest data_preprocess/tests/test_commodity_main_contract.py -q`

预期：通过，5 个测试通过。

- [ ] **Step 5: 提交 Task 2**

运行：

```bash
git add data_preprocess/operator_futures/commodity/main_contract.py data_preprocess/tests/test_commodity_main_contract.py
git commit -m "feat: stitch commodity main contracts"
```

预期：提交成功或由用户明确选择延后。

- [ ] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 3: 商品期货下采样

> **trace:** plan-ready.md → `### Task 3: 商品期货下采样` | tasks.md → ``- [ ] 3.1 实现商品期货参考价下采样：`LastPrice` 优先生成 `mark_price/index_price`，异常时回退 midprice，输出 funding 兼容列并携带 funding disabled 语义。``
> **sync:** tasks.md → ``- [ ] 3.1 实现商品期货参考价下采样：`LastPrice` 优先生成 `mark_price/index_price`，异常时回退 midprice，输出 funding 兼容列并携带 funding disabled 语义。`` | plan-ready.md → `### Task 3: 商品期货下采样`

**文件：**
- 新增：`data_preprocess/operator_futures/commodity/downscale.py`
- 新增：`data_preprocess/operator_futures/commodity/downscale_single_day.py`
- 新增测试：`data_preprocess/tests/test_commodity_downscale.py`

- [ ] **Step 1: 编写失败的下采样测试**

创建 `data_preprocess/tests/test_commodity_downscale.py`，测试覆盖异常最优报价、同秒取最后一条、参考价回退、真实 5 档输出、秒均价、tick rule 和空 quote 窗口报错。测试代码使用 `docs/上海商品交易所/fu2302.csv` 或构造出的最小 DataFrame，必须断言输出中不存在 `ask6_price` 与 `bid25_price`。

```python
import pandas as pd
import pytest

from operator_futures.commodity.downscale import (
    create_second_level_snapshots,
    downscale_base_features,
    downscale_derivative_reference,
    downscale_orderbook,
    downscale_quote_features,
    validate_best_quotes,
)


def test_sample_file_can_create_depth_five_outputs():
    raw = pd.read_csv("docs/上海商品交易所/fu2302.csv").head(20)
    second = create_second_level_snapshots(raw)
    orderbook = downscale_orderbook(second, "5min", depth=5)
    derivative = downscale_derivative_reference(second, "5min", "fu")
    base = downscale_base_features(second, "5min")
    quote = downscale_quote_features(second, "5min")

    assert "ask5_price" in orderbook.columns
    assert "ask6_price" not in orderbook.columns
    assert "mark_price" in derivative.columns
    assert "ntrade_estimated" in base.columns
    assert "nquote" in quote.columns


def test_invalid_best_quote_fails_fast():
    raw = pd.read_csv("docs/上海商品交易所/fu2302.csv").head(2)
    raw.loc[0, "BidPrice1"] = raw.loc[0, "AskPrice1"]
    with pytest.raises(ValueError, match="BidPrice1"):
        validate_best_quotes(raw, "fu2302")


def test_empty_quote_window_fails_fast():
    raw = pd.read_csv("docs/上海商品交易所/fu2302.csv").head(2)
    second = create_second_level_snapshots(raw)
    with pytest.raises(ValueError, match="no quote snapshots"):
        downscale_quote_features(second.iloc[0:0], "5min")
```

- [ ] **Step 2: 运行下采样测试并确认失败**

运行：`PYTHONPATH=data_preprocess python -m pytest data_preprocess/tests/test_commodity_downscale.py -q`

预期：失败，缺少 `operator_futures.commodity.downscale`。

- [ ] **Step 3: 实现下采样核心模块**

创建 `data_preprocess/operator_futures/commodity/downscale.py`，实现以下公开函数：

```python
def validate_best_quotes(df: pd.DataFrame, contract: str) -> None: ...
def create_second_level_snapshots(df: pd.DataFrame) -> pd.DataFrame: ...
def downscale_derivative_reference(second_df: pd.DataFrame, target_freq: str, symbol: str) -> pd.DataFrame: ...
def downscale_orderbook(second_df: pd.DataFrame, target_freq: str, depth: int = 5) -> pd.DataFrame: ...
def downscale_base_features(second_df: pd.DataFrame, target_freq: str) -> pd.DataFrame: ...
def downscale_quote_features(second_df: pd.DataFrame, target_freq: str) -> pd.DataFrame: ...
```

实现要求：
- `create_second_level_snapshots` 使用 `ActionDay + UpdateTime` 生成 timestamp，同秒保留最后一条。
- `downscale_derivative_reference` 输出 `timestamp/symbol/funding_timestamp/funding_rate/index_price/mark_price`，`funding_rate=0`。
- `downscale_orderbook` 只输出 1-5 档。
- `downscale_base_features` 使用 `Turnover.diff()/Volume.diff()`，价格不变计入 flat，不归入 buy/sell。
- `downscale_quote_features` 使用右闭右标聚合，空窗口 fail-fast。

- [ ] **Step 4: 新增单日 CLI 包装**

创建 `data_preprocess/operator_futures/commodity/downscale_single_day.py`：

```python
import argparse
from pathlib import Path

import pandas as pd

from .downscale import (
    create_second_level_snapshots,
    downscale_base_features,
    downscale_derivative_reference,
    downscale_orderbook,
    downscale_quote_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Downscale one commodity futures day")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--symbol", default="fu")
    parser.add_argument("--target_freq", default="5min")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.input)
    second = create_second_level_snapshots(raw)
    downscale_derivative_reference(second, args.target_freq, args.symbol).to_feather(output_dir / "derivative_reference.feather")
    downscale_orderbook(second, args.target_freq, depth=5).to_feather(output_dir / "orderbook_5.feather")
    downscale_base_features(second, args.target_freq).to_feather(output_dir / "base_feature.feather")
    downscale_quote_features(second, args.target_freq).to_feather(output_dir / "quote_feature.feather")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 运行下采样测试并确认通过**

运行：`PYTHONPATH=data_preprocess python -m pytest data_preprocess/tests/test_commodity_downscale.py -q`

预期：商品期货下采样测试全部通过。

- [ ] **Step 6: 运行样例 CLI smoke test**

运行：`PYTHONPATH=data_preprocess python -m operator_futures.commodity.downscale_single_day --input docs/上海商品交易所/fu2302.csv --output_dir /tmp/fu_downscale_smoke --symbol fu --target_freq 5min`

预期：命令退出码为 0，并写出 `/tmp/fu_downscale_smoke/derivative_reference.feather`、`orderbook_5.feather`、`base_feature.feather` 和 `quote_feature.feather`。

- [ ] **Step 7: 提交 Task 3**

运行：

```bash
git add data_preprocess/operator_futures/commodity/downscale.py data_preprocess/operator_futures/commodity/downscale_single_day.py data_preprocess/tests/test_commodity_downscale.py
git commit -m "feat: downscale commodity futures snapshots"
```

预期：提交成功或由用户明确选择延后。

- [ ] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 4: 特征管线适配

> **trace:** plan-ready.md → `### Task 4: 特征管线适配` | tasks.md → ``- [ ] 4.1 更新 cross-section 特征生成，使用可配置 orderbook depth；商品数据跳过 funding、真实逐笔、真实主动买卖和 6-25 档特征。``
> **sync:** tasks.md → ``- [ ] 4.1 更新 cross-section 特征生成，使用可配置 orderbook depth；商品数据跳过 funding、真实逐笔、真实主动买卖和 6-25 档特征。`` | plan-ready.md → `### Task 4: 特征管线适配`

**文件：**
- 修改：`data_preprocess/operator_futures/cross_section/base_feature_util.py`
- 修改：`data_preprocess/operator_futures/cross_section/create_feature.py`
- 修改：`data_preprocess/operator_futures/merge_concat/merge.py`
- 修改：`data_preprocess/operator_futures/time_operator/create_feature_multi_processing.py`
- 修改：`data_preprocess/operator_futures/feature_selection/ic_correlation.py`
- 修改：`data_preprocess/operator_futures/scale_describe_save/scale_save.py`
- 新增测试：`data_preprocess/tests/test_commodity_feature_pipeline.py`

- [ ] **Step 1: 编写失败的 depth-aware 特征测试**

创建 `data_preprocess/tests/test_commodity_feature_pipeline.py`：

```python
import pandas as pd

from operator_futures.commodity.schema import get_reward_execution_columns
from operator_futures.cross_section.base_feature_util import process_snapshot_features
from operator_futures.feature_selection.ic_correlation import calculate_target


def _snapshot():
    row = {"timestamp": pd.Timestamp("2023-01-03 21:05:00")}
    for level in range(1, 6):
        row[f"ask{level}_price"] = 2600 + level
        row[f"ask{level}_size"] = level
        row[f"bid{level}_price"] = 2600 - level
        row[f"bid{level}_size"] = level + 1
    return pd.DataFrame([row])


def test_snapshot_features_accept_depth_five_without_level_25():
    features = process_snapshot_features(_snapshot(), topk=3, depth=5)

    assert "midprice" in features.columns
    assert "buy_volume_oe" in features.columns
    assert "ask5_size_n" in features.columns
    assert "ask6_size_n" not in features.columns


def test_manifest_replaces_first_106_reward_columns():
    reward_columns = get_reward_execution_columns(depth=5)
    assert len(reward_columns) == 26
    assert "ask5_price" in reward_columns
    assert "ask25_price" not in reward_columns


def test_feature_selection_target_remains_price_difference():
    df = pd.DataFrame({"mark_price": [10.0, 12.5, 11.0]})
    target = calculate_target(df, "mark_price", 1)
    assert target.tolist() == [2.5, -1.5]
```

- [ ] **Step 2: 运行测试并确认失败**

运行：`PYTHONPATH=data_preprocess python -m pytest data_preprocess/tests/test_commodity_feature_pipeline.py -q`

预期：失败，因为 `process_snapshot_features` 尚不支持 `depth` 参数。

- [ ] **Step 3: 将 snapshot 特征改为 depth-aware**

修改 `data_preprocess/operator_futures/cross_section/base_feature_util.py` 中的 `process_snapshot_features`：

```python
def process_snapshot_features(df: pd.DataFrame, topk=5, depth=25):
    ask_size_array = df[[f"ask{i}_size" for i in range(1, depth + 1)]].values
    bid_size_array = df[[f"bid{i}_size" for i in range(1, depth + 1)]].values
    ask_price_array = df[[f"ask{i}_price" for i in range(1, depth + 1)]].values
    bid_price_array = df[[f"bid{i}_price" for i in range(1, depth + 1)]].values
    topk = min(topk, depth)
```

同时将函数内剩余硬编码 `25`、`ask25_price`、`bid25_price` 改为 `depth`、`ask{depth}_price`、`bid{depth}_price`。

- [ ] **Step 4: 给 cross-section 脚本新增市场类型和深度参数**

修改 `data_preprocess/operator_futures/cross_section/create_feature.py`：

```python
parser.add_argument("--market_type", type=str, default="crypto_futures", choices=["crypto_futures", "commodity_futures"])
parser.add_argument("--orderbook_depth", type=int, default=25)
```

并将调用改为：

```python
snapshot_feature = process_snapshot_features(snapshot, depth=args.orderbook_depth)
```

- [ ] **Step 5: 替换 scale/save 的前 106 列假设**

修改 `data_preprocess/operator_futures/scale_describe_save/scale_save.py`，增加参数：

```python
parser.add_argument("--market_type", type=str, default="crypto_futures", choices=["crypto_futures", "commodity_futures"])
parser.add_argument("--orderbook_depth", type=int, default=25)
```

将：

```python
reward_features = df.columns[:106]
```

替换为：

```python
if args.market_type == "commodity_futures":
    from operator_futures.commodity.schema import get_reward_execution_columns
    reward_features = [col for col in get_reward_execution_columns(args.orderbook_depth) if col in df.columns]
else:
    reward_features = df.columns[:106]
```

- [ ] **Step 6: 保持 feature selection target 为价差**

确认 `data_preprocess/operator_futures/feature_selection/ic_correlation.py` 保持：

```python
def calculate_target(df, reward_feature, window_length):
    target = df[reward_feature].shift(-window_length) - df[reward_feature]
    target = target[:-window_length]
    return target
```

若 rank/catboost/lasso 变体重复 target 逻辑，也保持同样的价差公式。

- [ ] **Step 7: 运行特征管线测试**

运行：`PYTHONPATH=data_preprocess python -m pytest data_preprocess/tests/test_commodity_feature_pipeline.py -q`

预期：通过，depth=5 特征和 target 计算测试通过。

- [ ] **Step 8: 运行 depth=25 加密货币兼容 smoke test**

运行：

```bash
PYTHONPATH=data_preprocess python - <<'PY'
import pandas as pd
from operator_futures.cross_section.base_feature_util import process_snapshot_features
row = {}
for i in range(1, 26):
    row[f"ask{i}_price"] = 100 + i
    row[f"ask{i}_size"] = i
    row[f"bid{i}_price"] = 100 - i
    row[f"bid{i}_size"] = i
df = pd.DataFrame([row])
out = process_snapshot_features(df, depth=25)
assert "ask25_size_n" in out.columns
print("depth-25 regression smoke passed")
PY
```

预期：输出 `depth-25 regression smoke passed`。

- [ ] **Step 9: 提交 Task 4**

运行：

```bash
git add data_preprocess/operator_futures/cross_section/base_feature_util.py data_preprocess/operator_futures/cross_section/create_feature.py data_preprocess/operator_futures/scale_describe_save/scale_save.py data_preprocess/operator_futures/feature_selection/ic_correlation.py data_preprocess/tests/test_commodity_feature_pipeline.py
git commit -m "feat: make feature pipeline commodity depth aware"
```

预期：提交成功或由用户明确选择延后。

- [ ] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 5: 商品期货环境支持

> **trace:** plan-ready.md → `### Task 5: 商品期货环境支持` | tasks.md → ``- [ ] 5.1 新增商品期货环境初始化，读取 depth=5 数组，关闭 funding countdown 状态，并加载商品手续费配置。``
> **sync:** tasks.md → ``- [ ] 5.1 新增商品期货环境初始化，读取 depth=5 数组，关闭 funding countdown 状态，并加载商品手续费配置。`` | plan-ready.md → `### Task 5: 商品期货环境支持`

**文件：**
- 新增：`FineFT/env/env_initiate/commodity_initiate.py`
- 新增：`FineFT/env/env_class/commodity_env.py`
- 修改：`FineFT/env/env_class/futures_util.py`
- 新增测试：`FineFT/env/test_commodity_env.py`

- [ ] **Step 1: 编写失败的商品环境测试**

创建 `FineFT/env/test_commodity_env.py`：

```python
import pandas as pd

from env.env_initiate.commodity_initiate import initiate_commodity_env


def _df():
    rows = []
    for t, price in enumerate([2600.0, 2601.0]):
        row = {
            "timestamp": pd.Timestamp("2023-01-03 21:00:00") + pd.Timedelta(minutes=5 * t),
            "mark_price": price,
            "feature_a": float(t),
        }
        for level in range(1, 6):
            row[f"ask{level}_price"] = price + level
            row[f"ask{level}_size"] = 10
            row[f"bid{level}_price"] = price - level
            row[f"bid{level}_size"] = 10
        rows.append(row)
    return pd.DataFrame(rows)


def test_commodity_env_reset_has_no_funding_countdown():
    env = initiate_commodity_env(_df(), ["feature_a"], max_holding_number=1, position_choices=3)
    state, info = env.reset()

    assert state.shape == (1,)
    assert "funding_count_down_hour" not in info
    assert "funding_count_down_minute" not in info
    assert len(info["ask_qyts"]) == 5
    assert len(info["bid_qyts"]) == 5


def test_commodity_env_step_uses_configured_fees_and_no_funding():
    env = initiate_commodity_env(
        _df(),
        ["feature_a"],
        max_holding_number=1,
        position_choices=3,
        buy_fee_rate=0.0001,
        sell_fee_rate=0.0003,
    )
    env.reset()
    _, _, _, info = env.step(env.env_map_position_leverage_to_action(1, env.leverage_choices[0]))

    assert env.buy_fee_rate == 0.0001
    assert env.sell_fee_rate == 0.0003
    assert "funding_count_down_hour" not in info
```

- [ ] **Step 2: 运行环境测试并确认失败**

运行：`cd FineFT && PYTHONPATH=. python -m pytest env/test_commodity_env.py -q`

预期：失败，缺少 `env.env_initiate.commodity_initiate`。

- [ ] **Step 3: 新增商品环境类**

创建 `FineFT/env/env_class/commodity_env.py`，显式传入 depth=5 数据，funding array 使用 0，返回 info 时移除 funding countdown 字段。构造函数必须保留 `buy_fee_rate` 和 `sell_fee_rate` 属性，供手续费逻辑使用。

- [ ] **Step 4: 新增商品环境 initializer**

创建 `FineFT/env/env_initiate/commodity_initiate.py`，读取 `ask1-5/bid1-5` 价格与数量列、`mark_price`、`timestamp` 和 state features；缺列时抛出 `ValueError`；默认 `buy_fee_rate=0.0001`、`sell_fee_rate=0.0003`。

- [ ] **Step 5: 增加买卖方向手续费支持**

修改 `FineFT/env/env_class/futures_util.py`，为 wallet-change 相关函数增加可选 `buy_fee_rate` 与 `sell_fee_rate`。当参数为 `None` 时使用原 `commission_rate`，保持加密货币路径行为不变；ask-side 成交使用买入费率，bid-side 成交使用卖出费率。

- [ ] **Step 6: 运行商品环境测试**

运行：`cd FineFT && PYTHONPATH=. python -m pytest env/test_commodity_env.py -q`

预期：通过，商品环境 reset/step 测试通过，info 中不包含 funding countdown。

- [ ] **Step 7: 提交 Task 5**

运行：

```bash
git add FineFT/env/env_initiate/commodity_initiate.py FineFT/env/env_class/commodity_env.py FineFT/env/env_class/futures_util.py FineFT/env/test_commodity_env.py
git commit -m "feat: add commodity futures environment"
```

预期：提交成功或由用户明确选择延后。

- [ ] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 6: 脚本入口与文档

> **trace:** plan-ready.md → `### Task 6: 脚本入口与文档` | tasks.md → ``- [ ] 6.1 新增商品期货预处理脚本入口，串联主力拼接、三条商品下采样、cross-section、merge/concat、time feature、feature selection 和 scale/save。``
> **sync:** tasks.md → ``- [ ] 6.1 新增商品期货预处理脚本入口，串联主力拼接、三条商品下采样、cross-section、merge/concat、time feature、feature selection 和 scale/save。`` | plan-ready.md → `### Task 6: 脚本入口与文档`

**文件：**
- 新增：`data_preprocess/script_preprocess/future_upgraded/commodity/commodity_process.sh`
- 修改：`data_preprocess/README.zh_CN.md`
- 修改：`data_preprocess/README.md`
- 新增：`docs/上海商品交易所/commodity_futures_preprocess.md`

- [ ] **Step 1: 新增商品期货脚本入口**

创建 `data_preprocess/script_preprocess/future_upgraded/commodity/commodity_process.sh`：

```bash
run_commodity_downscale_single_day() {
    local input_file=$1
    local output_dir=$2
    local target_freq=$3
    local symbol=$4
    local root_path=$5

    PYTHONPATH="${root_path}/data_preprocess" python -m operator_futures.commodity.downscale_single_day \
        --input "${input_file}" \
        --output_dir "${output_dir}" \
        --symbol "${symbol}" \
        --target_freq "${target_freq}"
}

run_commodity_smoke_fu() {
    local root_path=$1
    local target_freq=${2:-5min}
    local output_dir="${root_path}/PREPROCESS_DATASET/commodity-futures/fu/${target_freq}/sample"

    mkdir -p "${output_dir}"
    run_commodity_downscale_single_day \
        "${root_path}/docs/上海商品交易所/fu2302.csv" \
        "${output_dir}" \
        "${target_freq}" \
        "fu" \
        "${root_path}"
}
```

- [ ] **Step 2: 检查 shell 语法**

运行：`bash -n data_preprocess/script_preprocess/future_upgraded/commodity/commodity_process.sh`

预期：无输出且退出码为 0。

- [ ] **Step 3: 新增中文商品期货预处理文档**

创建 `docs/上海商品交易所/commodity_futures_preprocess.md`，说明本地五档 CSV 输入、`fu` 输出命名、时间归属、主力合约选择、真实 depth=5、estimated 特征、funding 关闭和燃料油手续费。

- [ ] **Step 4: 更新 README**

在 `data_preprocess/README.zh_CN.md` 和 `data_preprocess/README.md` 的数据预处理概览后补充商品期货章节，指向 `docs/上海商品交易所/commodity_futures_preprocess.md`。

- [ ] **Step 5: 运行文档与脚本检查**

运行：

```bash
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/commodity_process.sh
test -f docs/上海商品交易所/commodity_futures_preprocess.md
rg -n "Commodity Futures|商品期货" data_preprocess/README.md data_preprocess/README.zh_CN.md docs/上海商品交易所/commodity_futures_preprocess.md
```

预期：shell 语法通过，文档存在，`rg` 输出商品期货文档行。

- [ ] **Step 6: 提交 Task 6**

运行：

```bash
git add data_preprocess/script_preprocess/future_upgraded/commodity/commodity_process.sh data_preprocess/README.zh_CN.md data_preprocess/README.md docs/上海商品交易所/commodity_futures_preprocess.md
git commit -m "docs: add commodity futures preprocessing entry point"
```

预期：提交成功或由用户明确选择延后。

- [ ] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 7: 验证与回归

> **trace:** plan-ready.md → `### Task 7: 验证与回归` | tasks.md → ``- [ ] 7.1 运行 `fu` 样例数据的商品期货单元测试和 smoke test。``
> **sync:** tasks.md → ``- [ ] 7.1 运行 `fu` 样例数据的商品期货单元测试和 smoke test。`` | plan-ready.md → `### Task 7: 验证与回归`

**文件：**
- 修改：`openspec/changes/add-commodity-futures-support/tasks.md`
- 新增：`openspec/changes/add-commodity-futures-support/verification.md`

- [ ] **Step 1: 运行所有商品期货数据测试**

运行：

```bash
PYTHONPATH=data_preprocess python -m pytest \
  data_preprocess/tests/test_commodity_config_schema.py \
  data_preprocess/tests/test_commodity_main_contract.py \
  data_preprocess/tests/test_commodity_downscale.py \
  data_preprocess/tests/test_commodity_feature_pipeline.py \
  -q
```

预期：商品期货数据预处理测试全部通过。

- [ ] **Step 2: 运行商品环境测试**

运行：

```bash
cd FineFT
PYTHONPATH=. python -m pytest env/test_commodity_env.py -q
```

预期：商品环境测试通过。

- [ ] **Step 3: 运行商品样例 smoke 命令**

运行：

```bash
source data_preprocess/script_preprocess/future_upgraded/commodity/commodity_process.sh
run_commodity_smoke_fu "$(pwd)" "5min"
test -f PREPROCESS_DATASET/commodity-futures/fu/5min/sample/orderbook_5.feather
```

预期：命令退出码为 0，且 `orderbook_5.feather` 存在。

- [ ] **Step 4: 运行加密货币 depth=25 回归 smoke**

运行：

```bash
PYTHONPATH=data_preprocess python - <<'PY'
import pandas as pd
from operator_futures.cross_section.base_feature_util import process_snapshot_features
row = {}
for i in range(1, 26):
    row[f"ask{i}_price"] = 100 + i
    row[f"ask{i}_size"] = i
    row[f"bid{i}_price"] = 100 - i
    row[f"bid{i}_size"] = i
out = process_snapshot_features(pd.DataFrame([row]), depth=25)
assert "ask25_size_n" in out.columns
print("depth-25 regression smoke passed")
PY
```

预期：输出 `depth-25 regression smoke passed`。

- [ ] **Step 5: 校验 OpenSpec**

运行：`openspec validate add-commodity-futures-support --strict`

预期：输出 `Change 'add-commodity-futures-support' is valid`。

- [ ] **Step 6: 写入验证记录**

创建 `openspec/changes/add-commodity-futures-support/verification.md`：

```markdown
# 验证记录：add-commodity-futures-support

## 已运行命令

- `PYTHONPATH=data_preprocess python -m pytest data_preprocess/tests/test_commodity_config_schema.py data_preprocess/tests/test_commodity_main_contract.py data_preprocess/tests/test_commodity_downscale.py data_preprocess/tests/test_commodity_feature_pipeline.py -q`
- `cd FineFT && PYTHONPATH=. python -m pytest env/test_commodity_env.py -q`
- `source data_preprocess/script_preprocess/future_upgraded/commodity/commodity_process.sh && run_commodity_smoke_fu "$(pwd)" "5min"`
- `openspec validate add-commodity-futures-support --strict`

## 跳过项

- 全年燃料油预处理：需要完整本地原始数据。
- 训练、验证、测试 RL 运行：需要完整处理后数据集和较长训练时间。
- GPU 相关检查：本变更聚焦 CPU 预处理和环境初始化。
```

- [ ] **Step 7: 实现完成后更新 checkbox**

当 Task 1-7 全部完成后，同步更新以下文件的 task-level checkbox：

- `openspec/changes/add-commodity-futures-support/tasks.md`
- `openspec/changes/add-commodity-futures-support/plan-ready.md`
- `docs/superpowers/plans/2026-06-20-add-commodity-futures-support.md`

预期：所有完成任务的 task-level checkbox 为 `[x]`，本计划中已完成 step 也为 `[x]`。

- [ ] **Step 8: 提交验证记录**

运行：

```bash
git add openspec/changes/add-commodity-futures-support/tasks.md openspec/changes/add-commodity-futures-support/plan-ready.md docs/superpowers/plans/2026-06-20-add-commodity-futures-support.md openspec/changes/add-commodity-futures-support/verification.md
git commit -m "test: verify commodity futures support"
```

预期：提交成功或由用户明确选择延后。

- [ ] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）
