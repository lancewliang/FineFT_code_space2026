# 商品期货日期范围支持实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让商品期货燃料油主流程通过 `START_DATE` / `END_DATE` 一次性覆盖跨年训练窗口，并生成单条跨年连续主力数据供后续特征流水线使用。

**Architecture:** 保留既有商品期货数据契约、五档盘口和特征逻辑不变，只把主力拼接入口从单年扩展为日期范围。`main_contract.py` 负责按左闭右开日期范围推导年份、跨年扫描和选择主力；`stitch_main_contract.py` 与 shell 主流程只负责传参和日期范围命名。

**Tech Stack:** Python 3.10、pandas、pytest、bash、现有 `operator_futures.commodity` 商品期货模块、OpenSpec。

**Traceability (sddflow):**
- plan-ready: `openspec/changes/add-commodity-futures-date-range-support/plan-ready.md`
- tasks: `openspec/changes/add-commodity-futures-date-range-support/tasks.md`
- plan: `docs/superpowers/plans/2026-06-21-add-commodity-futures-date-range-support.md`

---

## 文件结构

- 修改：`data_preprocess/operator_futures/commodity/main_contract.py`
  - 负责日期范围解析、年份推导、跨年原始文件扫描、按 `TradingDay` 左闭右开过滤和连续主力拼接。
- 修改：`data_preprocess/operator_futures/commodity/stitch_main_contract.py`
  - 负责 CLI 参数兼容：新路径使用 `--start_date` / `--end_date`，旧路径继续支持 `--year`。
- 修改：`data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh`
  - 负责生成日期范围命名的连续主力原始文件，并把该文件传入下采样流程。
- 修改：`data_preprocess/script_preprocess/future_upgraded/commodity/main.sh`
  - 负责主入口默认参数和 `run_commodity_full_process` 调用顺序。
- 修改测试：`data_preprocess/tests/test_commodity_main_contract.py`
  - 覆盖年份推导、右开 `END_DATE`、跨年 `previous_frames` 连续性和输出过滤。
- 修改测试：`data_preprocess/tests/test_commodity_main_contract_cli.py`
  - 覆盖 CLI 日期范围调用和 shell 脚本文本契约。

---

### Task 1: 主力连续化实现

> **trace:** plan-ready.md → `### Task 1: 主力连续化实现` | tasks.md → ``- [ ] 1.0 主力连续化实现完成（与 `plan-ready.md` Task 1 和 superpowers plan Task 1 同步）``
> **sync:** tasks.md → ``- [ ] 1.0 主力连续化实现完成（与 `plan-ready.md` Task 1 和 superpowers plan Task 1 同步）`` | plan-ready.md → `### Task 1: 主力连续化实现`

**Files:**
- Modify: `data_preprocess/operator_futures/commodity/main_contract.py`
- Modify test: `data_preprocess/tests/test_commodity_main_contract.py`

- [x] **Step 1: 写年份推导失败测试**

在 `data_preprocess/tests/test_commodity_main_contract.py` 的 import 列表中加入新函数，并新增测试：

```python
from operator_futures.commodity.main_contract import (
    build_main_contract_continuous_frame_for_date_range,
    calculate_contract_volume,
    infer_years_for_date_range,
    iter_contract_files,
    normalize_timestamp,
    select_main_contract_for_day,
    stitch_main_contract_frames,
)


def test_infer_years_for_left_closed_right_open_date_range():
    assert infer_years_for_date_range("2023-01-01", "2026-03-01") == [
        "2023",
        "2024",
        "2025",
        "2026",
    ]
    assert infer_years_for_date_range("2023-01-01", "2024-01-01") == ["2023"]
    assert infer_years_for_date_range("2026-02-28", "2026-03-01") == ["2026"]
```

- [x] **Step 2: 运行测试并确认失败**

Run: `conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py::test_infer_years_for_left_closed_right_open_date_range -q`

Expected: FAIL，错误包含 `ImportError` 或 `AttributeError`，说明 `infer_years_for_date_range` 尚未实现。

- [x] **Step 3: 实现日期范围年份推导**

在 `data_preprocess/operator_futures/commodity/main_contract.py` 中补充 `Sequence` import，并新增函数：

```python
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
```

```python
def _parse_date(value: str) -> pd.Timestamp:
    parsed = pd.to_datetime(value, format="%Y-%m-%d", errors="raise")
    return parsed.normalize()


def infer_years_for_date_range(start_date: str, end_date: str) -> List[str]:
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if end <= start:
        raise ValueError(
            f"end_date must be greater than start_date for left-open range: "
            f"{start_date} -> {end_date}"
        )

    last_included = end - pd.Timedelta(days=1)
    return [str(year) for year in range(start.year, last_included.year + 1)]
```

- [x] **Step 4: 运行年份推导测试并确认通过**

Run: `conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py::test_infer_years_for_left_closed_right_open_date_range -q`

Expected: PASS。

- [x] **Step 5: 写跨年主力连续性失败测试**

在 `data_preprocess/tests/test_commodity_main_contract.py` 中新增写文件 helper 和跨年测试：

```python
def _write_contract_file(
    root: Path,
    commodity_name: str,
    year: str,
    month: str,
    trading_day: str,
    contract: str,
    action_day: str,
    volumes,
) -> Path:
    path = root / commodity_name / year / month / trading_day / f"{contract}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    _frame(contract, trading_day, action_day, volumes).to_csv(path, index=False)
    return path


def test_build_main_contract_for_date_range_keeps_previous_frames_across_years(
    tmp_path,
):
    raw_root = tmp_path / "data" / "原始下载"
    _write_contract_file(
        raw_root, "燃料油", "2023", "12", "20231229", "fu2401", "20231228", [0, 50]
    )
    _write_contract_file(
        raw_root, "燃料油", "2023", "12", "20231229", "fu2402", "20231228", [0, 10]
    )
    _write_contract_file(
        raw_root, "燃料油", "2024", "01", "20240102", "fu2401", "20240101", [0, 2]
    )
    _write_contract_file(
        raw_root, "燃料油", "2024", "01", "20240102", "fu2402", "20240101", [0, 20]
    )

    stitched = build_main_contract_continuous_frame_for_date_range(
        raw_root,
        "燃料油",
        "2023-12-29",
        "2024-01-03",
        "fu",
    )

    assert stitched["main_contract_trading_day"].astype(str).tolist() == [
        "20231229",
        "20231229",
        "20240102",
        "20240102",
    ]
    assert stitched["main_contract"].tolist() == [
        "fu2401",
        "fu2401",
        "fu2401",
        "fu2401",
    ]
    assert stitched["main_contract_selection_reason"].tolist() == [
        "current_trading_day_fallback",
        "current_trading_day_fallback",
        "previous_trading_day_volume",
        "previous_trading_day_volume",
    ]
```

- [x] **Step 6: 运行跨年测试并确认失败**

Run: `conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py::test_build_main_contract_for_date_range_keeps_previous_frames_across_years -q`

Expected: FAIL，错误包含 `ImportError`、`AttributeError` 或 `NameError`，说明日期范围构建函数尚未实现。

- [x] **Step 7: 实现跨年扫描与日期范围构建**

在 `data_preprocess/operator_futures/commodity/main_contract.py` 中保留现有单年函数，新增这些函数：

```python
def load_contract_frames_by_trading_day_for_years(
    raw_root: Path, commodity_name: str, years: Sequence[str]
) -> Dict[str, Dict[str, Tuple[pd.DataFrame, Path]]]:
    days: Dict[str, Dict[str, Tuple[pd.DataFrame, Path]]] = {}
    for year in years:
        year_days = load_contract_frames_by_trading_day(
            raw_root, commodity_name, str(year)
        )
        for trading_day, contracts in year_days.items():
            if trading_day in days:
                overlap = sorted(set(days[trading_day]).intersection(contracts))
                if overlap:
                    raise ValueError(
                        f"Duplicate contract data for TradingDay {trading_day}: "
                        f"{overlap}"
                    )
            days.setdefault(trading_day, {}).update(contracts)
    return days


def _trading_day_in_range(
    trading_day: str, start_date: str, end_date: str
) -> bool:
    trading_ts = pd.to_datetime(trading_day, format="%Y%m%d", errors="raise")
    return _parse_date(start_date) <= trading_ts < _parse_date(end_date)


def build_main_contract_continuous_frame_for_date_range(
    raw_root: Path,
    commodity_name: str,
    start_date: str,
    end_date: str,
    symbol: str,
) -> pd.DataFrame:
    years = infer_years_for_date_range(start_date, end_date)
    days = load_contract_frames_by_trading_day_for_years(
        raw_root, commodity_name, years
    )
    selected = []
    previous_frames: Dict[str, pd.DataFrame] = {}
    for trading_day in sorted(days):
        current_items = days[trading_day]
        current_frames = {
            contract: frame for contract, (frame, _) in current_items.items()
        }
        if not _trading_day_in_range(trading_day, start_date, end_date):
            previous_frames = current_frames
            continue

        contract, reason = select_main_contract_for_day(
            previous_frames, current_frames, symbol
        )
        frame, source_file = current_items[contract]
        copied = frame.copy()
        copied["main_contract_selection_reason"] = reason
        selected.append((trading_day, contract, copied, source_file))
        previous_frames = current_frames

    return stitch_main_contract_frames(selected)
```

- [x] **Step 8: 运行主力拼接测试文件**

Run: `conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py -q`

Expected: PASS，既有单年测试和新增跨年测试都通过。

- [x] **Step 9: 提交 Task 1（已跳过：工作区已有用户未提交改动，本轮不自动提交）**

Run:

```bash
git add data_preprocess/operator_futures/commodity/main_contract.py data_preprocess/tests/test_commodity_main_contract.py
git commit -m "feat: support commodity main contract date ranges"
```

Expected: commit 成功。提交前用 `git diff --staged --stat` 确认只包含 Task 1 文件。

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

---

### Task 2: 商品主流程脚本适配

> **trace:** plan-ready.md → `### Task 2: 商品主流程脚本适配` | tasks.md → ``- [ ] 2.0 商品主流程脚本适配完成（与 `plan-ready.md` Task 2 和 superpowers plan Task 2 同步）``
> **sync:** tasks.md → ``- [ ] 2.0 商品主流程脚本适配完成（与 `plan-ready.md` Task 2 和 superpowers plan Task 2 同步）`` | plan-ready.md → `### Task 2: 商品主流程脚本适配`

**Files:**
- Modify: `data_preprocess/operator_futures/commodity/stitch_main_contract.py`
- Modify: `data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh`
- Modify: `data_preprocess/script_preprocess/future_upgraded/commodity/main.sh`
- Modify test: `data_preprocess/tests/test_commodity_main_contract_cli.py`

- [x] **Step 1: 写 CLI 日期范围失败测试**

在 `data_preprocess/tests/test_commodity_main_contract_cli.py` 中新增跨年 CLI 测试：

```python
def test_stitch_main_contract_cli_accepts_date_range(tmp_path):
    raw_root = tmp_path / "data" / "原始下载"
    _write_contract(
        raw_root / "燃料油" / "2023" / "12" / "20231229" / "fu2401.csv",
        "fu2401",
        "20231229",
        "20231228",
        [0, 50],
    )
    _write_contract(
        raw_root / "燃料油" / "2024" / "01" / "20240102" / "fu2401.csv",
        "fu2401",
        "20240102",
        "20240101",
        [0, 2],
    )
    _write_contract(
        raw_root / "燃料油" / "2024" / "01" / "20240102" / "fu2402.csv",
        "fu2402",
        "20240102",
        "20240101",
        [0, 20],
    )

    output = tmp_path / "continuous" / "fu_2023-12-29_2024-01-03.csv"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "operator_futures.commodity.stitch_main_contract",
            "--raw_root",
            str(raw_root),
            "--commodity_name",
            "燃料油",
            "--start_date",
            "2023-12-29",
            "--end_date",
            "2024-01-03",
            "--symbol",
            "fu",
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "data_preprocess")},
        check=True,
    )

    stitched = pd.read_csv(output)
    assert stitched["main_contract_trading_day"].astype(str).tolist() == [
        "20231229",
        "20231229",
        "20240102",
        "20240102",
    ]
    assert stitched["main_contract"].tolist() == [
        "fu2401",
        "fu2401",
        "fu2401",
        "fu2401",
    ]
```

- [x] **Step 2: 运行 CLI 测试并确认失败**

Run: `conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract_cli.py::test_stitch_main_contract_cli_accepts_date_range -q`

Expected: FAIL，错误包含 `unrecognized arguments: --start_date --end_date`。

- [x] **Step 3: 改造 stitch_main_contract.py CLI**

将 `data_preprocess/operator_futures/commodity/stitch_main_contract.py` 调整为以下结构，保留旧 `--year` 路径：

```python
import argparse
from pathlib import Path

from .main_contract import (
    build_main_contract_continuous_frame,
    build_main_contract_continuous_frame_for_date_range,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stitch local commodity futures contracts into a main-contract series"
    )
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--commodity_name", required=True)
    parser.add_argument("--year")
    parser.add_argument("--start_date")
    parser.add_argument("--end_date")
    parser.add_argument("--symbol", default="fu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    has_range = bool(args.start_date or args.end_date)
    if has_range and not (args.start_date and args.end_date):
        parser.error("--start_date and --end_date must be provided together")
    if not has_range and not args.year:
        parser.error("either --year or --start_date/--end_date is required")
    return args


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.start_date and args.end_date:
        stitched = build_main_contract_continuous_frame_for_date_range(
            Path(args.raw_root),
            args.commodity_name,
            args.start_date,
            args.end_date,
            args.symbol,
        )
    else:
        stitched = build_main_contract_continuous_frame(
            Path(args.raw_root), args.commodity_name, args.year, args.symbol
        )
    stitched.to_csv(output, index=False)


if __name__ == "__main__":
    main()
```

- [x] **Step 4: 运行 CLI 测试并确认通过**

Run: `conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract_cli.py::test_stitch_main_contract_cli_accepts_date_range data_preprocess/tests/test_commodity_main_contract_cli.py::test_stitch_main_contract_cli_outputs_continuous_file -q`

Expected: PASS，新日期范围 CLI 和旧 `--year` CLI 都通过。

- [x] **Step 5: 写 shell 文本契约失败测试**

更新 `data_preprocess/tests/test_commodity_main_contract_cli.py` 中脚本测试，明确日期范围命名和无 heredoc Python：

```python
def test_commodity_full_process_shell_exposes_expected_functions():
    script = (
        REPO_ROOT
        / "data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh"
    )
    assert script.exists()
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text(encoding="utf-8")
    assert "run_commodity_stitch_main_contract" in text
    assert "run_commodity_full_process" in text
    assert "run_commodity_merge_process" in text
    assert "PREPROCESS_DATASET/commodity-futures/MERGE_CONCAT" in text
    assert "--market_type commodity_futures" in text
    assert "--orderbook_depth 5" in text
    assert "--start_date" in text
    assert "--end_date" in text
    assert "${symbol}_${start_date}_${end_date}.csv" in text
    assert "run_merge_process " not in text
    assert "python - <<" not in text
    assert "operator_futures.commodity.downscale_continuous_by_trading_day" in text


def test_commodity_main_script_uses_date_range_full_process_entrypoint():
    script = (
        REPO_ROOT
        / "data_preprocess/script_preprocess/future_upgraded/commodity/main.sh"
    )
    assert script.exists()
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text(encoding="utf-8")
    assert "fu_full_process.sh" in text
    assert "run_commodity_full_process" in text
    assert "START_DATE" in text
    assert "END_DATE" in text
    assert "TARGET_FREQ" in text
    assert "COMMODITY_NAME" in text
    assert '"${YEAR}"' not in text
```

- [x] **Step 6: 运行 shell 文本契约测试并确认失败**

Run: `conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract_cli.py::test_commodity_full_process_shell_exposes_expected_functions data_preprocess/tests/test_commodity_main_contract_cli.py::test_commodity_main_script_uses_date_range_full_process_entrypoint -q`

Expected: FAIL，错误显示 shell 仍使用 `${symbol}_${year}.csv` 或 `main.sh` 仍把 `"${YEAR}"` 传给 `run_commodity_full_process`。

- [x] **Step 7: 修改 fu_full_process.sh 的日期范围入口**

在 `data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh` 中替换这两个函数：

```bash
run_commodity_stitch_main_contract() {
    local root_path=$1
    local commodity_name=${2:-燃料油}
    local start_date=$3
    local end_date=$4
    local symbol=${5:-fu}
    local output_dir="${root_path}/PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/${symbol}"
    local output_file="${output_dir}/${symbol}_${start_date}_${end_date}.csv"

    mkdir -p "${output_dir}"
    PYTHONPATH="${root_path}/data_preprocess" python -m operator_futures.commodity.stitch_main_contract \
        --raw_root "${root_path}/data/原始下载" \
        --commodity_name "${commodity_name}" \
        --start_date "${start_date}" \
        --end_date "${end_date}" \
        --symbol "${symbol}" \
        --output "${output_file}"
}

run_commodity_full_process() {
    local root_path=$1
    local start_date=$2
    local end_date=$3
    local target_freq=${4:-5min}
    local symbol=${5:-fu}
    local commodity_name=${6:-燃料油}
    local max_processes=${7:-4}

    run_commodity_stitch_main_contract "$root_path" "$commodity_name" "$start_date" "$end_date" "$symbol"
    local continuous_file="${root_path}/PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/${symbol}/${symbol}_${start_date}_${end_date}.csv"
    run_commodity_downscale_continuous_by_trading_day "$root_path" "$continuous_file" "$target_freq" "$symbol"
    run_commodity_cross_section_process "$start_date" "$end_date" "$max_processes" "$target_freq" "$symbol" "$root_path"
    run_commodity_merge_process "$start_date" "$end_date" "$max_processes" "$target_freq" "$symbol" "$root_path"
    run_commodity_concat_process "$target_freq" "$start_date" "$end_date" "$symbol" "$root_path"
    run_commodity_time_feature "$target_freq" "$start_date" "$end_date" "$symbol" "$root_path"
    run_commodity_merge_and_clean "$target_freq" "$start_date" "$end_date" "$symbol" "$root_path"
    run_commodity_ic_correlation "$target_freq" "$start_date" "$end_date" "$symbol" "$root_path"
    run_commodity_scale_save "$target_freq" "$start_date" "$end_date" "$symbol" "$root_path"
}
```

- [x] **Step 8: 修改 main.sh 的调用参数**

在 `data_preprocess/script_preprocess/future_upgraded/commodity/main.sh` 中保留 `YEAR=${YEAR:-2023}` 作为兼容环境变量也可以删除；主流程调用必须改为日期范围参数：

```bash
run_commodity_full_process \
    "${ROOTPATH}" \
    "${START_DATE}" \
    "${END_DATE}" \
    "${TARGET_FREQ}" \
    "${SYMBOL}" \
    "${COMMODITY_NAME}" \
    "${MAX_PROCESSES}" \
    >"${LOG_DIR}/${SYMBOL}_${TARGET_FREQ}_${START_DATE}_${END_DATE}.log" 2>&1
```

- [x] **Step 9: 运行 CLI 与 shell 测试并确认通过**

Run: `conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract_cli.py -q`

Expected: PASS。

- [x] **Step 10: 运行 shell 语法检查**

Run:

```bash
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/main.sh
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh
```

Expected: 两条命令退出码为 0，无输出。

- [x] **Step 11: 提交 Task 2（已跳过：工作区已有用户未提交改动，本轮不自动提交）**

Run:

```bash
git add \
  data_preprocess/operator_futures/commodity/stitch_main_contract.py \
  data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh \
  data_preprocess/script_preprocess/future_upgraded/commodity/main.sh \
  data_preprocess/tests/test_commodity_main_contract_cli.py
git commit -m "feat: drive commodity preprocessing by date range"
```

Expected: commit 成功。提交前用 `git diff --staged --stat` 确认只包含 Task 2 文件。

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

---

### Task 3: 验证与回归

> **trace:** plan-ready.md → `### Task 3: 验证与回归` | tasks.md → ``- [ ] 3.0 验证与回归完成（与 `plan-ready.md` Task 3 和 superpowers plan Task 3 同步）``
> **sync:** tasks.md → ``- [ ] 3.0 验证与回归完成（与 `plan-ready.md` Task 3 和 superpowers plan Task 3 同步）`` | plan-ready.md → `### Task 3: 验证与回归`

**Files:**
- Modify: `openspec/changes/add-commodity-futures-date-range-support/tasks.md`
- Modify: `openspec/changes/add-commodity-futures-date-range-support/plan-ready.md`
- Modify: `docs/superpowers/plans/2026-06-21-add-commodity-futures-date-range-support.md`

- [x] **Step 1: 运行主力拼接回归测试**

Run: `conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py data_preprocess/tests/test_commodity_main_contract_cli.py -q`

Expected: PASS，输出包含所有测试通过的摘要。

- [x] **Step 2: 运行商品期货相关聚焦回归**

Run:

```bash
conda run -n finetf pytest \
  data_preprocess/tests/test_commodity_config_schema.py \
  data_preprocess/tests/test_commodity_downscale.py \
  data_preprocess/tests/test_commodity_feature_pipeline.py \
  data_preprocess/tests/test_commodity_scripts_docs.py \
  FineFT/tests/env/test_commodity_env.py \
  -q
```

Expected: PASS。如果环境缺少可选依赖导致失败，记录完整错误和缺失依赖名称，不把失败测试标记为已完成。

- [x] **Step 3: 运行 OpenSpec 严格校验**

Run: `openspec validate add-commodity-futures-date-range-support --strict`

Expected: PASS，输出包含 `Totals: 1 passed` 或等价成功信息。

- [x] **Step 4: 运行格式与占位符检查**

Run:

```bash
git diff --check
rg -n "TO""DO|TB""D|YYYY""-MM-DD|add-commodity-futures-support""/specs" \
  openspec/changes/add-commodity-futures-date-range-support \
  docs/superpowers/plans/2026-06-21-add-commodity-futures-date-range-support.md
```

Expected: `git diff --check` 无输出且退出码为 0；`rg` 无匹配且退出码为 1。

- [x] **Step 5: 同步三份文档 checkbox**

在所有验证通过后，同步勾选本 Task 的任务级 checkbox：

```markdown
openspec/changes/add-commodity-futures-date-range-support/tasks.md:
将 `- [ ] 3.0 验证与回归完成（与 `plan-ready.md` Task 3 和 superpowers plan Task 3 同步）`
改为 `- [x] 3.0 验证与回归完成（与 `plan-ready.md` Task 3 和 superpowers plan Task 3 同步）`

openspec/changes/add-commodity-futures-date-range-support/plan-ready.md:
在 `### Task 3: 验证与回归` 下，将任务完成 checkbox 从未完成改为已完成。

docs/superpowers/plans/2026-06-21-add-commodity-futures-date-range-support.md:
在 `### Task 3: 验证与回归` 末尾，将 Task complete checkbox 从未完成改为已完成。
```

- [x] **Step 6: 提交验证状态（已跳过：工作区已有用户未提交改动，本轮不自动提交）**

Run:

```bash
git add \
  openspec/changes/add-commodity-futures-date-range-support/tasks.md \
  openspec/changes/add-commodity-futures-date-range-support/plan-ready.md \
  docs/superpowers/plans/2026-06-21-add-commodity-futures-date-range-support.md
git commit -m "chore: record commodity date range verification"
```

Expected: commit 成功。提交前用 `git diff --staged --stat` 确认只包含 checkbox 状态更新和验证记录。

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）
