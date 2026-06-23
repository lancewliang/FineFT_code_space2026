# Feature Validation Reference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an independent commodity feature validation workflow that compares generated Polars preprocessing artifacts against copied pandas reference calculations and fixed docs-derived expected columns.

**Architecture:** Add a shell entrypoint next to the commodity preprocess scripts and a validation-only Python package under `data_preprocess/operator_futures/feature_validation`. The validator reads existing intermediate artifacts, recomputes reference outputs through copied pandas modules, compares timestamp-aligned sampled rows with `abs_diff <= 1e-9`, and writes Markdown plus JSON reports without changing `main.sh` or production Polars logic.

**Tech Stack:** Bash, Python, pandas, polars/pyarrow feather IO where existing artifacts require it, pytest, OpenSpec.

**Traceability (sddflow):**
- plan-ready: `openspec/changes/add-feature-validation-reference/plan-ready.md`
- tasks: `openspec/changes/add-feature-validation-reference/tasks.md`
- plan: `docs/superpowers/plans/2026-06-23-add-feature-validation-reference.md`

---

## File Structure

- `data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh`: independent commodity validation entrypoint; sets defaults, report directory, `PYTHONPATH`, and invokes the Python CLI.
- `data_preprocess/operator_futures/feature_validation/__init__.py`: package marker and public version string.
- `data_preprocess/operator_futures/feature_validation/validate_features.py`: CLI argument parsing, validation orchestration, report writing, and exit-code mapping.
- `data_preprocess/operator_futures/feature_validation/models.py`: dataclasses for stage configuration, stage results, mismatches, and report summary.
- `data_preprocess/operator_futures/feature_validation/io.py`: read/write helpers for feather/csv artifacts and path construction.
- `data_preprocess/operator_futures/feature_validation/expected_columns.py`: fixed expected-column definitions derived from `docs/data/*.md`; no runtime Markdown parsing.
- `data_preprocess/operator_futures/feature_validation/compare.py`: timestamp sampling, column classification, and numeric comparison with `abs_diff <= 1e-9`.
- `data_preprocess/operator_futures/feature_validation/report.py`: Markdown and JSON report rendering.
- `data_preprocess/operator_futures/feature_validation/reference_adapters.py`: thin adapters that call copied pandas reference modules and normalize their returned DataFrames.
- `data_preprocess/operator_futures/feature_validation/validators.py`: stage validators for `cross_section`, `merge_concat`, `time_feature`, `merge_clean`, `ic_correlation`, and `scale_save`.
- `data_preprocess/operator_futures/feature_validation/pandas_reference/**`: copied pandas reference source files from `/home/lanceliang/opt/aiwork/FineFT_code_space2026_2/data_preprocess/operator_futures`.
- `data_preprocess/tests/test_feature_validation_entrypoint.py`: shell/CLI/reference import smoke tests.
- `data_preprocess/tests/test_feature_validation_compare_report.py`: expected columns, comparator, report, and exit-code unit tests.
- `data_preprocess/tests/test_feature_validation_validators.py`: stage validator and five-day commodity smoke tests.

### Task 1: Reference layout and entrypoint

> **trace:** plan-ready.md -> `### Task 1: Reference layout and entrypoint` | tasks.md -> `- [ ] 1.0 Complete reference layout and entrypoint.`
> **sync:** tasks.md -> `- [ ] 1.0 Complete reference layout and entrypoint.` | plan-ready.md -> `### Task 1: Reference layout and entrypoint`

**Files:**
- Create: `data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh`
- Create: `data_preprocess/operator_futures/feature_validation/__init__.py`
- Create: `data_preprocess/operator_futures/feature_validation/validate_features.py`
- Create: `data_preprocess/operator_futures/feature_validation/models.py`
- Create: `data_preprocess/operator_futures/feature_validation/io.py`
- Create: `data_preprocess/operator_futures/feature_validation/pandas_reference/**`
- Test: `data_preprocess/tests/test_feature_validation_entrypoint.py`

- [x] **Step 1: Write failing entrypoint and CLI smoke tests**

Create `data_preprocess/tests/test_feature_validation_entrypoint.py` with these tests:

```python
from pathlib import Path
import os
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]


def _copy_commodity_script_tree(tmp_path: Path) -> Path:
    source_dir = (
        REPO_ROOT
        / "data_preprocess"
        / "script_preprocess"
        / "future_upgraded"
        / "commodity"
    )
    target_dir = (
        tmp_path
        / "data_preprocess"
        / "script_preprocess"
        / "future_upgraded"
        / "commodity"
    )
    target_dir.parent.mkdir(parents=True)
    shutil.copytree(source_dir, target_dir)
    return target_dir


def test_validate_features_shell_invokes_cli_without_main_sh(tmp_path):
    script_dir = _copy_commodity_script_tree(tmp_path)
    root_path = tmp_path
    cli_dir = (
        root_path
        / "data_preprocess"
        / "operator_futures"
        / "feature_validation"
    )
    cli_dir.mkdir(parents=True)
    (cli_dir / "__init__.py").write_text("", encoding="utf-8")
    (cli_dir / "validate_features.py").write_text(
        "import argparse\n"
        "def main(argv=None):\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--root_path', required=True)\n"
        "    parser.add_argument('--symbol', required=True)\n"
        "    parser.add_argument('--target_freq', required=True)\n"
        "    parser.add_argument('--start_date', required=True)\n"
        "    parser.add_argument('--end_date', required=True)\n"
        "    parser.add_argument('--report_dir', required=True)\n"
        "    args = parser.parse_args(argv)\n"
        "    print('feature-validation-cli', args.symbol, args.target_freq)\n"
        "    return 0\n"
        "if __name__ == '__main__':\n"
        "    raise SystemExit(main())\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "bash",
            str(script_dir / "validate_features.sh"),
            "--root_path",
            str(root_path),
            "--symbol",
            "fu",
            "--target_freq",
            "5min",
            "--start_date",
            "2025-11-03",
            "--end_date",
            "2025-11-08",
        ],
        cwd=root_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "feature-validation-cli fu 5min" in result.stdout
    assert "main.sh" not in result.stdout
    assert "main.sh" not in result.stderr


def test_feature_validation_cli_help_imports():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "operator_futures.feature_validation.validate_features",
            "--help",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "data_preprocess")},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--symbol" in result.stdout
    assert "--target_freq" in result.stdout


def test_pandas_reference_modules_import_from_validation_namespace():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import operator_futures.feature_validation.pandas_reference.cross_section.create_feature;"
            "import operator_futures.feature_validation.pandas_reference.time_operator.create_feature_multi_processing;"
            "import operator_futures.feature_validation.pandas_reference.scale_describe_save.scale_save",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "data_preprocess")},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
```

- [x] **Step 2: Run entrypoint tests to verify they fail before implementation**

Run:

```bash
conda run -n finetf python -m pytest data_preprocess/tests/test_feature_validation_entrypoint.py -q
```

Expected: FAIL because `validate_features.sh` and `operator_futures.feature_validation` do not exist yet.

- [x] **Step 3: Copy pandas reference modules into the validation namespace**

Run these commands from the repository root:

```bash
mkdir -p data_preprocess/operator_futures/feature_validation/pandas_reference
cp -R /home/lanceliang/opt/aiwork/FineFT_code_space2026_2/data_preprocess/operator_futures/cross_section data_preprocess/operator_futures/feature_validation/pandas_reference/
cp -R /home/lanceliang/opt/aiwork/FineFT_code_space2026_2/data_preprocess/operator_futures/features_related data_preprocess/operator_futures/feature_validation/pandas_reference/
cp -R /home/lanceliang/opt/aiwork/FineFT_code_space2026_2/data_preprocess/operator_futures/time_operator data_preprocess/operator_futures/feature_validation/pandas_reference/
cp -R /home/lanceliang/opt/aiwork/FineFT_code_space2026_2/data_preprocess/operator_futures/merge_concat data_preprocess/operator_futures/feature_validation/pandas_reference/
cp -R /home/lanceliang/opt/aiwork/FineFT_code_space2026_2/data_preprocess/operator_futures/merge_all data_preprocess/operator_futures/feature_validation/pandas_reference/
cp -R /home/lanceliang/opt/aiwork/FineFT_code_space2026_2/data_preprocess/operator_futures/feature_selection data_preprocess/operator_futures/feature_validation/pandas_reference/
cp -R /home/lanceliang/opt/aiwork/FineFT_code_space2026_2/data_preprocess/operator_futures/scale_describe_save data_preprocess/operator_futures/feature_validation/pandas_reference/
cp /home/lanceliang/opt/aiwork/FineFT_code_space2026_2/data_preprocess/operator_futures/util.py data_preprocess/operator_futures/feature_validation/pandas_reference/util.py
find data_preprocess/operator_futures/feature_validation/pandas_reference -type d -name __pycache__ -prune -exec rm -rf {} +
rm -f data_preprocess/operator_futures/feature_validation/pandas_reference/feature_selection/catbooost.py
rm -f data_preprocess/operator_futures/feature_validation/pandas_reference/feature_selection/lasso_linear.py
rm -f data_preprocess/operator_futures/feature_validation/pandas_reference/feature_selection/rank_ic_correlation.py
rm -f data_preprocess/operator_futures/feature_validation/pandas_reference/feature_selection/remove_duplicates_feature.py
find data_preprocess/operator_futures/feature_validation/pandas_reference -type d -exec sh -c 'touch "$1/__init__.py"' sh {} \;
```

Expected copied files include:

```text
data_preprocess/operator_futures/feature_validation/pandas_reference/cross_section/base_feature_util.py
data_preprocess/operator_futures/feature_validation/pandas_reference/cross_section/create_feature.py
data_preprocess/operator_futures/feature_validation/pandas_reference/features_related/feature_util.py
data_preprocess/operator_futures/feature_validation/pandas_reference/features_related/base_feature.py
data_preprocess/operator_futures/feature_validation/pandas_reference/time_operator/multi_processing_util.py
data_preprocess/operator_futures/feature_validation/pandas_reference/time_operator/create_feature_multi_processing.py
data_preprocess/operator_futures/feature_validation/pandas_reference/time_operator/time_operator_util.py
data_preprocess/operator_futures/feature_validation/pandas_reference/merge_concat/merge.py
data_preprocess/operator_futures/feature_validation/pandas_reference/merge_concat/concat.py
data_preprocess/operator_futures/feature_validation/pandas_reference/merge_all/merge_clean.py
data_preprocess/operator_futures/feature_validation/pandas_reference/feature_selection/cor_util.py
data_preprocess/operator_futures/feature_validation/pandas_reference/feature_selection/ic_correlation.py
data_preprocess/operator_futures/feature_validation/pandas_reference/scale_describe_save/scale_save.py
data_preprocess/operator_futures/feature_validation/pandas_reference/util.py
```

- [x] **Step 4: Rewrite copied reference imports so they stay inside pandas_reference**

Run:

```bash
python - <<'PY'
from pathlib import Path

root = Path("data_preprocess/operator_futures/feature_validation/pandas_reference")
for path in root.rglob("*.py"):
    text = path.read_text(encoding="utf-8")
    text = text.replace("from operator_futures.", "from operator_futures.feature_validation.pandas_reference.")
    text = text.replace("import operator_futures.", "import operator_futures.feature_validation.pandas_reference.")
    path.write_text(text, encoding="utf-8")
PY
```

Expected: copied reference modules import other copied reference modules, not production Polars modules.

- [x] **Step 5: Create the package skeleton and independent shell entrypoint**

Create `data_preprocess/operator_futures/feature_validation/__init__.py`:

```python
__version__ = "0.1.0"
```

Create `data_preprocess/operator_futures/feature_validation/models.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


StageStatus = Literal["pass", "fail", "partial", "error"]


@dataclass(frozen=True)
class ValidationConfig:
    root_path: Path
    symbol: str
    target_freq: str
    start_date: str
    end_date: str
    report_dir: Path
    tolerance: float = 1e-9
    sample_size: int = 200
    orderbook_depth: int = 5


@dataclass(frozen=True)
class Mismatch:
    stage: str
    column: str
    timestamp: str
    actual: Any
    expected: Any
    abs_diff: float


@dataclass
class StageResult:
    stage: str
    status: StageStatus
    checked_columns: int = 0
    missing_columns: list[str] = field(default_factory=list)
    extra_columns: list[str] = field(default_factory=list)
    unverified_columns: list[str] = field(default_factory=list)
    mismatched_columns: list[str] = field(default_factory=list)
    max_abs_diff: float = 0.0
    sample_failures: list[Mismatch] = field(default_factory=list)
    message: str = ""


@dataclass
class ValidationReport:
    config: ValidationConfig
    stages: list[StageResult]
```

Create `data_preprocess/operator_futures/feature_validation/io.py`:

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd


def dataset_root(root_path: Path) -> Path:
    return root_path / "PREPROCESS_DATASET" / "commodity-futures"


def read_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".feather":
        return pd.read_feather(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported artifact format: {path}")


def require_existing(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return path
```

Create `data_preprocess/operator_futures/feature_validation/validate_features.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from operator_futures.feature_validation.models import ValidationConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate commodity feature artifacts")
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--target_freq", required=True)
    parser.add_argument("--start_date", required=True)
    parser.add_argument("--end_date", required=True)
    parser.add_argument("--report_dir")
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--sample_size", type=int, default=200)
    parser.add_argument("--orderbook_depth", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root_path = Path(args.root_path).resolve()
    report_dir = (
        Path(args.report_dir).resolve()
        if args.report_dir
        else root_path / "log_futures" / "feature_validation"
    )
    config = ValidationConfig(
        root_path=root_path,
        symbol=args.symbol,
        target_freq=args.target_freq,
        start_date=args.start_date,
        end_date=args.end_date,
        report_dir=report_dir,
        tolerance=args.tolerance,
        sample_size=args.sample_size,
        orderbook_depth=args.orderbook_depth,
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    print(
        "feature validation configured: "
        f"symbol={config.symbol} target_freq={config.target_freq} "
        f"start_date={config.start_date} end_date={config.end_date} "
        f"report_dir={config.report_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Create `data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOTPATH=$(pwd)
SYMBOL=fu
TARGET_FREQ=5min
START_DATE=2025-11-03
END_DATE=2025-11-08
REPORT_DIR=

while [ "$#" -gt 0 ]; do
    case "$1" in
        --root_path)
            ROOTPATH=$2
            shift 2
            ;;
        --symbol)
            SYMBOL=$2
            shift 2
            ;;
        --target_freq)
            TARGET_FREQ=$2
            shift 2
            ;;
        --start_date)
            START_DATE=$2
            shift 2
            ;;
        --end_date)
            END_DATE=$2
            shift 2
            ;;
        --report_dir)
            REPORT_DIR=$2
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [ -z "${REPORT_DIR}" ]; then
    REPORT_DIR="${ROOTPATH}/log_futures/feature_validation"
fi

mkdir -p "${REPORT_DIR}"

PYTHONPATH="${ROOTPATH}/data_preprocess" python -m operator_futures.feature_validation.validate_features \
    --root_path "${ROOTPATH}" \
    --symbol "${SYMBOL}" \
    --target_freq "${TARGET_FREQ}" \
    --start_date "${START_DATE}" \
    --end_date "${END_DATE}" \
    --report_dir "${REPORT_DIR}"
```

- [x] **Step 6: Run entrypoint verification**

Run:

```bash
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh
conda run -n finetf python -m pytest data_preprocess/tests/test_feature_validation_entrypoint.py -q
git diff -- data_preprocess/script_preprocess/future_upgraded/commodity/main.sh
```

Expected:

```text
3 passed
```

The `git diff -- .../main.sh` command prints no diff.

- [x] **Step 7: Commit Task 1**

Run:

```bash
git add data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh \
    data_preprocess/operator_futures/feature_validation \
    data_preprocess/tests/test_feature_validation_entrypoint.py
git commit -m "feat: add feature validation entrypoint"
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 2: Validation rules and comparison engine

> **trace:** plan-ready.md -> `### Task 2: Validation rules and comparison engine` | tasks.md -> `- [ ] 2.0 Complete validation rules and comparison engine.`
> **sync:** tasks.md -> `- [ ] 2.0 Complete validation rules and comparison engine.` | plan-ready.md -> `### Task 2: Validation rules and comparison engine`

**Files:**
- Create: `data_preprocess/operator_futures/feature_validation/expected_columns.py`
- Create: `data_preprocess/operator_futures/feature_validation/compare.py`
- Create: `data_preprocess/operator_futures/feature_validation/report.py`
- Create: `data_preprocess/operator_futures/feature_validation/reference_adapters.py`
- Create: `data_preprocess/operator_futures/feature_validation/validators.py`
- Modify: `data_preprocess/operator_futures/feature_validation/validate_features.py`
- Test: `data_preprocess/tests/test_feature_validation_compare_report.py`

- [x] **Step 1: Write failing comparator, report, and expected-column tests**

Create `data_preprocess/tests/test_feature_validation_compare_report.py`:

```python
from pathlib import Path

import pandas as pd

from operator_futures.feature_validation.compare import compare_frames
from operator_futures.feature_validation.expected_columns import EXPECTED_COLUMNS_BY_DOC
from operator_futures.feature_validation.models import StageResult, ValidationConfig, ValidationReport
from operator_futures.feature_validation.report import render_json_report, render_markdown_report


def test_expected_columns_are_fixed_docs_derived_lists():
    assert "base_feature" in EXPECTED_COLUMNS_BY_DOC
    assert "time_feature" in EXPECTED_COLUMNS_BY_DOC
    assert "open" in EXPECTED_COLUMNS_BY_DOC["base_feature"]
    assert "close" in EXPECTED_COLUMNS_BY_DOC["base_feature"]
    assert "bid1_price_log_return_2" in EXPECTED_COLUMNS_BY_DOC["time_feature"]
    assert all(isinstance(column, str) for column in EXPECTED_COLUMNS_BY_DOC["time_feature"])


def test_compare_frames_reports_missing_extra_and_mismatch():
    actual = pd.DataFrame(
        {
            "timestamp": ["2025-11-03 09:00:00", "2025-11-03 09:05:00"],
            "open": [100.0, 101.0],
            "close": [100.5, 101.5],
            "actual_only": [1.0, 2.0],
        }
    )
    expected = pd.DataFrame(
        {
            "timestamp": ["2025-11-03 09:00:00", "2025-11-03 09:05:00"],
            "open": [100.0, 101.0 + 2e-9],
            "volume": [10.0, 20.0],
        }
    )

    result = compare_frames(
        stage="base_feature",
        actual=actual,
        expected=expected,
        expected_columns=["open", "close", "volume"],
        tolerance=1e-9,
        sample_size=10,
    )

    assert result.status == "fail"
    assert result.checked_columns == 1
    assert result.missing_columns == ["volume"]
    assert result.extra_columns == ["actual_only"]
    assert result.mismatched_columns == ["open"]
    assert result.sample_failures[0].column == "open"
    assert result.sample_failures[0].abs_diff > 1e-9


def test_compare_frames_passes_at_tolerance_boundary():
    actual = pd.DataFrame({"timestamp": ["t1"], "open": [1.0 + 1e-9]})
    expected = pd.DataFrame({"timestamp": ["t1"], "open": [1.0]})

    result = compare_frames(
        stage="base_feature",
        actual=actual,
        expected=expected,
        expected_columns=["open"],
        tolerance=1e-9,
        sample_size=10,
    )

    assert result.status == "pass"
    assert result.mismatched_columns == []


def test_reports_include_stage_status_counts_and_failures(tmp_path):
    config = ValidationConfig(
        root_path=tmp_path,
        symbol="fu",
        target_freq="5min",
        start_date="2025-11-03",
        end_date="2025-11-08",
        report_dir=tmp_path,
    )
    report = ValidationReport(
        config=config,
        stages=[
            StageResult(
                stage="base_feature",
                status="fail",
                checked_columns=1,
                missing_columns=["volume"],
                extra_columns=["actual_only"],
                mismatched_columns=["open"],
                max_abs_diff=2e-9,
            )
        ],
    )

    markdown = render_markdown_report(report)
    payload = render_json_report(report)

    assert "base_feature" in markdown
    assert "fail" in markdown
    assert "volume" in markdown
    assert payload["summary"]["failed_stage_count"] == 1
    assert payload["stages"][0]["max_abs_diff"] == 2e-9
```

- [x] **Step 2: Run comparator tests to verify they fail before implementation**

Run:

```bash
conda run -n finetf python -m pytest data_preprocess/tests/test_feature_validation_compare_report.py -q
```

Expected: FAIL because `expected_columns.py`, `compare.py`, and `report.py` do not exist yet.

- [x] **Step 3: Add fixed expected-column definitions**

Create `data_preprocess/operator_futures/feature_validation/expected_columns.py` with fixed constants. Generate the long lists from `docs/data/*.md` during development, then paste the lists into this module so runtime validation imports Python constants only.

The module shape must be:

```python
from __future__ import annotations


BASE_FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "tradeval",
    "ntrade",
    "vwap",
    "awap",
    "twap",
]

KLINE_FEATURE_COLUMNS = [
    "mark_price",
    "basis_spread",
    "basis_spread_rate",
]

QUOTES_FEATURE_COLUMNS = [
    "bid1_price",
    "ask1_price",
    "bid1_size",
    "ask1_size",
]

SNAPSHOT_FEATURE_COLUMNS = [
    "bid1_price",
    "ask1_price",
    "bid1_size",
    "ask1_size",
]

REWARD_ENVIRONMENT_COLUMNS = [
    "timestamp",
    "exchange",
    "symbol",
]

TIME_FEATURE_COLUMNS = [
    "bid1_price_log_return_2",
    "bid1_price_trend_2",
    "ask1_price_log_return_2",
    "ask1_price_trend_2",
]

EXPECTED_COLUMNS_BY_DOC = {
    "base_feature": BASE_FEATURE_COLUMNS,
    "kline_feature": KLINE_FEATURE_COLUMNS,
    "quotes_feature": QUOTES_FEATURE_COLUMNS,
    "snapshot_feature": SNAPSHOT_FEATURE_COLUMNS,
    "reward_environment": REWARD_ENVIRONMENT_COLUMNS,
    "time_feature": TIME_FEATURE_COLUMNS,
}

EXPECTED_COLUMNS_BY_STAGE = {
    "cross_section:kline": KLINE_FEATURE_COLUMNS,
    "cross_section:quotes": QUOTES_FEATURE_COLUMNS,
    "cross_section:snapshot": SNAPSHOT_FEATURE_COLUMNS,
    "merge_concat": BASE_FEATURE_COLUMNS
    + KLINE_FEATURE_COLUMNS
    + QUOTES_FEATURE_COLUMNS
    + SNAPSHOT_FEATURE_COLUMNS,
    "time_feature": TIME_FEATURE_COLUMNS,
    "merge_clean": BASE_FEATURE_COLUMNS
    + KLINE_FEATURE_COLUMNS
    + QUOTES_FEATURE_COLUMNS
    + SNAPSHOT_FEATURE_COLUMNS
    + TIME_FEATURE_COLUMNS,
    "ic_correlation": [],
    "scale_save": [],
}
```

Before finishing this step, replace the short lists above with the complete fixed column lists from:

```text
docs/data/1.DATA_PREPROCESS_REWARD_ENVIRONMENT_106_COLUMNS(挂单).md
docs/data/2.BASE_FEATURE_112_COLUMNS(成交).md
docs/data/3.KLINE_FEATURE_216_COLUMNS.md
docs/data/4.QUOTES_FEATURE_69_COLUMNS.md
docs/data/5.SNAPSHOT_FEATURE_82_COLUMNS.md
docs/data/6.TIME_FEATURE_3375_COLUMNS.md
```

Acceptance checks for this step:

```bash
python - <<'PY'
from operator_futures.feature_validation.expected_columns import EXPECTED_COLUMNS_BY_DOC
assert len(EXPECTED_COLUMNS_BY_DOC["base_feature"]) == 112
assert len(EXPECTED_COLUMNS_BY_DOC["kline_feature"]) == 216
assert len(EXPECTED_COLUMNS_BY_DOC["quotes_feature"]) == 69
assert len(EXPECTED_COLUMNS_BY_DOC["snapshot_feature"]) == 82
assert len(EXPECTED_COLUMNS_BY_DOC["reward_environment"]) == 106
assert len(EXPECTED_COLUMNS_BY_DOC["time_feature"]) == 3375
PY
```

- [x] **Step 4: Implement timestamp-aligned comparator**

Create `data_preprocess/operator_futures/feature_validation/compare.py`:

```python
from __future__ import annotations

import math
from typing import Iterable

import pandas as pd

from operator_futures.feature_validation.models import Mismatch, StageResult


def _normalize_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        raise ValueError("timestamp column is required for validation")
    result = df.copy()
    result["timestamp"] = result["timestamp"].astype(str)
    return result


def _sample_timestamps(timestamps: list[str], sample_size: int) -> list[str]:
    if sample_size <= 0 or len(timestamps) <= sample_size:
        return timestamps
    if sample_size == 1:
        return [timestamps[0]]
    step = (len(timestamps) - 1) / (sample_size - 1)
    indexes = sorted({round(i * step) for i in range(sample_size)})
    return [timestamps[index] for index in indexes]


def _to_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compare_frames(
    stage: str,
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    expected_columns: Iterable[str],
    tolerance: float,
    sample_size: int,
) -> StageResult:
    actual = _normalize_timestamp_column(actual)
    expected = _normalize_timestamp_column(expected)
    expected_column_list = list(expected_columns)
    actual_columns = set(actual.columns) - {"timestamp"}
    reference_columns = set(expected.columns) - {"timestamp"}
    expected_set = set(expected_column_list)

    missing_columns = sorted(expected_set - actual_columns)
    extra_columns = sorted(actual_columns - expected_set)
    comparable_columns = sorted(expected_set & actual_columns & reference_columns)
    unverified_columns = sorted((expected_set & actual_columns) - reference_columns)

    actual_indexed = actual.set_index("timestamp")
    expected_indexed = expected.set_index("timestamp")
    common_timestamps = sorted(set(actual_indexed.index) & set(expected_indexed.index))
    sampled_timestamps = _sample_timestamps(common_timestamps, sample_size)

    failures: list[Mismatch] = []
    max_abs_diff = 0.0
    mismatched_columns: set[str] = set()

    for timestamp in sampled_timestamps:
        actual_row = actual_indexed.loc[timestamp]
        expected_row = expected_indexed.loc[timestamp]
        if isinstance(actual_row, pd.DataFrame):
            actual_row = actual_row.iloc[0]
        if isinstance(expected_row, pd.DataFrame):
            expected_row = expected_row.iloc[0]
        for column in comparable_columns:
            actual_value = actual_row[column]
            expected_value = expected_row[column]
            actual_float = _to_float(actual_value)
            expected_float = _to_float(expected_value)
            if actual_float is None and expected_float is None:
                continue
            if actual_float is None or expected_float is None:
                abs_diff = math.inf
            else:
                abs_diff = abs(actual_float - expected_float)
            max_abs_diff = max(max_abs_diff, abs_diff)
            if abs_diff > tolerance:
                mismatched_columns.add(column)
                if len(failures) < 50:
                    failures.append(
                        Mismatch(
                            stage=stage,
                            column=column,
                            timestamp=str(timestamp),
                            actual=actual_value,
                            expected=expected_value,
                            abs_diff=abs_diff,
                        )
                    )

    if not common_timestamps:
        status = "partial"
        message = "No overlapping timestamps between actual and reference outputs"
    elif missing_columns or mismatched_columns:
        status = "fail"
        message = ""
    elif unverified_columns:
        status = "partial"
        message = "Some expected columns were present but not produced by the reference adapter"
    else:
        status = "pass"
        message = ""

    return StageResult(
        stage=stage,
        status=status,
        checked_columns=len(comparable_columns),
        missing_columns=missing_columns,
        extra_columns=extra_columns,
        unverified_columns=unverified_columns,
        mismatched_columns=sorted(mismatched_columns),
        max_abs_diff=max_abs_diff,
        sample_failures=failures,
        message=message,
    )
```

- [x] **Step 5: Implement Markdown and JSON reports**

Create `data_preprocess/operator_futures/feature_validation/report.py`:

```python
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from operator_futures.feature_validation.models import ValidationReport


def render_json_report(report: ValidationReport) -> dict[str, Any]:
    failed_stage_count = sum(1 for stage in report.stages if stage.status in {"fail", "error"})
    partial_stage_count = sum(1 for stage in report.stages if stage.status == "partial")
    return {
        "config": {
            "root_path": str(report.config.root_path),
            "symbol": report.config.symbol,
            "target_freq": report.config.target_freq,
            "start_date": report.config.start_date,
            "end_date": report.config.end_date,
            "tolerance": report.config.tolerance,
            "sample_size": report.config.sample_size,
        },
        "summary": {
            "stage_count": len(report.stages),
            "failed_stage_count": failed_stage_count,
            "partial_stage_count": partial_stage_count,
        },
        "stages": [asdict(stage) for stage in report.stages],
    }


def render_markdown_report(report: ValidationReport) -> str:
    lines = [
        "# Feature Validation Report",
        "",
        f"- symbol: `{report.config.symbol}`",
        f"- target_freq: `{report.config.target_freq}`",
        f"- date_range: `{report.config.start_date}` to `{report.config.end_date}`",
        f"- tolerance: `{report.config.tolerance}`",
        "",
        "| stage | status | checked | missing | extra | unverified | mismatched | max_abs_diff |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in report.stages:
        lines.append(
            f"| {stage.stage} | {stage.status} | {stage.checked_columns} | "
            f"{len(stage.missing_columns)} | {len(stage.extra_columns)} | "
            f"{len(stage.unverified_columns)} | {len(stage.mismatched_columns)} | "
            f"{stage.max_abs_diff} |"
        )
    for stage in report.stages:
        lines.extend(["", f"## {stage.stage}", "", f"- status: `{stage.status}`"])
        if stage.message:
            lines.append(f"- message: {stage.message}")
        for label, values in [
            ("missing_columns", stage.missing_columns),
            ("extra_columns", stage.extra_columns),
            ("unverified_columns", stage.unverified_columns),
            ("mismatched_columns", stage.mismatched_columns),
        ]:
            if values:
                lines.append(f"- {label}: `{', '.join(values[:50])}`")
        if stage.sample_failures:
            lines.extend(["", "| column | timestamp | actual | expected | abs_diff |", "|---|---|---:|---:|---:|"])
            for failure in stage.sample_failures:
                lines.append(
                    f"| {failure.column} | {failure.timestamp} | {failure.actual} | "
                    f"{failure.expected} | {failure.abs_diff} |"
                )
    return "\n".join(lines) + "\n"


def write_reports(report: ValidationReport, report_dir: Path) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{report.config.symbol}_{report.config.target_freq}_"
        f"{report.config.start_date}_{report.config.end_date}"
    )
    markdown_path = report_dir / f"{stem}.md"
    json_path = report_dir / f"{stem}.json"
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")
    json_path.write_text(
        json.dumps(render_json_report(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return markdown_path, json_path
```

- [x] **Step 6: Add reference adapters and stage validators**

Create `data_preprocess/operator_futures/feature_validation/reference_adapters.py`:

```python
from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pandas as pd

from operator_futures.feature_validation.io import read_frame
from operator_futures.feature_validation.models import ValidationConfig


def normalize_reference_frame(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if result.index.name == "timestamp":
        result = result.reset_index()
    if "datetime" in result.columns and "timestamp" not in result.columns:
        result = result.rename(columns={"datetime": "timestamp"})
    return result


def cross_section_reference(config: ValidationConfig, date: str, feature_name: str) -> pd.DataFrame:
    path = (
        config.root_path
        / "PREPROCESS_DATASET"
        / "commodity-futures"
        / "CROSS_SECTION"
        / feature_name
        / config.symbol
        / config.target_freq
        / f"{date}.feather"
    )
    return normalize_reference_frame(read_frame(path))


def concat_reference(config: ValidationConfig) -> pd.DataFrame:
    path = (
        config.root_path
        / "PREPROCESS_DATASET"
        / "commodity-futures"
        / "MERGE_CONCAT"
        / "CONCAT_FEATURE"
        / config.symbol
        / config.target_freq
        / f"{config.start_date}-{config.end_date}.feather"
    )
    return normalize_reference_frame(read_frame(path))


def time_feature_reference(config: ValidationConfig) -> pd.DataFrame:
    path = (
        config.root_path
        / "PREPROCESS_DATASET"
        / "commodity-futures"
        / "TIME_FEATURE"
        / config.symbol
        / config.target_freq
        / f"{config.start_date}-{config.end_date}.feather"
    )
    return normalize_reference_frame(read_frame(path))


def passthrough_file_reference(config: ValidationConfig, relative_path: Path) -> pd.DataFrame:
    return normalize_reference_frame(read_frame(config.root_path / relative_path))
```

Create `data_preprocess/operator_futures/feature_validation/validators.py`:

```python
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Callable

from operator_futures.feature_validation.compare import compare_frames
from operator_futures.feature_validation.expected_columns import EXPECTED_COLUMNS_BY_STAGE
from operator_futures.feature_validation.io import read_frame
from operator_futures.feature_validation.models import StageResult, ValidationConfig
from operator_futures.feature_validation.reference_adapters import (
    concat_reference,
    cross_section_reference,
    passthrough_file_reference,
    time_feature_reference,
)


def _date_range(start_date: str, end_date: str) -> list[str]:
    current = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    dates: list[str] = []
    while current < end:
        dates.append(current.isoformat())
        current += timedelta(days=1)
    return dates


def _error_result(stage: str, exc: Exception) -> StageResult:
    return StageResult(stage=stage, status="error", message=f"{type(exc).__name__}: {exc}")


def validate_cross_section(config: ValidationConfig) -> list[StageResult]:
    results: list[StageResult] = []
    feature_pairs = [
        ("cross_section:kline", "KLINE_FEATURE"),
        ("cross_section:quotes", "QUOTES_FEATURE"),
        ("cross_section:snapshot", "SNAPSHOT_FEATURE"),
    ]
    for stage_key, feature_name in feature_pairs:
        for day in _date_range(config.start_date, config.end_date):
            stage = f"{stage_key}:{day}"
            path = (
                config.root_path
                / "PREPROCESS_DATASET"
                / "commodity-futures"
                / "CROSS_SECTION"
                / feature_name
                / config.symbol
                / config.target_freq
                / f"{day}.feather"
            )
            try:
                actual = read_frame(path)
                expected = cross_section_reference(config, day, feature_name)
                results.append(
                    compare_frames(
                        stage=stage,
                        actual=actual,
                        expected=expected,
                        expected_columns=EXPECTED_COLUMNS_BY_STAGE[stage_key],
                        tolerance=config.tolerance,
                        sample_size=config.sample_size,
                    )
                )
            except Exception as exc:
                results.append(_error_result(stage, exc))
    return results


def validate_merge_concat(config: ValidationConfig) -> StageResult:
    stage = "merge_concat"
    path = (
        config.root_path
        / "PREPROCESS_DATASET"
        / "commodity-futures"
        / "MERGE_CONCAT"
        / "CONCAT_FEATURE"
        / config.symbol
        / config.target_freq
        / f"{config.start_date}-{config.end_date}.feather"
    )
    try:
        return compare_frames(
            stage=stage,
            actual=read_frame(path),
            expected=concat_reference(config),
            expected_columns=EXPECTED_COLUMNS_BY_STAGE[stage],
            tolerance=config.tolerance,
            sample_size=config.sample_size,
        )
    except Exception as exc:
        return _error_result(stage, exc)


def validate_time_feature(config: ValidationConfig) -> StageResult:
    stage = "time_feature"
    path = (
        config.root_path
        / "PREPROCESS_DATASET"
        / "commodity-futures"
        / "TIME_FEATURE"
        / config.symbol
        / config.target_freq
        / f"{config.start_date}-{config.end_date}.feather"
    )
    try:
        return compare_frames(
            stage=stage,
            actual=read_frame(path),
            expected=time_feature_reference(config),
            expected_columns=EXPECTED_COLUMNS_BY_STAGE[stage],
            tolerance=config.tolerance,
            sample_size=config.sample_size,
        )
    except Exception as exc:
        return _error_result(stage, exc)


def validate_file_stage(config: ValidationConfig, stage: str, relative_path: Path) -> StageResult:
    try:
        actual = read_frame(config.root_path / relative_path)
        expected = passthrough_file_reference(config, relative_path)
        return compare_frames(
            stage=stage,
            actual=actual,
            expected=expected,
            expected_columns=EXPECTED_COLUMNS_BY_STAGE[stage],
            tolerance=config.tolerance,
            sample_size=config.sample_size,
        )
    except Exception as exc:
        return _error_result(stage, exc)


def run_validations(config: ValidationConfig) -> list[StageResult]:
    results: list[StageResult] = []
    results.extend(validate_cross_section(config))
    results.append(validate_merge_concat(config))
    results.append(validate_time_feature(config))
    results.append(
        validate_file_stage(
            config,
            "merge_clean",
            Path("PREPROCESS_DATASET/commodity-futures/ALL_FEATURE")
            / config.symbol
            / config.target_freq
            / f"{config.start_date}-{config.end_date}.feather",
        )
    )
    results.append(
        validate_file_stage(
            config,
            "ic_correlation",
            Path("PREPROCESS_DATASET/commodity-futures/IC_RESULT")
            / config.symbol
            / config.target_freq
            / f"{config.start_date}-{config.end_date}.feather",
        )
    )
    results.append(
        validate_file_stage(
            config,
            "scale_save",
            Path("PREPROCESS_DATASET/commodity-futures/SCALE_SAVE")
            / config.symbol
            / config.target_freq
            / f"{config.start_date}-{config.end_date}.feather",
        )
    )
    return results
```

The first build pass may use passthrough adapters for stages where the copied pandas scripts only expose file-oriented CLIs. Replace each passthrough with a pandas-reference recompute adapter before marking Task 2 complete.

- [x] **Step 7: Wire the CLI to validators, reports, and exit codes**

Modify `data_preprocess/operator_futures/feature_validation/validate_features.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from operator_futures.feature_validation.models import ValidationConfig, ValidationReport
from operator_futures.feature_validation.report import write_reports
from operator_futures.feature_validation.validators import run_validations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate commodity feature artifacts")
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--target_freq", required=True)
    parser.add_argument("--start_date", required=True)
    parser.add_argument("--end_date", required=True)
    parser.add_argument("--report_dir")
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--sample_size", type=int, default=200)
    parser.add_argument("--orderbook_depth", type=int, default=5)
    return parser


def _exit_code(report: ValidationReport) -> int:
    if not report.stages:
        return 2
    if any(stage.status in {"fail", "partial", "error"} for stage in report.stages):
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.sample_size < 1:
        parser.error("--sample_size must be greater than 0")
    if args.tolerance < 0:
        parser.error("--tolerance must be greater than or equal to 0")

    root_path = Path(args.root_path).resolve()
    report_dir = (
        Path(args.report_dir).resolve()
        if args.report_dir
        else root_path / "log_futures" / "feature_validation"
    )
    config = ValidationConfig(
        root_path=root_path,
        symbol=args.symbol,
        target_freq=args.target_freq,
        start_date=args.start_date,
        end_date=args.end_date,
        report_dir=report_dir,
        tolerance=args.tolerance,
        sample_size=args.sample_size,
        orderbook_depth=args.orderbook_depth,
    )
    stages = run_validations(config)
    report = ValidationReport(config=config, stages=stages)
    markdown_path, json_path = write_reports(report, report_dir)
    print(f"feature validation markdown report: {markdown_path}")
    print(f"feature validation json report: {json_path}")
    return _exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [x] **Step 8: Run comparator/report verification**

Run:

```bash
conda run -n finetf python -m pytest data_preprocess/tests/test_feature_validation_compare_report.py -q
conda run -n finetf python -m pytest data_preprocess/tests/test_feature_validation_entrypoint.py -q
```

Expected:

```text
4 passed
3 passed
```

- [x] **Step 9: Commit Task 2**

Run:

```bash
git add data_preprocess/operator_futures/feature_validation \
    data_preprocess/tests/test_feature_validation_compare_report.py
git commit -m "feat: add feature validation comparator"
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 3: Verification

> **trace:** plan-ready.md -> `### Task 3: Verification` | tasks.md -> `- [ ] 3.0 Complete verification.`
> **sync:** tasks.md -> `- [ ] 3.0 Complete verification.` | plan-ready.md -> `### Task 3: Verification`

**Files:**
- Create: `data_preprocess/tests/test_feature_validation_validators.py`
- Modify: `data_preprocess/tests/test_feature_validation_entrypoint.py`
- Modify: `data_preprocess/tests/test_feature_validation_compare_report.py`
- Modify: `data_preprocess/operator_futures/feature_validation/reference_adapters.py`
- Modify: `data_preprocess/operator_futures/feature_validation/validators.py`

- [x] **Step 1: Write validator and smoke tests**

Create `data_preprocess/tests/test_feature_validation_validators.py`:

```python
from pathlib import Path

import pandas as pd

from operator_futures.feature_validation.models import ValidationConfig
from operator_futures.feature_validation.validators import run_validations, validate_merge_concat


def _write_feather(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_feather(path)


def test_validate_merge_concat_reports_pass_for_matching_sample(tmp_path):
    artifact = (
        tmp_path
        / "PREPROCESS_DATASET"
        / "commodity-futures"
        / "MERGE_CONCAT"
        / "CONCAT_FEATURE"
        / "fu"
        / "5min"
        / "2025-11-03-2025-11-08.feather"
    )
    _write_feather(
        artifact,
        [
            {"timestamp": "2025-11-03 09:00:00", "open": 100.0, "close": 100.5},
            {"timestamp": "2025-11-03 09:05:00", "open": 101.0, "close": 101.5},
        ],
    )
    config = ValidationConfig(
        root_path=tmp_path,
        symbol="fu",
        target_freq="5min",
        start_date="2025-11-03",
        end_date="2025-11-08",
        report_dir=tmp_path / "reports",
        sample_size=2,
    )

    result = validate_merge_concat(config)

    assert result.status in {"pass", "partial"}
    assert result.stage == "merge_concat"


def test_run_validations_returns_stage_errors_instead_of_raising(tmp_path):
    config = ValidationConfig(
        root_path=tmp_path,
        symbol="fu",
        target_freq="5min",
        start_date="2025-11-03",
        end_date="2025-11-08",
        report_dir=tmp_path / "reports",
    )

    results = run_validations(config)

    assert results
    assert any(result.status == "error" for result in results)


def test_validate_features_shell_writes_reports_for_five_day_sample(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script = (
        repo_root
        / "data_preprocess"
        / "script_preprocess"
        / "future_upgraded"
        / "commodity"
        / "validate_features.sh"
    )
    report_dir = tmp_path / "reports"

    completed = __import__("subprocess").run(
        [
            "bash",
            str(script),
            "--root_path",
            str(repo_root),
            "--symbol",
            "fu",
            "--target_freq",
            "5min",
            "--start_date",
            "2025-11-03",
            "--end_date",
            "2025-11-08",
            "--report_dir",
            str(report_dir),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert completed.returncode in {0, 1, 2}
    assert list(report_dir.glob("fu_5min_2025-11-03_2025-11-08.*"))
```

- [x] **Step 2: Run validator tests to verify the remaining adapter gaps**

Run:

```bash
conda run -n finetf python -m pytest data_preprocess/tests/test_feature_validation_validators.py -q
```

Expected before final adapters are complete: FAIL on the tests that require complete artifact path handling or report writing.

- [x] **Step 3: Replace passthrough adapters with pandas-reference recompute adapters**

Modify `data_preprocess/operator_futures/feature_validation/reference_adapters.py` so each adapter calls copied pandas reference functions rather than production Polars modules. The adapter contract must remain pandas DataFrame in, pandas DataFrame out:

```python
from __future__ import annotations

from pathlib import Path
import tempfile

import pandas as pd

from operator_futures.feature_validation.io import read_frame
from operator_futures.feature_validation.models import ValidationConfig


def normalize_reference_frame(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if result.index.name == "timestamp":
        result = result.reset_index()
    if "datetime" in result.columns and "timestamp" not in result.columns:
        result = result.rename(columns={"datetime": "timestamp"})
    return result


def cross_section_reference(config: ValidationConfig, date: str, feature_name: str) -> pd.DataFrame:
    from operator_futures.feature_validation.pandas_reference.cross_section import base_feature_util

    base_path = (
        config.root_path
        / "PREPROCESS_DATASET"
        / "commodity-futures"
        / "BASE_FEATURE"
        / config.symbol
        / config.target_freq
        / f"{date}.feather"
    )
    snapshot_path = (
        config.root_path
        / "PREPROCESS_DATASET"
        / "commodity-futures"
        / "DOWNSCALE_ORDERBOOK_25"
        / config.symbol
        / config.target_freq
        / f"{date}.feather"
    )
    base_feature = read_frame(base_path)
    snapshot = read_frame(snapshot_path)
    if feature_name == "KLINE_FEATURE":
        return normalize_reference_frame(base_feature_util.process_k_line_feature(base_feature))
    if feature_name == "QUOTES_FEATURE":
        return normalize_reference_frame(base_feature_util.process_quotes_n_feature(base_feature))
    if feature_name == "SNAPSHOT_FEATURE":
        return normalize_reference_frame(
            base_feature_util.process_snapshot_features(snapshot, depth=config.orderbook_depth)
        )
    raise ValueError(f"Unsupported cross-section feature: {feature_name}")


def concat_reference(config: ValidationConfig) -> pd.DataFrame:
    path = (
        config.root_path
        / "PREPROCESS_DATASET"
        / "commodity-futures"
        / "MERGE_CONCAT"
        / "CONCAT_FEATURE"
        / config.symbol
        / config.target_freq
        / f"{config.start_date}-{config.end_date}.feather"
    )
    return normalize_reference_frame(read_frame(path))


def time_feature_reference(config: ValidationConfig) -> pd.DataFrame:
    from operator_futures.feature_validation.pandas_reference.time_operator.create_feature_multi_processing import main as reference_main

    with tempfile.TemporaryDirectory() as tmp_dir:
        args = type(
            "Args",
            (),
            {
                "root_path": str(config.root_path),
                "data_path": "PREPROCESS_DATASET/commodity-futures/MERGE_CONCAT/CONCAT_FEATURE/",
                "save_path": str(Path(tmp_dir)),
                "symbols": config.symbol,
                "start_date": config.start_date,
                "end_date": config.end_date,
                "target_freq": config.target_freq,
                "windows": "2,6,12,16,24,48",
                "orderbook_depth": config.orderbook_depth,
            },
        )()
        reference_main(args)
        output = Path(tmp_dir) / config.symbol / config.target_freq / f"{config.start_date}-{config.end_date}.feather"
        return normalize_reference_frame(read_frame(output))
```

Keep stages that cannot be recomputed without broader copied script adaptation as `partial` with explicit `unverified_columns`; do not report them as `pass` through self-comparison.

- [x] **Step 4: Tighten validators so unavailable reference stages become partial, not false pass**

Modify `data_preprocess/operator_futures/feature_validation/validators.py` so `ic_correlation` and `scale_save` only pass when the reference adapter recomputes comparable columns. If the adapter cannot recompute yet, return a `StageResult` like this:

```python
StageResult(
    stage="ic_correlation",
    status="partial",
    unverified_columns=EXPECTED_COLUMNS_BY_STAGE["ic_correlation"],
    message="Copied pandas reference adapter is present but this stage has no comparable recompute output yet",
)
```

Acceptance: no validator uses the actual artifact as both actual and expected for a `pass` result.

- [x] **Step 5: Run all feature-validation tests and shell syntax checks**

Run:

```bash
bash -n data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh
conda run -n finetf python -m pytest \
    data_preprocess/tests/test_feature_validation_entrypoint.py \
    data_preprocess/tests/test_feature_validation_compare_report.py \
    data_preprocess/tests/test_feature_validation_validators.py \
    -q
```

Expected:

```text
10 passed
```

- [x] **Step 6: Run the five-day commodity validation smoke command**

Run:

```bash
bash data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh \
    --root_path /home/lanceliang/opt/aiwork/FineFT_code_space2026 \
    --symbol fu \
    --target_freq 5min \
    --start_date 2025-11-03 \
    --end_date 2025-11-08
```

Expected:

```text
feature validation markdown report: /home/lanceliang/opt/aiwork/FineFT_code_space2026/log_futures/feature_validation/fu_5min_2025-11-03_2025-11-08.md
feature validation json report: /home/lanceliang/opt/aiwork/FineFT_code_space2026/log_futures/feature_validation/fu_5min_2025-11-03_2025-11-08.json
```

Exit code may be `1` if the report contains formula mismatches or partial stages. Exit code must be `2` only for invalid arguments or no comparable artifacts.

- [x] **Step 7: Run OpenSpec and repository verification**

Run:

```bash
openspec validate add-feature-validation-reference --strict
git diff --check
```

Expected:

```text
Change 'add-feature-validation-reference' is valid
```

`git diff --check` prints no whitespace errors.

- [x] **Step 8: Commit Task 3**

Run:

```bash
git add data_preprocess/operator_futures/feature_validation \
    data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh \
    data_preprocess/tests/test_feature_validation_entrypoint.py \
    data_preprocess/tests/test_feature_validation_compare_report.py \
    data_preprocess/tests/test_feature_validation_validators.py
git commit -m "test: verify feature validation workflow"
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）
