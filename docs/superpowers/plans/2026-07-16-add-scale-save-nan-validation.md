# Scale Save NaN Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `scale_save.py` fail fast when the main input DataFrame or final output DataFrame contains floating-point NaN values.

**Architecture:** Keep the change local to the scale-save script and its focused CLI tests. Add one small Polars DataFrame validation helper, call it immediately after reading the input feather and immediately before writing output files, and preserve all existing successful output behavior.

**Tech Stack:** Python, Polars, NumPy, pytest, subprocess CLI tests, OpenSpec.

**Traceability (sddflow):**
- plan-ready: `openspec/changes/add-scale-save-nan-validation/plan-ready.md`
- tasks: `openspec/changes/add-scale-save-nan-validation/tasks.md`
- plan: `docs/superpowers/plans/2026-07-16-add-scale-save-nan-validation.md`

---

### Task 1: Add focused scale-save CLI tests

> **trace:** plan-ready.md → `### Task 1: Add focused scale-save CLI tests` | tasks.md → `- [ ] 1.1 Add focused scale-save CLI tests covering input-stage NaN failure, output-stage NaN failure, and the existing successful output path.`
> **sync:** tasks.md → `- [ ] 1.1 Add focused scale-save CLI tests covering input-stage NaN failure, output-stage NaN failure, and the existing successful output path.` | plan-ready.md → `### Task 1: Add focused scale-save CLI tests`

**Files:**
- Modify: `data_preprocess/tests/test_feature_selection_polars.py`

- [x] **Step 1: Replace the scale-save fixture helper with a configurable version**

In `data_preprocess/tests/test_feature_selection_polars.py`, replace `_write_scale_fixture` with:

```python
def _write_scale_fixture(path: Path, feature_values=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if feature_values is None:
        feature_values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0]
    frame = pl.DataFrame(
        {
            "timestamp": list(range(12)),
            "mark_price": [100.0 + i for i in range(12)],
            "bid1_price": [99.0 + i for i in range(12)],
            "ask1_price": [101.0 + i for i in range(12)],
            "feature_a": feature_values,
        }
    )
    frame.write_ipc(path)
```

- [x] **Step 2: Add path and runner helpers for scale-save CLI tests**

Add these helpers immediately after `_write_scale_fixture`:

```python
def _scale_input_file(tmp_path: Path) -> Path:
    return (
        tmp_path
        / "PREPROCESS_DATASET/commodity-futures/IC_RESULT/fu/5min"
        / "2026-01-05-2026-01-06/df.feather"
    )


def _scale_output_dir(tmp_path: Path) -> Path:
    return tmp_path / "PREPROCESS_DATASET/commodity-futures/SCALE_SAVE/fu/5min/2026-01-05-2026-01-06"


def _write_scale_state_features(input_file: Path) -> None:
    np.save(
        input_file.parent / "state_features.npy",
        np.array(["feature_a"]),
    )


def _run_scale_save_cli(tmp_path: Path, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "data_preprocess/operator_futures/scale_describe_save/scale_save.py",
            "--root_path",
            str(tmp_path),
            "--data_path",
            "PREPROCESS_DATASET/commodity-futures/IC_RESULT",
            "--save_path",
            "PREPROCESS_DATASET/commodity-futures/SCALE_SAVE/",
            "--symbols",
            "fu",
            "--target_freq",
            "5min",
            "--start_date",
            "2026-01-05",
            "--end_date",
            "2026-01-06",
            "--market_type",
            "commodity_futures",
            "--orderbook_depth",
            "5",
            "--ic_choice",
            "ic",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "data_preprocess")},
        check=check,
        text=True,
        capture_output=True,
    )
```

- [x] **Step 3: Update the existing success test to use the helpers**

Replace the body of `test_scale_save_cli_writes_expected_files` with:

```python
def test_scale_save_cli_writes_expected_files(tmp_path):
    input_file = _scale_input_file(tmp_path)
    _write_scale_fixture(input_file)
    _write_scale_state_features(input_file)

    _run_scale_save_cli(tmp_path)

    output_dir = _scale_output_dir(tmp_path)
    assert (output_dir / "df.feather").exists()
    assert (output_dir / "df.csv").exists()
    assert pl.read_csv(output_dir / "df.csv").shape == pl.read_ipc(output_dir / "df.feather").shape
    assert (output_dir / "state_features.npy").exists()
    assert (output_dir / "df_describe.csv").exists()
    df = pl.read_ipc(output_dir / "df.feather")
    assert "symbol" in df.columns
    assert df["symbol"].unique().to_list() == ["fu"]
```

- [x] **Step 4: Add input-stage NaN failure test**

Add this test after `test_scale_save_cli_writes_expected_files`:

```python
def test_scale_save_cli_rejects_input_nan_before_writing_outputs(tmp_path):
    input_file = _scale_input_file(tmp_path)
    _write_scale_fixture(input_file, feature_values=[10.0, 20.0, float("nan"), 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0])
    _write_scale_state_features(input_file)

    result = _run_scale_save_cli(tmp_path, check=False)

    output_dir = _scale_output_dir(tmp_path)
    combined_output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "input" in combined_output
    assert str(input_file) in combined_output
    assert "feature_a" in combined_output
    assert not (output_dir / "df.feather").exists()
    assert not (output_dir / "df.csv").exists()
    assert not (output_dir / "state_features.npy").exists()
    assert not (output_dir / "df_describe.csv").exists()
```

- [x] **Step 5: Add output-stage NaN failure test**

Add this test after the input-stage test:

```python
def test_scale_save_cli_rejects_output_nan_before_writing_outputs(tmp_path):
    input_file = _scale_input_file(tmp_path)
    _write_scale_fixture(input_file, feature_values=[0.0 for _ in range(12)])
    _write_scale_state_features(input_file)

    result = _run_scale_save_cli(tmp_path, check=False)

    output_dir = _scale_output_dir(tmp_path)
    combined_output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "output" in combined_output
    assert str(output_dir / "df.feather") in combined_output
    assert "feature_a" in combined_output
    assert not (output_dir / "df.feather").exists()
    assert not (output_dir / "df.csv").exists()
    assert not (output_dir / "state_features.npy").exists()
    assert not (output_dir / "df_describe.csv").exists()
```

- [x] **Step 6: Run the new failure-path tests and confirm they fail before implementation**

Run:

```bash
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest data_preprocess/tests/test_feature_selection_polars.py::test_scale_save_cli_rejects_input_nan_before_writing_outputs data_preprocess/tests/test_feature_selection_polars.py::test_scale_save_cli_rejects_output_nan_before_writing_outputs -q'
```

Expected: FAIL because `scale_save.py` has not yet added the fail-fast NaN validation.

- [x] **Step 7: Commit the test changes**

Run:

```bash
git add data_preprocess/tests/test_feature_selection_polars.py
git commit -m "test: cover scale-save nan validation"
```

Expected: Commit succeeds if the working tree policy allows commits in this session. Skipped in this run because the git index already contains unrelated staged files.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 2: Add scale-save NaN validation helper

> **trace:** plan-ready.md → `### Task 2: Add scale-save NaN validation helper` | tasks.md → ``- [ ] 1.2 Add a small Polars DataFrame NaN validation helper in `data_preprocess/operator_futures/scale_describe_save/scale_save.py` that reports stage, path, and NaN columns.``
> **sync:** tasks.md → ``- [ ] 1.2 Add a small Polars DataFrame NaN validation helper in `data_preprocess/operator_futures/scale_describe_save/scale_save.py` that reports stage, path, and NaN columns.`` | plan-ready.md → `### Task 2: Add scale-save NaN validation helper`

**Files:**
- Modify: `data_preprocess/operator_futures/scale_describe_save/scale_save.py`

- [x] **Step 1: Add a numeric dtype constant near the logger**

Add this directly after `logger = logging.getLogger(__name__)`:

```python
NAN_CHECK_DTYPES = {
    pl.Float32,
    pl.Float64,
}
```

- [x] **Step 2: Add the helper that finds columns containing NaN**

Add this function after `configure_logging()`:

```python
def columns_with_nan(df: pl.DataFrame) -> list[str]:
    nan_columns = []
    for column, dtype in df.schema.items():
        if dtype not in NAN_CHECK_DTYPES:
            continue
        if df.select(pl.col(column).is_nan().any()).item():
            nan_columns.append(column)
    return nan_columns
```

- [x] **Step 3: Add the validation helper that raises `ValueError`**

Add this function immediately after `columns_with_nan`:

```python
def validate_no_nan(df: pl.DataFrame, *, path: Path, stage: str) -> None:
    nan_columns = columns_with_nan(df)
    if not nan_columns:
        return
    columns = ", ".join(nan_columns)
    raise ValueError(f"NaN detected during {stage} validation for {path}: columns={columns}")
```

- [x] **Step 4: Run a syntax/import check**

Run:

```bash
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && python -m py_compile data_preprocess/operator_futures/scale_describe_save/scale_save.py'
```

Expected: PASS with no output.

- [x] **Step 5: Commit the helper change**

Run:

```bash
git add data_preprocess/operator_futures/scale_describe_save/scale_save.py
git commit -m "feat: add scale-save nan validation helper"
```

Expected: Commit succeeds if the working tree policy allows commits in this session. Skipped in this run because the git index already contains unrelated staged files.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 3: Wire validation into scale-save flow

> **trace:** plan-ready.md → `### Task 3: Wire validation into scale-save flow` | tasks.md → ``- [ ] 1.3 Call the validation helper immediately after reading the main input feather and after building final `out`, before any output file is written.``
> **sync:** tasks.md → ``- [ ] 1.3 Call the validation helper immediately after reading the main input feather and after building final `out`, before any output file is written.`` | plan-ready.md → `### Task 3: Wire validation into scale-save flow`

**Files:**
- Modify: `data_preprocess/operator_futures/scale_describe_save/scale_save.py`

- [x] **Step 1: Store the main input file path and validate after reading**

Replace:

```python
    df = pl.read_ipc(input_dir / f"{df_name}.feather")
```

with:

```python
    input_file = input_dir / f"{df_name}.feather"
    df = pl.read_ipc(input_file)
    validate_no_nan(df, path=input_file, stage="input")
```

- [x] **Step 2: Store the output feather path and validate before logging/writing outputs**

Replace:

```python
    logger.info(
        "Writing scale-save outputs: output_dir=%s rows=%d columns=%d",
        output_dir,
        out.height,
        len(out.columns),
    )
    out.write_ipc(output_dir / "df.feather")
```

with:

```python
    output_file = output_dir / "df.feather"
    validate_no_nan(out, path=output_file, stage="output")
    logger.info(
        "Writing scale-save outputs: output_dir=%s rows=%d columns=%d",
        output_dir,
        out.height,
        len(out.columns),
    )
    out.write_ipc(output_file)
```

- [x] **Step 3: Run the failure-path tests and confirm they pass**

Run:

```bash
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest data_preprocess/tests/test_feature_selection_polars.py::test_scale_save_cli_rejects_input_nan_before_writing_outputs data_preprocess/tests/test_feature_selection_polars.py::test_scale_save_cli_rejects_output_nan_before_writing_outputs -q'
```

Expected: PASS. Both CLI runs fail fast and leave no scale-save output files.

- [x] **Step 4: Run the existing success-path scale-save test**

Run:

```bash
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest data_preprocess/tests/test_feature_selection_polars.py::test_scale_save_cli_writes_expected_files -q'
```

Expected: PASS. The successful path still writes `df.feather`, `df.csv`, `state_features.npy`, and `df_describe.csv`.

- [x] **Step 5: Commit the wiring change**

Run:

```bash
git add data_preprocess/operator_futures/scale_describe_save/scale_save.py data_preprocess/tests/test_feature_selection_polars.py
git commit -m "feat: fail fast on scale-save nan values"
```

Expected: Commit succeeds if the working tree policy allows commits in this session. Skipped in this run because the git index already contains unrelated staged files.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 4: Run focused verification

> **trace:** plan-ready.md → `### Task 4: Run focused verification` | tasks.md → ``- [ ] 1.4 Run focused scale-save tests and a syntax/import check under the `finetf` conda environment.``
> **sync:** tasks.md → ``- [ ] 1.4 Run focused scale-save tests and a syntax/import check under the `finetf` conda environment.`` | plan-ready.md → `### Task 4: Run focused verification`

**Files:**
- Verify: `data_preprocess/tests/test_feature_selection_polars.py`
- Verify: `data_preprocess/operator_futures/scale_describe_save/scale_save.py`
- Verify: `openspec/changes/add-scale-save-nan-validation/specs/operator-futures-polars-preprocessing/spec.md`
- Verify: `openspec/changes/add-scale-save-nan-validation/tasks.md`

- [x] **Step 1: Run all scale-save focused tests**

Run:

```bash
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest data_preprocess/tests/test_feature_selection_polars.py::test_scale_helpers_ignore_nan_like_pandas data_preprocess/tests/test_feature_selection_polars.py::test_scale_helpers_match_reference_for_tiny_std_large_mean_adjustment data_preprocess/tests/test_feature_selection_polars.py::test_scale_save_cli_writes_expected_files data_preprocess/tests/test_feature_selection_polars.py::test_scale_save_cli_rejects_input_nan_before_writing_outputs data_preprocess/tests/test_feature_selection_polars.py::test_scale_save_cli_rejects_output_nan_before_writing_outputs -q'
```

Expected: PASS.

- [x] **Step 2: Run syntax checks for the touched runtime and test files**

Run:

```bash
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && python -m py_compile data_preprocess/operator_futures/scale_describe_save/scale_save.py data_preprocess/tests/test_feature_selection_polars.py'
```

Expected: PASS with no output.

- [x] **Step 3: Run OpenSpec strict validation**

Run:

```bash
openspec validate add-scale-save-nan-validation --strict
```

Expected: PASS with `Change 'add-scale-save-nan-validation' is valid`.

- [x] **Step 4: Check the final diff is scoped**

Run:

```bash
git diff -- data_preprocess/operator_futures/scale_describe_save/scale_save.py data_preprocess/tests/test_feature_selection_polars.py openspec/changes/add-scale-save-nan-validation docs/superpowers/plans/2026-07-16-add-scale-save-nan-validation.md
```

Expected: Diff only contains the NaN validation behavior, focused tests, OpenSpec artifacts, and this implementation plan.

- [x] **Step 5: Commit verification bookkeeping if commits are being used**

Run:

```bash
git status --short
```

Expected: No unexpected modified files outside the scoped paths. This run found unrelated staged files, so commit bookkeeping was intentionally skipped. If a later implementation session is committing changes, commit any remaining scoped documentation or checkbox updates with:

```bash
git add openspec/changes/add-scale-save-nan-validation docs/superpowers/plans/2026-07-16-add-scale-save-nan-validation.md
git commit -m "docs: add scale-save nan validation plan"
```

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）
