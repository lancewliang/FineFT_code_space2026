# Refactor Multi-Contract Scale Save Robust Scaler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace commodity split-stage multi-contract scale-save with a train-only robust scaler while preserving the existing output layout.

**Architecture:** `muti_contract_scale_save.py` remains the only commodity split-stage scale-save entry point. It fits one robust scaler from all train split rows, writes an auditable manifest, applies the same scaler to train/valid/test, and writes diagnostics for clipping and output accounting.

**Tech Stack:** Python 3, standard-library `dataclasses`, `json`, `pathlib`; existing NumPy, Polars, Pytest, and OpenSpec.

**Traceability (sddflow):**
- plan-ready: `openspec/changes/refactor-multi-contract-scale-save-robust-scaler/plan-ready.md`
- tasks: `openspec/changes/refactor-multi-contract-scale-save-robust-scaler/tasks.md`
- plan: `docs/superpowers/plans/2026-07-23-refactor-multi-contract-scale-save-robust-scaler.md`

---

### Task 1: Add robust scale-save regression tests

> **trace:** plan-ready.md -> ### Task 1: Add robust scale-save regression tests | tasks.md -> - [ ] 1.1 Add focused regression tests for train-only robust scaling in `data_preprocess/tests/test_feature_selection_polars.py`, covering train-fit once, consistent apply across train/valid/test, clip behavior, manifest/diagnostics files, and fail-fast behavior for missing train inputs, invalid clip bounds, and missing selected feature columns.
> **sync:** tasks.md -> - [ ] 1.1 Add focused regression tests for train-only robust scaling in `data_preprocess/tests/test_feature_selection_polars.py`, covering train-fit once, consistent apply across train/valid/test, clip behavior, manifest/diagnostics files, and fail-fast behavior for missing train inputs, invalid clip bounds, and missing selected feature columns. | plan-ready.md -> ### Task 1: Add robust scale-save regression tests

**Files:**
- Modify: `data_preprocess/tests/test_feature_selection_polars.py`

- [x] **Step 1: Extend the multi-contract CLI test helper to accept extra CLI args**

In `data_preprocess/tests/test_feature_selection_polars.py`, update `_run_multi_contract_scale_save_cli` so tests can pass clip options:

```python
def _run_multi_contract_scale_save_cli(
    tmp_path: Path,
    *,
    check: bool = True,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "data_preprocess/operator_futures/scale_describe_save/muti_contract_scale_save.py",
        "--root_path",
        str(tmp_path),
        "--data_path",
        "PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST",
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
        "--feature_list_path",
        "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/5min/fu/train/state_features.npy",
    ]
    if extra_args:
        command.extend(extra_args)
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "data_preprocess")},
        check=check,
        text=True,
        capture_output=True,
    )
```

- [x] **Step 2: Add a fixture helper for price-core and clip test data**

Add this helper near `_write_scale_fixture`:

```python
def _write_multi_scale_fixture(
    path: Path,
    *,
    wap_values: list[float],
    awap_values: list[float],
    spike_values: list[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(
        {
            "timestamp": list(range(len(wap_values))),
            "mark_price": wap_values,
            "bid1_price": [value - 1.0 for value in wap_values],
            "ask1_price": [value + 1.0 for value in wap_values],
            "wap_1": wap_values,
            "awap": awap_values,
            "spike_feature": spike_values,
        }
    )
    frame.write_ipc(path)
```

- [x] **Step 3: Add the train-only robust scaler regression test**

Add this test after `test_multi_contract_scale_save_cli_scans_all_split_stage_contracts`:

```python
def test_multi_contract_scale_save_cli_uses_train_only_robust_scaler(tmp_path):
    feature_file = (
        tmp_path
        / "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/5min/fu/train/state_features.npy"
    )
    feature_file.parent.mkdir(parents=True)
    np.save(feature_file, np.array(["wap_1", "awap", "spike_feature"]))

    _write_multi_scale_fixture(
        tmp_path
        / "PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/5min/fu/train/fu2601.feather",
        wap_values=[2600.0, 2700.0, 2900.0, 3000.0],
        awap_values=[2600.0, 2700.0, 2900.0, 3000.0],
        spike_values=[0.0, 1.0, 2.0, 3.0],
    )
    _write_multi_scale_fixture(
        tmp_path
        / "PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/5min/fu/test/fu2510.feather",
        wap_values=[2848.0, 2849.0, 2850.0, 2851.0],
        awap_values=[2848.0, 2849.0, 2850.0, 2851.0],
        spike_values=[1000.0, 1000.0, 1000.0, 1000.0],
    )

    _run_multi_contract_scale_save_cli(tmp_path)

    output_root = tmp_path / "PREPROCESS_DATASET/commodity-futures/SCALE_SAVE/fu/5min"
    test_output = pl.read_ipc(output_root / "test/fu2510.feather")
    manifest = json.loads((output_root / "scaler_manifest.json").read_text(encoding="utf-8"))
    diagnostics = pl.read_csv(output_root / "scale_diagnostics.csv")

    expected_wap = (2849.5 - 2800.0) / 250.0
    assert abs(float(test_output["wap_1"].median()) - expected_wap) < 1e-9
    assert abs(float(test_output["awap"].median()) - expected_wap) < 1e-9
    assert float(test_output["spike_feature"].max()) == 20.0
    assert manifest["scaler_version"] == "robust_v1"
    assert manifest["fit_scope"] == "train_all_contracts"
    assert manifest["clip"]["enabled"] is True
    assert manifest["clip"]["min"] == -20.0
    assert manifest["clip"]["max"] == 20.0
    assert {item["feature"] for item in manifest["features"]} == {
        "wap_1",
        "awap",
        "spike_feature",
    }
    assert set(diagnostics["stage"].to_list()) == {"train", "test"}
    assert (
        diagnostics.filter(
            (pl.col("stage") == "test") & (pl.col("contract") == "fu2510")
        )["total_clipped_values"].item()
        == 4
    )
```

- [x] **Step 4: Add fail-fast tests for train inputs and clip bounds**

Add these tests after the existing missing-feature tests:

```python
def test_multi_contract_scale_save_cli_rejects_missing_train_split_inputs(tmp_path):
    input_file = (
        tmp_path
        / "PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/5min/fu/valid/fu2601.feather"
    )
    _write_scale_fixture(input_file)
    feature_file = (
        tmp_path
        / "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/5min/fu/train/state_features.npy"
    )
    feature_file.parent.mkdir(parents=True)
    np.save(feature_file, np.array(["feature_a"]))

    result = _run_multi_contract_scale_save_cli(tmp_path, check=False)

    assert result.returncode != 0
    assert "no train split-stage inputs found" in (result.stdout + result.stderr)


def test_multi_contract_scale_save_cli_rejects_invalid_clip_bounds(tmp_path):
    input_file = (
        tmp_path
        / "PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/5min/fu/train/fu2601.feather"
    )
    _write_scale_fixture(input_file)
    feature_file = (
        tmp_path
        / "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/5min/fu/train/state_features.npy"
    )
    feature_file.parent.mkdir(parents=True)
    np.save(feature_file, np.array(["feature_a"]))

    result = _run_multi_contract_scale_save_cli(
        tmp_path,
        check=False,
        extra_args=["--clip_min", "20", "--clip_max", "20"],
    )

    assert result.returncode != 0
    assert "clip_min must be less than clip_max" in (result.stdout + result.stderr)
```

- [x] **Step 5: Run the new tests and confirm they fail before implementation**

Run:

```bash
conda run -n finetf pytest \
  data_preprocess/tests/test_feature_selection_polars.py::test_multi_contract_scale_save_cli_uses_train_only_robust_scaler \
  data_preprocess/tests/test_feature_selection_polars.py::test_multi_contract_scale_save_cli_rejects_missing_train_split_inputs \
  data_preprocess/tests/test_feature_selection_polars.py::test_multi_contract_scale_save_cli_rejects_invalid_clip_bounds \
  -q
```

Expected: FAIL because `--clip_min` / `--clip_max` are not implemented and `muti_contract_scale_save.py` still uses per-file `scale_std` / `scale_mean`.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 2: Implement train-only robust scaler in multi-contract scale save

> **trace:** plan-ready.md -> ### Task 2: Implement train-only robust scaler in multi-contract scale save | tasks.md -> - [ ] 1.2 Implement the train-only robust scaler path in `data_preprocess/operator_futures/scale_describe_save/muti_contract_scale_save.py`, including train split fit, reusable manifest generation, apply-to-all-splits output writing, same-basename CSV debug output, and original `SCALE_SAVE/{symbol}/{target_freq}/{stage}/{contract}.feather` layout.
> **sync:** tasks.md -> - [ ] 1.2 Implement the train-only robust scaler path in `data_preprocess/operator_futures/scale_describe_save/muti_contract_scale_save.py`, including train split fit, reusable manifest generation, apply-to-all-splits output writing, same-basename CSV debug output, and original `SCALE_SAVE/{symbol}/{target_freq}/{stage}/{contract}.feather` layout. | plan-ready.md -> ### Task 2: Implement train-only robust scaler in multi-contract scale save

**Files:**
- Modify: `data_preprocess/operator_futures/scale_describe_save/muti_contract_scale_save.py`

- [x] **Step 1: Replace legacy scaling imports and add robust scaler configuration**

In `muti_contract_scale_save.py`, remove `scale_mean` and `scale_std` from the import from `scale_save.py`, keeping `configure_logging` and `validate_no_nan`. Add imports:

```python
from dataclasses import asdict, dataclass
import json
import math
```

Add parser arguments after the existing legacy `--base` argument:

```python
parser.add_argument("--clip_min", type=float, default=-20.0, help="minimum clipped robust-scaled value")
parser.add_argument("--clip_max", type=float, default=20.0, help="maximum clipped robust-scaled value")
parser.add_argument("--disable_clip", action="store_true", help="disable robust scaler clipping")
parser.add_argument("--iqr_epsilon", type=float, default=1e-8, help="minimum usable IQR scale")
parser.add_argument("--std_epsilon", type=float, default=1e-8, help="minimum usable std fallback scale")
```

- [x] **Step 2: Add scaler dataclasses and validation helpers**

Add these dataclasses below `SPLIT_STAGES`:

```python
@dataclass(frozen=True)
class ScalerFeatureStats:
    feature: str
    center: float
    scale: float
    scale_method: str
    q25: float
    q50: float
    q75: float
    std: float
    fallback_reason: str | None = None


@dataclass(frozen=True)
class ScaleManifest:
    scaler_version: str
    fit_scope: str
    symbol: str
    target_freq: str
    feature_list_path: str
    train_input_files: list[str]
    row_count: int
    clip: dict[str, float | bool]
    features: list[ScalerFeatureStats]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["features"] = [asdict(item) for item in self.features]
        return payload
```

Add:

```python
def validate_clip_args(args) -> None:
    if args.disable_clip:
        return
    if args.clip_min >= args.clip_max:
        raise ValueError("clip_min must be less than clip_max")


def ensure_finite(value: float, *, feature: str, field: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"non-finite scaler statistic: feature={feature} field={field} value={value}")
    return value
```

- [x] **Step 3: Implement train-only fit helpers**

Add:

```python
def fit_feature_stats(
    feature: str,
    values: np.ndarray,
    *,
    iqr_epsilon: float,
    std_epsilon: float,
) -> ScalerFeatureStats:
    values = values.astype(float, copy=False)
    q25, q50, q75 = np.nanquantile(values, [0.25, 0.5, 0.75])
    std = float(np.nanstd(values, ddof=1))
    iqr = float(q75 - q25)
    scale_method = "iqr"
    fallback_reason = None
    scale = iqr
    if not math.isfinite(scale) or abs(scale) <= iqr_epsilon:
        scale = std
        scale_method = "std"
        fallback_reason = "iqr_too_small"
    if not math.isfinite(scale) or abs(scale) <= std_epsilon:
        scale = 1.0
        scale_method = "constant"
        fallback_reason = "std_too_small"

    return ScalerFeatureStats(
        feature=feature,
        center=ensure_finite(float(q50), feature=feature, field="center"),
        scale=ensure_finite(float(scale), feature=feature, field="scale"),
        scale_method=scale_method,
        q25=ensure_finite(float(q25), feature=feature, field="q25"),
        q50=ensure_finite(float(q50), feature=feature, field="q50"),
        q75=ensure_finite(float(q75), feature=feature, field="q75"),
        std=ensure_finite(std, feature=feature, field="std"),
        fallback_reason=fallback_reason,
    )
```

Add:

```python
def fit_robust_scaler(
    *,
    train_inputs: list[tuple[str, str, Path]],
    state_features: list[str],
    feature_list_path: Path,
    args,
) -> ScaleManifest:
    if not train_inputs:
        raise ValueError(f"no train split-stage inputs found for symbol={args.symbols}")

    frames = []
    for _, _, input_file in train_inputs:
        df = pl.read_ipc(input_file)
        missing_features = [feature for feature in state_features if feature not in df.columns]
        if missing_features:
            raise ValueError(f"missing selected state feature columns in {input_file}: {missing_features}")
        frames.append(df.select(state_features))

    train_df = pl.concat(frames, how="vertical")
    feature_stats = [
        fit_feature_stats(
            feature,
            train_df.get_column(feature).to_numpy(),
            iqr_epsilon=args.iqr_epsilon,
            std_epsilon=args.std_epsilon,
        )
        for feature in state_features
    ]
    return ScaleManifest(
        scaler_version="robust_v1",
        fit_scope="train_all_contracts",
        symbol=args.symbols,
        target_freq=args.target_freq,
        feature_list_path=str(feature_list_path),
        train_input_files=[str(path) for _, _, path in train_inputs],
        row_count=train_df.height,
        clip={
            "enabled": not args.disable_clip,
            "min": float(args.clip_min),
            "max": float(args.clip_max),
        },
        features=feature_stats,
    )
```

- [x] **Step 4: Implement apply, manifest writing, and diagnostics writing**

Add:

```python
def apply_robust_scaler(
    df_state: pl.DataFrame,
    manifest: ScaleManifest,
) -> tuple[pl.DataFrame, dict[str, float | int | str | bool]]:
    columns = {}
    total_clipped_values = 0
    total_values = df_state.height * len(manifest.features)
    top_feature = ""
    top_feature_clip_ratio = 0.0
    clip_enabled = bool(manifest.clip["enabled"])
    clip_min = float(manifest.clip["min"])
    clip_max = float(manifest.clip["max"])

    for stats in manifest.features:
        values = df_state.get_column(stats.feature).to_numpy().astype(float, copy=False)
        scaled = (values - stats.center) / stats.scale
        clipped_count = 0
        if clip_enabled:
            clipped = np.clip(scaled, clip_min, clip_max)
            clipped_count = int(np.sum(clipped != scaled))
            scaled = clipped
        ratio = clipped_count / len(values) if len(values) else 0.0
        if ratio > top_feature_clip_ratio:
            top_feature = stats.feature
            top_feature_clip_ratio = ratio
        total_clipped_values += clipped_count
        columns[stats.feature] = scaled

    diagnostics = {
        "state_feature_count": len(manifest.features),
        "clip_enabled": clip_enabled,
        "total_clipped_values": total_clipped_values,
        "total_values": total_values,
        "max_feature_clip_ratio": top_feature_clip_ratio,
        "top_clipped_feature": top_feature,
    }
    return pl.DataFrame(columns), diagnostics
```

Add:

```python
def write_manifest(manifest: ScaleManifest, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "scaler_manifest.json").write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_diagnostics(rows: list[dict[str, object]], output_root: Path) -> None:
    pl.DataFrame(rows).write_csv(output_root / "scale_diagnostics.csv")
```

- [x] **Step 5: Wire the new scaler through `scale_one_input` and `main`**

Update `scale_one_input` to accept `manifest: ScaleManifest` and return one diagnostics row. Replace:

```python
df_state = scale_std(df_state, args.base)
df_state = scale_mean(df_state, args.base, args.clip_theshold)
```

with:

```python
df_state, diagnostics = apply_robust_scaler(df_state, manifest)
```

Return diagnostics merged with file metadata:

```python
return {
    "stage": stage,
    "contract": contract,
    "input_file": str(input_file),
    "output_file": str(output_file),
    "rows": out.height,
    **diagnostics,
}
```

In `main`, before the output loop:

```python
validate_clip_args(args)
train_inputs = [item for item in inputs if item[0] == "train"]
manifest = fit_robust_scaler(
    train_inputs=train_inputs,
    state_features=state_features,
    feature_list_path=feature_list_path,
    args=args,
)
output_root = save_root / args.symbols / args.target_freq
write_manifest(manifest, output_root)
diagnostics_rows = []
```

During the loop, append the returned diagnostics row and after the loop:

```python
write_diagnostics(diagnostics_rows, output_root)
```

- [x] **Step 6: Run focused tests until green**

Run:

```bash
conda run -n finetf pytest \
  data_preprocess/tests/test_feature_selection_polars.py -k "multi_contract_scale_save" \
  -q
```

Expected: PASS for all multi-contract scale-save tests.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 3: Validate spec and Python artifacts

> **trace:** plan-ready.md -> ### Task 3: Validate spec and Python artifacts | tasks.md -> - [ ] 1.3 Run focused validation for the new scaler contract, including `pytest` for the updated scale-save tests, `python -m py_compile` for changed Python files, and `openspec validate --strict` for this change.
> **sync:** tasks.md -> - [ ] 1.3 Run focused validation for the new scaler contract, including `pytest` for the updated scale-save tests, `python -m py_compile` for changed Python files, and `openspec validate --strict` for this change. | plan-ready.md -> ### Task 3: Validate spec and Python artifacts

**Files:**
- Verify: `openspec/changes/refactor-multi-contract-scale-save-robust-scaler/proposal.md`
- Verify: `openspec/changes/refactor-multi-contract-scale-save-robust-scaler/design.md`
- Verify: `openspec/changes/refactor-multi-contract-scale-save-robust-scaler/specs/`
- Verify: `openspec/changes/refactor-multi-contract-scale-save-robust-scaler/tasks.md`
- Verify: `openspec/changes/refactor-multi-contract-scale-save-robust-scaler/plan-ready.md`
- Verify: `data_preprocess/operator_futures/scale_describe_save/muti_contract_scale_save.py`
- Verify: `data_preprocess/tests/test_feature_selection_polars.py`

- [x] **Step 1: Run py_compile on changed Python files**

Run:

```bash
conda run -n finetf python -m py_compile \
  data_preprocess/operator_futures/scale_describe_save/muti_contract_scale_save.py \
  data_preprocess/tests/test_feature_selection_polars.py
```

Expected: exit code 0 with no syntax errors.

- [x] **Step 2: Run focused scale-save tests**

Run:

```bash
conda run -n finetf pytest \
  data_preprocess/tests/test_feature_selection_polars.py -k "multi_contract_scale_save or scale_save" \
  -q
```

Expected: PASS for the focused scale-save tests. Existing `scale_save.py` tests should still pass because the legacy entry point was not modified.

- [x] **Step 3: Run strict OpenSpec validation**

Run:

```bash
openspec validate refactor-multi-contract-scale-save-robust-scaler --strict
```

Expected: `Change 'refactor-multi-contract-scale-save-robust-scaler' is valid`.

- [x] **Step 4: Inspect generated artifacts on a tiny fixture run**

After the focused CLI tests pass, inspect the temporary test output or run an equivalent fixture command and confirm:

```text
SCALE_SAVE/fu/5min/scaler_manifest.json exists
SCALE_SAVE/fu/5min/scale_diagnostics.csv exists
SCALE_SAVE/fu/5min/{stage}/{contract}.feather exists
SCALE_SAVE/fu/5min/{stage}/{contract}.csv exists
```

Expected: manifest records `scaler_version=robust_v1`, `fit_scope=train_all_contracts`, and clip `[-20, 20]`; diagnostics contains at least one row per processed output file.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）
