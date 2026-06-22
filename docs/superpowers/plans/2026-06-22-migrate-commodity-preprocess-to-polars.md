# Migrate Commodity Preprocess To Polars Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace pandas usage in the shared post-merge preprocessing layer with Polars while preserving existing CLI and output contracts.

**Architecture:** Keep the current script entrypoints and file layout, but replace their internals with Polars helpers. Time feature generation uses Polars rolling expressions rather than Python multiprocessing; feature selection and scale/save use Polars with NumPy or library-native conversion only at ML boundaries.

**Tech Stack:** Python, Polars, NumPy, PyArrow/Feather, CatBoost/sklearn boundaries, pytest, OpenSpec.

**Traceability (sddflow):**
- plan-ready: `openspec/changes/migrate-commodity-preprocess-to-polars/plan-ready.md`
- tasks: `openspec/changes/migrate-commodity-preprocess-to-polars/tasks.md`
- plan: `docs/superpowers/plans/2026-06-22-migrate-commodity-preprocess-to-polars.md`

---

### Task 1: Time feature Polars migration

> **trace:** plan-ready.md → `### Task 1: Time feature Polars migration` | tasks.md → ``- [ ] 1.0 Time feature Polars migration complete（与 `plan-ready.md` Task 1 和 superpowers plan Task 1 同步）``
> **sync:** tasks.md → ``- [ ] 1.0 Time feature Polars migration complete（与 `plan-ready.md` Task 1 和 superpowers plan Task 1 同步）`` | plan-ready.md → `### Task 1: Time feature Polars migration`

**Files:**
- Modify: `data_preprocess/operator_futures/time_operator/create_feature_multi_processing.py`
- Modify: `data_preprocess/operator_futures/time_operator/multi_processing_util.py`
- Test: `data_preprocess/tests/test_commodity_main_contract_cli.py` or `data_preprocess/tests/test_time_operator_polars.py`

- [x] **Step 1: Write failing tests for depth-aware time features**

Add or extend tests with a 5-level commodity fixture and a 25-level generic fixture. The commodity fixture should call the CLI with `--orderbook_depth 5` and assert the output contains `bid5_price_log_return_2` but not `bid6_price_log_return_2`. The generic fixture should use `--orderbook_depth 25` and assert `bid25_price_log_return_2` exists.

```python
def test_time_feature_polars_respects_orderbook_depth(tmp_path):
    input_dir = (
        tmp_path
        / "PREPROCESS_DATASET"
        / "commodity-futures"
        / "MERGE_CONCAT"
        / "CONCAT_FEATURE"
        / "fu"
        / "5min"
    )
    input_dir.mkdir(parents=True)
    rows = []
    for idx in range(20):
        row = {
            "timestamp": idx,
            "open": 2600.0 + idx,
            "high": 2601.0 + idx,
            "low": 2599.0 + idx,
            "close": 2600.5 + idx,
            "volume": 100.0 + idx,
            "mark_price": 2600.25 + idx,
            "buy_spread_oe_max": 4.0,
            "sell_spread_oe_max": 4.0,
            "wap_1": 2600.2 + idx,
            "wap_2": 2600.3 + idx,
            "buy_wap": 2600.1 + idx,
            "sell_wap": 2600.4 + idx,
            "buy_volume_oe": 20.0 + idx,
            "sell_volume_oe": 21.0 + idx,
            "imblance_volume_oe": 1.0,
        }
        for level in range(1, 6):
            row[f"bid{level}_price"] = 2600.0 + idx - level
            row[f"ask{level}_price"] = 2600.0 + idx + level
            row[f"bid{level}_size_n"] = 0.1 * level
            row[f"ask{level}_size_n"] = 0.1 * level
        rows.append(row)
    pl.DataFrame(rows).write_ipc(input_dir / "2026-01-05-2026-01-06.feather")

    subprocess.run(
        [
            sys.executable,
            "data_preprocess/operator_futures/time_operator/create_feature_multi_processing.py",
            "--root_path",
            str(tmp_path),
            "--data_path",
            "PREPROCESS_DATASET/commodity-futures/MERGE_CONCAT/CONCAT_FEATURE/",
            "--save_path",
            "PREPROCESS_DATASET/commodity-futures/TIME_FEATURE/",
            "--symbols",
            "fu",
            "--target_freq",
            "5min",
            "--start_date",
            "2026-01-05",
            "--end_date",
            "2026-01-06",
            "--windows",
            "2",
            "--orderbook_depth",
            "5",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "data_preprocess")},
        check=True,
    )

    out = pl.read_ipc(
        tmp_path
        / "PREPROCESS_DATASET/commodity-futures/TIME_FEATURE/fu/5min/2026-01-05-2026-01-06.feather"
    )
    assert "bid5_price_log_return_2" in out.columns
    assert "bid6_price_log_return_2" not in out.columns
```

- [x] **Step 2: Run tests to verify they fail on pandas imports or old behavior**

Run:

```bash
conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract_cli.py::test_time_feature_polars_respects_orderbook_depth -q
```

Expected: fail until the Polars implementation and test imports are aligned.

- [x] **Step 3: Replace time feature helpers with Polars**

Implement Polars helper functions in `multi_processing_util.py` that accept a Polars `DataFrame` and return a Polars `DataFrame`. Preserve existing output suffixes such as `log_return_<w>`, `trend_<w>`, `roc_<w>`, and `ma_<w>`.

```python
import numpy as np
import polars as pl

MIN_VALUE = 1e-12


def process_price_windows(df: pl.DataFrame, windows: list[int], features: list[str]) -> pl.DataFrame:
    pieces = [df.select("timestamp")]
    for feature in features:
        for window in windows:
            exprs = [
                (pl.col(feature) / (pl.col(feature).shift(1) + MIN_VALUE))
                .log()
                .mul(1000)
                .alias(f"{feature}_log_return_{window}")
            ]
            if window != 1:
                rolling_mean = pl.col(feature).rolling_mean(window)
                rolling_std = pl.col(feature).rolling_std(window)
                exprs.append(
                    ((pl.col(feature) - rolling_mean) / (rolling_std + MIN_VALUE))
                    .alias(f"{feature}_trend_{window}")
                )
            pieces.append(df.select(["timestamp", *exprs]).slice(window + 1))
    return _inner_join_on_timestamp(pieces).fill_nan(0).fill_null(0)
```

Implement OHLCV/OHLC helpers with explicit rolling expressions and timestamp slicing compatible with current output. Keep the existing trim offsets: OHLCV uses `w + 10`, OHLC uses `w + 1`, and price-window helpers use their current offset from the pandas implementation.

```python
def _clean_time_features(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.all().exclude("timestamp").replace([float("inf"), float("-inf")], None)
    ).fill_nan(0).fill_null(0)


def _rank_pct_last_expr(column: str, window: int, alias: str) -> pl.Expr:
    return (
        pl.col(column)
        .rolling_map(lambda values: float((values <= values[-1]).sum()) / len(values), window)
        .alias(alias)
    )


def process_ohlc_windows(df: pl.DataFrame, windows: list[int]) -> pl.DataFrame:
    pieces = []
    for w in windows:
        close = pl.col("close")
        close_shift = close.shift(w)
        close_std = close.rolling_std(w) + MIN_VALUE
        low_min = pl.min_horizontal(pl.col("low"), close_shift)
        high_max = pl.max_horizontal(pl.col("high"), close_shift)
        ret1 = (close / close.shift(1) - 1).alias("__ret1")
        base = df.with_columns(
            ret1,
            pl.when(pl.col("__ret1") > 0).then(pl.col("__ret1")).otherwise(0).alias("__pos_ret1"),
            pl.col("__ret1").abs().alias("__abs_ret1"),
        )
        pieces.append(
            base.select(
                "timestamp",
                (close_shift / close).alias(f"roc_{w}"),
                (close_shift / close_std).alias(f"roc_{w}_std_norm"),
                (close.rolling_mean(w) / close).alias(f"ma_{w}"),
                (close.rolling_std(w) / close).alias(f"std_{w}"),
                ((close - low_min) / (high_max - low_min + MIN_VALUE)).alias(f"rsv_{w}"),
                (pl.col("__pos_ret1").rolling_sum(w) / (pl.col("__abs_ret1").rolling_sum(w) + MIN_VALUE)).alias(f"sump_{w}"),
            ).slice(w + 1)
        )
    return _clean_time_features(_inner_join_on_timestamp(pieces))
```

- [x] **Step 4: Update `create_feature_multi_processing.py` to use Polars I/O**

Read and write IPC with Polars. Build depth-aware feature lists and call the helper functions.

```python
original_df = pl.read_ipc(input_path)
price_features = build_price_features(original_df.columns, args.orderbook_depth)
time_frames = [
    process_price_windows(original_df, windows, price_features),
    process_ohlcv_windows(original_df, windows),
    process_ohlc_windows(original_df, windows),
]
time_df = _inner_join_on_timestamp(time_frames)
time_df.write_ipc(output_path)
```

- [x] **Step 5: Run focused verification**

Run:

```bash
conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract_cli.py::test_time_feature_polars_respects_orderbook_depth -q
```

Expected: pass.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 2: Core feature selection Polars migration

> **trace:** plan-ready.md → `### Task 2: Core feature selection Polars migration` | tasks.md → ``- [ ] 2.0 Core feature selection Polars migration complete（与 `plan-ready.md` Task 2 和 superpowers plan Task 2 同步）``
> **sync:** tasks.md → ``- [ ] 2.0 Core feature selection Polars migration complete（与 `plan-ready.md` Task 2 和 superpowers plan Task 2 同步）`` | plan-ready.md → `### Task 2: Core feature selection Polars migration`

**Files:**
- Modify: `data_preprocess/operator_futures/feature_selection/ic_correlation.py`
- Modify: `data_preprocess/operator_futures/feature_selection/rank_ic_correlation.py`
- Modify: `data_preprocess/operator_futures/feature_selection/cor_util.py`
- Test: `data_preprocess/tests/test_commodity_feature_pipeline.py` or `data_preprocess/tests/test_feature_selection_polars.py`

- [x] **Step 1: Write failing tests for IC artifacts**

Create a small all-feature IPC file with `timestamp`, `mark_price`, reward columns, and state columns. Run `ic_correlation.py` with `--market_type commodity_futures --orderbook_depth 5`.

```python
def test_ic_correlation_polars_writes_expected_artifacts(tmp_path):
    data_dir = tmp_path / "PREPROCESS_DATASET/commodity-futures/ALL_FEATURE/fu/5min"
    data_dir.mkdir(parents=True)
    frame = pl.DataFrame(
        {
            "timestamp": list(range(12)),
            "mark_price": [100.0 + i for i in range(12)],
            "bid1_price": [99.0 + i for i in range(12)],
            "ask1_price": [101.0 + i for i in range(12)],
            "feature_a": [float(i) for i in range(12)],
            "feature_b": [float(12 - i) for i in range(12)],
        }
    )
    frame.write_ipc(data_dir / "2026-01-05-2026-01-06.feather")
    subprocess.run(
        [
            sys.executable,
            "data_preprocess/operator_futures/feature_selection/ic_correlation.py",
            "--root_path",
            str(tmp_path),
            "--data_path",
            "PREPROCESS_DATASET/commodity-futures/ALL_FEATURE/",
            "--save_path",
            "PREPROCESS_DATASET/commodity-futures/IC_RESULT/",
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
            "--windows_list",
            "1",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "data_preprocess")},
        check=True,
    )
    result_dir = tmp_path / "PREPROCESS_DATASET/commodity-futures/IC_RESULT/fu/5min/2026-01-05-2026-01-06"
    assert (result_dir / "df.feather").exists()
    assert (result_dir / "state_features.npy").exists()
    assert (result_dir / "correlation.csv").exists()
```

- [x] **Step 2: Run tests to verify failure before migration**

Run:

```bash
conda run -n finetf pytest data_preprocess/tests/test_feature_selection_polars.py::test_ic_correlation_polars_writes_expected_artifacts -q
```

Expected: fail while pandas-based code or missing Polars behavior remains.

- [x] **Step 3: Port `ic_correlation.py` and `rank_ic_correlation.py`**

Use Polars for reads, target creation, feature selection, and writes. Convert only correlation arrays to NumPy when simpler.

```python
df = pl.read_ipc(input_path)
reward_features, state_features = select_reward_state_features(df, args.market_type, args.orderbook_depth)
target = (pl.col("mark_price").shift(-window_length) - pl.col("mark_price")).alias("__target")
work = df.with_columns(target).slice(0, max(df.height - window_length, 0))
correlations = {
    feature: work.select(pl.corr(feature, "__target")).item()
    for feature in state_features
}
```

- [x] **Step 4: Port `cor_util.py` to pandas-free correlation filtering**

Accept a Polars DataFrame or a NumPy matrix and return selected features. Keep deterministic ordering.

```python
def select_feature(corre_df: pl.DataFrame, theshold: float = 0.5) -> list[str]:
    features = [c for c in corre_df.columns if c != "feature"]
    selected: list[str] = []
    for feature in features:
        if all(abs(corre_df.select(pl.col(feature).filter(pl.col("feature") == prev)).item()) <= theshold for prev in selected):
            selected.append(feature)
    return selected
```

- [x] **Step 5: Run focused verification**

Run:

```bash
conda run -n finetf pytest data_preprocess/tests/test_feature_selection_polars.py -q
```

Expected: IC/rank IC tests pass and output artifacts exist.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 3: ML feature selection Polars migration

> **trace:** plan-ready.md → `### Task 3: ML feature selection Polars migration` | tasks.md → ``- [ ] 3.0 ML feature selection Polars migration complete（与 `plan-ready.md` Task 3 和 superpowers plan Task 3 同步）``
> **sync:** tasks.md → ``- [ ] 3.0 ML feature selection Polars migration complete（与 `plan-ready.md` Task 3 和 superpowers plan Task 3 同步）`` | plan-ready.md → `### Task 3: ML feature selection Polars migration`

**Files:**
- Modify: `data_preprocess/operator_futures/feature_selection/catbooost.py`
- Modify: `data_preprocess/operator_futures/feature_selection/lasso_linear.py`
- Test: `data_preprocess/tests/test_feature_selection_polars.py`

- [x] **Step 1: Write failing tests for pandas-free ML scripts**

Add import-scan assertions and small fixture tests that run each script with minimal data or call extracted helpers.

```python
def test_ml_feature_selection_files_do_not_import_pandas():
    files = [
        REPO_ROOT / "data_preprocess/operator_futures/feature_selection/catbooost.py",
        REPO_ROOT / "data_preprocess/operator_futures/feature_selection/lasso_linear.py",
    ]
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert "import pandas" not in text
        assert "from pandas" not in text
```

- [x] **Step 2: Run tests to verify failure**

Run:

```bash
conda run -n finetf pytest data_preprocess/tests/test_feature_selection_polars.py::test_ml_feature_selection_files_do_not_import_pandas -q
```

Expected: fail until imports are removed.

- [x] **Step 3: Port ML data preparation to Polars**

Read features with `pl.read_ipc`, choose reward/state columns with Polars lists, then pass NumPy arrays to model APIs. Write CatBoost feature importance with Polars CSV, and write selected datasets with Polars IPC.

```python
df = pl.read_ipc(input_path)
y = df.select(target_column).to_numpy().ravel()
x = df.select(state_features).to_numpy()
train_pool = Pool(x, y)
model.fit(train_pool, eval_set=train_pool, verbose=100)
feature_importance_df = pl.DataFrame(
    {"Feature": state_features, "Importance": model.get_feature_importance(train_pool)}
).sort("Importance", descending=True)
feature_importance_df.write_csv(output_dir / f"cat_boost_feature_importance_{window_length}.csv")
selected = feature_importance_df.filter(pl.col("Importance") > args.ic_theshold)["Feature"].to_list()
df.select([*reward_features, *selected]).write_ipc(output_dir / "df_catboost.feather")
np.save(output_dir / "state_features_catboost.npy", np.array(selected))
```

For `lasso_linear.py`, keep the same shape: Polars read and feature selection, `to_numpy()` at sklearn boundaries, Polars CSV/IPC for outputs, and `np.save` for selected feature names.

- [x] **Step 4: Run focused verification**

Run:

```bash
conda run -n finetf pytest data_preprocess/tests/test_feature_selection_polars.py -q
```

Expected: ML import-scan and artifact tests pass.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 4: Remove-duplicates and scale/save Polars migration

> **trace:** plan-ready.md → `### Task 4: Remove-duplicates and scale/save Polars migration` | tasks.md → ``- [ ] 4.0 Remove-duplicates and scale/save Polars migration complete（与 `plan-ready.md` Task 4 和 superpowers plan Task 4 同步）``
> **sync:** tasks.md → ``- [ ] 4.0 Remove-duplicates and scale/save Polars migration complete（与 `plan-ready.md` Task 4 和 superpowers plan Task 4 同步）`` | plan-ready.md → `### Task 4: Remove-duplicates and scale/save Polars migration`

**Files:**
- Modify: `data_preprocess/operator_futures/feature_selection/remove_duplicates_feature.py`
- Modify: `data_preprocess/operator_futures/scale_describe_save/scale_save.py`
- Test: `data_preprocess/tests/test_feature_selection_polars.py`
- Test: `data_preprocess/tests/test_commodity_feature_pipeline.py`

- [x] **Step 1: Write failing tests for scale/save outputs**

Use a small IC_RESULT fixture and assert reward columns are preserved, state columns are scaled, and expected files are written.

```python
def test_scale_save_polars_preserves_commodity_reward_columns(tmp_path):
    result_dir = tmp_path / "PREPROCESS_DATASET/commodity-futures/IC_RESULT/fu/5min/2026-01-05-2026-01-06"
    result_dir.mkdir(parents=True)
    frame = pl.DataFrame(
        {
            "bid1_price": [99.0, 100.0, 101.0],
            "ask1_price": [101.0, 102.0, 103.0],
            "feature_a": [10.0, 20.0, 30.0],
        }
    )
    frame.write_ipc(result_dir / "df.feather")
    np.save(result_dir / "state_features.npy", np.array(["feature_a"]))
    subprocess.run(
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
        check=True,
    )
    out_dir = tmp_path / "PREPROCESS_DATASET/commodity-futures/SCALE_SAVE/fu/5min/2026-01-05-2026-01-06"
    assert (out_dir / "df.feather").exists()
    assert (out_dir / "state_features.npy").exists()
    assert (out_dir / "df_describe.csv").exists()
```

- [x] **Step 2: Run tests to verify failure**

Run:

```bash
conda run -n finetf pytest data_preprocess/tests/test_feature_selection_polars.py::test_scale_save_polars_preserves_commodity_reward_columns -q
```

Expected: fail until `scale_save.py` no longer depends on pandas.

- [x] **Step 3: Port `scale_save.py`**

Use Polars for read, select, scaling, concat, describe, and write.

```python
df = pl.read_ipc(input_path)
reward = df.select(reward_features)
state = df.select(state_features)
scaled = scale_state_polars(state, args.base, args.clip_theshold)
pl.concat([reward, scaled], how="horizontal").write_ipc(output_dir / "df.feather")
scaled.describe().write_csv(output_dir / "df_describe.csv")
np.save(output_dir / "state_features.npy", state_features)
```

Implement the scaling helpers without pandas. Preserve the existing std-scale then mean-scale order.

```python
def scale_std(df: pl.DataFrame, log_base: float = 10) -> pl.DataFrame:
    scales = {}
    for name in df.columns:
        std = float(df.select(pl.col(name).std()).item() or 0.0)
        scale = log_base ** np.floor(np.log10(std) * np.log10(log_base) / np.log10(10)) if std > 0 else 1.0
        scales[name] = scale
    return df.with_columns([(pl.col(name) / scale).alias(name) for name, scale in scales.items()])


def scale_mean(df: pl.DataFrame, log_base: float = 10, clip_theshold: float = 10) -> pl.DataFrame:
    exprs = []
    for name in df.columns:
        mean_value = float(df.select(pl.col(name).mean()).item() or 0.0)
        if abs(mean_value) > clip_theshold:
            adjustment = log_base ** round(np.log10(abs(mean_value)) * np.log10(log_base) / np.log10(10))
            adjustment = -adjustment if mean_value > 0 else adjustment
        else:
            adjustment = 0.0
        exprs.append((pl.col(name) + adjustment).alias(name))
    return df.with_columns(exprs)
```

- [x] **Step 4: Port `remove_duplicates_feature.py`**

Replace CSV/Feather reads with Polars, preserve the first 106 reward-column fallback for non-commodity inputs, and preserve commodity reward selection by `get_reward_execution_columns(args.orderbook_depth)`. Correlation CSVs should be normalized to a Polars table with a `feature` column before calling `select_feature`.

```python
df = pl.read_ipc(input_path)
reward_features = df.columns[:106]
state_features = [col for col in df.columns if col not in reward_features]
selected_from_ic: list[str] = []
for ic_file_name in ic_file_name_list:
    with open(output_dir / ic_file_name, "r", encoding="utf-8") as handle:
        scores = json.load(handle)
    selected_from_ic.extend(
        feature for feature, score in scores.items() if abs(float(score)) > args.ic_theshold
    )
ic_selection_key = remove_duplicates_preserve_order(selected_from_ic)
cor_df = pl.read_csv(output_dir / cor_file_name)
if "feature" not in cor_df.columns:
    cor_df = cor_df.rename({cor_df.columns[0]: "feature"})
cor_df = cor_df.filter(pl.col("feature").is_in(ic_selection_key)).select(["feature", *ic_selection_key])
selected_feature_names = select_feature(corre_df=cor_df, theshold=args.cor_theshold)
df.select([*reward_features, *selected_feature_names]).write_ipc(output_dir / target_df_name)
np.save(output_dir / target_state_name, np.array(selected_feature_names))
```

- [x] **Step 5: Run focused verification**

Run:

```bash
conda run -n finetf pytest data_preprocess/tests/test_feature_selection_polars.py data_preprocess/tests/test_commodity_feature_pipeline.py -q
```

Expected: remove-duplicates and scale/save tests pass.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 5: Validation and smoke

> **trace:** plan-ready.md → `### Task 5: Validation and smoke` | tasks.md → ``- [ ] 5.0 Validation and smoke complete（与 `plan-ready.md` Task 5 和 superpowers plan Task 5 同步）``
> **sync:** tasks.md → ``- [ ] 5.0 Validation and smoke complete（与 `plan-ready.md` Task 5 和 superpowers plan Task 5 同步）`` | plan-ready.md → `### Task 5: Validation and smoke`

**Files:**
- Modify: `data_preprocess/tests/test_polars_compat.py` or a new import-scan test file
- Modify: `openspec/changes/migrate-commodity-preprocess-to-polars/tasks.md`
- Modify: `openspec/changes/migrate-commodity-preprocess-to-polars/plan-ready.md`
- Modify: `docs/superpowers/plans/2026-06-22-migrate-commodity-preprocess-to-polars.md`

- [x] **Step 1: Add target import scan test**

Add a test that scans the exact target files and fails on pandas imports.

```python
def test_post_merge_polars_targets_do_not_import_pandas():
    targets = [
        REPO_ROOT / "data_preprocess/operator_futures/time_operator/create_feature_multi_processing.py",
        REPO_ROOT / "data_preprocess/operator_futures/time_operator/multi_processing_util.py",
        REPO_ROOT / "data_preprocess/operator_futures/scale_describe_save/scale_save.py",
        *sorted((REPO_ROOT / "data_preprocess/operator_futures/feature_selection").glob("*.py")),
    ]
    for path in targets:
        text = path.read_text(encoding="utf-8")
        assert "import pandas" not in text, path
        assert "from pandas" not in text, path
```

- [x] **Step 2: Run focused test suite**

Run:

```bash
conda run -n finetf pytest data_preprocess/tests/test_feature_selection_polars.py data_preprocess/tests/test_commodity_main_contract_cli.py data_preprocess/tests/test_commodity_feature_pipeline.py -q
```

Expected: all focused tests pass.

- [x] **Step 3: Run commodity five-day main**

Run:

```bash
source /home/lanceliang/miniconda3/etc/profile.d/conda.sh
conda activate finetf
bash data_preprocess/script_preprocess/future_upgraded/commodity/main.sh
```

Expected: exit code 0 and `PREPROCESS_DATASET/commodity-futures/SCALE_SAVE/fu/5min/2025-11-03-2025-11-08/df.feather` exists.

- [x] **Step 4: Scan logs for known fatal errors**

Run:

```bash
rg -n "Traceback|FileNotFound|KeyError|ModuleNotFoundError|ValueError" \
  log_futures/ticker_result/commodity/fu_5min_2025-11-03_2025-11-08.log \
  log_futures/downscale/cross_section/5min/fu/2025-11-0{3,4,5,6,7}.log \
  log_futures/merge/5min/fu/2025-11-0{3,4,5,6,7}.log
```

Expected: no matches. `rg` may exit 1 when there are no matches; that is acceptable for this check.

- [x] **Step 5: Run final metadata checks**

Run:

```bash
openspec validate migrate-commodity-preprocess-to-polars --strict
git diff --check
```

Expected: OpenSpec reports the change is valid, and `git diff --check` prints no whitespace errors.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）
