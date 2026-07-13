# Refactor Commodity Feature Selection Union Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor commodity multi-contract feature selection so all contracts share one union state feature list while each contract still gets a filtered standard `IC_RESULT/df.feather`.

**Architecture:** Split commodity IC selection into candidate and finalize phases. `ic_correlation.py` keeps the existing default output contract but gains a candidate-only mode. `contract_feature_union.py` gains an IC-candidate finalize path that builds the union, validates every contract has every union feature in `ALL_FEATURE`, and writes standard per-contract `IC_RESULT` outputs before `scale_save` runs.

**Tech Stack:** Python 3.10, Polars, NumPy, pytest, Bash, OpenSpec.

**Traceability (sddflow):**
- plan-ready: `openspec/changes/refactor-commodity-feature-selection-union/plan-ready.md`
- tasks: `openspec/changes/refactor-commodity-feature-selection-union/tasks.md`
- plan: `docs/superpowers/plans/2026-07-13-refactor-commodity-feature-selection-union.md`

---

### Task 1: Candidate-only artifact tests

> **trace:** plan-ready.md → `### Task 1: Candidate-only artifact tests` | tasks.md → `- [ ] 1.1 Add focused tests for commodity IC candidate-only output artifacts and absence of final `df.feather` / `state_features.npy`.`
> **sync:** tasks.md → `- [ ] 1.1 Add focused tests for commodity IC candidate-only output artifacts and absence of final `df.feather` / `state_features.npy`.` | plan-ready.md → `### Task 1: Candidate-only artifact tests`

**Files:**
- Modify: `data_preprocess/tests/test_feature_selection_polars.py`

- [x] **Step 1: Add a commodity contract IC fixture helper**

Add this helper near `_write_ic_fixture` in `data_preprocess/tests/test_feature_selection_polars.py`:

```python
def _write_contract_ic_fixture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(
        {
            "timestamp": list(range(12)),
            "mark_price": [100.0 + i for i in range(12)],
            "index_price": [100.0 + i for i in range(12)],
            "funding_timestamp": list(range(12)),
            "funding_rate": [0.0 for _ in range(12)],
            "ask1_price": [101.0 + i for i in range(12)],
            "ask1_size": [10.0 for _ in range(12)],
            "bid1_price": [99.0 + i for i in range(12)],
            "bid1_size": [11.0 for _ in range(12)],
            "feature_a": [float(i) for i in range(12)],
            "feature_b": [float(12 - i) for i in range(12)],
        }
    )
    frame.write_ipc(path)
```

- [x] **Step 2: Add the failing candidate-only CLI test**

Add this test below `test_ic_correlation_cli_writes_expected_files`:

```python
def test_ic_correlation_candidate_only_writes_candidate_artifacts(tmp_path):
    input_file = (
        tmp_path
        / "PREPROCESS_DATASET/commodity-futures/ALL_FEATURE/fu/fu2601/5min"
        / "2026-01-05-2026-01-06.feather"
    )
    _write_contract_ic_fixture(input_file)

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
            "--contract",
            "fu2601",
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
            "--candidate_only",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "data_preprocess")},
        check=True,
    )

    output_dir = (
        tmp_path
        / "PREPROCESS_DATASET/commodity-futures/IC_RESULT/fu/fu2601/5min"
        / "2026-01-05-2026-01-06"
    )
    assert (output_dir / "state_features_candidate.npy").exists()
    assert (output_dir / "candidate_manifest.json").exists()
    assert (output_dir / "ic_window_1.json").exists()
    assert (output_dir / "correlation.csv").exists()
    assert not (output_dir / "df.feather").exists()
    assert not (output_dir / "state_features.npy").exists()
```

- [x] **Step 3: Run the candidate-only test and confirm it fails**

Run: `conda run -n finetf pytest data_preprocess/tests/test_feature_selection_polars.py::test_ic_correlation_candidate_only_writes_candidate_artifacts -q`

Expected: FAIL with an argparse error for unknown `--candidate_only`.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 2: Candidate-only implementation

> **trace:** plan-ready.md → `### Task 2: Candidate-only implementation` | tasks.md → `- [ ] 1.2 Implement commodity candidate-only mode in `ic_correlation.py` while preserving default IC output compatibility.`
> **sync:** tasks.md → `- [ ] 1.2 Implement commodity candidate-only mode in `ic_correlation.py` while preserving default IC output compatibility.` | plan-ready.md → `### Task 2: Candidate-only implementation`

**Files:**
- Modify: `data_preprocess/operator_futures/feature_selection/ic_correlation.py`
- Test: `data_preprocess/tests/test_feature_selection_polars.py`

- [x] **Step 1: Add the CLI flag**

Add this argument after `--orderbook_depth` in `ic_correlation.py`:

```python
parser.add_argument(
    "--candidate_only",
    action="store_true",
    help="write selected feature candidates and reports without writing final df.feather",
)
```

- [x] **Step 2: Add a helper for candidate manifest writing**

Add this helper above `main`:

```python
def write_candidate_outputs(
    output_dir: Path,
    symbol: str,
    contract: str | None,
    target_freq: str,
    start_date: str,
    end_date: str,
    state_features: list[str],
    reward_features: list[str],
    input_path: Path,
) -> None:
    np.save(output_dir / "state_features_candidate.npy", np.array(state_features))
    manifest = {
        "symbol": symbol,
        "contract": contract,
        "target_freq": target_freq,
        "start_date": start_date,
        "end_date": end_date,
        "input_path": str(input_path),
        "candidate_state_feature_count": len(state_features),
        "reward_feature_count": len(reward_features),
        "state_features_candidate_path": str(output_dir / "state_features_candidate.npy"),
        "state_features": state_features,
    }
    (output_dir / "candidate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
```

- [x] **Step 3: Branch before final `df.feather` writing**

Replace the final output block in `main` after `state_feature = selected_feature_names` with this:

```python
    state_feature = selected_feature_names
    if args.candidate_only:
        logger.info(
            "Writing IC candidate outputs: selected_state_features=%d output_dir=%s",
            len(state_feature),
            output_dir,
        )
        write_candidate_outputs(
            output_dir=output_dir,
            symbol=args.symbols,
            contract=args.contract,
            target_freq=args.target_freq,
            start_date=args.start_date,
            end_date=args.end_date,
            state_features=state_feature,
            reward_features=reward_features,
            input_path=input_path,
        )
        logger.info(
            "Finished IC candidate process: rows=%d candidate_features=%d elapsed_seconds=%.2f",
            df.height,
            len(state_feature),
            time.monotonic() - started_at,
        )
        return df

    out = df.select([*reward_features, *state_feature])
    logger.info(
        "Writing IC outputs: selected_state_features=%d total_columns=%d output_dir=%s",
        len(state_feature),
        len(out.columns),
        output_dir,
    )
    out.write_ipc(output_dir / "df.feather")
    np.save(output_dir / "state_features.npy", np.array(state_feature))
```

- [x] **Step 4: Run candidate-only and default IC tests**

Run: `conda run -n finetf pytest data_preprocess/tests/test_feature_selection_polars.py::test_ic_correlation_candidate_only_writes_candidate_artifacts data_preprocess/tests/test_feature_selection_polars.py::test_ic_correlation_cli_writes_expected_files -q`

Expected: PASS. The candidate-only test has no `df.feather`; the default test still has `df.feather` and `state_features.npy`.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 3: Union finalize happy-path tests

> **trace:** plan-ready.md → `### Task 3: Union finalize happy-path tests` | tasks.md → `- [ ] 1.3 Add focused tests for union finalize loading candidate features, writing品种级 `FEATURE_UNION`, and writing per-contract filtered `IC_RESULT`.`
> **sync:** tasks.md → `- [ ] 1.3 Add focused tests for union finalize loading candidate features, writing品种级 `FEATURE_UNION`, and writing per-contract filtered `IC_RESULT`.` | plan-ready.md → `### Task 3: Union finalize happy-path tests`

**Files:**
- Modify: `data_preprocess/tests/test_commodity_feature_pipeline.py`

- [x] **Step 1: Import Polars**

At the top of `data_preprocess/tests/test_commodity_feature_pipeline.py`, add:

```python
import polars as pl
```

- [x] **Step 2: Add a reusable summary helper**

Add this helper near `test_write_contract_feature_union_writes_symbol_level_manifest`:

```python
def _write_two_contract_summary(path):
    path.write_text(
        json.dumps(
            {
                "symbol": "fu",
                "commodity_name": "燃料油",
                "start_date": "2026-01-01",
                "end_date": "2026-04-01",
                "contracts": [
                    {
                        "contract": "fu2601",
                        "start_trading_day": "20260101",
                        "end_trading_day": "20260102",
                        "trading_day_count": 1,
                        "selected_months": ["2026-01"],
                        "trading_days": [
                            {
                                "trading_day": "20260101",
                                "date": "2026-01-01",
                                "source_file": "fu2601.csv",
                                "daily_volume": 1.0,
                            }
                        ],
                    },
                    {
                        "contract": "fu2605",
                        "start_trading_day": "20260201",
                        "end_trading_day": "20260202",
                        "trading_day_count": 1,
                        "selected_months": ["2026-02"],
                        "trading_days": [
                            {
                                "trading_day": "20260201",
                                "date": "2026-02-01",
                                "source_file": "fu2605.csv",
                                "daily_volume": 1.0,
                            }
                        ],
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
```

- [x] **Step 3: Add the finalize happy-path test**

Add this test:

```python
def test_write_contract_feature_union_finalizes_ic_result_from_candidates(tmp_path):
    summary_path = tmp_path / "main_contract_summary.json"
    _write_two_contract_summary(summary_path)
    base = tmp_path / "PREPROCESS_DATASET" / "commodity-futures"
    date_range = "2026-01-01-2026-04-01"

    first_candidate = base / "IC_RESULT" / "fu" / "fu2601" / "5min" / date_range
    second_candidate = base / "IC_RESULT" / "fu" / "fu2605" / "5min" / date_range
    first_candidate.mkdir(parents=True)
    second_candidate.mkdir(parents=True)
    np.save(first_candidate / "state_features_candidate.npy", np.array(["alpha", "beta"]))
    np.save(second_candidate / "state_features_candidate.npy", np.array(["beta", "gamma"]))

    for contract in ["fu2601", "fu2605"]:
        all_feature_dir = base / "ALL_FEATURE" / "fu" / contract / "5min"
        all_feature_dir.mkdir(parents=True)
        pl.DataFrame(
            {
                "timestamp": [1, 2],
                "mark_price": [100.0, 101.0],
                "index_price": [100.0, 101.0],
                "funding_timestamp": [1, 2],
                "funding_rate": [0.0, 0.0],
                "ask1_price": [101.0, 102.0],
                "ask1_size": [10.0, 11.0],
                "bid1_price": [99.0, 100.0],
                "bid1_size": [12.0, 13.0],
                "alpha": [1.0, 2.0],
                "beta": [3.0, 4.0],
                "gamma": [5.0, 6.0],
            }
        ).write_ipc(all_feature_dir / f"{date_range}.feather")

    output_dir = write_contract_feature_union(
        root_path=tmp_path,
        summary_path=summary_path,
        symbol="fu",
        target_freq="5min",
        start_date="2026-01-01",
        end_date="2026-04-01",
        candidate_path="PREPROCESS_DATASET/commodity-futures/IC_RESULT",
        all_feature_path="PREPROCESS_DATASET/commodity-futures/ALL_FEATURE",
        ic_result_path="PREPROCESS_DATASET/commodity-futures/IC_RESULT",
        finalize_filtered_df=True,
        market_type="commodity_futures",
        orderbook_depth=5,
    )

    assert np.load(output_dir / "state_features.npy", allow_pickle=True).tolist() == [
        "alpha",
        "beta",
        "gamma",
    ]
    for contract in ["fu2601", "fu2605"]:
        contract_dir = base / "IC_RESULT" / "fu" / contract / "5min" / date_range
        assert np.load(contract_dir / "state_features.npy", allow_pickle=True).tolist() == [
            "alpha",
            "beta",
            "gamma",
        ]
        frame = pl.read_ipc(contract_dir / "df.feather")
        assert frame.columns == [
            "timestamp",
            "mark_price",
            "index_price",
            "funding_timestamp",
            "funding_rate",
            "ask1_price",
            "ask1_size",
            "bid1_price",
            "bid1_size",
            "alpha",
            "beta",
            "gamma",
        ]

    manifest = json.loads(
        (output_dir / "feature_union_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["state_feature_count"] == 3
    assert manifest["per_contract_output_shapes"]["fu2601"]["rows"] == 2
    assert manifest["per_contract_output_shapes"]["fu2605"]["columns"] == 12
```

- [x] **Step 4: Run the happy-path finalize test and confirm it fails**

Run: `conda run -n finetf pytest data_preprocess/tests/test_commodity_feature_pipeline.py::test_write_contract_feature_union_finalizes_ic_result_from_candidates -q`

Expected: FAIL because `write_contract_feature_union` does not accept `candidate_path`, `all_feature_path`, `ic_result_path`, `finalize_filtered_df`, `market_type`, or `orderbook_depth`.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 4: Union finalize implementation

> **trace:** plan-ready.md → `### Task 4: Union finalize implementation` | tasks.md → `- [ ] 1.4 Extend `contract_feature_union.py` to read IC candidates, build union, validate all contract columns, and write per-contract filtered `IC_RESULT` outputs.`
> **sync:** tasks.md → `- [ ] 1.4 Extend `contract_feature_union.py` to read IC candidates, build union, validate all contract columns, and write per-contract filtered `IC_RESULT` outputs.` | plan-ready.md → `### Task 4: Union finalize implementation`

**Files:**
- Modify: `data_preprocess/operator_futures/feature_selection/contract_feature_union.py`
- Test: `data_preprocess/tests/test_commodity_feature_pipeline.py`

- [x] **Step 1: Add Polars and util imports**

At the top of `contract_feature_union.py`, add:

```python
import polars as pl

from operator_futures.feature_selection.ic_correlation import select_reward_state_features
```

- [x] **Step 2: Add path and validation helpers**

Add these helpers below `_load_state_features`:

```python
def _feature_path(
    root_path: Path,
    base_path: str,
    symbol: str,
    contract: str,
    target_freq: str,
    date_range: str,
    file_name: str,
) -> Path:
    return root_path / base_path / symbol / contract / target_freq / date_range / file_name


def _all_feature_path(
    root_path: Path,
    all_feature_path: str,
    symbol: str,
    contract: str,
    target_freq: str,
    date_range: str,
) -> Path:
    return root_path / all_feature_path / symbol / contract / target_freq / f"{date_range}.feather"


def _missing_features(df: pl.DataFrame, required: Sequence[str]) -> list[str]:
    available = set(df.columns)
    return [feature for feature in required if feature not in available]
```

- [x] **Step 3: Extend `write_contract_feature_union` signature and feature source selection**

Change the function signature to:

```python
def write_contract_feature_union(
    root_path: Path,
    summary_path: Path,
    symbol: str,
    target_freq: str,
    start_date: str,
    end_date: str,
    scale_save_path: str = "PREPROCESS_DATASET/commodity-futures/SCALE_SAVE",
    save_path: str = "PREPROCESS_DATASET/commodity-futures/FEATURE_UNION",
    candidate_path: str | None = None,
    all_feature_path: str = "PREPROCESS_DATASET/commodity-futures/ALL_FEATURE",
    ic_result_path: str = "PREPROCESS_DATASET/commodity-futures/IC_RESULT",
    finalize_filtered_df: bool = False,
    market_type: str = "commodity_futures",
    orderbook_depth: int = 5,
) -> Path:
```

Inside the contract loop, replace the current `scale_root` feature path with:

```python
    feature_base_path = candidate_path if candidate_path is not None else scale_save_path
    feature_file_name = (
        "state_features_candidate.npy" if candidate_path is not None else "state_features.npy"
    )

    contract_features: dict[str, list[str]] = {}
    contract_feature_paths: dict[str, str] = {}
    for contract in summary.contracts:
        feature_path = _feature_path(
            root_path,
            feature_base_path,
            symbol,
            contract.contract,
            target_freq,
            date_range,
            feature_file_name,
        )
        contract_features[contract.contract] = _load_state_features(
            feature_path, contract.contract
        )
        contract_feature_paths[contract.contract] = str(feature_path)
```

- [x] **Step 4: Add finalize writing after union manifest setup**

After `union = build_union_state_features(...)`, add this block before writing the manifest:

```python
    if finalize_filtered_df and not union:
        raise ValueError("Feature union is empty; cannot finalize filtered IC_RESULT files")

    per_contract_outputs: dict[str, str] = {}
    per_contract_output_shapes: dict[str, dict[str, int]] = {}
    if finalize_filtered_df:
        for contract in summary.contracts:
            input_path = _all_feature_path(
                root_path,
                all_feature_path,
                symbol,
                contract.contract,
                target_freq,
                date_range,
            )
            if not input_path.exists():
                raise FileNotFoundError(
                    f"Missing ALL_FEATURE input for contract {contract.contract}: {input_path}"
                )
            df = pl.read_ipc(input_path)
            reward_features, _ = select_reward_state_features(
                df, market_type=market_type, orderbook_depth=orderbook_depth
            )
            required_columns = [*reward_features, *union]
            missing = _missing_features(df, required_columns)
            if missing:
                raise ValueError(
                    f"Contract {contract.contract} missing union features: {missing}"
                )
            contract_output_dir = (
                root_path
                / ic_result_path
                / symbol
                / contract.contract
                / target_freq
                / date_range
            )
            contract_output_dir.mkdir(parents=True, exist_ok=True)
            out = df.select(required_columns)
            out.write_ipc(contract_output_dir / "df.feather")
            np.save(contract_output_dir / "state_features.npy", np.array(union))
            per_contract_outputs[contract.contract] = str(contract_output_dir / "df.feather")
            per_contract_output_shapes[contract.contract] = {
                "rows": out.height,
                "columns": len(out.columns),
            }
```

- [x] **Step 5: Extend manifest**

Add these keys to the existing `manifest` dictionary:

```python
        "candidate_source_path": candidate_path,
        "all_feature_path": all_feature_path,
        "ic_result_path": ic_result_path,
        "finalize_filtered_df": finalize_filtered_df,
        "per_contract_output_paths": per_contract_outputs,
        "per_contract_output_shapes": per_contract_output_shapes,
```

- [x] **Step 6: Add CLI arguments and pass them through**

Add parser arguments:

```python
parser.add_argument("--candidate_path", type=str, default=None)
parser.add_argument(
    "--all_feature_path",
    type=str,
    default="PREPROCESS_DATASET/commodity-futures/ALL_FEATURE",
)
parser.add_argument(
    "--ic_result_path",
    type=str,
    default="PREPROCESS_DATASET/commodity-futures/IC_RESULT",
)
parser.add_argument("--finalize_filtered_df", action="store_true")
parser.add_argument(
    "--market_type",
    type=str,
    default="commodity_futures",
    choices=["crypto_futures", "commodity_futures"],
)
parser.add_argument("--orderbook_depth", type=int, default=5)
```

Pass them in `main`:

```python
        candidate_path=args.candidate_path,
        all_feature_path=args.all_feature_path,
        ic_result_path=args.ic_result_path,
        finalize_filtered_df=args.finalize_filtered_df,
        market_type=args.market_type,
        orderbook_depth=args.orderbook_depth,
```

- [x] **Step 7: Run happy-path and existing union tests**

Run: `conda run -n finetf pytest data_preprocess/tests/test_commodity_feature_pipeline.py::test_write_contract_feature_union_finalizes_ic_result_from_candidates data_preprocess/tests/test_commodity_feature_pipeline.py::test_write_contract_feature_union_writes_symbol_level_manifest -q`

Expected: PASS. Existing scale-save based union behavior remains compatible; new candidate finalize path writes `FEATURE_UNION` and per-contract `IC_RESULT`.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 5: Union finalize fail-fast tests

> **trace:** plan-ready.md → `### Task 5: Union finalize fail-fast tests` | tasks.md → `- [ ] 1.5 Add fail-fast tests for missing candidate files, empty union, and union features missing from a contract `ALL_FEATURE`.`
> **sync:** tasks.md → `- [ ] 1.5 Add fail-fast tests for missing candidate files, empty union, and union features missing from a contract `ALL_FEATURE`.` | plan-ready.md → `### Task 5: Union finalize fail-fast tests`

**Files:**
- Modify: `data_preprocess/tests/test_commodity_feature_pipeline.py`
- Modify: `data_preprocess/operator_futures/feature_selection/contract_feature_union.py`

- [x] **Step 1: Add missing candidate assertion**

Update `test_write_contract_feature_union_fails_when_contract_state_features_missing` by calling `write_contract_feature_union` with `candidate_path="PREPROCESS_DATASET/commodity-futures/IC_RESULT"` and assert the message contains `state_features_candidate.npy`:

```python
    with pytest.raises(FileNotFoundError) as excinfo:
        write_contract_feature_union(
            root_path=tmp_path,
            summary_path=summary_path,
            symbol="fu",
            target_freq="5min",
            start_date="2026-01-01",
            end_date="2026-04-01",
            candidate_path="PREPROCESS_DATASET/commodity-futures/IC_RESULT",
            finalize_filtered_df=True,
        )

    message = str(excinfo.value)
    assert "fu2605" in message
    assert "state_features_candidate.npy" in message
```

- [x] **Step 2: Add empty union failure test**

Add this test:

```python
def test_write_contract_feature_union_fails_when_candidate_union_empty(tmp_path):
    summary_path = tmp_path / "main_contract_summary.json"
    _write_two_contract_summary(summary_path)
    date_range = "2026-01-01-2026-04-01"
    for contract in ["fu2601", "fu2605"]:
        candidate_dir = (
            tmp_path
            / "PREPROCESS_DATASET/commodity-futures/IC_RESULT/fu"
            / contract
            / "5min"
            / date_range
        )
        candidate_dir.mkdir(parents=True)
        np.save(candidate_dir / "state_features_candidate.npy", np.array([]))

    with pytest.raises(ValueError) as excinfo:
        write_contract_feature_union(
            root_path=tmp_path,
            summary_path=summary_path,
            symbol="fu",
            target_freq="5min",
            start_date="2026-01-01",
            end_date="2026-04-01",
            candidate_path="PREPROCESS_DATASET/commodity-futures/IC_RESULT",
            finalize_filtered_df=True,
        )

    assert "Feature union is empty" in str(excinfo.value)
```

- [x] **Step 3: Add missing union column failure test**

Add this test:

```python
def test_write_contract_feature_union_fails_when_union_feature_missing_from_contract(
    tmp_path,
):
    summary_path = tmp_path / "main_contract_summary.json"
    _write_two_contract_summary(summary_path)
    base = tmp_path / "PREPROCESS_DATASET" / "commodity-futures"
    date_range = "2026-01-01-2026-04-01"

    for contract, features in {
        "fu2601": ["alpha", "gamma"],
        "fu2605": ["gamma"],
    }.items():
        candidate_dir = base / "IC_RESULT" / "fu" / contract / "5min" / date_range
        candidate_dir.mkdir(parents=True)
        np.save(candidate_dir / "state_features_candidate.npy", np.array(features))

    all_feature_dir = base / "ALL_FEATURE" / "fu" / "fu2601" / "5min"
    all_feature_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "timestamp": [1],
            "mark_price": [100.0],
            "alpha": [1.0],
            "gamma": [2.0],
        }
    ).write_ipc(all_feature_dir / f"{date_range}.feather")

    missing_dir = base / "ALL_FEATURE" / "fu" / "fu2605" / "5min"
    missing_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "timestamp": [1],
            "mark_price": [100.0],
            "gamma": [2.0],
        }
    ).write_ipc(missing_dir / f"{date_range}.feather")

    with pytest.raises(ValueError) as excinfo:
        write_contract_feature_union(
            root_path=tmp_path,
            summary_path=summary_path,
            symbol="fu",
            target_freq="5min",
            start_date="2026-01-01",
            end_date="2026-04-01",
            candidate_path="PREPROCESS_DATASET/commodity-futures/IC_RESULT",
            finalize_filtered_df=True,
            market_type="commodity_futures",
            orderbook_depth=5,
        )

    message = str(excinfo.value)
    assert "fu2605" in message
    assert "alpha" in message
```

- [x] **Step 4: Run fail-fast tests**

Run: `conda run -n finetf pytest data_preprocess/tests/test_commodity_feature_pipeline.py::test_write_contract_feature_union_fails_when_candidate_union_empty data_preprocess/tests/test_commodity_feature_pipeline.py::test_write_contract_feature_union_fails_when_union_feature_missing_from_contract data_preprocess/tests/test_commodity_feature_pipeline.py::test_write_contract_feature_union_fails_when_contract_state_features_missing -q`

Expected: PASS. Failures mention the contract and missing candidate or feature.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 6: Commodity full-process shell ordering

> **trace:** plan-ready.md → `### Task 6: Commodity full-process shell ordering` | tasks.md → `- [ ] 1.6 Update `fu_full_process.sh` tests and shell flow so candidate runs inside the contract loop, union finalize runs once after all candidates, and `scale_save` runs per contract after finalize.`
> **sync:** tasks.md → `- [ ] 1.6 Update `fu_full_process.sh` tests and shell flow so candidate runs inside the contract loop, union finalize runs once after all candidates, and `scale_save` runs per contract after finalize.` | plan-ready.md → `### Task 6: Commodity full-process shell ordering`

**Files:**
- Modify: `data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh`
- Modify: `data_preprocess/tests/test_commodity_main_contract_cli.py`

- [x] **Step 1: Update the step logging test stubs**

In `test_commodity_full_process_writes_step_logs_and_preserves_child_log_paths`, replace the IC/scale/union stubs with:

```bash
run_commodity_ic_candidate() { echo "ic candidate stdout"; }
run_commodity_ic_union_finalize() {
    local summary_path=$1
    local target_freq=$2
    local start_date=$3
    local end_date=$4
    local symbol=$5
    echo "ic_union_finalize:${symbol}:${target_freq}:${start_date}:${end_date}:${summary_path}"
}
run_commodity_scale_save() { echo "scale stdout"; }
```

Update `symbol_by_step`:

```python
    symbol_by_step = {
        "stitch_main_contract": "fu",
        "downscale_continuous_by_trading_day": "fu",
        "cross_section": "fu_fu2601",
        "merge": "fu_fu2601",
        "concat": "fu_fu2601",
        "time_feature": "fu_fu2601",
        "merge_clean": "fu_fu2601",
        "ic_candidate": "fu_fu2601",
        "ic_union_finalize": "fu",
        "scale_save": "fu_fu2601",
        "maintenance_margin_dict": "fu",
    }
```

Update the log assertion:

```python
    union_log = (
        tmp_path
        / "log_futures"
        / "ticker_result"
        / "commodity"
        / "steps"
        / "fu_5min_2026-01-05_2026-01-07_ic_union_finalize.log"
    )
    assert "ic_union_finalize:fu:5min:2026-01-05:2026-01-07:" in (
        union_log.read_text(encoding="utf-8")
    )
```

- [x] **Step 2: Add a static ordering test**

Add this test near `test_commodity_full_process_shell_scales_ic_selection_output`:

```python
def test_commodity_full_process_shell_runs_scale_after_ic_union_finalize():
    script = (
        REPO_ROOT
        / "data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh"
    )
    text = script.read_text(encoding="utf-8")

    assert "run_commodity_ic_candidate()" in text
    assert "run_commodity_ic_union_finalize()" in text
    assert '"ic_candidate"' in text
    assert '"ic_union_finalize"' in text
    assert '"feature_union"' not in text
    assert text.index('"ic_union_finalize"') < text.rindex('"scale_save"')
```

- [x] **Step 3: Run shell tests and confirm they fail**

Run: `conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract_cli.py::test_commodity_full_process_writes_step_logs_and_preserves_child_log_paths data_preprocess/tests/test_commodity_main_contract_cli.py::test_commodity_full_process_shell_runs_scale_after_ic_union_finalize -q`

Expected: FAIL because the shell still uses `ic_correlation`, runs `scale_save` inside the first loop, and has the old `feature_union` stage.

- [x] **Step 4: Rename commodity IC shell function and pass candidate flag**

In `fu_full_process.sh`, replace `run_commodity_ic_correlation()` with:

```bash
run_commodity_ic_candidate() {
    local target_freq=$1
    local start_date=$2
    local end_date=$3
    local symbol=$4
    local root_path=$5
    local contract=${6:-}
    local contract_args=()
    if [ -n "${contract}" ]; then
        contract_args=(--contract "${contract}")
    fi

    PYTHONPATH="${root_path}/data_preprocess" python -u data_preprocess/operator_futures/feature_selection/ic_correlation.py \
        --symbols "$symbol" \
        "${contract_args[@]}" \
        --target_freq "$target_freq" \
        --start_date "$start_date" \
        --end_date "$end_date" \
        --root_path "$root_path" \
        --data_path "PREPROCESS_DATASET/commodity-futures/ALL_FEATURE/" \
        --save_path "PREPROCESS_DATASET/commodity-futures/IC_RESULT/" \
        --market_type commodity_futures \
        --orderbook_depth 5 \
        --candidate_only
}
```

- [x] **Step 5: Replace feature union function with finalize function**

Replace `run_commodity_feature_union()` with:

```bash
run_commodity_ic_union_finalize() {
    local summary_path=$1
    local target_freq=$2
    local start_date=$3
    local end_date=$4
    local symbol=$5
    local root_path=$6

    PYTHONPATH="${root_path}/data_preprocess" python -u -m operator_futures.feature_selection.contract_feature_union \
        --summary "${summary_path}" \
        --symbols "${symbol}" \
        --target_freq "${target_freq}" \
        --start_date "${start_date}" \
        --end_date "${end_date}" \
        --root_path "${root_path}" \
        --candidate_path "PREPROCESS_DATASET/commodity-futures/IC_RESULT" \
        --all_feature_path "PREPROCESS_DATASET/commodity-futures/ALL_FEATURE" \
        --ic_result_path "PREPROCESS_DATASET/commodity-futures/IC_RESULT" \
        --save_path "PREPROCESS_DATASET/commodity-futures/FEATURE_UNION" \
        --finalize_filtered_df \
        --market_type commodity_futures \
        --orderbook_depth 5
}
```

- [x] **Step 6: Split the full process loops**

In `run_commodity_full_process`, keep the first contract loop through `ic_candidate` and remove `scale_save` from it. After the loop, add union finalize and a second scale-save loop:

```bash
        run_commodity_logged_step \
            "$log_dir" "${symbol}_${contract}" "$target_freq" "$start_date" "$end_date" \
            "ic_candidate" \
            run_commodity_ic_candidate "$target_freq" "$start_date" "$end_date" "$symbol" "$root_path" "$contract"
    done < <(run_commodity_summary_contracts "$summary_path")

    run_commodity_logged_step \
        "$log_dir" "$symbol" "$target_freq" "$start_date" "$end_date" \
        "ic_union_finalize" \
        run_commodity_ic_union_finalize "$summary_path" "$target_freq" "$start_date" "$end_date" "$symbol" "$root_path"

    while IFS= read -r contract; do
        [ -n "$contract" ] || continue
        run_commodity_logged_step \
            "$log_dir" "${symbol}_${contract}" "$target_freq" "$start_date" "$end_date" \
            "scale_save" \
            run_commodity_scale_save "$target_freq" "$start_date" "$end_date" "$symbol" "$root_path" "$contract"
    done < <(run_commodity_summary_contracts "$summary_path")
```

Remove the old logged `"feature_union"` call.

- [x] **Step 7: Run shell tests**

Run: `conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract_cli.py::test_commodity_full_process_writes_step_logs_and_preserves_child_log_paths data_preprocess/tests/test_commodity_main_contract_cli.py::test_commodity_full_process_shell_runs_scale_after_ic_union_finalize data_preprocess/tests/test_commodity_main_contract_cli.py::test_commodity_full_process_shell_sets_pythonpath_for_operator_scripts -q`

Expected: PASS. The total log contains `ic_candidate`, `ic_union_finalize`, and `scale_save`, with `scale_save` after finalize.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 7: Pipeline artifact and manifest regression

> **trace:** plan-ready.md → `### Task 7: Pipeline artifact and manifest regression` | tasks.md → `- [ ] 1.7 Update validation entrypoint or commodity feature pipeline tests to cover the new final artifact layout and manifest content.`
> **sync:** tasks.md → `- [ ] 1.7 Update validation entrypoint or commodity feature pipeline tests to cover the new final artifact layout and manifest content.` | plan-ready.md → `### Task 7: Pipeline artifact and manifest regression`

**Files:**
- Modify: `data_preprocess/tests/test_commodity_feature_pipeline.py`
- Modify: `data_preprocess/tests/test_commodity_main_contract_cli.py`
- Review: `data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh`

- [x] **Step 1: Strengthen manifest assertions in the happy-path test**

In `test_write_contract_feature_union_finalizes_ic_result_from_candidates`, add:

```python
    assert manifest["candidate_source_path"] == "PREPROCESS_DATASET/commodity-futures/IC_RESULT"
    assert manifest["all_feature_path"] == "PREPROCESS_DATASET/commodity-futures/ALL_FEATURE"
    assert manifest["ic_result_path"] == "PREPROCESS_DATASET/commodity-futures/IC_RESULT"
    assert manifest["finalize_filtered_df"] is True
    assert set(manifest["per_contract_output_paths"]) == {"fu2601", "fu2605"}
    for output_path in manifest["per_contract_output_paths"].values():
        assert Path(output_path).exists()
```

- [x] **Step 2: Update validate feature shell static assertion if needed**

If `validate_features.sh` checks only the品种级 `FEATURE_UNION`, keep the existing behavior and add a test assertion that the full process creates standard per-contract IC outputs before scale-save. Add this assertion to `test_validate_features_checks_feature_union_outputs`:

```python
    assert "FEATURE_UNION" in text
    assert "feature_union_manifest.json" in text
    assert "state_features.npy" in text
```

The current test already has these assertions; if no script change is needed, leave `validate_features.sh` unchanged.

- [x] **Step 3: Run pipeline artifact tests**

Run: `conda run -n finetf pytest data_preprocess/tests/test_commodity_feature_pipeline.py::test_write_contract_feature_union_finalizes_ic_result_from_candidates data_preprocess/tests/test_commodity_main_contract_cli.py::test_validate_features_checks_feature_union_outputs -q`

Expected: PASS. The manifest records candidate source, ALL_FEATURE source, IC_RESULT output root, finalization flag, and real per-contract output paths.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 8: OpenSpec validation

> **trace:** plan-ready.md → `### Task 8: OpenSpec validation` | tasks.md → `- [ ] 2.1 Run `openspec validate refactor-commodity-feature-selection-union --strict`.`
> **sync:** tasks.md → `- [ ] 2.1 Run `openspec validate refactor-commodity-feature-selection-union --strict`.` | plan-ready.md → `### Task 8: OpenSpec validation`

**Files:**
- Read: `openspec/changes/refactor-commodity-feature-selection-union/proposal.md`
- Read: `openspec/changes/refactor-commodity-feature-selection-union/design.md`
- Read: `openspec/changes/refactor-commodity-feature-selection-union/specs/commodity-futures-support/spec.md`
- Read: `openspec/changes/refactor-commodity-feature-selection-union/tasks.md`

- [x] **Step 1: Run strict OpenSpec validation**

Run: `openspec validate refactor-commodity-feature-selection-union --strict`

Expected: PASS with `Change 'refactor-commodity-feature-selection-union' is valid`.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 9: Focused pytest regression

> **trace:** plan-ready.md → `### Task 9: Focused pytest regression` | tasks.md → `- [ ] 2.2 Run `conda run -n finetf pytest data_preprocess/tests/test_feature_selection_polars.py data_preprocess/tests/test_commodity_feature_pipeline.py data_preprocess/tests/test_commodity_main_contract_cli.py -q`.`
> **sync:** tasks.md → `- [ ] 2.2 Run `conda run -n finetf pytest data_preprocess/tests/test_feature_selection_polars.py data_preprocess/tests/test_commodity_feature_pipeline.py data_preprocess/tests/test_commodity_main_contract_cli.py -q`.` | plan-ready.md → `### Task 9: Focused pytest regression`

**Files:**
- Test: `data_preprocess/tests/test_feature_selection_polars.py`
- Test: `data_preprocess/tests/test_commodity_feature_pipeline.py`
- Test: `data_preprocess/tests/test_commodity_main_contract_cli.py`

- [x] **Step 1: Run focused regression tests**

Run: `conda run -n finetf pytest data_preprocess/tests/test_feature_selection_polars.py data_preprocess/tests/test_commodity_feature_pipeline.py data_preprocess/tests/test_commodity_main_contract_cli.py -q`

Expected: PASS. Failures in unrelated tests should be inspected before marking this task complete because this change touches shared commodity preprocessing entry points.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 10: Static shell assertions

> **trace:** plan-ready.md → `### Task 10: Static shell assertions` | tasks.md → `- [ ] 2.3 Run any focused static shell assertions that verify `fu_full_process.sh` no longer has the old separate post-loop `feature_union` stage and runs `scale_save` only after `ic_union_finalize`.`
> **sync:** tasks.md → `- [ ] 2.3 Run any focused static shell assertions that verify `fu_full_process.sh` no longer has the old separate post-loop `feature_union` stage and runs `scale_save` only after `ic_union_finalize`.` | plan-ready.md → `### Task 10: Static shell assertions`

**Files:**
- Test: `data_preprocess/tests/test_commodity_main_contract_cli.py`
- Read: `data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh`

- [x] **Step 1: Run the shell static ordering test**

Run: `conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract_cli.py::test_commodity_full_process_shell_runs_scale_after_ic_union_finalize -q`

Expected: PASS. The test confirms the shell has `ic_candidate`, `ic_union_finalize`, no old `"feature_union"` stage string, and `scale_save` appears after `ic_union_finalize`.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）
