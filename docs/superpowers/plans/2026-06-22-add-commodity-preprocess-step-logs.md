# add-commodity-preprocess-step-logs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add dedicated logs for each major step in the commodity preprocess flow while preserving the existing total log and fail-fast behavior.

**Architecture:** Keep the logging change at the shell pipeline layer so the commodity preprocess scripts remain data-pipeline focused. Wrap each major stage in `fu_full_process.sh` with a small logging helper that prints stage start/end to the total log and redirects stdout/stderr to a stage-specific log file. Preserve the existing per-date child logs already emitted by cross-section and merge subprocesses.

**Tech Stack:** Bash, Python shell entry points, pytest, OpenSpec.

**Traceability (sddflow):**
- plan-ready: `openspec/changes/add-commodity-preprocess-step-logs/plan-ready.md`
- tasks: `openspec/changes/add-commodity-preprocess-step-logs/tasks.md`
- plan: `docs/superpowers/plans/2026-06-22-add-commodity-preprocess-step-logs.md`

---

### Task 1: Step-level logging implementation

> **trace:** plan-ready.md → `### Task 1: Step-level logging implementation` | tasks.md → `- [ ] 1.1 Add focused shell/script tests for commodity step log naming, stage status output, stderr capture, and fail-fast behavior.`
> **sync:** tasks.md → `- [ ] 1.1 Add focused shell/script tests for commodity step log naming, stage status output, stderr capture, and fail-fast behavior.` | plan-ready.md → `### Task 1: Step-level logging implementation`

**Files:**
- Modify: `data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh`
- Test: `data_preprocess/tests/test_commodity_scripts_docs.py` or a new focused commodity script test file

- [x] **Step 1: Write the failing test**

```python
import os
import subprocess
from pathlib import Path


def test_commodity_full_process_writes_step_logs(tmp_path):
    root = tmp_path
    log_dir = root / "log_futures/ticker_result/commodity"
    log_dir.mkdir(parents=True)
    cmd = [
        "bash",
        "data_preprocess/script_preprocess/future_upgraded/commodity/main.sh",
    ]
    env = {
        **os.environ,
        "ROOTPATH": str(root),
        "START_DATE": "2025-11-03",
        "END_DATE": "2025-11-08",
        "TARGET_FREQ": "5min",
        "SYMBOL": "fu",
        "COMMODITY_NAME": "燃料油",
    }
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
    total_log = log_dir / "fu_5min_2025-11-03_2025-11-08.log"
    assert total_log.exists()
    text = total_log.read_text(encoding="utf-8")
    assert "stitch_main_contract" in text
    assert "scale_save" in text
    assert "step log" in text
```

- [x] **Step 2: Run the test to verify it fails**

Run: `conda run -n finetf pytest data_preprocess/tests/test_commodity_scripts_docs.py::test_commodity_full_process_writes_step_logs -q`
Expected: FAIL because the current pipeline only writes a total log and does not emit stage-specific log paths.

- [x] **Step 3: Implement the minimal shell logging wrapper**

```bash
run_logged_step() {
    local step_name=$1
    shift
    local step_log_dir="${LOG_DIR}/steps"
    local step_log="${step_log_dir}/${SYMBOL}_${TARGET_FREQ}_${START_DATE}_${END_DATE}_${step_name}.log"
    mkdir -p "${step_log_dir}"
    echo "[commodity][${step_name}] start -> ${step_log}"
    if "$@" >"${step_log}" 2>&1; then
        echo "[commodity][${step_name}] success -> ${step_log}"
    else
        local status=$?
        echo "[commodity][${step_name}] failed(${status}) -> ${step_log}"
        return "${status}"
    fi
}

run_logged_step stitch_main_contract run_commodity_stitch_main_contract "${ROOTPATH}" "${START_DATE}" "${END_DATE}" "${SYMBOL}" "${COMMODITY_NAME}"
```

Update `run_commodity_full_process` to call `run_logged_step` for each major step in order: `stitch_main_contract`, `downscale_continuous_by_trading_day`, `cross_section`, `merge`, `concat`, `time_feature`, `merge_clean`, `ic_correlation`, `scale_save`.

- [x] **Step 4: Run the test to verify it passes**

Run: `conda run -n finetf pytest data_preprocess/tests/test_commodity_scripts_docs.py::test_commodity_full_process_writes_step_logs -q`
Expected: PASS and the total log contains stage start/success messages plus per-step log paths.

- [x] **Step 5: Commit not performed**

No git commit was created because this build request did not ask for one.

- [x] **Task complete**

### Task 2: Preserve total log and child logs

> **trace:** plan-ready.md → `### Task 2: Verification` | tasks.md → `- [ ] 1.4 Preserve existing `main.sh` total log behavior and existing cross-section/merge per-date child logs.`
> **sync:** tasks.md → `- [ ] 1.4 Preserve existing `main.sh` total log behavior and existing cross-section/merge per-date child logs.` | plan-ready.md → `### Task 2: Verification`

**Files:**
- Modify: `data_preprocess/script_preprocess/future_upgraded/commodity/main.sh`
- Modify: `data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh`
- Test: `data_preprocess/tests/test_commodity_scripts_docs.py`

- [x] **Step 1: Add an assertion that the total log file still exists after the pipeline run**

```python
assert total_log.exists()
assert "ROOTPATH:" in total_log.read_text(encoding="utf-8")
```

- [x] **Step 2: Add an assertion that stage-specific child logs are still present for cross-section or merge**

```python
child_log = root / "log_futures/downscale/cross_section/5min/fu/2025-11-03.log"
assert child_log.exists()
```

- [x] **Step 3: Run the targeted tests**

Run: `conda run -n finetf pytest data_preprocess/tests/test_commodity_scripts_docs.py -q`
Expected: PASS with the new step-log assertions and no regression in the existing script smoke coverage.

- [x] **Step 4: Verify shell syntax and OpenSpec consistency**

Run: `bash -n data_preprocess/script_preprocess/future_upgraded/commodity/main.sh data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh && openspec validate add-commodity-preprocess-step-logs --strict`
Expected: exit code 0 for both commands.

- [x] **Task complete**
