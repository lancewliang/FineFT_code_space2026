"""Controlled command-substitution tests for the fu 10-minute controller.

The tests copy only shell entrypoints into a temporary root and substitute every
step with a harmless recorder.  They never import or start the RL workloads.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE = REPO_ROOT / "run_fu_10min_pipeline.sh"
STEP_PATHS = (
    "data_preprocess/script_preprocess/future_upgraded/commodity/main_fu_10.sh",
    "FineFT/script/data/commodity_data_handler_fu_10.sh",
    "FineFT/script/train/train_commodity_fu_10.sh",
    "FineFT/script/test/DiHFT/low_level/test_util_fu_10.sh",
    "FineFT/script/analysis/pick_agent/low_level_fu_10.sh",
    "FineFT/script/train/DiHFT/low_level/VAE_util_fu_10.sh",
    "FineFT/script/test/DiHFT/high_level/vae_optuna_fu_10.sh",
    "FineFT/script/analysis/pick_agent/high_level_heurstic_fu_10.sh",
    "FineFT/script/test/DiHFT/high_level/final_result_fu_10_main_sub.sh",
    "FineFT/script/test/DiHFT/high_level/final_result_fu_10.sh",
)
HIGH_LEVEL_ENTRYPOINTS = (
    "FineFT/script/test/DiHFT/high_level/vae_optuna_fu_10.sh",
    "FineFT/script/analysis/pick_agent/high_level_heurstic_fu_10.sh",
    "FineFT/script/test/DiHFT/high_level/final_result_fu_10_main_sub.sh",
    "FineFT/script/test/DiHFT/high_level/final_result_fu_10.sh",
)


def _write(path: Path, content: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if executable:
        path.chmod(0o755)


@pytest.fixture
def controlled_root(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    root = tmp_path / "workspace with spaces"
    root.mkdir()
    shutil.copy2(PIPELINE, root / "run_fu_10min_pipeline.sh")
    for helper in ("require_finetf.sh", "foreground_task.sh"):
        source = REPO_ROOT / "FineFT/script/lib" / helper
        target = root / "FineFT/script/lib" / helper
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    events = root / "events.txt"
    for number, relative_path in enumerate(STEP_PATHS, start=1):
        body = (
            "#!/usr/bin/env bash\nset -euo pipefail\nprintf '%s\\n' '"
            + str(number)
            + "' >> \"$STEP_EVENTS\"\n"
        )
        if number == 7:
            body += (
                "mkdir -p \"$(dirname \"$SELECTION_EVIDENCE\")\"\n"
                "printf '{}' > \"$SELECTION_EVIDENCE\"\n"
                "printf '%s\\n' '{\"experiment_identity\":\"controlled\",\"selection\":{\"trial_number\":0},\"config_hash\":\"90f0833dda6fd2421dc8f709ec010956564dd9f3163cb1f7455db9d1896c19cb\"}' > \"$FROZEN_CONFIG\"\n"
            )
        if number == 8:
            body += (
                "mkdir -p \"$PUBLICATION_DIR\"\n"
                "cp \"$FROZEN_CONFIG\" \"$PUBLICATION_DIR/frozen_config.json\"\n"
                "printf '%s\\n' '{\"document_type\":\"expert_hysteresis_step8_publication_receipt\",\"experiment_identity\":\"controlled\",\"config_hash\":\"90f0833dda6fd2421dc8f709ec010956564dd9f3163cb1f7455db9d1896c19cb\",\"selected_trial\":{\"trial_number\":0}}' > \"$PUBLICATION_DIR/step8_publication_receipt.json\"\n"
            )
        _write(root / relative_path, body, executable=True)

    conda_base = root / "fake conda"
    _write(
        conda_base / "etc/profile.d/conda.sh",
        "conda() {\n"
        "  if [[ $1 == activate && $2 == finetf ]]; then\n"
        "    export CONDA_DEFAULT_ENV=finetf; return 0\n"
        "  fi\n"
        "  return 1\n"
        "}\n",
    )
    fake_bin = root / "bin"
    _write(fake_bin / "conda", f"#!/usr/bin/env bash\necho '{conda_base}'\n", executable=True)
    manifest = root / "inputs/experiment manifest.json"
    selection = root / "inputs/selection manifest.json"
    _write(manifest, '{"experiment_id": "controlled-run"}\n')
    _write(selection, '{}\n')
    env = {
        **os.environ,
        "ROOTPATH": str(root),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "STEP_EVENTS": str(events),
        "EXPERIMENT_MANIFEST": str(manifest),
        "SELECTION_MANIFEST": str(selection),
        "CALIBRATION_RUN_ID": "controlled-run",
    }
    return root, env


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ([], list(range(1, 11))),
        (["-s", "1-9"], list(range(1, 10))),
        (["-s", "10,7,8,9"], [7, 8, 9, 10]),
    ],
)
def test_controller_runs_default_validation_and_explicit_high_level_ranges(
    controlled_root: tuple[Path, dict[str, str]], arguments: list[str], expected: list[int]
) -> None:
    root, env = controlled_root
    result = subprocess.run(
        ["bash", str(root / "run_fu_10min_pipeline.sh"), *arguments],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    events = (root / "events.txt").read_text(encoding="utf-8").splitlines()
    assert events == [str(item) for item in expected]


def test_controller_dry_run_is_read_only_and_reports_step_ten(
    controlled_root: tuple[Path, dict[str, str]],
) -> None:
    root, env = controlled_root
    result = subprocess.run(
        ["bash", str(root / "run_fu_10min_pipeline.sh"), "--dry-run", "-s", "7-10"],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Planned steps: 7 8 9 10" in result.stdout
    assert "最终 test 评估与审计" in result.stdout
    assert not (root / "events.txt").exists()


def test_controller_stops_after_nonzero_child_status(
    controlled_root: tuple[Path, dict[str, str]],
) -> None:
    root, env = controlled_root
    step_eight = root / STEP_PATHS[7]
    _write(
        step_eight,
        "#!/usr/bin/env bash\nprintf '8\\n' >> \"$STEP_EVENTS\"\nexit 17\n",
        executable=True,
    )
    result = subprocess.run(
        ["bash", str(root / "run_fu_10min_pipeline.sh"), "-s", "7-10"],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    assert (root / "events.txt").read_text(encoding="utf-8").splitlines() == ["7", "8"]
    assert "Step 8 failed" in result.stderr


def test_controller_term_reaps_its_running_child(
    controlled_root: tuple[Path, dict[str, str]],
) -> None:
    root, env = controlled_root
    step_one = root / STEP_PATHS[0]
    _write(
        step_one,
        "#!/usr/bin/env bash\n"
        "trap 'printf reaped\\n >> \"$STEP_EVENTS\"; exit 143' TERM\n"
        "printf started\\n >> \"$STEP_EVENTS\"\n"
        "sleep 30 &\nwait\n",
        executable=True,
    )
    process = subprocess.Popen(
        ["bash", str(root / "run_fu_10min_pipeline.sh"), "-s", "1"],
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    for _ in range(50):
        if (root / "events.txt").exists() and "started" in (
            root / "events.txt"
        ).read_text(encoding="utf-8"):
            break
        time.sleep(0.05)
    process.send_signal(signal.SIGTERM)
    _, stderr = process.communicate(timeout=10)
    assert process.returncode == 143
    assert "reaped" in (root / "events.txt").read_text(encoding="utf-8")
    assert "stopping pipeline child group" in stderr


@pytest.mark.parametrize(
    ("entrypoint", "extra_env", "expected_tokens"),
    [
        (
            HIGH_LEVEL_ENTRYPOINTS[0],
            {
                "EXPERIMENT_MANIFEST": "inputs/experiment manifest.json",
                "CALIBRATION_RUN_ID": "run-a",
                "SELECTION_MANIFEST": "inputs/selection manifest.json",
            },
            (
                "vae_routing_optuna.py",
                "--gating_strategy expert_hysteresis",
                "--experiment_manifest inputs/experiment manifest.json",
            ),
        ),
        (
            HIGH_LEVEL_ENTRYPOINTS[1],
            {
                "SELECTION_EVIDENCE": "inputs/selection evidence.json",
                "FROZEN_CONFIG": "inputs/frozen config.json",
            },
            (
                "DiHFT_high_level_heurstic.py",
                "--selection-evidence inputs/selection evidence.json",
                "--frozen-config inputs/frozen config.json",
            ),
        ),
        (
            HIGH_LEVEL_ENTRYPOINTS[2],
            {"FROZEN_CONFIG": "inputs/frozen config.json", "EVAL_STAGE": "valid"},
            (
                "vae_routing_final_result_macro_action.py",
                "--eval_stage valid",
                "--evaluation_purpose main_sub",
                "daily_trading_audit.py",
            ),
        ),
        (
            HIGH_LEVEL_ENTRYPOINTS[3],
            {"FROZEN_CONFIG": "inputs/frozen config.json"},
            (
                "vae_routing_final_result_macro_action.py",
                "--eval_stage test",
                "--evaluation_purpose final",
                "daily_trading_audit.py",
            ),
        ),
    ],
)
def test_high_level_entrypoints_use_controlled_finetf_python_and_wait(
    tmp_path: Path, entrypoint: str, extra_env: dict[str, str], expected_tokens: tuple[str, ...]
) -> None:
    """Each real shell receives a fake finetf activation and fake Python only."""
    root = tmp_path / "entrypoint root"
    root.mkdir()
    sources = (*HIGH_LEVEL_ENTRYPOINTS, "FineFT/script/lib/require_finetf.sh")
    sources += ("FineFT/script/lib/foreground_task.sh",)
    for source_relative in sources:
        source = REPO_ROOT / source_relative
        target = root / source_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    inputs = root / "inputs"
    inputs.mkdir()
    filenames = (
        "experiment manifest.json",
        "selection manifest.json",
            "selection evidence.json",
            "frozen config.json",
            "step8_publication_receipt.json",
    )
    for filename in filenames:
        _write(inputs / filename, "{}\n")
    (inputs / "best replay").mkdir()
    conda_base = root / "fake conda"
    _write(
        conda_base / "etc/profile.d/conda.sh",
        "conda() {\n"
        "  [[ $1 == activate && $2 == finetf ]] && {\n"
        "    export CONDA_DEFAULT_ENV=finetf; return 0\n"
        "  }\n"
        "  return 1\n"
        "}\n",
    )
    trace = root / "python trace.txt"
    fake_bin = root / "bin"
    _write(fake_bin / "conda", f"#!/usr/bin/env bash\necho '{conda_base}'\n", executable=True)
    _write(
        fake_bin / "python",
        "#!/usr/bin/env bash\n"
        "if [[ $1 == -c ]]; then echo controlled-hash; exit 0; fi\n"
        "printf '%s\\n' \"$*\" >> \"$PYTHON_TRACE\"\n"
        "sleep 0.05\n",
        executable=True,
    )
    env = {
        **os.environ,
        "ROOTPATH": str(root),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "PYTHON_TRACE": str(trace),
        **extra_env,
    }
    result = subprocess.run(
        ["bash", str(root / entrypoint)],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    commands = trace.read_text(encoding="utf-8")
    for expected in expected_tokens:
        assert expected in commands
