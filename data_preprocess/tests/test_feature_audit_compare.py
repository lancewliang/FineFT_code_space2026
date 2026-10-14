import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import polars as pl
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_preprocess.operator_futures.feature_audit.comparisons import (
    ComparisonConfig,
    ComparisonRunner,
)


def _create_mock_run(run_dir: Path, freq: str, schema_ver="multi_horizon_audit.v1", protocol_ver="multi_horizon_audit.research.v1"):
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": schema_ver,
        "protocol_version": protocol_ver,
        "symbol": "fu",
        "frequency": freq,
        "horizons": ["1h", "today"],
        "status": "completed",
    }
    with open(run_dir / "audit_run_manifest.json", "w") as f:
        json.dump(manifest, f)
    (run_dir / "_SUCCESS").touch()

    # Create mock row_labels.parquet
    # 10min has timestamps at 09:10, 09:20, 09:30
    # 30min has timestamps at 09:30
    if freq == "10min":
        ts_list = ["2024-01-02 09:10:00", "2024-01-02 09:20:00", "2024-01-02 09:30:00"]
    else:
        ts_list = ["2024-01-02 09:30:00"]

    labels = []
    feat_data = []
    for ts in ts_list:
        labels.append({
            "contract": "fu2405",
            "timestamp": ts,
            "trading_day": "20240102",
            "horizon": "1h",
            "target_return": 0.01,
            "abs_target_return": 0.01,
            "valid_mask": True,
            "invalid_reason": "none",
        })
        feat_data.append({
            "contract": "fu2405",
            "timestamp": ts,
            "trading_day": "20240102",
            "feature_x": 1.5,
        })
    pl.DataFrame(labels).write_parquet(run_dir / "row_labels.parquet")
    pl.DataFrame(feat_data).write_ipc(run_dir / "features.feather")


def test_comparison_runner_and_cli(tmp_path):
    work_dir = tmp_path / "cmp_work"
    work_dir.mkdir()

    run_10min = work_dir / "run_10min"
    run_30min = work_dir / "run_30min"
    _create_mock_run(run_10min, "10min")
    _create_mock_run(run_30min, "30min")

    cmp_out = work_dir / "cmp_out"
    cmp_config = {
        "schema_version": "multi_horizon_audit.comparison.v1",
        "protocol_version": "multi_horizon_audit.research.v1",
        "comparison_type": "matched_frequency",
        "run_a": str(run_10min),
        "run_b": str(run_30min),
        "output_dir": str(cmp_out),
    }
    cmp_config_file = work_dir / "cmp_cfg.yaml"
    with open(cmp_config_file, "w") as f:
        yaml.safe_dump(cmp_config, f)

    cmd = [
        sys.executable, "-m",
        "data_preprocess.operator_futures.feature_audit.cli",
        "compare",
        "--manifest", str(cmp_config_file)
    ]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    # Verify comparison outputs exist
    assert (cmp_out / "audit_comparison_manifest.json").exists()
    assert (cmp_out / "comparison_metrics.csv").exists()
    assert (cmp_out / "comparison_report.md").exists()

    with open(cmp_out / "audit_comparison_manifest.json") as f:
        cmp_man = json.load(f)
    assert cmp_man["status"] == "completed"
    assert cmp_man["matched_samples_count"] >= 1


def test_comparison_incompatible_protocol_rejection(tmp_path):
    """Verifies that compare rejects runs with incompatible protocol versions."""
    work_dir = tmp_path / "cmp_fail_work"
    work_dir.mkdir()

    run_a = work_dir / "run_a"
    run_b = work_dir / "run_b"
    _create_mock_run(run_a, "10min", protocol_ver="v1.0")
    _create_mock_run(run_b, "30min", protocol_ver="v2.0")

    cmp_out = work_dir / "cmp_out"
    cmp_config = {
        "schema_version": "multi_horizon_audit.comparison.v1",
        "protocol_version": "multi_horizon_audit.research.v1",
        "comparison_type": "matched_frequency",
        "run_a": str(run_a),
        "run_b": str(run_b),
        "output_dir": str(cmp_out),
    }
    cmp_config_file = work_dir / "cmp_cfg.yaml"
    with open(cmp_config_file, "w") as f:
        yaml.safe_dump(cmp_config, f)

    cmd = [
        sys.executable, "-m",
        "data_preprocess.operator_futures.feature_audit.cli",
        "compare",
        "--manifest", str(cmp_config_file)
    ]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert res.returncode != 0
    assert "protocol" in res.stderr.lower() or "incompatible" in res.stderr.lower()
