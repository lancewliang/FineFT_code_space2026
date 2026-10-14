import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, time
from pathlib import Path
import numpy as np
import polars as pl
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def sample_audit_env(tmp_path):
    """Creates a self-contained fixture environment for multi-horizon audit."""
    work_dir = tmp_path / "audit_work"
    work_dir.mkdir(parents=True)

    # 1. Calendar manifest (fu trading sessions and trading days)
    # fu sessions: 09:00-10:15, 10:30-11:30, 13:30-15:00, 21:00-23:00
    calendar_data = {
        "symbol": "fu",
        "timezone": "Asia/Shanghai",
        "trading_days": [
            {
                "trading_day": "20240102",
                "sessions": [
                    {"start": "2023-12-29 21:00:00", "end": "2023-12-29 23:00:00", "name": "night"},
                    {"start": "2024-01-02 09:00:00", "end": "2024-01-02 10:15:00", "name": "day_morning_1"},
                    {"start": "2024-01-02 10:30:00", "end": "2024-01-02 11:30:00", "name": "day_morning_2"},
                    {"start": "2024-01-02 13:30:00", "end": "2024-01-02 15:00:00", "name": "day_afternoon"},
                ]
            },
            {
                "trading_day": "20240103",
                "sessions": [
                    {"start": "2024-01-02 21:00:00", "end": "2024-01-02 23:00:00", "name": "night"},
                    {"start": "2024-01-03 09:00:00", "end": "2024-01-03 10:15:00", "name": "day_morning_1"},
                    {"start": "2024-01-03 10:30:00", "end": "2024-01-03 11:30:00", "name": "day_morning_2"},
                    {"start": "2024-01-03 13:30:00", "end": "2024-01-03 15:00:00", "name": "day_afternoon"},
                ]
            }
        ]
    }
    calendar_file = work_dir / "calendar_manifest.json"
    calendar_file.write_text(json.dumps(calendar_data), encoding="utf-8")

    # 2. Stage 6 input data (features)
    # Timestamps in session 1: 09:00, 09:10, 09:20, 09:30, 09:40, 09:50, 10:00, 10:10
    # In session 2: 10:30, 10:40, 10:50, 11:00, 11:10, 11:20
    # In session 3: 13:30, 13:40, 13:50, 14:00, 14:10, 14:20, 14:30, 14:40, 14:50, 15:00
    timestamps = [
        "2024-01-02 09:10:00", "2024-01-02 09:20:00", "2024-01-02 09:30:00",
        "2024-01-02 09:40:00", "2024-01-02 09:50:00", "2024-01-02 10:00:00",
        "2024-01-02 10:10:00", "2024-01-02 10:30:00", "2024-01-02 10:40:00",
        "2024-01-02 10:50:00", "2024-01-02 11:00:00", "2024-01-02 11:10:00",
        "2024-01-02 11:20:00", "2024-01-02 13:30:00", "2024-01-02 13:40:00",
        "2024-01-02 13:50:00", "2024-01-02 14:00:00", "2024-01-02 14:10:00",
        "2024-01-02 14:20:00", "2024-01-02 14:30:00", "2024-01-02 14:40:00",
        "2024-01-02 14:50:00", "2024-01-02 15:00:00"
    ]
    n_rows = len(timestamps)
    rng = np.random.RandomState(42)
    prices = 3000.0 + np.cumsum(rng.normal(0.5, 2.0, size=n_rows))
    alpha_feat = prices * 0.01 + rng.normal(0, 0.1, size=n_rows)
    noise_feat = rng.normal(0, 1.0, size=n_rows)

    df_stage6 = pl.DataFrame({
        "contract": ["fu2405"] * n_rows,
        "timestamp": timestamps,
        "trading_day": ["20240102"] * n_rows,
        "mark_price": prices,
        "close": prices,
        "alpha_feature": alpha_feat,
        "noise_feature": noise_feat,
        "str_col_id": ["ignore_me"] * n_rows,
        "target_return": rng.normal(0, 0.01, size=n_rows),  # future label to exclude
    })
    data_dir = work_dir / "stage6_data"
    data_dir.mkdir()
    train_file = data_dir / "fu2405.feather"
    df_stage6.write_ipc(train_file)

    # 3. Snapshot Price Source (fine-grained quotes for as-of reference pricing)
    # We provide snapshot quotes matching the timestamps and target endpoints
    snapshot_df = pl.DataFrame({
        "contract": ["fu2405"] * n_rows,
        "timestamp": timestamps,
        "trading_day": ["20240102"] * n_rows,
        "LastPrice": prices,
    })
    price_file = work_dir / "snapshots.feather"
    snapshot_df.write_ipc(price_file)

    price_source_manifest = {
        "symbol": "fu",
        "price_file": str(price_file),
        "price_field": "LastPrice",
        "max_quote_age_seconds": 60
    }
    price_manifest_file = work_dir / "price_source_manifest.json"
    price_manifest_file.write_text(json.dumps(price_source_manifest), encoding="utf-8")

    # 4. Feature availability manifest
    feat_manifest = {
        "symbol": "fu",
        "frequency": "10min",
        "data_path": str(data_dir),
        "available_features": ["alpha_feature", "noise_feature"]
    }
    feat_manifest_file = work_dir / "feature_availability_manifest.json"
    feat_manifest_file.write_text(json.dumps(feat_manifest), encoding="utf-8")

    # 5. Split manifest
    split_manifest = {
        "train_contracts": ["fu2405"],
        "valid_contracts": [],
        "test_contracts": []
    }
    split_manifest_file = work_dir / "split_manifest.json"
    split_manifest_file.write_text(json.dumps(split_manifest), encoding="utf-8")

    # 6. Audit config
    result_root = work_dir / "output"
    audit_config = {
        "schema_version": "multi_horizon_audit.v1",
        "protocol_version": "multi_horizon_audit.research.v1",
        "symbol": "fu",
        "frequency": "10min",
        "horizons": ["1h", "today"],
        "hour_clock": "active_trading_minutes",
        "trading_day_offsets": {"today": 0},
        "contract_policy": "same_contract",
        "candidate_policy": "causal_quality_only",
        "targets": ["return", "abs_return"],
        "calendar_manifest": str(calendar_file),
        "feature_availability_manifest": str(feat_manifest_file),
        "price_source_manifest": str(price_manifest_file),
        "split_manifest": str(split_manifest_file),
        "read_final_test": False,
        "purge_by_label_available_at": True,
        "write_selection_outputs": False,
        "result_root": str(result_root),
        "reference_price_protocol": {
            "field": "LastPrice",
            "observation": "latest_valid_snapshot_at_or_before_target",
            "same_contract": True,
            "same_session": True,
            "max_quote_age_seconds": 60,
            "fallback": "none"
        },
        "evidence_protocol": {
            "method": "joint_trading_day_moving_block_bootstrap",
            "confidence_level": 0.95,
            "replicates": 50,  # small for fast test
            "seed": 42
        }
    }
    config_file = work_dir / "audit_config.yaml"
    with open(config_file, "w") as f:
        yaml.safe_dump(audit_config, f)

    return {
        "work_dir": work_dir,
        "config_file": config_file,
        "result_root": result_root
    }


def test_cli_validate_inputs_success(sample_audit_env):
    """Tests that validate-inputs passes on valid configuration."""
    config_file = sample_audit_env["config_file"]
    cmd = [
        sys.executable, "-m",
        "data_preprocess.operator_futures.feature_audit.cli",
        "validate-inputs",
        "--config", str(config_file)
    ]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert "Input validation passed" in res.stdout


def test_cli_validate_inputs_missing_manifest(sample_audit_env):
    """Tests that validate-inputs fails clearly on missing manifest."""
    work_dir = sample_audit_env["work_dir"]
    bad_config = {
        "schema_version": "multi_horizon_audit.v1",
        "protocol_version": "multi_horizon_audit.research.v1",
        "symbol": "fu",
        "calendar_manifest": str(work_dir / "non_existent.json"),
    }
    bad_config_file = work_dir / "bad_config.yaml"
    with open(bad_config_file, "w") as f:
        yaml.safe_dump(bad_config, f)

    cmd = [
        sys.executable, "-m",
        "data_preprocess.operator_futures.feature_audit.cli",
        "validate-inputs",
        "--config", str(bad_config_file)
    ]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert res.returncode != 0
    assert "missing" in res.stderr.lower() or "missing" in res.stdout.lower() or "not found" in res.stderr.lower()


def test_cli_run_minimal_tracer_bullet(sample_audit_env):
    """Tests that run executes end-to-end and creates expected artifacts."""
    config_file = sample_audit_env["config_file"]
    result_root = sample_audit_env["result_root"]

    cmd = [
        sys.executable, "-m",
        "data_preprocess.operator_futures.feature_audit.cli",
        "run",
        "--config", str(config_file)
    ]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    # Verify artifacts exist
    manifest_file = result_root / "audit_run_manifest.json"
    success_file = result_root / "_SUCCESS"
    metrics_file = result_root / "feature_metrics.csv"
    row_labels_file = result_root / "row_labels.parquet"

    assert manifest_file.exists()
    assert success_file.exists()
    assert metrics_file.exists()
    assert row_labels_file.exists()

    # Verify manifest content
    with open(manifest_file, "r") as f:
        manifest = json.load(f)
    assert manifest["status"] == "completed"
    assert manifest["symbol"] == "fu"
    assert manifest["frequency"] == "10min"

    # Verify metrics content
    metrics_df = pl.read_csv(metrics_file)
    assert "feature" in metrics_df.columns
    assert "horizon" in metrics_df.columns
    assert "target_kind" in metrics_df.columns
    assert "spearman_corr" in metrics_df.columns

    # Verify candidates excluded non-numeric, id, target_return
    features = metrics_df["feature"].unique().to_list()
    assert "alpha_feature" in features
    assert "noise_feature" in features
    assert "str_col_id" not in features
    assert "target_return" not in features
    assert "contract" not in features


def test_calendar_adapter_active_time_math(sample_audit_env):
    """Unit test for active trading time math across sessions and night breaks."""
    from data_preprocess.operator_futures.feature_audit.calendar_adapter import CalendarAdapter
    calendar_file = sample_audit_env["work_dir"] / "calendar_manifest.json"
    adapter = CalendarAdapter.from_manifest(calendar_file)

    # 1. Intra-session accumulation: 09:10 + 30m -> 09:40
    t1 = datetime(2024, 1, 2, 9, 10, 0)
    target1, td1 = adapter.add_active_trading_minutes(t1, 30)
    assert target1 == datetime(2024, 1, 2, 9, 40, 0)
    assert td1 == "20240102"

    # 2. Crossing lunch break (morning session 1 ends at 10:15, session 2 starts at 10:30):
    # 10:00 + 60m active: 15m in session 1 (until 10:15), remaining 45m in session 2 (10:30 + 45m = 11:15)
    t2 = datetime(2024, 1, 2, 10, 0, 0)
    target2, td2 = adapter.add_active_trading_minutes(t2, 60)
    assert target2 == datetime(2024, 1, 2, 11, 15, 0)
    assert td2 == "20240102"

    # 3. Night session crossing into next day:
    # 2023-12-29 22:30:00 (night ends at 23:00 -> 30m). Remaining 30m into next session (2024-01-02 09:00 -> 09:30)
    t3 = datetime(2023, 12, 29, 22, 30, 0)
    target3, td3 = adapter.add_active_trading_minutes(t3, 60)
    assert target3 == datetime(2024, 1, 2, 9, 30, 0)
    assert td3 == "20240102"

    # 4. Today endpoint: 20240102 close is 15:00:00
    close_dt, close_td = adapter.get_trading_day_close("20240102", offset=0)
    assert close_dt == datetime(2024, 1, 2, 15, 0, 0)
    assert close_td == "20240102"


def test_spearman_metrics_ties_and_constants():
    """Unit test for Spearman rank calculation with ties and edge cases."""
    from data_preprocess.operator_futures.feature_audit.metrics import compute_spearman_correlation
    from scipy import stats

    # Vector with ties
    x = np.array([1.0, 2.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 3.0, 4.0, 4.0, 6.0, 7.0])
    corr, null_reason = compute_spearman_correlation(x, y)
    expected_corr = float(stats.spearmanr(x, y).statistic)
    assert null_reason is None
    assert np.isclose(corr, expected_corr)

    # Constant series
    x_const = np.array([1.0, 1.0, 1.0, 1.0])
    y_var = np.array([1.0, 2.0, 3.0, 4.0])
    corr_const, null_reason_const = compute_spearman_correlation(x_const, y_var)
    assert corr_const is None
    assert null_reason_const == "CONSTANT_SERIES"

    # Insufficient pairs (< 3)
    corr_few, null_reason_few = compute_spearman_correlation(np.array([1.0, 2.0]), np.array([2.0, 3.0]))
    assert corr_few is None
    assert null_reason_few == "INSUFFICIENT_PAIRS"
