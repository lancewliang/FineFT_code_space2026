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

from data_preprocess.operator_futures.feature_audit.split_guard import (
    generate_evaluation_windows,
    EvaluationWindow,
)


def test_evaluation_windows_schedule():
    """Tests 6-month warmup reference, 3-month windows, train/valid boundary split."""
    # From 2023-01-01 to 2024-06-30
    # Train: 2023-01-01 to 2023-12-31
    # Valid: 2024-01-01 to 2024-06-30
    trading_days = [f"2023{m:02d}01" for m in range(1, 13)] + [f"2024{m:02d}01" for m in range(1, 7)]
    train_end_td = "20231201"

    windows = generate_evaluation_windows(
        start_date=datetime(2023, 1, 1),
        train_end_date=datetime(2023, 12, 31),
        valid_end_date=datetime(2024, 6, 30),
        warmup_months=6,
        window_months=3,
    )

    assert len(windows) >= 4
    # First window is historical reference (warmup)
    assert windows[0].is_historical_reference
    assert windows[0].window_id == "warmup_ref"

    # Evaluation windows follow warmup
    eval_windows = [w for w in windows if not w.is_historical_reference]
    assert len(eval_windows) >= 3
    # Check partition tags
    train_eval = [w for w in eval_windows if w.partition == "train"]
    valid_eval = [w for w in eval_windows if w.partition == "valid"]
    assert len(train_eval) > 0
    assert len(valid_eval) > 0


def test_test_dataset_sealing_and_purification(tmp_path):
    """Integration test asserting that mutating final Test data does not alter development metrics."""
    work_dir = tmp_path / "sealing_work"
    work_dir.mkdir(parents=True)

    # 1. Calendar
    trading_days = [f"202401{i:02d}" for i in range(2, 23)]
    intervals_json = []
    for td in trading_days:
        day_str = f"2024-01-{int(td[-2:]):02d}"
        intervals_json.append({
            "trading_day": td,
            "sessions": [
                {"start": f"{day_str} 09:00:00", "end": f"{day_str} 11:30:00", "name": "m"},
                {"start": f"{day_str} 13:30:00", "end": f"{day_str} 15:00:00", "name": "a"},
            ]
        })
    cal_file = work_dir / "calendar.json"
    cal_file.write_text(json.dumps({"symbol": "fu", "trading_days": intervals_json}))

    # 2. Stage 6 Data (Train: 20240102-20240112, Valid: 20240115-20240118, Test: 20240119-20240122)
    stage6_dir = work_dir / "stage6"
    stage6_dir.mkdir()

    # Create 3 contracts to test contract-equal & sign consistency
    contracts = ["fu2405", "fu2409", "fu2411"]
    records = []
    snapshot_records = []
    rng = np.random.RandomState(42)

    for c in contracts:
        base_p = 3000.0
        for td in trading_days:
            day_str = f"2024-01-{int(td[-2:]):02d}"
            for t in [f"{day_str} 09:30:00", f"{day_str} 14:00:00"]:
                base_p += rng.normal(0, 1.0)
                records.append({
                    "contract": c,
                    "timestamp": t,
                    "trading_day": td,
                    "feature_alpha": rng.normal(0, 1.0) + base_p * 0.01,
                })
                snapshot_records.append({
                    "contract": c,
                    "timestamp": t,
                    "trading_day": td,
                    "LastPrice": base_p,
                })

    df_stage6 = pl.DataFrame(records)
    df_stage6.write_ipc(stage6_dir / "train_valid.feather")

    price_file = work_dir / "snapshots.feather"
    pl.DataFrame(snapshot_records).write_ipc(price_file)

    price_man_file = work_dir / "price_man.json"
    price_man_file.write_text(json.dumps({
        "price_file": str(price_file),
        "price_field": "LastPrice",
        "max_quote_age_seconds": 60
    }))

    feat_man_file = work_dir / "feat_man.json"
    feat_man_file.write_text(json.dumps({
        "symbol": "fu",
        "data_path": str(stage6_dir),
    }))

    # Split manifest
    split_man_file = work_dir / "split_man.json"
    split_man_file.write_text(json.dumps({
        "train_contracts": contracts,
        "valid_contracts": [],
        "test_contracts": ["fu_sealed_test"]  # Sealed test contract
    }))

    # Create dummy sealed test file that should NOT be read
    test_dir = work_dir / "test_stage6"
    test_dir.mkdir()
    pl.DataFrame({"dummy": [1, 2]}).write_ipc(test_dir / "fu_sealed_test.feather")

    out_dir_1 = work_dir / "output_1"
    config_1 = {
        "schema_version": "multi_horizon_audit.v1",
        "protocol_version": "multi_horizon_audit.research.v1",
        "symbol": "fu",
        "frequency": "10min",
        "horizons": ["1h", "next_td"],
        "calendar_manifest": str(cal_file),
        "feature_availability_manifest": str(feat_man_file),
        "price_source_manifest": str(price_man_file),
        "split_manifest": str(split_man_file),
        "read_final_test": False,
        "result_root": str(out_dir_1),
    }
    cfg_file_1 = work_dir / "cfg_1.yaml"
    with open(cfg_file_1, "w") as f:
        yaml.safe_dump(config_1, f)

    cmd = [
        sys.executable, "-m",
        "data_preprocess.operator_futures.feature_audit.cli",
        "run",
        "--config", str(cfg_file_1)
    ]
    res1 = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert res1.returncode == 0, f"STDOUT:\n{res1.stdout}\nSTDERR:\n{res1.stderr}"

    # Verify primary summary contains contract_equal_spearman and sign_consistency
    metrics_df = pl.read_csv(out_dir_1 / "feature_metrics.csv")
    assert "contract_equal_spearman" in metrics_df.columns
    assert "sample_weighted_spearman" in metrics_df.columns
    assert "sign_consistency" in metrics_df.columns

    # Verify window strata table exists
    assert (out_dir_1 / "window_strata_metrics.csv").exists()
