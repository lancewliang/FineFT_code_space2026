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

from data_preprocess.operator_futures.feature_audit.calendar_adapter import CalendarAdapter, SessionInterval


def test_seven_horizons_active_time_and_multiday_offsets():
    """Tests 1h, 2h, 4h, today, next_td, next_5td, next_15td endpoints."""
    # Build 20 consecutive trading days
    trading_days = [f"202401{i:02d}" for i in range(2, 23)]
    intervals = []
    for td in trading_days:
        day_str = f"2024-01-{int(td[-2:]):02d}"
        intervals.extend([
            SessionInterval(td, "day_m1", datetime.strptime(f"{day_str} 09:00:00", "%Y-%m-%d %H:%M:%S"), datetime.strptime(f"{day_str} 10:15:00", "%Y-%m-%d %H:%M:%S")),
            SessionInterval(td, "day_m2", datetime.strptime(f"{day_str} 10:30:00", "%Y-%m-%d %H:%M:%S"), datetime.strptime(f"{day_str} 11:30:00", "%Y-%m-%d %H:%M:%S")),
            SessionInterval(td, "day_aft", datetime.strptime(f"{day_str} 13:30:00", "%Y-%m-%d %H:%M:%S"), datetime.strptime(f"{day_str} 15:00:00", "%Y-%m-%d %H:%M:%S")),
        ])
    adapter = CalendarAdapter(intervals, trading_days)

    start_dt = datetime(2024, 1, 2, 9, 30, 0)
    # 1h = 60 mins -> 09:30 + 45m (to 10:15) + 15m in session 2 (10:30 + 15m) = 10:45
    t_1h, _ = adapter.add_active_trading_minutes(start_dt, 60)
    assert t_1h == datetime(2024, 1, 2, 10, 45, 0)

    # 2h = 120 mins -> 45m (to 10:15) + 60m (10:30-11:30) + 15m in afternoon (13:30 + 15m) = 13:45
    t_2h, _ = adapter.add_active_trading_minutes(start_dt, 120)
    assert t_2h == datetime(2024, 1, 2, 13, 45, 0)

    # 4h = 240 mins -> total daily active = 75 + 60 + 90 = 225 mins.
    # From 09:30, available today = 45 + 60 + 90 = 195 mins. Remaining 45 mins onto next day (09:00 + 45m = 09:45)
    t_4h, td_4h = adapter.add_active_trading_minutes(start_dt, 240)
    assert t_4h == datetime(2024, 1, 3, 9, 45, 0)
    assert td_4h == "20240103"

    # Multi-day offsets
    t_today, td_today = adapter.get_trading_day_close("20240102", offset=0)
    assert t_today == datetime(2024, 1, 2, 15, 0, 0)
    assert td_today == "20240102"

    t_next, td_next = adapter.get_trading_day_close("20240102", offset=1)
    assert t_next == datetime(2024, 1, 3, 15, 0, 0)
    assert td_next == "20240103"

    t_5td, td_5td = adapter.get_trading_day_close("20240102", offset=5)
    assert t_5td == datetime(2024, 1, 7, 15, 0, 0)
    assert td_5td == "20240107"

    t_15td, td_15td = adapter.get_trading_day_close("20240102", offset=15)
    assert t_15td == datetime(2024, 1, 17, 15, 0, 0)
    assert td_15td == "20240117"


def test_full_seven_horizons_execution(tmp_path):
    """Integration test verifying all seven horizons run and generate invalid reason breakdown."""
    work_dir = tmp_path / "seven_h_work"
    work_dir.mkdir(parents=True)

    trading_days = [f"202401{i:02d}" for i in range(2, 23)]
    intervals_json = []
    for td in trading_days:
        day_str = f"2024-01-{int(td[-2:]):02d}"
        intervals_json.append({
            "trading_day": td,
            "sessions": [
                {"start": f"{day_str} 09:00:00", "end": f"{day_str} 10:15:00", "name": "m1"},
                {"start": f"{day_str} 10:30:00", "end": f"{day_str} 11:30:00", "name": "m2"},
                {"start": f"{day_str} 13:30:00", "end": f"{day_str} 15:00:00", "name": "aft"},
            ]
        })

    cal_file = work_dir / "calendar_manifest.json"
    cal_file.write_text(json.dumps({"symbol": "fu", "trading_days": intervals_json}))

    # Create feature data across 20 trading days
    records = []
    rng = np.random.RandomState(42)
    price_val = 3000.0

    snapshot_records = []

    for td in trading_days:
        day_str = f"2024-01-{int(td[-2:]):02d}"
        times = [f"{day_str} 09:10:00", f"{day_str} 11:00:00", f"{day_str} 14:00:00", f"{day_str} 15:00:00"]
        for t in times:
            price_val += rng.normal(0, 2.0)
            records.append({
                "contract": "fu2405",
                "timestamp": t,
                "trading_day": td,
                "feature_a": rng.normal(0, 1.0),
                "feature_b": price_val * 0.05,
            })
            snapshot_records.append({
                "contract": "fu2405",
                "timestamp": t,
                "trading_day": td,
                "LastPrice": price_val,
            })

    data_dir = work_dir / "stage6"
    data_dir.mkdir()
    pl.DataFrame(records).write_ipc(data_dir / "fu2405.feather")

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
        "data_path": str(data_dir),
    }))

    split_man_file = work_dir / "split_man.json"
    split_man_file.write_text(json.dumps({"train_contracts": ["fu2405"]}))

    out_dir = work_dir / "output"
    config = {
        "schema_version": "multi_horizon_audit.v1",
        "protocol_version": "multi_horizon_audit.research.v1",
        "symbol": "fu",
        "frequency": "10min",
        "horizons": ["1h", "2h", "4h", "today", "next_td", "next_5td", "next_15td"],
        "calendar_manifest": str(cal_file),
        "feature_availability_manifest": str(feat_man_file),
        "price_source_manifest": str(price_man_file),
        "split_manifest": str(split_man_file),
        "result_root": str(out_dir),
    }
    cfg_file = work_dir / "audit_cfg.yaml"
    with open(cfg_file, "w") as f:
        yaml.safe_dump(config, f)

    cmd = [
        sys.executable, "-m",
        "data_preprocess.operator_futures.feature_audit.cli",
        "run",
        "--config", str(cfg_file)
    ]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    # Verify all seven horizons in metrics
    metrics_df = pl.read_csv(out_dir / "feature_metrics.csv")
    horizons = set(metrics_df["horizon"].unique().to_list())
    assert horizons == {"1h", "2h", "4h", "today", "next_td", "next_5td", "next_15td"}

    # Verify coverage_ratio column in metrics
    assert "coverage_ratio" in metrics_df.columns

    # Verify invalid reasons summary file exists
    invalid_summary_file = out_dir / "invalid_reasons_summary.csv"
    assert invalid_summary_file.exists()
    inv_df = pl.read_csv(invalid_summary_file)
    assert "horizon" in inv_df.columns
    assert "invalid_reason" in inv_df.columns
    assert "count" in inv_df.columns

    # Verify ZERO_LENGTH_HORIZON appears for today at 15:00:00
    today_inv = inv_df.filter((pl.col("horizon") == "today") & (pl.col("invalid_reason") == "ZERO_LENGTH_HORIZON"))
    assert today_inv["count"].sum() > 0
