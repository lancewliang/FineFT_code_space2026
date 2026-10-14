import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import polars as pl
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_preprocess.operator_futures.feature_audit.prepare_30min_inputs import prepare_30min_data


def _compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def test_real_data_acceptance_fu_dual_frequency(tmp_path):
    """End-to-end acceptance test with real fu data for both 10min and 30min across all seven horizons."""
    work_dir = tmp_path / "fu_real_acceptance"
    work_dir.mkdir(parents=True)

    real_10min_dir = REPO_ROOT / "PREPROCESS_DATASET" / "commodity-futures" / "SPLIT-TRAIN-VALID-TEST" / "10min" / "fu"
    real_train_file = real_10min_dir / "train.feather"
    if not real_train_file.exists():
        pytest.skip("Real fu 10min dataset not available in local environment")

    # Record SHA256 of real dataset before test
    real_train_sha_before = _compute_sha256(real_train_file)

    # 1. Prepare isolated acceptance data directory using real fu2305 contract
    acc_10min_dir = work_dir / "10min_fu"
    acc_10min_train = acc_10min_dir / "train"
    acc_10min_train.mkdir(parents=True)

    # Read real fu2305 contract
    src_contract_file = real_10min_dir / "train" / "fu2305.feather"
    df_real = pl.read_ipc(src_contract_file)
    # Ensure LastPrice exists for reference price provider
    if "LastPrice" not in df_real.columns and "close" in df_real.columns:
        df_real = df_real.with_columns(pl.col("close").alias("LastPrice"))

    acc_contract_file = acc_10min_train / "fu2305.feather"
    df_real.write_ipc(acc_contract_file)
    df_real.write_ipc(acc_10min_dir / "train.feather")

    manifest_10 = {
        "symbol": "fu",
        "target_freq": "10min",
        "sets": {"train": ["fu2305"], "valid": [], "test": []}
    }
    with open(acc_10min_dir / "dataset_split_manifest.json", "w") as f:
        json.dump(manifest_10, f)

    # 2. Run isolated 30min data preparation
    acc_30min_dir = work_dir / "30min_fu"
    prepare_30min_data(acc_10min_dir, acc_30min_dir)
    assert (acc_30min_dir / "train.feather").exists()
    assert (acc_30min_dir / "train" / "fu2305.feather").exists()

    # 3. Build trading calendar spanning fu2305 observation range (2023-01 to 2023-05)
    calendar_days = []
    # Dates from 2023-01-01 to 2023-05-15
    cur_d = datetime(2023, 1, 3)
    end_d = datetime(2023, 5, 15)
    while cur_d <= end_d:
        if cur_d.weekday() < 5:  # Mon-Fri
            td_str = cur_d.strftime("%Y%m%d")
            d_str = cur_d.strftime("%Y-%m-%d")
            calendar_days.append({
                "trading_day": td_str,
                "sessions": [
                    {"start": f"{d_str} 09:00:00", "end": f"{d_str} 10:15:00", "name": "m1"},
                    {"start": f"{d_str} 10:30:00", "end": f"{d_str} 11:30:00", "name": "m2"},
                    {"start": f"{d_str} 13:30:00", "end": f"{d_str} 15:00:00", "name": "aft"},
                    {"start": f"{d_str} 21:00:00", "end": f"{d_str} 23:00:00", "name": "night"},
                ]
            })
        cur_d = cur_d.fromordinal(cur_d.toordinal() + 1)

    cal_manifest_file = work_dir / "calendar_manifest.json"
    with open(cal_manifest_file, "w") as f:
        json.dump({"symbol": "fu", "trading_days": calendar_days}, f)

    # 4. Snapshots price source
    snapshot_file = work_dir / "fu2305_snapshots.feather"
    df_real.select(["contract", "timestamp", "LastPrice"]).write_ipc(snapshot_file)

    price_manifest_file = work_dir / "price_source_manifest.json"
    with open(price_manifest_file, "w") as f:
        json.dump({
            "price_file": str(snapshot_file),
            "price_field": "LastPrice",
            "max_quote_age_seconds": 60
        }, f)

    feat_manifest_10_file = work_dir / "feat_man_10.json"
    with open(feat_manifest_10_file, "w") as f:
        json.dump({"symbol": "fu", "data_path": str(acc_10min_train)}, f)

    feat_manifest_30_file = work_dir / "feat_man_30.json"
    with open(feat_manifest_30_file, "w") as f:
        json.dump({"symbol": "fu", "data_path": str(acc_30min_dir / "train")}, f)

    split_manifest_file = work_dir / "split_manifest.json"
    with open(split_manifest_file, "w") as f:
        json.dump({"train_contracts": ["fu2305"], "test_contracts": []}, f)

    # 5. Run full 10min audit across all seven horizons
    out_10min = work_dir / "output_10min"
    config_10 = {
        "schema_version": "multi_horizon_audit.v1",
        "protocol_version": "multi_horizon_audit.research.v1",
        "symbol": "fu",
        "frequency": "10min",
        "horizons": ["1h", "2h", "4h", "today", "next_td", "next_5td", "next_15td"],
        "calendar_manifest": str(cal_manifest_file),
        "feature_availability_manifest": str(feat_manifest_10_file),
        "price_source_manifest": str(price_manifest_file),
        "split_manifest": str(split_manifest_file),
        "result_root": str(out_10min),
        "evidence_protocol": {
            "method": "joint_trading_day_moving_block_bootstrap",
            "confidence_level": 0.95,
            "replicates": 50,
            "seed": 42,
            "minimum_block_trading_days": 5,
            "minimum_effective_blocks": 2,
        }
    }
    cfg_10_file = work_dir / "cfg_10min.yaml"
    with open(cfg_10_file, "w") as f:
        yaml.safe_dump(config_10, f)

    t0_10 = time.time()
    cmd_10 = [sys.executable, "-m", "data_preprocess.operator_futures.feature_audit.cli", "run", "--config", str(cfg_10_file)]
    res_10 = subprocess.run(cmd_10, cwd=REPO_ROOT, capture_output=True, text=True)
    elapsed_10 = time.time() - t0_10
    assert res_10.returncode == 0, f"10min failed:\nSTDOUT: {res_10.stdout}\nSTDERR: {res_10.stderr}"

    # Verify 10min outputs
    assert (out_10min / "_SUCCESS").exists()
    assert (out_10min / "audit_report.md").exists()
    assert (out_10min / "feature_metrics.csv").exists()
    assert (out_10min / "invalid_reasons_summary.csv").exists()

    # 6. Run full 30min audit across all seven horizons
    out_30min = work_dir / "output_30min"
    config_30 = {
        "schema_version": "multi_horizon_audit.v1",
        "protocol_version": "multi_horizon_audit.research.v1",
        "symbol": "fu",
        "frequency": "30min",
        "horizons": ["1h", "2h", "4h", "today", "next_td", "next_5td", "next_15td"],
        "calendar_manifest": str(cal_manifest_file),
        "feature_availability_manifest": str(feat_manifest_30_file),
        "price_source_manifest": str(price_manifest_file),
        "split_manifest": str(split_manifest_file),
        "result_root": str(out_30min),
        "evidence_protocol": {
            "method": "joint_trading_day_moving_block_bootstrap",
            "confidence_level": 0.95,
            "replicates": 50,
            "seed": 42,
            "minimum_block_trading_days": 5,
            "minimum_effective_blocks": 2,
        }
    }
    cfg_30_file = work_dir / "cfg_30min.yaml"
    with open(cfg_30_file, "w") as f:
        yaml.safe_dump(config_30, f)

    t0_30 = time.time()
    cmd_30 = [sys.executable, "-m", "data_preprocess.operator_futures.feature_audit.cli", "run", "--config", str(cfg_30_file)]
    res_30 = subprocess.run(cmd_30, cwd=REPO_ROOT, capture_output=True, text=True)
    elapsed_30 = time.time() - t0_30
    assert res_30.returncode == 0, f"30min failed:\nSTDOUT: {res_30.stdout}\nSTDERR: {res_30.stderr}"

    # Verify 30min outputs
    assert (out_30min / "_SUCCESS").exists()
    assert (out_30min / "audit_report.md").exists()
    assert (out_30min / "feature_metrics.csv").exists()

    # 7. Run compare CLI
    cmp_out = work_dir / "cmp_fu_10_vs_30"
    cmp_config = {
        "schema_version": "multi_horizon_audit.comparison.v1",
        "protocol_version": "multi_horizon_audit.research.v1",
        "comparison_type": "matched_frequency",
        "run_a": str(out_10min),
        "run_b": str(out_30min),
        "output_dir": str(cmp_out),
    }
    cmp_cfg_file = work_dir / "cmp_cfg.yaml"
    with open(cmp_cfg_file, "w") as f:
        yaml.safe_dump(cmp_config, f)

    cmd_cmp = [sys.executable, "-m", "data_preprocess.operator_futures.feature_audit.cli", "compare", "--manifest", str(cmp_cfg_file)]
    res_cmp = subprocess.run(cmd_cmp, cwd=REPO_ROOT, capture_output=True, text=True)
    assert res_cmp.returncode == 0, f"Compare failed:\nSTDOUT: {res_cmp.stdout}\nSTDERR: {res_cmp.stderr}"

    assert (cmp_out / "audit_comparison_manifest.json").exists()
    assert (cmp_out / "comparison_report.md").exists()

    # 8. Verify legacy file isolation (zero change to original real data)
    real_train_sha_after = _compute_sha256(real_train_file)
    assert real_train_sha_before == real_train_sha_after

    # Peak memory diagnostic
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"\n[Real Acceptance Performance] 10min: {elapsed_10:.2f}s, 30min: {elapsed_30:.2f}s, Peak RSS: {peak_kb / 1024:.2f} MB")
