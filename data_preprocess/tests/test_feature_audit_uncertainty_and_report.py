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

from data_preprocess.operator_futures.feature_audit.uncertainty import (
    joint_moving_block_bootstrap,
    BootstrapResult,
)


def test_joint_moving_block_bootstrap_unit():
    """Unit test verifying deterministic moving block bootstrap under fixed seed."""
    trading_days = [f"202401{i:02d}" for i in range(1, 41)]  # 40 trading days
    rng = np.random.RandomState(42)

    # Generate synthetic x and y by trading day
    day_data = {}
    for td in trading_days:
        n_obs = 10
        x = rng.normal(0, 1.0, size=n_obs)
        y = 0.5 * x + rng.normal(0, 0.5, size=n_obs)
        day_data[td] = (x, y)

    # 1. Base test: block_length = 20
    # Effective blocks = 40 / 20 = 2.0 < 20 -> should trigger INSUFFICIENT_DATA status
    res_insuf = joint_moving_block_bootstrap(
        trading_days=trading_days,
        day_data=day_data,
        block_length=20,
        min_effective_blocks=20,
        n_replicates=100,
        seed=42
    )
    assert res_insuf.status == "INSUFFICIENT_DATA"
    assert res_insuf.ci_lower is None

    # 2. Relax min_effective_blocks = 2 to verify calculation
    res_calc = joint_moving_block_bootstrap(
        trading_days=trading_days,
        day_data=day_data,
        block_length=5,
        min_effective_blocks=2,
        n_replicates=200,
        seed=42
    )
    assert res_calc.status == "OK"
    assert res_calc.ci_lower is not None
    assert res_calc.ci_upper is not None
    assert res_calc.ci_lower <= res_calc.ci_upper

    # Repeat with same seed to verify deterministic reproducibility
    res_calc_repeat = joint_moving_block_bootstrap(
        trading_days=trading_days,
        day_data=day_data,
        block_length=5,
        min_effective_blocks=2,
        n_replicates=200,
        seed=42
    )
    assert np.isclose(res_calc.ci_lower, res_calc_repeat.ci_lower)
    assert np.isclose(res_calc.ci_upper, res_calc_repeat.ci_upper)


def test_audit_report_and_legacy_annotation(tmp_path):
    """Integration test verifying report rendering, heatmaps, and legacy annotations."""
    work_dir = tmp_path / "report_work"
    work_dir.mkdir(parents=True)

    trading_days = [f"202401{i:02d}" for i in range(2, 23)]
    intervals_json = []
    for td in trading_days:
        day_str = f"2024-01-{int(td[-2:]):02d}"
        intervals_json.append({
            "trading_day": td,
            "sessions": [
                {"start": f"{day_str} 09:00:00", "end": f"{day_str} 15:00:00", "name": "all_day"},
            ]
        })
    cal_file = work_dir / "calendar.json"
    cal_file.write_text(json.dumps({"symbol": "fu", "trading_days": intervals_json}))

    records = []
    snapshot_records = []
    rng = np.random.RandomState(42)
    p = 3000.0
    for td in trading_days:
        day_str = f"2024-01-{int(td[-2:]):02d}"
        for t in [f"{day_str} 10:00:00", f"{day_str} 14:00:00"]:
            p += rng.normal(0, 1.0)
            records.append({
                "contract": "fu2405",
                "timestamp": t,
                "trading_day": td,
                "alpha_slope": rng.normal(0, 1.0) + p * 0.01,
                "beta_vol": abs(rng.normal(0, 1.0)),
                "gamma_rl": rng.normal(0, 0.5),
            })
            snapshot_records.append({
                "contract": "fu2405",
                "timestamp": t,
                "trading_day": td,
                "LastPrice": p,
            })

    stage6_dir = work_dir / "stage6"
    stage6_dir.mkdir()
    pl.DataFrame(records).write_ipc(stage6_dir / "fu2405.feather")

    price_file = work_dir / "snapshots.feather"
    pl.DataFrame(snapshot_records).write_ipc(price_file)

    price_man_file = work_dir / "price_man.json"
    price_man_file.write_text(json.dumps({
        "price_file": str(price_file),
        "price_field": "LastPrice",
        "max_quote_age_seconds": 60
    }))

    feat_man_file = work_dir / "feat_man.json"
    feat_man_file.write_text(json.dumps({"symbol": "fu", "data_path": str(stage6_dir)}))

    split_man_file = work_dir / "split_man.json"
    split_man_file.write_text(json.dumps({"train_contracts": ["fu2405"]}))

    # Legacy selection manifest for annotation
    legacy_manifest = {
        "slope_vae_features": ["alpha_slope"],
        "volatility_vae_features": ["beta_vol"],
        "rl_decision_features": ["gamma_rl"],
        "rejected_features": {}
    }
    legacy_file = work_dir / "legacy_selection.json"
    legacy_file.write_text(json.dumps(legacy_manifest))

    out_dir = work_dir / "output"
    config = {
        "schema_version": "multi_horizon_audit.v1",
        "protocol_version": "multi_horizon_audit.research.v1",
        "symbol": "fu",
        "frequency": "10min",
        "horizons": ["1h", "today"],
        "calendar_manifest": str(cal_file),
        "feature_availability_manifest": str(feat_man_file),
        "price_source_manifest": str(price_man_file),
        "split_manifest": str(split_man_file),
        "legacy_selection_manifest": str(legacy_file),
        "result_root": str(out_dir),
        "evidence_protocol": {
            "method": "joint_trading_day_moving_block_bootstrap",
            "confidence_level": 0.95,
            "replicates": 50,
            "seed": 42,
            "minimum_block_trading_days": 5,
            "minimum_effective_blocks": 2,  # set low for test fixture
        }
    }
    cfg_file = work_dir / "cfg.yaml"
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

    # Check metrics table has legacy membership and bootstrap CI columns
    metrics_df = pl.read_csv(out_dir / "feature_metrics.csv")
    assert "legacy_stream_membership" in metrics_df.columns
    assert "ci_lower" in metrics_df.columns
    assert "ci_upper" in metrics_df.columns
    assert "ci_lower_2x" in metrics_df.columns
    assert "ci_upper_2x" in metrics_df.columns

    # Verify alpha_slope has legacy membership 'slope_vae'
    slope_row = metrics_df.filter(pl.col("feature") == "alpha_slope").head(1)
    assert slope_row["legacy_stream_membership"][0] == "slope_vae"

    # Check report file exists and has exploratory disclaimers
    report_file = out_dir / "audit_report.md"
    assert report_file.exists()
    report_text = report_file.read_text(encoding="utf-8")
    assert "Exploratory" in report_text or "探索性" in report_text
    assert "Multi-Horizon Feature Audit Report" in report_text

    # Check matrix files exist
    assert (out_dir / "feature_horizon_corr_matrix.csv").exists()
    assert (out_dir / "feature_horizon_coverage_matrix.csv").exists()


def test_strata_and_metrics_schema_robustness_with_null_prefixes(tmp_path):
    """Regression test ensuring Polars schema inference does not crash when first 100+ rows contain nulls."""
    work_dir = tmp_path / "schema_robustness_work"
    work_dir.mkdir(parents=True)

    # Create 30 trading days
    trading_days = [f"20230{m}15" for m in range(1, 10)] + [f"20231{m}15" for m in range(0, 3)]
    intervals_json = []
    for td in trading_days:
        day_str = f"{td[:4]}-{td[4:6]}-{td[6:]}"
        intervals_json.append({
            "trading_day": td,
            "sessions": [
                {"start": f"{day_str} 09:00:00", "end": f"{day_str} 15:00:00", "name": "all_day"},
            ]
        })
    cal_file = work_dir / "calendar.json"
    cal_file.write_text(json.dumps({"symbol": "fu", "trading_days": intervals_json}))

    stage6_dir = work_dir / "stage6"
    stage6_dir.mkdir()

    # Contract 1 has data only in early window
    # Contract 2 has data only in late window
    # Over 120 features to exceed Polars default 100-row infer_schema_length
    n_features = 120
    feat_names = [f"feat_{i:03d}" for i in range(n_features)]
    rng = np.random.RandomState(42)

    records_c1 = []
    snapshot_records = []
    p = 3000.0
    for td in trading_days[:4]:  # Early days
        day_str = f"{td[:4]}-{td[4:6]}-{td[6:]}"
        for t in [f"{day_str} 09:30:00", f"{day_str} 10:00:00", f"{day_str} 10:30:00", f"{day_str} 11:00:00", f"{day_str} 11:30:00"]:
            p += rng.normal(0, 1.0)
            row = {"contract": "fu2305", "timestamp": t, "trading_day": td}
            for fn in feat_names:
                row[fn] = rng.normal(0, 1.0)
            records_c1.append(row)
            snapshot_records.append({"contract": "fu2305", "timestamp": t, "trading_day": td, "LastPrice": p})

    records_c2 = []
    for td in trading_days[8:]:  # Late days
        day_str = f"{td[:4]}-{td[4:6]}-{td[6:]}"
        for t in [f"{day_str} 09:30:00", f"{day_str} 10:00:00", f"{day_str} 10:30:00", f"{day_str} 11:00:00", f"{day_str} 11:30:00"]:
            p += rng.normal(0, 1.0)
            row = {"contract": "fu2309", "timestamp": t, "trading_day": td}
            for fn in feat_names:
                row[fn] = rng.normal(0, 1.0)
            records_c2.append(row)
            snapshot_records.append({"contract": "fu2309", "timestamp": t, "trading_day": td, "LastPrice": p})

    pl.DataFrame(records_c1).write_ipc(stage6_dir / "fu2305.feather")
    pl.DataFrame(records_c2).write_ipc(stage6_dir / "fu2309.feather")

    price_file = work_dir / "snapshots.feather"
    pl.DataFrame(snapshot_records).write_ipc(price_file)

    price_man_file = work_dir / "price_man.json"
    price_man_file.write_text(json.dumps({
        "price_file": str(price_file),
        "price_field": "LastPrice",
        "max_quote_age_seconds": 3600
    }))

    feat_man_file = work_dir / "feat_man.json"
    feat_man_file.write_text(json.dumps({"symbol": "fu", "data_path": str(stage6_dir)}))

    split_man_file = work_dir / "split_man.json"
    split_man_file.write_text(json.dumps({"train_contracts": ["fu2305", "fu2309"]}))

    out_dir = work_dir / "output"
    config = {
        "schema_version": "multi_horizon_audit.v1",
        "protocol_version": "multi_horizon_audit.research.v1",
        "symbol": "fu",
        "frequency": "10min",
        "horizons": ["1h", "today"],
        "calendar_manifest": str(cal_file),
        "feature_availability_manifest": str(feat_man_file),
        "price_source_manifest": str(price_man_file),
        "split_manifest": str(split_man_file),
        "result_root": str(out_dir),
        "evidence_protocol": {
            "method": "joint_trading_day_moving_block_bootstrap",
            "confidence_level": 0.95,
            "replicates": 20,
            "seed": 42,
            "minimum_block_trading_days": 2,
            "minimum_effective_blocks": 1,
        }
    }
    cfg_file = work_dir / "cfg.yaml"
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

    # Check that strata table loaded correctly as f64
    strata_df = pl.read_csv(out_dir / "window_strata_metrics.csv", infer_schema_length=None)
    assert strata_df["spearman_corr"].dtype == pl.Float64
    assert len(strata_df) > 100
    # Ensure some are null and some are not null
    assert strata_df["spearman_corr"].null_count() > 0
    assert (out_dir / "_SUCCESS").exists()
