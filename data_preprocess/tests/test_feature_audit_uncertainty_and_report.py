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


@pytest.mark.parametrize("manifest_format", ["flat", "native"])
def test_audit_report_and_legacy_annotation(tmp_path, manifest_format):
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
        "max_quote_age_seconds": 600
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
    if manifest_format == "native":
        legacy_manifest = {
            "stream_mode": "triple",
            "vae_slope_stream": {"selected_features": ["alpha_slope"]},
            "vae_volatility_stream": {"selected_features": ["beta_vol"]},
            "rl_stream": {
                "selected_features": ["gamma_rl", "alpha_slope"],
                "filter_results": {"Hard Filter Dropped": ["beta_vol"]},
            },
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
    expected = "slope_vae,rl_decision" if manifest_format == "native" else "slope_vae"
    assert slope_row["legacy_stream_membership"][0] == expected
    comparison = pl.read_csv(out_dir / "legacy_selection_comparison.csv")
    beta = comparison.filter(pl.col("feature") == "beta_vol").row(0, named=True)
    assert beta["is_selected"] is True
    assert beta["legacy_stream_membership"] == "vol_vae"
    if manifest_format == "native":
        assert json.loads(beta["drop_reasons"]) == [
            {"scope": "rl_decision", "reason": "Hard Filter Dropped"}
        ]
    run_manifest = json.loads((out_dir / "audit_run_manifest.json").read_text())
    assert run_manifest["legacy_selection_source"]["path"] == str(legacy_file.resolve())

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


def test_ci_diagnostics_and_three_way_matched_comparisons(tmp_path):
    """End-to-end verification of CI diagnostic fields and three-way matched comparisons."""
    work_dir = tmp_path / "ci_matched_work"
    work_dir.mkdir(parents=True)

    # 15 trading days
    trading_days = [f"202402{i:02d}" for i in range(1, 16)]
    intervals_json = []
    for td in trading_days:
        day_str = f"2024-02-{int(td[-2:]):02d}"
        intervals_json.append({
            "trading_day": td,
            "sessions": [
                {"start": f"{day_str} 09:00:00", "end": f"{day_str} 10:15:00", "name": "m1"},
                {"start": f"{day_str} 10:30:00", "end": f"{day_str} 11:30:00", "name": "m2"},
                {"start": f"{day_str} 13:30:00", "end": f"{day_str} 15:00:00", "name": "aft"},
            ]
        })
    cal_file = work_dir / "calendar.json"
    cal_file.write_text(json.dumps({"symbol": "fu", "trading_days": intervals_json}))

    records = []
    snapshot_records = []
    rng = np.random.RandomState(42)
    price_val = 3000.0

    for td in trading_days:
        day_str = f"2024-02-{int(td[-2:]):02d}"
        for t in [f"{day_str} 09:10:00", f"{day_str} 10:40:00", f"{day_str} 14:00:00", f"{day_str} 15:00:00"]:
            price_val += rng.normal(0, 1.5)
            row = {
                "contract": "fu2405",
                "timestamp": t,
                "trading_day": td,
                "alpha_decay": rng.normal(0, 1.0),
                "trend_emerge": price_val * 0.01,
            }
            records.append(row)
            snapshot_records.append({
                "contract": "fu2405",
                "timestamp": t,
                "trading_day": td,
                "LastPrice": price_val,
            })

    stage6_dir = work_dir / "stage6"
    stage6_dir.mkdir()
    pl.DataFrame(records).write_ipc(stage6_dir / "fu2405.feather")

    price_file = work_dir / "snapshots.feather"
    pl.DataFrame(snapshot_records).write_ipc(price_file)

    price_man = work_dir / "price_man.json"
    price_man.write_text(json.dumps({
        "price_file": str(price_file),
        "price_field": "LastPrice",
        "max_quote_age_seconds": 600
    }))

    feat_man = work_dir / "feat_man.json"
    feat_man.write_text(json.dumps({"symbol": "fu", "data_path": str(stage6_dir)}))

    split_man = work_dir / "split_man.json"
    split_man.write_text(json.dumps({"train_contracts": ["fu2405"]}))

    out_dir = work_dir / "audit_out"
    config = {
        "schema_version": "multi_horizon_audit.v1",
        "protocol_version": "multi_horizon_audit.research.v1",
        "symbol": "fu",
        "frequency": "10min",
        "horizons": ["1h", "next_td", "next_5td"],
        "calendar_manifest": str(cal_file),
        "feature_availability_manifest": str(feat_man),
        "price_source_manifest": str(price_man),
        "split_manifest": str(split_man),
        "result_root": str(out_dir),
        "evidence_protocol": {
            "method": "joint_trading_day_moving_block_bootstrap",
            "confidence_level": 0.95,
            "replicates": 30,
            "seed": 42,
            "minimum_block_trading_days": 5,
            "minimum_effective_blocks": 20,
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
    assert res.returncode == 0, f"run failed: {res.stderr}"

    # 1. Verify feature_metrics.csv has all 11 CI diagnostic columns
    fm = pl.read_csv(out_dir / "feature_metrics.csv")
    expected_ci_cols = [
        "ci_status", "ci_status_2x", "ci_valid_trading_days",
        "ci_block_length", "ci_block_length_2x", "ci_effective_blocks",
        "ci_effective_blocks_2x", "ci_min_effective_blocks", "ci_min_effective_blocks_2x",
        "ci_null_reason", "ci_null_reason_2x",
    ]
    for col in expected_ci_cols:
        assert col in fm.columns, f"Missing CI diagnostic column {col}"
    assert "INSUFFICIENT_EFFECTIVE_BLOCKS" in fm["ci_status"].unique().to_list()

    # 2. Verify ci_diagnostics.csv exists and has expected schema
    diag_file = out_dir / "ci_diagnostics.csv"
    assert diag_file.exists()
    diag_df = pl.read_csv(diag_file)
    assert "horizon" in diag_df.columns
    assert "num_valid_days" in diag_df.columns
    assert "block_length" in diag_df.columns
    assert "effective_blocks" in diag_df.columns
    assert "ci_status" in diag_df.columns

    # 3. Verify matched_horizon_comparisons.csv exists in single-row wide format
    matched_file = out_dir / "matched_horizon_comparisons.csv"
    assert matched_file.exists()
    m_df = pl.read_csv(matched_file)
    expected_matched_cols = [
        "feature", "target_kind", "legacy_stream_membership",
        "matched_samples", "sample_retention_ratio",
        "ic_common_1h", "ic_common_1day", "ic_common_5day",
        "delta_1h_to_1day", "delta_1day_to_5day", "delta_1h_to_5day",
        "trajectory_category",
    ]
    for col in expected_matched_cols:
        assert col in m_df.columns, f"Missing matched comparison column {col}"
    assert m_df["matched_samples"][0] > 0
    assert 0.0 < m_df["sample_retention_ratio"][0] <= 1.0

    # 4. Verify audit_report.md includes the new sections
    report_text = (out_dir / "audit_report.md").read_text(encoding="utf-8")
    assert "Bootstrap Uncertainty & Horizon Block Diagnostics (Protocol Q14)" in report_text
    assert "Three-Way Matched Common-Sample Horizon Comparisons (1h ↔ 1day ↔ 5day)" in report_text

    # 5. Verify analyze_audit_results updates matched_horizon_status
    insights_dir = work_dir / "insights"
    cmd_analyze = [
        sys.executable, "-m",
        "data_preprocess.operator_futures.feature_audit.analyze_audit_results",
        "--audit-dir", str(out_dir),
        "--output-dir", str(insights_dir),
        "--horizons", "1h,next_td,next_5td",
    ]
    res_a = subprocess.run(cmd_analyze, cwd=REPO_ROOT, capture_output=True, text=True)
    assert res_a.returncode == 0, f"analyze failed: {res_a.stderr}"

    manifest_json = json.loads((insights_dir / "analysis_manifest.json").read_text())
    assert manifest_json["matched_horizon_status"] == "COMPLETED_MATCHED_EVALUATION"
    summary_md = (insights_dir / "executive_summary.md").read_text()
    assert "COMPLETED_MATCHED_EVALUATION" in summary_md
    assert "共同决策点" in summary_md
    assert "完整审计报告" in summary_md
    assert "| 特征 | 目标 / 期限 |" not in summary_md
