"""Reading reports stay bounded while evidence files remain authoritative."""

import json
import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_preprocess.operator_futures.feature_audit.report import generate_audit_report


def test_compact_report_keeps_scope_and_reasons_without_dumping_all_features(tmp_path):
    rows = [
        {
            "feature": f"feature_{i:04d}",
            "horizon": "1h",
            "target_kind": "return",
            "spearman_corr": 0.1,
            "coverage_ratio": 0.9,
            "ci_status": "INSUFFICIENT_EFFECTIVE_BLOCKS",
        }
        for i in range(1000)
    ]
    rows.append({**rows[0], "feature": "ask1_price_trend_2", "spearman_corr": None})
    metrics = pl.DataFrame(rows)
    before = metrics.clone()
    pl.DataFrame([{
        "feature": "ask1_price_trend_2",
        "legacy_stream_membership": "slope_vae,rl_decision",
        "drop_reasons": json.dumps([{"scope": "vol_vae", "reason": "Stream Blacklist Dropped"}]),
    }]).write_csv(tmp_path / "legacy_selection_comparison.csv")
    metrics.write_csv(tmp_path / "feature_metrics.csv")
    source = (tmp_path / "feature_metrics.csv").read_bytes()
    output = tmp_path / "audit_report.md"
    generate_audit_report({"symbol": "fu", "frequency": "10min"}, metrics, None, output)
    report = output.read_text()
    assert len(report.splitlines()) < 150
    feature_rows = [line for line in report.splitlines() if "| return / 1h |" in line]
    assert len(feature_rows) == 50
    assert len({line.split("`")[1] for line in feature_rows}) == 50
    assert "feature_0999" not in report
    assert "ask1_price_trend_2" in report
    assert "Slope/RL；Vol：流黑名单" in report
    assert "IC 使用各期限原生有效样本" in report
    assert "INSUFFICIENT_EFFECTIVE_BLOCKS" in report
    assert "[feature_metrics.csv](<feature_metrics.csv>)" in report
    assert "`insights/feature_evidence.csv`（未生成）" in report
    assert "(<insights/feature_evidence.csv>)" not in report
    assert (tmp_path / "feature_metrics.csv").read_bytes() == source
    assert metrics.equals(before)
