"""Analysis must carry selection evidence without inventing statistical conclusions."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_preprocess.operator_futures.feature_audit.analyze_audit_results import (
    annotate_ci_evidence,
    classify_feature_dynamics,
    main,
    selection_annotations,
)


def test_analysis_refresh_preserves_statistics_and_reports_real_selection(tmp_path):
    audit = tmp_path / "audit"
    audit.mkdir()
    rows = []
    windows = []
    for feature, long_ic in [("existing", .3), ("dropped", .25), ("unknown", -.4)]:
        for horizon, ic in [("1h", .02), ("next_15td", long_ic)]:
            for target in ["return", "abs_return"]:
                rows.append({
                    "feature": feature, "horizon": horizon, "target_kind": target,
                    "spearman_corr": ic, "legacy_stream_membership": "unassigned",
                    "coverage_ratio": .9, "sign_consistency": 1.,
                    "ci_lower": np.nan, "ci_upper": np.nan,
                    "ci_lower_2x": np.nan, "ci_upper_2x": np.nan,
                })
                windows.append({
                    "feature": feature, "horizon": horizon, "target_kind": target,
                    "spearman_corr": ic, "contract": "fu_example", "window_id": "one",
                    "is_historical_reference": False,
                })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(audit / "feature_metrics.csv", index=False)
    pd.DataFrame(windows).to_csv(audit / "window_strata_metrics.csv", index=False)
    directional = metrics[metrics.target_kind == "return"]
    for field, filename in [
        ("spearman_corr", "feature_horizon_corr_matrix.csv"),
        ("coverage_ratio", "feature_horizon_coverage_matrix.csv"),
    ]:
        directional.pivot(index="feature", columns="horizon", values=field).to_csv(audit / filename)
    pd.DataFrame([{"horizon": "next_15td", "invalid_reason": "STALE_PRICE", "count": 2}]).to_csv(
        audit / "invalid_reasons_summary.csv", index=False
    )
    (audit / "audit_run_manifest.json").write_text(json.dumps({
        "symbol": "fu", "frequency": "10min", "protocol_version": "example.v1",
    }))
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({
        "vae_slope_stream": {"selected_features": ["existing"]},
        "vae_volatility_stream": {"selected_features": []},
        "rl_stream": {
            "selected_features": ["existing"],
            "filter_results": {"Correlation Filter Dropped": ["dropped"]},
        },
    }))
    before = (audit / "feature_metrics.csv").read_bytes()
    output = tmp_path / "insights"
    assert main([
        "--audit-dir", str(audit), "--output-dir", str(output),
        "--legacy-selection-manifest", str(selection), "--horizons", "1h,next15d",
        "--min-coverage", "0.7", "--min-sign-consistency", "0.6",
    ]) == 0
    assert (audit / "feature_metrics.csv").read_bytes() == before
    evidence = pd.read_csv(output / "feature_evidence.csv")
    pd.testing.assert_series_equal(evidence.spearman_corr, metrics.spearman_corr)
    assert set(evidence.ci_status) == {"MISSING_CI_REASON_UNRECORDED"}
    selected = evidence[evidence.feature == "existing"].iloc[0]
    assert selected.legacy_stream_membership == "slope_vae,rl_decision"
    ranking = pd.read_csv(output / "top_long_horizon_features.csv")
    assert ranking.iloc[0].spearman_corr == -.4
    summary = (output / "executive_summary.md").read_text()
    assert "入选并集 **1**" in summary
    assert "有淘汰记录 **1**" in summary
    assert "缺少淘汰原因 **1**" in summary
    assert "**1** 个评价窗口" in summary and "**1** 个合约" in summary
    assert "符号一致率 ≥ 60%" in summary and "覆盖率 ≥ 70%" in summary
    assert "NOT_AVAILABLE" in summary
    assert "previously discarded under legacy" not in summary
    assert len(summary.splitlines()) < 100
    assert "[window_strata_metrics.csv](<../audit/window_strata_metrics.csv>)" in summary
    assert "[insights/feature_evidence.csv](<feature_evidence.csv>)" in summary
    manifest = json.loads((output / "analysis_manifest.json").read_text())
    assert manifest["selection_source"]["mode"] == "explicit_retrospective"
    assert "feature_metrics.csv" in manifest["source_artifacts"]
    # The next analysis can consume the saved annotations without re-reading a live manifest.
    annotations, provenance = selection_annotations(output, ["existing", "dropped", "unknown"], None, None)
    assert provenance["mode"] == "saved_audit_annotations"
    assert annotations.loc[annotations.feature == "existing", "is_selected"].item()


def test_ci_missing_reason_is_not_inferred_and_recorded_status_is_preserved():
    frame = pd.DataFrame({
        "ci_lower": [np.nan, np.nan, -.2], "ci_upper": [np.nan, np.nan, .1],
        "ci_status": [None, "INSUFFICIENT_DATA", "OK"],
    })
    result = annotate_ci_evidence(frame)
    assert result.ci_status.tolist() == [
        "MISSING_CI_REASON_UNRECORDED", "INSUFFICIENT_DATA", "AVAILABLE_EXPLORATORY",
    ]


def test_missing_endpoints_and_intermediate_sign_reversal_are_not_persistence():
    corr = pd.DataFrame({
        "feature": ["missing", "reversal"], "1h": [np.nan, .12],
        "today": [.3, -.5], "next_15td": [.3, .12],
    })
    result = classify_feature_dynamics(corr, corr)
    assert result.category.tolist() == ["INSUFFICIENT_HORIZON_DATA", "ENDPOINT_SIGN_ALIGNED"]


def test_absent_selection_evidence_remains_unknown(tmp_path):
    annotations, provenance = selection_annotations(tmp_path, ["example"], None, None)
    assert annotations.selection_status.tolist() == ["unknown"]
    assert provenance["mode"] == "unavailable"


def test_changed_recorded_selection_manifest_is_rejected(tmp_path):
    source = tmp_path / "selection.json"
    source.write_text(json.dumps({"slope_vae_features": ["example"]}))
    with pytest.raises(ValueError, match="changed since audit"):
        selection_annotations(tmp_path, ["example"], {
            "legacy_selection_source": {"path": str(source), "sha256": "wrong"},
        }, None)
