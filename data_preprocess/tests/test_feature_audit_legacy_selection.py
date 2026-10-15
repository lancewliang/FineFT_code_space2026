"""Selection provenance must reflect the native selector's manifest semantics."""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_preprocess.operator_futures.feature_audit.legacy_selection import load_legacy_selection


def test_native_drop_evidence_does_not_treat_survivors_as_rejected(tmp_path):
    manifest = {
        "filter_results": {
            "Feature Blacklist Dropped": ["blacklisted"],
            "Hard Filter": ["survivor"],
        },
        "vae_slope_stream": {"selected_features": ["shared"]},
        "vae_volatility_stream": {"selected_features": []},
        "rl_stream": {
            "selected_features": ["shared"],
            "filter_results": {
                "Composite Score": ["survivor", "shared"],
                "Correlation Filter Dropped": ["redundant"],
            },
        },
    }
    source = tmp_path / "selection.json"
    source.write_text(json.dumps(manifest))
    result = load_legacy_selection(source)
    assert result.membership("shared") == "slope_vae,rl_decision"
    assert result.membership("blacklisted") == "rejected"
    assert result.membership("redundant") == "rejected"
    assert result.membership("survivor") == "not_selected"
    assert "survivor" not in result.drop_reasons
    unknown = result.comparison_rows(["unrecorded"])[0]
    assert unknown["is_selected"] is False
    assert unknown["drop_reason_status"] == "unknown"
    assert len(result.sha256) == 64


@pytest.mark.parametrize("payload", [{}, {"vae_slope_stream": {"selected_features": []}}])
def test_unknown_or_incomplete_format_is_not_silently_ignored(tmp_path, payload):
    source = tmp_path / "selection.json"
    source.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_legacy_selection(source)


def test_explicit_missing_manifest_is_not_silently_ignored(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_legacy_selection(tmp_path / "missing.json")


def test_partial_flat_manifest_keeps_unknown_selection_status(tmp_path):
    source = tmp_path / "selection.json"
    source.write_text(json.dumps({"slope_vae_features": ["slope"]}))
    result = load_legacy_selection(source)
    row = result.comparison_rows(["unrecorded"])[0]
    assert row["is_selected"] is None
    assert row["legacy_stream_membership"] == "unassigned"
