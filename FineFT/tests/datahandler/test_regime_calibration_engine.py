from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from datahandler import regime_calibration_engine as engine


def test_calibrate_regime_thresholds_computes_terciles_and_enforces_invariants():
    slopes = np.array([-1.5, -0.8, -0.2, 0.1, 0.5, 1.2])
    vols = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    manifest = engine.calibrate_regime_thresholds(slopes, vols, dynamic_number=3)
    s_thresh = manifest["slope_thresholds"]
    v_thresh = manifest["vol_thresholds"]

    assert len(s_thresh) == 2
    assert len(v_thresh) == 2
    assert s_thresh[0] < 0.0
    assert s_thresh[1] > 0.0
    assert v_thresh[0] > 0.0
    assert v_thresh[1] > v_thresh[0]
    assert manifest["sample_count"] == 6
    assert manifest["invariant_checked"] is True


def test_calibrate_regime_thresholds_rejects_non_negative_lower_slope():
    slopes = np.array([0.0, 0.1, 0.2, 0.5, 1.0, 2.0])
    vols = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    with pytest.raises(ValueError, match="lower slope threshold .* must be strictly negative"):
        engine.calibrate_regime_thresholds(slopes, vols, dynamic_number=3)


def test_calibrate_regime_thresholds_rejects_non_positive_upper_slope():
    slopes = np.array([-2.0, -1.5, -1.0, -0.5, -0.2, 0.0])
    vols = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    with pytest.raises(ValueError, match="upper slope threshold .* must be strictly positive"):
        engine.calibrate_regime_thresholds(slopes, vols, dynamic_number=3)


def test_extract_and_apply_regime_labels_end_to_end(tmp_path):
    rows = 120
    increments = np.resize(np.array([1.0] * 10 + [-1.0] * 10 + [0.1] * 10), rows)
    prices = 100.0 + np.cumsum(increments)

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="10min"),
            "symbol": ["fu2605"] * rows,
            "mark_price": prices,
            "bid1_price": prices,
            "limit_up_single_sided_ratio": 0.0,
            "limit_down_single_sided_ratio": 0.0,
        }
    )
    df.loc[10, "limit_up_single_sided_ratio"] = 0.5
    df.loc[20, "limit_down_single_sided_ratio"] = 0.5

    stage_root = tmp_path / "stage"
    stage_root.mkdir()

    segments = engine.extract_contract_segments(
        df,
        contract="fu2605",
        stage_root=stage_root,
        key_indicator="mark_price",
        min_length_limit=4,
        filter_padlen=5,
        merging_threshold=-1.0,
    )

    assert segments.contract == "fu2605"
    assert len(segments.turning_points) >= 2
    assert len(segments.slopes) == len(segments.turning_points) - 1
    assert len(segments.vols) == len(segments.slopes)

    # In our test distribution, ensure negative lower slope and positive upper slope
    extended_slopes = np.array(segments.slopes + [-1.0, 1.0])
    extended_vols = np.array(segments.vols + [0.01, 1.0])
    manifest = engine.calibrate_regime_thresholds(extended_slopes, extended_vols, dynamic_number=3)

    labeled_df = engine.apply_regime_labels_to_dataframe(
        df,
        segments,
        slope_thresholds=manifest["slope_thresholds"],
        vol_thresholds=manifest["vol_thresholds"],
        key_indicator="mark_price",
    )

    assert "slope_label" in labeled_df.columns
    assert "volatility_label" in labeled_df.columns
    assert "regime_grid_id" in labeled_df.columns

    grid_ids = labeled_df["regime_grid_id"].to_numpy()
    slope_labels = labeled_df["slope_label"].to_numpy()
    vol_labels = labeled_df["volatility_label"].to_numpy()

    assert not (grid_ids < 0).any()
    assert (grid_ids == vol_labels * 3 + slope_labels).all()

    # Limit up row: slope_label=2, vol_label=2, grid_id=8
    assert slope_labels[10] == 2
    assert vol_labels[10] == 2
    assert grid_ids[10] == 8

    # Limit down row: slope_label=0, vol_label=2, grid_id=6
    assert slope_labels[20] == 0
    assert vol_labels[20] == 2
    assert grid_ids[20] == 6


def test_extract_contract_segments_fails_fast_on_missing_price_column(tmp_path):
    df = pd.DataFrame({"symbol": ["fu2605"] * 10, "close": [100.0] * 10})
    stage_root = tmp_path / "stage"
    stage_root.mkdir()

    with pytest.raises(ValueError, match="missing required key_indicator column"):
        engine.extract_contract_segments(
            df,
            contract="fu2605",
            stage_root=stage_root,
            key_indicator="mark_price",
        )
