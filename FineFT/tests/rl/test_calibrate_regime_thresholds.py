import numpy as np
import pytest
import pandas as pd

from RL.util.calibrate_regime_thresholds import (
    calibrate_regime_thresholds,
    map_regime_grid_id,
    ensure_regime_grid_id_column,
)


def test_calibrate_regime_thresholds_enforces_negative_slope_invariant():
    slopes = np.array([-0.05, -0.02, -0.01, 0.0, 0.01, 0.04, 0.08, 0.10, 0.15])
    vols = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    manifest = calibrate_regime_thresholds(slopes, vols)

    s_thresh = manifest["slope_thresholds"]
    v_thresh = manifest["vol_thresholds"]
    assert s_thresh[0] < 0.0
    assert s_thresh[1] > 0.0
    assert s_thresh[0] < s_thresh[1]
    assert v_thresh[0] < v_thresh[1]
    assert manifest["invariant_checked"] is True


def test_calibrate_regime_thresholds_fails_when_slope_0_is_not_negative():
    slopes = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09])
    vols = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    with pytest.raises(ValueError, match="lower slope threshold.*must be strictly negative"):
        calibrate_regime_thresholds(slopes, vols)


def test_map_regime_grid_id_boundary_and_coordinates():
    s_thresh = (-0.01, 0.04)
    v_thresh = (0.30, 0.50)

    # (v=0, s=0) -> 0
    assert map_regime_grid_id(-0.02, 0.20, s_thresh, v_thresh) == 0
    # (v=0, s=1) -> 1
    assert map_regime_grid_id(0.0, 0.20, s_thresh, v_thresh) == 1
    # (v=0, s=2) -> 2
    assert map_regime_grid_id(0.05, 0.20, s_thresh, v_thresh) == 2
    # (v=1, s=0) -> 3
    assert map_regime_grid_id(-0.02, 0.40, s_thresh, v_thresh) == 3
    # (v=1, s=1) -> 4
    assert map_regime_grid_id(0.0, 0.40, s_thresh, v_thresh) == 4
    # (v=1, s=2) -> 5
    assert map_regime_grid_id(0.05, 0.40, s_thresh, v_thresh) == 5
    # (v=2, s=0) -> 6
    assert map_regime_grid_id(-0.02, 0.60, s_thresh, v_thresh) == 6
    # (v=2, s=1) -> 7
    assert map_regime_grid_id(0.0, 0.60, s_thresh, v_thresh) == 7
    # (v=2, s=2) -> 8
    assert map_regime_grid_id(0.05, 0.60, s_thresh, v_thresh) == 8


def test_map_regime_grid_id_vectorized():
    s_thresh = (-0.01, 0.04)
    v_thresh = (0.30, 0.50)
    slopes = np.array([-0.02, 0.0, 0.05])
    vols = np.array([0.20, 0.40, 0.60])
    grids = map_regime_grid_id(slopes, vols, s_thresh, v_thresh)
    assert list(grids) == [0, 4, 8]


def test_ensure_regime_grid_id_column_passes_when_present():
    df = pd.DataFrame({"regime_grid_id": [0, 1, 2]})
    res_df = ensure_regime_grid_id_column(df)
    assert "regime_grid_id" in res_df.columns
    assert list(res_df["regime_grid_id"]) == [0, 1, 2]


def test_ensure_regime_grid_id_column_fails_fast_when_missing():
    df = pd.DataFrame({"close": [100.0, 101.0]})
    with pytest.raises(ValueError, match="DataFrame lacks pre-computed 'regime_grid_id' column"):
        ensure_regime_grid_id_column(df)
