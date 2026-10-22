from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from RL.DiHFT.high_level.trading_utility.prequential import (
    PrequentialProtocol,
    run_strict_prequential,
)


@dataclass
class _DeterministicModel:
    train_mean: float

    def predict_components(self, frame: pd.DataFrame) -> pd.DataFrame:
        signal = frame["signal"].to_numpy(dtype=float)
        return pd.DataFrame(
            {
                "mean_net_pnl_bps": self.train_mean + signal,
                "cvar_loss_bps": np.full(len(frame), 2.0),
                "mae_loss_bps": np.full(len(frame), 1.0),
                "win_probability_raw": 1.0 / (1.0 + np.exp(-signal)),
            },
            index=frame.index,
        )


def _frame() -> pd.DataFrame:
    dates = pd.date_range("2024-07-23", "2025-06-27", freq="D")
    index = np.arange(len(dates))
    return pd.DataFrame(
        {
            "trading_day": dates,
            "contract": [f"fu{value % 4}" for value in index],
            "slot_id": index % 2,
            "direction": np.where(index % 3 == 0, -1, 1),
            "signal": np.where(index % 2 == 0, 1.0, -1.0),
            "net_pnl_bps": np.where(index % 2 == 0, 8.0, -6.0),
            "win": index % 2 == 0,
        }
    )


def test_two_outer_folds_rebuild_prefix_assets_and_keep_test_unseen() -> None:
    selector_prefix_ends: list[pd.Timestamp] = []
    model_prefix_ends: list[pd.Timestamp] = []

    def build_selector(prefix: pd.DataFrame, fold_name: str) -> str:
        selector_prefix_ends.append(prefix["trading_day"].max())
        return f"selector-{fold_name}"

    def fit_model(prefix: pd.DataFrame, lambda_cvar: float, lambda_mae: float):
        model_prefix_ends.append(prefix["trading_day"].max())
        return _DeterministicModel(float(prefix["net_pnl_bps"].mean()))

    result = run_strict_prequential(
        _frame(),
        feature_names=("signal",),
        fit_model=fit_model,
        build_selector=build_selector,
        protocol=PrequentialProtocol(inner_splits=2),
    )

    assert list(result.outer_predictions["outer_fold"].unique()) == [1, 2]
    fold_1 = result.outer_predictions[result.outer_predictions["outer_fold"] == 1]
    fold_2 = result.outer_predictions[result.outer_predictions["outer_fold"] == 2]
    assert fold_1["trading_day"].min() == pd.Timestamp("2024-11-01")
    assert fold_1["trading_day"].max() == pd.Timestamp("2025-02-28")
    assert fold_2["trading_day"].min() == pd.Timestamp("2025-03-01")
    assert fold_2["trading_day"].max() == pd.Timestamp("2025-06-27")
    assert selector_prefix_ends[0] < pd.Timestamp("2024-11-01")
    assert selector_prefix_ends[1] < pd.Timestamp("2025-03-01")
    assert selector_prefix_ends[-1] == pd.Timestamp("2025-06-27")
    assert max(model_prefix_ends) <= pd.Timestamp("2025-06-27")
    assert result.final_calibration_source == "outer_unseen_predictions"
    assert result.protocol_manifest["outer_fold_count"] == 2
    assert result.protocol_manifest["test_accessed"] is False


def test_future_fold_sentinel_cannot_change_fold_one_predictions() -> None:
    frame = _frame()

    def fit_model(prefix: pd.DataFrame, lambda_cvar: float, lambda_mae: float):
        return _DeterministicModel(float(prefix["net_pnl_bps"].mean()))

    baseline = run_strict_prequential(
        frame,
        feature_names=("signal",),
        fit_model=fit_model,
        protocol=PrequentialProtocol(inner_splits=2),
    )
    mutated = frame.copy()
    mutated.loc[
        mutated["trading_day"] >= pd.Timestamp("2025-03-01"), "net_pnl_bps"
    ] = 1_000_000.0
    changed = run_strict_prequential(
        mutated,
        feature_names=("signal",),
        fit_model=fit_model,
        protocol=PrequentialProtocol(inner_splits=2),
    )

    baseline_fold_1 = baseline.outer_predictions.query("outer_fold == 1")
    changed_fold_1 = changed.outer_predictions.query("outer_fold == 1")
    assert baseline_fold_1["utility_raw_bps"].tolist() == pytest.approx(
        changed_fold_1["utility_raw_bps"].tolist()
    )


def test_protocol_rejects_rows_from_final_test_period() -> None:
    frame = _frame()
    frame.loc[len(frame)] = {
        "trading_day": pd.Timestamp("2025-06-30"),
        "contract": "fu9999",
        "slot_id": 0,
        "direction": 1,
        "signal": 1.0,
        "net_pnl_bps": 1.0,
        "win": True,
    }

    with pytest.raises(ValueError, match="outside the frozen valid range"):
        run_strict_prequential(
            frame,
            feature_names=("signal",),
            fit_model=lambda prefix, lambda_cvar, lambda_mae: _DeterministicModel(0.0),
            protocol=PrequentialProtocol(inner_splits=2),
        )


def test_frozen_protocol_change_invalidates_fingerprint() -> None:
    baseline = PrequentialProtocol()
    changed = PrequentialProtocol(ood_recover_threshold=0.02)

    assert baseline.fingerprint != changed.fingerprint
