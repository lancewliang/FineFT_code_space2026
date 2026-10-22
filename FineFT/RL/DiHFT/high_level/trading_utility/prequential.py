from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from RL.DiHFT.high_level.trading_utility.model_bundle import (
    UtilityModelBundle,
    UtilityTrainingData,
)


ModelFit = Callable[[pd.DataFrame, float, float], Any]
SelectorBuild = Callable[[pd.DataFrame, str], Any]


@dataclass(frozen=True)
class PrequentialProtocol:
    initial_start: str = "2024-07-23"
    initial_end: str = "2024-10-31"
    fold_1_start: str = "2024-11-01"
    fold_1_end: str = "2025-02-28"
    fold_2_start: str = "2025-03-01"
    fold_2_end: str = "2025-06-27"
    horizon: int = 3
    embargo_steps: int = 3
    inner_splits: int = 3
    lambda_grid: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
    feature_schema_version: str = "utility_features_v1"
    quintile_method: str = "pooled_outer_unseen_equal_frequency"
    estimator_family: str = "catboost_preferred_replaceable"
    cost_model_version: str = "after_cost_bps_v1"
    friction_stress: float = 1.5
    ood_enter_threshold: float = 0.005
    ood_recover_threshold: float = 0.01
    minimum_channel_episodes: int = 150
    minimum_quintile_episodes: int = 30
    minimum_contracts: int = 3
    minimum_trading_day_blocks: int = 20
    fdr_alpha: float = 0.05
    minimum_spearman: float = 0.80
    protocol_version: str = "tri_module_prequential_v1"

    def __post_init__(self) -> None:
        if self.horizon <= 0 or self.embargo_steps < self.horizon:
            raise ValueError("embargo_steps must be at least the positive horizon")
        if self.inner_splits <= 0:
            raise ValueError("inner_splits must be positive")
        if not self.lambda_grid or any(value <= 0.0 for value in self.lambda_grid):
            raise ValueError("lambda_grid must contain only non-zero positive values")
        if not 0.0 <= self.ood_enter_threshold < self.ood_recover_threshold <= 1.0:
            raise ValueError("frozen OOD thresholds are invalid")

    @property
    def valid_start(self) -> pd.Timestamp:
        return pd.Timestamp(self.initial_start)

    @property
    def valid_end(self) -> pd.Timestamp:
        return pd.Timestamp(self.fold_2_end)

    @property
    def outer_folds(self) -> tuple[tuple[int, pd.Timestamp, pd.Timestamp], ...]:
        return (
            (1, pd.Timestamp(self.fold_1_start), pd.Timestamp(self.fold_1_end)),
            (2, pd.Timestamp(self.fold_2_start), pd.Timestamp(self.fold_2_end)),
        )

    @property
    def fingerprint(self) -> str:
        payload = {
            key: value
            for key, value in self.__dict__.items()
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=list).encode("utf-8")
        ).hexdigest()


@dataclass
class UtilityCalibrator:
    utility_model: IsotonicRegression
    win_model: LogisticRegression

    @classmethod
    def fit(
        cls,
        raw_utility: Sequence[float],
        raw_win_probability: Sequence[float],
        realized_net_pnl: Sequence[float],
        wins: Sequence[bool],
    ) -> "UtilityCalibrator":
        utility = np.asarray(raw_utility, dtype=float)
        probability = np.asarray(raw_win_probability, dtype=float)
        net = np.asarray(realized_net_pnl, dtype=float)
        labels = np.asarray(wins, dtype=int)
        if len(utility) < 4:
            raise ValueError("calibration requires at least four OOF observations")
        if len(np.unique(labels)) < 2:
            raise ValueError("Platt calibration requires both outcome classes")
        isotonic = IsotonicRegression(increasing=True, out_of_bounds="clip")
        isotonic.fit(utility, net)
        platt = LogisticRegression(max_iter=1000, random_state=0)
        platt.fit(probability.reshape(-1, 1), labels)
        return cls(isotonic, platt)

    def predict_utility(self, raw_utility: Sequence[float]) -> np.ndarray:
        return np.asarray(self.utility_model.predict(raw_utility), dtype=float)

    def predict_win_probability(
        self, raw_probability: Sequence[float]
    ) -> np.ndarray:
        values = np.asarray(raw_probability, dtype=float).reshape(-1, 1)
        return np.asarray(self.win_model.predict_proba(values)[:, 1], dtype=float)


@dataclass(frozen=True)
class PrequentialResult:
    outer_predictions: pd.DataFrame
    final_model: Any
    final_selector: Any
    final_calibrator: UtilityCalibrator
    final_lambda_cvar: float
    final_lambda_mae: float
    final_calibration_source: str
    protocol_manifest: dict[str, Any]
    fold_selectors: tuple[Any, ...] = field(default_factory=tuple)


def run_strict_prequential(
    frame: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    fit_model: ModelFit | None = None,
    build_selector: SelectorBuild | None = None,
    protocol: PrequentialProtocol | None = None,
) -> PrequentialResult:
    active_protocol = protocol or PrequentialProtocol()
    data = _validated_frame(frame, feature_names, active_protocol)
    model_fit = fit_model or build_utility_model_fitter(feature_names)
    selector_build = build_selector or _default_selector_builder
    outer_results: list[pd.DataFrame] = []
    fold_selectors: list[Any] = []

    for fold_id, eval_start, eval_end in active_protocol.outer_folds:
        prefix = data[data["trading_day"] < eval_start].copy()
        prefix = _purge_contract_tails(prefix, active_protocol.embargo_steps)
        evaluation = data[
            (data["trading_day"] >= eval_start)
            & (data["trading_day"] <= eval_end)
        ].copy()
        if prefix.empty or evaluation.empty:
            raise ValueError(f"outer Fold {fold_id} has empty train or evaluation data")
        selector = selector_build(prefix.copy(), f"fold_{fold_id}")
        fold_selectors.append(selector)
        inner_oof = _build_inner_oof(
            prefix,
            model_fit,
            active_protocol,
        )
        lambda_cvar, lambda_mae = _select_lambda_pair(
            inner_oof, active_protocol.lambda_grid
        )
        inner_oof = _with_raw_utility(inner_oof, lambda_cvar, lambda_mae)
        calibrator = UtilityCalibrator.fit(
            inner_oof["utility_raw_bps"],
            inner_oof["win_probability_raw"],
            inner_oof["net_pnl_bps"],
            inner_oof["win"],
        )
        model = model_fit(prefix.copy(), lambda_cvar, lambda_mae)
        predictions = _predict_components(model, evaluation)
        predictions = _with_raw_utility(predictions, lambda_cvar, lambda_mae)
        predictions["utility_calibrated_bps"] = calibrator.predict_utility(
            predictions["utility_raw_bps"]
        )
        predictions["win_probability_calibrated"] = (
            calibrator.predict_win_probability(predictions["win_probability_raw"])
        )
        for column in (
            "trading_day",
            "contract",
            "slot_id",
            "direction",
            "net_pnl_bps",
            "win",
        ):
            predictions[column] = evaluation[column].to_numpy()
        predictions["outer_fold"] = fold_id
        predictions["lambda_cvar"] = lambda_cvar
        predictions["lambda_mae"] = lambda_mae
        predictions["selector_identity"] = str(selector)
        outer_results.append(predictions.reset_index(drop=True))

    outer_predictions = pd.concat(outer_results, ignore_index=True)
    final_lambda_cvar, final_lambda_mae = _select_lambda_pair(
        outer_predictions, active_protocol.lambda_grid
    )
    final_prediction_basis = _with_raw_utility(
        outer_predictions.copy(), final_lambda_cvar, final_lambda_mae
    )
    final_calibrator = UtilityCalibrator.fit(
        final_prediction_basis["utility_raw_bps"],
        final_prediction_basis["win_probability_raw"],
        final_prediction_basis["net_pnl_bps"],
        final_prediction_basis["win"],
    )
    final_selector = selector_build(data.copy(), "final_full_valid")
    final_model = model_fit(data.copy(), final_lambda_cvar, final_lambda_mae)
    manifest = {
        "schema_version": 1,
        "protocol_version": active_protocol.protocol_version,
        "protocol_fingerprint": active_protocol.fingerprint,
        "frozen_protocol": dict(active_protocol.__dict__),
        "official_trading_day_boundaries": {
            "initial_history": [active_protocol.initial_start, active_protocol.initial_end],
            "outer_fold_1": [active_protocol.fold_1_start, active_protocol.fold_1_end],
            "outer_fold_2": [active_protocol.fold_2_start, active_protocol.fold_2_end],
        },
        "outer_fold_count": 2,
        "horizon": active_protocol.horizon,
        "embargo_steps": active_protocol.embargo_steps,
        "inner_splits": active_protocol.inner_splits,
        "lambda_grid": list(active_protocol.lambda_grid),
        "feature_schema": list(feature_names),
        "feature_schema_version": active_protocol.feature_schema_version,
        "quintile_method": active_protocol.quintile_method,
        "final_calibration_source": "outer_unseen_predictions",
        "test_accessed": False,
    }
    return PrequentialResult(
        outer_predictions=outer_predictions,
        final_model=final_model,
        final_selector=final_selector,
        final_calibrator=final_calibrator,
        final_lambda_cvar=final_lambda_cvar,
        final_lambda_mae=final_lambda_mae,
        final_calibration_source="outer_unseen_predictions",
        protocol_manifest=manifest,
        fold_selectors=tuple(fold_selectors),
    )


def _validated_frame(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    protocol: PrequentialProtocol,
) -> pd.DataFrame:
    required = {
        "trading_day",
        "contract",
        "slot_id",
        "direction",
        "net_pnl_bps",
        "win",
        *feature_names,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"prequential data is missing columns: {missing}")
    data = frame.copy()
    data["trading_day"] = pd.to_datetime(data["trading_day"]).dt.normalize()
    if data["trading_day"].min() < protocol.valid_start or data["trading_day"].max() > protocol.valid_end:
        raise ValueError("prequential data contains rows outside the frozen valid range")
    if data[list(feature_names)].isna().any().any():
        raise ValueError("prequential features must be complete")
    return data.sort_values(["trading_day", "contract"], kind="stable").reset_index(drop=True)


def _purge_contract_tails(frame: pd.DataFrame, embargo_steps: int) -> pd.DataFrame:
    if embargo_steps <= 0:
        return frame.copy()
    keep = np.ones(len(frame), dtype=bool)
    for indices in frame.groupby("contract", sort=False).indices.values():
        ordered = np.asarray(indices, dtype=int)
        group = frame.iloc[ordered]
        if "step_index" in group.columns:
            decision_keys = group["step_index"].drop_duplicates().to_numpy()
            embargoed = set(decision_keys[-embargo_steps:].tolist())
            remove = group["step_index"].isin(embargoed).to_numpy()
        else:
            decision_keys = group["trading_day"].drop_duplicates().to_numpy()
            embargoed = set(decision_keys[-embargo_steps:].tolist())
            remove = group["trading_day"].isin(embargoed).to_numpy()
        keep[ordered[remove]] = False
    return frame.loc[keep].reset_index(drop=True)


def _build_inner_oof(
    prefix: pd.DataFrame,
    fit_model: ModelFit,
    protocol: PrequentialProtocol,
) -> pd.DataFrame:
    days = np.asarray(sorted(prefix["trading_day"].unique()))
    minimum_history = max(4, len(days) // (protocol.inner_splits + 1))
    future_days = days[minimum_history:]
    blocks = [block for block in np.array_split(future_days, protocol.inner_splits) if len(block)]
    outputs: list[pd.DataFrame] = []
    for block in blocks:
        block_start = pd.Timestamp(block[0])
        inner_train = prefix[prefix["trading_day"] < block_start].copy()
        inner_train = _purge_contract_tails(inner_train, protocol.embargo_steps)
        inner_eval = prefix[prefix["trading_day"].isin(block)].copy()
        if len(inner_train) < 8 or inner_eval.empty:
            continue
        model = fit_model(inner_train, 1.0, 1.0)
        predictions = _predict_components(model, inner_eval)
        predictions["net_pnl_bps"] = inner_eval["net_pnl_bps"].to_numpy()
        predictions["win"] = inner_eval["win"].astype(bool).to_numpy()
        outputs.append(predictions)
    if not outputs:
        raise ValueError("inner prequential split produced no OOF observations")
    return pd.concat(outputs, ignore_index=True)


def _predict_components(model: Any, frame: pd.DataFrame) -> pd.DataFrame:
    predictions = model.predict_components(frame)
    required = {
        "mean_net_pnl_bps",
        "cvar_loss_bps",
        "mae_loss_bps",
        "win_probability_raw",
    }
    if set(predictions.columns) < required:
        raise ValueError("utility model did not return all four prediction components")
    if len(predictions) != len(frame):
        raise ValueError("utility model prediction length mismatch")
    return predictions.reset_index(drop=True).copy()


def _with_raw_utility(
    predictions: pd.DataFrame, lambda_cvar: float, lambda_mae: float
) -> pd.DataFrame:
    result = predictions.copy()
    result["utility_raw_bps"] = (
        result["mean_net_pnl_bps"]
        - float(lambda_cvar) * result["cvar_loss_bps"]
        - float(lambda_mae) * result["mae_loss_bps"]
    )
    return result


def _select_lambda_pair(
    predictions: pd.DataFrame, lambda_grid: Sequence[float]
) -> tuple[float, float]:
    best_pair: tuple[float, float] | None = None
    best_score = float("-inf")
    for lambda_cvar in lambda_grid:
        for lambda_mae in lambda_grid:
            utility = (
                predictions["mean_net_pnl_bps"]
                - lambda_cvar * predictions["cvar_loss_bps"]
                - lambda_mae * predictions["mae_loss_bps"]
            )
            if not _passes_inner_risk_and_monotonicity_constraints(
                utility, predictions
            ):
                continue
            admitted = predictions.loc[utility > 0.0, "net_pnl_bps"]
            if admitted.empty:
                score = float("-inf")
            else:
                standard_error = float(admitted.std(ddof=1)) / max(
                    1.0, float(np.sqrt(len(admitted)))
                )
                score = float(admitted.mean()) - 1.645 * standard_error
            candidate = (float(lambda_cvar), float(lambda_mae))
            if score > best_score or (
                score == best_score and (best_pair is None or candidate < best_pair)
            ):
                best_score = score
                best_pair = candidate
    if best_pair is None:
        conservative = float(max(lambda_grid))
        return conservative, conservative
    return best_pair


def _passes_inner_risk_and_monotonicity_constraints(
    utility: pd.Series, predictions: pd.DataFrame
) -> bool:
    admitted = predictions.loc[utility > 0.0, "net_pnl_bps"]
    if admitted.empty or float(admitted.mean()) <= 0.0:
        return False
    if len(predictions) < 25:
        return True
    ranked = utility.rank(method="first")
    quintile = pd.qcut(ranked, 5, labels=False)
    net_means = predictions.groupby(quintile, observed=True)["net_pnl_bps"].mean()
    win_rates = predictions.groupby(quintile, observed=True)["win"].mean()
    if len(net_means) != 5 or len(win_rates) != 5:
        return False
    return bool(
        np.all(np.diff(net_means.to_numpy(dtype=float)) > 0.0)
        and np.all(np.diff(win_rates.to_numpy(dtype=float)) >= 0.0)
    )


def build_utility_model_fitter(
    feature_names: Sequence[str],
    *,
    backend: str = "auto",
    random_seed: int = 42,
) -> ModelFit:
    names = tuple(feature_names)

    def fit(frame: pd.DataFrame, lambda_cvar: float, lambda_mae: float) -> UtilityModelBundle:
        required_targets = {"downside_loss_bps", "mae_bps"}
        missing = sorted(required_targets - set(frame.columns))
        if missing:
            raise ValueError(f"default utility trainer is missing targets: {missing}")
        data = UtilityTrainingData(
            features=frame[list(names)].to_numpy(dtype=float),
            feature_names=names,
            net_pnl_bps=frame["net_pnl_bps"].to_numpy(dtype=float),
            downside_loss_bps=frame["downside_loss_bps"].to_numpy(dtype=float),
            mae_bps=frame["mae_bps"].to_numpy(dtype=float),
            wins=frame["win"].astype(bool).to_numpy(),
            contracts=tuple(frame["contract"].astype(str)),
            slots=frame["slot_id"].to_numpy(dtype=int),
            directions=frame["direction"].to_numpy(dtype=int),
        )
        return UtilityModelBundle.fit(
            data,
            lambda_cvar=lambda_cvar,
            lambda_mae=lambda_mae,
            backend=backend,
            random_seed=random_seed,
        )

    return fit


def _default_selector_builder(prefix: pd.DataFrame, fold_name: str) -> str:
    payload = {
        "fold": fold_name,
        "first_trading_day": str(prefix["trading_day"].min().date()),
        "last_trading_day": str(prefix["trading_day"].max().date()),
        "row_count": len(prefix),
        "contracts": sorted(prefix["contract"].astype(str).unique()),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
