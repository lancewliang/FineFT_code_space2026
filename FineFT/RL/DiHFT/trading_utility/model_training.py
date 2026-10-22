from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from RL.DiHFT.trading_utility.model_bundle import TAIL_QUANTILES, UtilityModelBundle


@dataclass(frozen=True)
class UtilityTrainingData:
    features: np.ndarray
    feature_names: tuple[str, ...]
    net_pnl_bps: np.ndarray
    downside_loss_bps: np.ndarray
    mae_bps: np.ndarray
    wins: np.ndarray
    contracts: tuple[str, ...]
    slots: np.ndarray
    directions: np.ndarray

    def validate(self) -> None:
        features = np.asarray(self.features, dtype=float)
        if features.ndim != 2 or features.shape[1] != len(self.feature_names):
            raise ValueError("features do not match feature_names")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be unique")
        row_count = features.shape[0]
        arrays = (
            self.net_pnl_bps,
            self.downside_loss_bps,
            self.mae_bps,
            self.wins,
            self.slots,
            self.directions,
        )
        if row_count == 0 or any(len(values) != row_count for values in arrays):
            raise ValueError("training targets must match feature rows")
        if len(self.contracts) != row_count:
            raise ValueError("contracts must match feature rows")
        if not np.all(np.isfinite(features)):
            raise ValueError("training features must be finite")
        for values in (self.net_pnl_bps, self.downside_loss_bps, self.mae_bps):
            if not np.all(np.isfinite(np.asarray(values, dtype=float))):
                raise ValueError("training targets must be finite")
        if len(np.unique(np.asarray(self.wins, dtype=int))) < 2:
            raise ValueError("win classifier requires both outcome classes")


class UtilityModelTrainer:
    """Fit utility-model heads and return the inference-only bundle."""

    @classmethod
    def fit(
        cls,
        data: UtilityTrainingData,
        *,
        lambda_cvar: float = 0.5,
        lambda_mae: float = 0.5,
        random_seed: int = 42,
        backend: str = "auto",
    ) -> UtilityModelBundle:
        data.validate()
        if float(lambda_cvar) <= 0.0 or float(lambda_mae) <= 0.0:
            raise ValueError("tail-risk coefficients must be non-zero and positive")
        features = np.asarray(data.features, dtype=float)
        net = np.asarray(data.net_pnl_bps, dtype=float)
        mae = np.asarray(data.mae_bps, dtype=float)
        wins = np.asarray(data.wins, dtype=int)
        sample_weight = cls._balanced_weights(
            data.contracts, data.slots, data.directions
        )
        resolved_backend = cls._resolve_backend(backend)
        if resolved_backend == "catboost":
            mean_model, tail_models, mae_model, win_model = cls._fit_catboost(
                features, net, mae, wins, sample_weight, random_seed
            )
        else:
            mean_model, tail_models, mae_model, win_model = cls._fit_sklearn(
                features, net, mae, wins, sample_weight, random_seed
            )
        return UtilityModelBundle(
            feature_names=data.feature_names,
            mean_model=mean_model,
            tail_models=tail_models,
            mae_model=mae_model,
            win_model=win_model,
            lambda_cvar=lambda_cvar,
            lambda_mae=lambda_mae,
            random_seed=random_seed,
            resolved_backend=resolved_backend,
            training_data_hash=cls._training_data_hash(data),
        )

    @staticmethod
    def _resolve_backend(backend: str) -> str:
        normalized = str(backend).strip().lower()
        if normalized == "auto":
            return (
                "catboost"
                if importlib.util.find_spec("catboost")
                else "sklearn_gradient_boosting"
            )
        if normalized == "catboost":
            if importlib.util.find_spec("catboost") is None:
                raise ValueError("catboost backend requested but catboost is unavailable")
            return "catboost"
        if normalized in {"sklearn", "sklearn_gradient_boosting"}:
            return "sklearn_gradient_boosting"
        raise ValueError(f"unsupported utility-model backend {backend!r}")

    @staticmethod
    def _balanced_weights(
        contracts: Sequence[str], slots: np.ndarray, directions: np.ndarray
    ) -> np.ndarray:
        keys = [
            (str(contract), int(slot), int(direction))
            for contract, slot, direction in zip(contracts, slots, directions)
        ]
        counts = Counter(keys)
        weights = np.asarray([1.0 / counts[key] for key in keys], dtype=float)
        return weights / float(np.mean(weights))

    @staticmethod
    def _fit_sklearn(
        features: np.ndarray,
        net: np.ndarray,
        mae: np.ndarray,
        wins: np.ndarray,
        sample_weight: np.ndarray,
        random_seed: int,
    ) -> tuple[Any, dict[float, Any], Any, Any]:
        common = {"n_estimators": 80, "max_depth": 2, "random_state": random_seed}
        mean_model = GradientBoostingRegressor(loss="huber", **common)
        mean_model.fit(features, net, sample_weight=sample_weight)
        tail_models: dict[float, Any] = {}
        for quantile in TAIL_QUANTILES:
            model = GradientBoostingRegressor(
                loss="quantile", alpha=quantile, **common
            )
            model.fit(features, net, sample_weight=sample_weight)
            tail_models[quantile] = model
        mae_model = GradientBoostingRegressor(loss="absolute_error", **common)
        mae_model.fit(features, mae, sample_weight=sample_weight)
        win_model = LogisticRegression(random_state=random_seed, max_iter=1000)
        win_model.fit(features, wins, sample_weight=sample_weight)
        return mean_model, tail_models, mae_model, win_model

    @staticmethod
    def _fit_catboost(
        features: np.ndarray,
        net: np.ndarray,
        mae: np.ndarray,
        wins: np.ndarray,
        sample_weight: np.ndarray,
        random_seed: int,
    ) -> tuple[Any, dict[float, Any], Any, Any]:
        from catboost import CatBoostClassifier, CatBoostRegressor

        common = {
            "iterations": 120,
            "depth": 5,
            "random_seed": random_seed,
            "verbose": False,
            "allow_writing_files": False,
        }
        mean_model = CatBoostRegressor(loss_function="RMSE", **common)
        mean_model.fit(features, net, sample_weight=sample_weight)
        tail_models: dict[float, Any] = {}
        for quantile in TAIL_QUANTILES:
            model = CatBoostRegressor(
                loss_function=f"Quantile:alpha={quantile}", **common
            )
            model.fit(features, net, sample_weight=sample_weight)
            tail_models[quantile] = model
        mae_model = CatBoostRegressor(loss_function="MAE", **common)
        mae_model.fit(features, mae, sample_weight=sample_weight)
        win_model = CatBoostClassifier(loss_function="Logloss", **common)
        win_model.fit(features, wins, sample_weight=sample_weight)
        return mean_model, tail_models, mae_model, win_model

    @staticmethod
    def _training_data_hash(data: UtilityTrainingData) -> str:
        digest = hashlib.sha256()
        digest.update(np.asarray(data.features, dtype=float).tobytes())
        for values in (
            data.net_pnl_bps,
            data.downside_loss_bps,
            data.mae_bps,
            data.wins,
            data.slots,
            data.directions,
        ):
            digest.update(np.asarray(values).tobytes())
        digest.update(json.dumps(list(data.feature_names)).encode("utf-8"))
        digest.update(json.dumps(list(data.contracts)).encode("utf-8"))
        return digest.hexdigest()
