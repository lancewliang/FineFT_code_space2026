from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression


TAIL_QUANTILES = (0.05, 0.10, 0.20)


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


@dataclass(frozen=True)
class UtilityPrediction:
    mean_net_pnl_bps: float
    cvar_loss_bps: float
    mae_loss_bps: float
    win_probability: float
    raw_utility_bps: float


class UtilityModelBundle:
    """Four-head supervised utility model behind one stable interface."""

    def __init__(
        self,
        *,
        feature_names: Sequence[str],
        mean_model: Any,
        tail_models: Mapping[float, Any],
        mae_model: Any,
        win_model: Any,
        lambda_cvar: float,
        lambda_mae: float,
        random_seed: int,
        resolved_backend: str,
        training_data_hash: str,
        artifact_id: str | None = None,
    ) -> None:
        self.feature_names = tuple(str(name) for name in feature_names)
        self.mean_model = mean_model
        self.tail_models = {float(key): value for key, value in tail_models.items()}
        self.mae_model = mae_model
        self.win_model = win_model
        self.lambda_cvar = float(lambda_cvar)
        self.lambda_mae = float(lambda_mae)
        self.random_seed = int(random_seed)
        self.resolved_backend = str(resolved_backend)
        self.training_data_hash = str(training_data_hash)
        self.artifact_id = artifact_id
        if self.lambda_cvar <= 0.0 or self.lambda_mae <= 0.0:
            raise ValueError("tail-risk coefficients must be non-zero and positive")
        if set(self.tail_models) != set(TAIL_QUANTILES):
            raise ValueError("lower-tail head must contain the registered quantiles")

    @classmethod
    def fit(
        cls,
        data: UtilityTrainingData,
        *,
        lambda_cvar: float = 0.5,
        lambda_mae: float = 0.5,
        random_seed: int = 42,
        backend: str = "auto",
    ) -> "UtilityModelBundle":
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
                features,
                net,
                mae,
                wins,
                sample_weight,
                random_seed,
            )
        else:
            mean_model, tail_models, mae_model, win_model = cls._fit_sklearn(
                features,
                net,
                mae,
                wins,
                sample_weight,
                random_seed,
            )
        return cls(
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
            return "catboost" if importlib.util.find_spec("catboost") else "sklearn_gradient_boosting"
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

    def predict_one(self, features: Mapping[str, float]) -> UtilityPrediction:
        missing = [name for name in self.feature_names if name not in features]
        if missing:
            raise ValueError(f"missing utility features: {missing}")
        row = np.asarray(
            [[float(features[name]) for name in self.feature_names]], dtype=float
        )
        if not np.all(np.isfinite(row)):
            raise ValueError("utility features must be finite")
        mean_net = float(np.asarray(self.mean_model.predict(row)).reshape(-1)[0])
        lower_quantiles = [
            float(np.asarray(self.tail_models[q].predict(row)).reshape(-1)[0])
            for q in TAIL_QUANTILES
        ]
        cvar_loss = max(0.0, -float(np.mean(lower_quantiles)))
        mae_loss = max(
            0.0, float(np.asarray(self.mae_model.predict(row)).reshape(-1)[0])
        )
        probability = float(
            np.asarray(self.win_model.predict_proba(row))[0, 1]
        )
        utility = mean_net - self.lambda_cvar * cvar_loss - self.lambda_mae * mae_loss
        return UtilityPrediction(
            mean_net_pnl_bps=mean_net,
            cvar_loss_bps=cvar_loss,
            mae_loss_bps=mae_loss,
            win_probability=min(1.0, max(0.0, probability)),
            raw_utility_bps=utility,
        )

    def predict_components(self, frame: Any) -> Any:
        """Return the four raw component columns for a tabular public seam."""
        import pandas as pd

        missing = [name for name in self.feature_names if name not in frame.columns]
        if missing:
            raise ValueError(f"missing utility features: {missing}")
        features = frame[list(self.feature_names)].to_numpy(dtype=float)
        mean_net = np.asarray(self.mean_model.predict(features), dtype=float)
        lower = np.vstack(
            [
                np.asarray(self.tail_models[q].predict(features), dtype=float)
                for q in TAIL_QUANTILES
            ]
        )
        cvar_loss = np.maximum(0.0, -np.mean(lower, axis=0))
        mae_loss = np.maximum(
            0.0, np.asarray(self.mae_model.predict(features), dtype=float)
        )
        win_probability = np.asarray(
            self.win_model.predict_proba(features)[:, 1], dtype=float
        )
        return pd.DataFrame(
            {
                "mean_net_pnl_bps": mean_net,
                "cvar_loss_bps": cvar_loss,
                "mae_loss_bps": mae_loss,
                "win_probability_raw": np.clip(win_probability, 0.0, 1.0),
            },
            index=frame.index,
        )

    def save(self, output_dir: str | Path) -> Path:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        assets = {
            "mean_net_pnl": ("mean_net_pnl.joblib", self.mean_model),
            "lower_tail": ("lower_tail.joblib", self.tail_models),
            "mae": ("mae.joblib", self.mae_model),
            "win_probability": ("win_probability.joblib", self.win_model),
        }
        heads: dict[str, dict[str, str]] = {}
        for name, (filename, model) in assets.items():
            path = target / filename
            joblib.dump(model, path)
            heads[name] = {"path": filename, "sha256": self._file_hash(path)}
        core = {
            "schema_version": 1,
            "preferred_backend": "catboost",
            "resolved_backend": self.resolved_backend,
            "feature_schema": list(self.feature_names),
            "lambda_cvar": self.lambda_cvar,
            "lambda_mae": self.lambda_mae,
            "tail_quantiles": list(TAIL_QUANTILES),
            "random_seed": self.random_seed,
            "training_data_hash": self.training_data_hash,
            "heads": heads,
        }
        artifact_id = hashlib.sha256(
            json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        manifest = {**core, "artifact_id": artifact_id}
        manifest_path = target / "utility_bundle_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        self.artifact_id = artifact_id
        return manifest_path

    @classmethod
    def load(cls, manifest_path: str | Path) -> "UtilityModelBundle":
        path = Path(manifest_path)
        if not path.is_file():
            raise ValueError(f"explicit utility manifest does not exist: {path}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        core = {key: value for key, value in manifest.items() if key != "artifact_id"}
        expected_artifact_id = hashlib.sha256(
            json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if manifest.get("artifact_id") != expected_artifact_id:
            raise ValueError("utility bundle manifest artifact_id mismatch")
        required_heads = {
            "mean_net_pnl",
            "lower_tail",
            "mae",
            "win_probability",
        }
        if set(manifest.get("heads", {})) != required_heads:
            raise ValueError("utility manifest does not contain all four heads")
        loaded: dict[str, Any] = {}
        for name, metadata in manifest["heads"].items():
            asset_path = path.parent / metadata["path"]
            if not asset_path.is_file():
                raise ValueError(f"utility head is missing: {name}")
            if cls._file_hash(asset_path) != metadata["sha256"]:
                raise ValueError(f"utility head hash mismatch: {name}")
            loaded[name] = joblib.load(asset_path)
        return cls(
            feature_names=manifest["feature_schema"],
            mean_model=loaded["mean_net_pnl"],
            tail_models=loaded["lower_tail"],
            mae_model=loaded["mae"],
            win_model=loaded["win_probability"],
            lambda_cvar=float(manifest["lambda_cvar"]),
            lambda_mae=float(manifest["lambda_mae"]),
            random_seed=int(manifest["random_seed"]),
            resolved_backend=str(manifest["resolved_backend"]),
            training_data_hash=str(manifest["training_data_hash"]),
            artifact_id=str(manifest["artifact_id"]),
        )

    @staticmethod
    def _file_hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

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
