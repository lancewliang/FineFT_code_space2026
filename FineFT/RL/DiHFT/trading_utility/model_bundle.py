from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np


TAIL_QUANTILES = (0.05, 0.10, 0.20)


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
