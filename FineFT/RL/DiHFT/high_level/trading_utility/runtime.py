from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib

from RL.DiHFT.high_level.trading_utility.eligibility import EligibilityReport
from RL.DiHFT.high_level.trading_utility.gate import TradingUtilityGate
from RL.DiHFT.high_level.trading_utility.model_bundle import UtilityModelBundle


@dataclass(frozen=True)
class LoadedUtilityRuntime:
    gate: TradingUtilityGate
    artifact_id: str
    protocol_fingerprint: str
    feature_schema: tuple[str, ...]
    horizon: int
    ood_enter_threshold: float
    ood_recover_threshold: float


def publish_runtime_manifest(
    *,
    output_dir: str | Path,
    bundle_manifest: str | Path,
    calibrator: Any,
    eligibility_manifest: str | Path,
    protocol_manifest: Mapping[str, Any],
) -> Path:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    bundle_path = Path(bundle_manifest).resolve()
    eligibility_path = Path(eligibility_manifest).resolve()
    if not bundle_path.is_file() or not eligibility_path.is_file():
        raise ValueError("runtime assets require explicit existing manifests")
    bundle = UtilityModelBundle.load(bundle_path)
    _validate_protocol_manifest(protocol_manifest)
    calibrator_path = target / "utility_calibrator.joblib"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".utility_calibrator.", suffix=".joblib", dir=target
    )
    os.close(descriptor)
    temporary_calibrator = Path(temporary_name)
    try:
        joblib.dump(calibrator, temporary_calibrator)
        os.replace(temporary_calibrator, calibrator_path)
    finally:
        temporary_calibrator.unlink(missing_ok=True)
    core = {
        "schema_version": 1,
        "bundle_manifest": _asset_entry(bundle_path),
        "calibrator": _asset_entry(calibrator_path.resolve()),
        "eligibility_manifest": _asset_entry(eligibility_path),
        "protocol_manifest": dict(protocol_manifest),
        "protocol_fingerprint": str(
            protocol_manifest.get("protocol_fingerprint", "")
        ),
        "feature_schema": list(bundle.feature_names),
    }
    artifact_id = _runtime_artifact_id(core)
    payload = {**core, "artifact_id": artifact_id}
    path = target / "trading_utility_runtime_manifest.json"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".utility_runtime.", suffix=".json", dir=target
    )
    os.close(descriptor)
    temporary_manifest = Path(temporary_name)
    try:
        temporary_manifest.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(temporary_manifest, path)
    finally:
        temporary_manifest.unlink(missing_ok=True)
    return path


def load_utility_runtime(manifest_path: str | Path) -> LoadedUtilityRuntime:
    path = Path(manifest_path)
    if not path.is_file():
        raise ValueError(f"explicit utility runtime manifest does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported utility runtime schema_version")
    required = {
        "artifact_id",
        "bundle_manifest",
        "calibrator",
        "eligibility_manifest",
        "protocol_manifest",
        "protocol_fingerprint",
        "feature_schema",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"utility runtime manifest is missing fields: {missing}")
    if not payload["protocol_fingerprint"]:
        raise ValueError("utility runtime protocol fingerprint is required")
    if (
        payload["protocol_manifest"].get("protocol_fingerprint")
        != payload["protocol_fingerprint"]
    ):
        raise ValueError("utility runtime protocol fingerprint mismatch")
    _validate_protocol_manifest(payload["protocol_manifest"])
    for key in ("bundle_manifest", "calibrator", "eligibility_manifest"):
        _verify_asset(payload[key], key)
    core = {key: value for key, value in payload.items() if key != "artifact_id"}
    expected_id = _runtime_artifact_id(core)
    legacy_path_bound_id = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if payload.get("artifact_id") not in {expected_id, legacy_path_bound_id}:
        raise ValueError("utility runtime manifest artifact_id mismatch")
    bundle = UtilityModelBundle.load(payload["bundle_manifest"]["path"])
    if list(bundle.feature_names) != list(payload["feature_schema"]):
        raise ValueError("utility runtime feature schema mismatch")
    calibrator = joblib.load(payload["calibrator"]["path"])
    if not callable(getattr(calibrator, "predict_utility", None)) or not callable(
        getattr(calibrator, "predict_win_probability", None)
    ):
        raise ValueError("utility runtime calibrator contract mismatch")
    eligibility = EligibilityReport.load(payload["eligibility_manifest"]["path"])
    evidence = eligibility.evidence_metadata
    protocol_manifest = payload["protocol_manifest"]
    if "final_calibration_source" in protocol_manifest:
        if evidence.get("protocol_fingerprint") != payload["protocol_fingerprint"]:
            raise ValueError("utility runtime eligibility protocol fingerprint mismatch")
        if int(evidence.get("outer_fold_count", -1)) != int(
            protocol_manifest.get("outer_fold_count", -2)
        ):
            raise ValueError("utility runtime eligibility outer-Fold mismatch")
        if evidence.get("calibration_source") != protocol_manifest.get(
            "final_calibration_source"
        ):
            raise ValueError("utility runtime eligibility calibration source mismatch")
        if (
            "lambda_cvar" in evidence
            and float(evidence["lambda_cvar"]) != bundle.lambda_cvar
        ):
            raise ValueError("utility runtime eligibility lambda_cvar mismatch")
        if (
            "lambda_mae" in evidence
            and float(evidence["lambda_mae"]) != bundle.lambda_mae
        ):
            raise ValueError("utility runtime eligibility lambda_mae mismatch")
    gate = TradingUtilityGate(bundle, calibrator, eligibility)
    return LoadedUtilityRuntime(
        gate=gate,
        artifact_id=str(payload["artifact_id"]),
        protocol_fingerprint=str(payload.get("protocol_fingerprint", "")),
        feature_schema=tuple(payload["feature_schema"]),
        horizon=int(payload["protocol_manifest"]["frozen_protocol"]["horizon"]),
        ood_enter_threshold=float(
            payload["protocol_manifest"]["frozen_protocol"][
                "ood_enter_threshold"
            ]
        ),
        ood_recover_threshold=float(
            payload["protocol_manifest"]["frozen_protocol"][
                "ood_recover_threshold"
            ]
        ),
    )


def _validate_protocol_manifest(protocol_manifest: Mapping[str, Any]) -> None:
    frozen = protocol_manifest.get("frozen_protocol")
    if not isinstance(frozen, Mapping):
        raise ValueError("utility runtime frozen protocol is required")
    required = {"horizon", "ood_enter_threshold", "ood_recover_threshold"}
    missing = sorted(required - set(frozen))
    if missing:
        raise ValueError(f"utility runtime frozen protocol is missing fields: {missing}")
    expected = hashlib.sha256(
        json.dumps(frozen, sort_keys=True, default=list).encode("utf-8")
    ).hexdigest()
    if protocol_manifest.get("protocol_fingerprint") != expected:
        raise ValueError("utility runtime frozen protocol hash mismatch")
    horizon = int(frozen["horizon"])
    enter = float(frozen["ood_enter_threshold"])
    recover = float(frozen["ood_recover_threshold"])
    if horizon <= 0 or not 0.0 <= enter < recover <= 1.0:
        raise ValueError("utility runtime frozen safety protocol is invalid")


def _asset_entry(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _file_hash(path)}


def _verify_asset(metadata: Mapping[str, str], name: str) -> None:
    path = Path(metadata["path"])
    if not path.is_file():
        raise ValueError(f"utility runtime asset is missing: {name}")
    if _file_hash(path) != metadata["sha256"]:
        raise ValueError(f"utility runtime asset hash mismatch: {name}")


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_artifact_id(core: Mapping[str, Any]) -> str:
    """Content identity that does not change when verified assets are relocated."""

    identity = json.loads(json.dumps(core))
    for name in ("bundle_manifest", "calibrator", "eligibility_manifest"):
        identity[name] = {"sha256": identity[name]["sha256"]}
    identity.get("protocol_manifest", {}).pop("candidate_episode_source", None)
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
