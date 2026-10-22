from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import joblib

from RL.DiHFT.trading_utility.model_bundle import UtilityModelBundle


def publish_runtime_manifest(
    *,
    output_dir: str | Path,
    bundle_manifest: str | Path,
    calibrator: Any,
    eligibility_manifest: str | Path,
    protocol_manifest: Mapping[str, Any],
) -> Path:
    """Publish immutable model, calibration, and eligibility references for inference."""
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
    payload = {**core, "artifact_id": _runtime_artifact_id(core)}
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
