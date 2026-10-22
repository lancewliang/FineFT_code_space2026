from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from common.artifacts import ArtifactNames
from common.audit_records import (
    ACCOUNT_RECORD_COLUMNS,
    PREDICTION_RECORD_COLUMNS,
    TRADE_RECORD_COLUMNS,
)
from RL.DiHFT.trading_utility.eligibility import EligibilityReport
from RL.DiHFT.trading_utility.model_bundle import UtilityModelBundle
from RL.DiHFT.trading_utility.prequential import PrequentialProtocol
from RL.DiHFT.high_level.trading_utility.runtime import load_utility_runtime
from RL.DiHFT.trading_utility.runtime_artifacts import publish_runtime_manifest


TEST_START = pd.Timestamp("2025-06-30")
TEST_END = pd.Timestamp("2026-02-05")
AUDIT_SCHEMA_VERSION = "1.1.0"
_OOD_ASSET_NAMES = {
    "vae_weights",
    "train_scaler",
    "train_id_likelihood_reference",
}
_REPORT_SECTIONS = {
    "performance",
    "risk",
    "coverage",
    "ood",
    "route",
    "utility_rejection",
    "channel_support",
    "audit_integrity",
}
_UTILITY_ONLY_FIELDS = {
    "denied_baseline_action",
    "utility_raw",
    "utility_calibrated",
    "utility_lower",
    "win_probability",
    "channel_eligible",
    "utility_reject_stage",
    "utility_reject_reason",
    "utility_artifact_id",
}


@dataclass(frozen=True)
class StackTestResult:
    """Public result contract returned by each independently executed stack."""

    stack_type: str
    test_input_identity: str
    cost_model_version: str
    audit_schema_version: str
    metrics: Mapping[str, Mapping[str, Any]]
    prediction_records: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    trade_records: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    account_records: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    scenario_coverage: Sequence[str] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


LegacyEvaluator = Callable[[pd.DataFrame], StackTestResult]
TriModuleEvaluator = Callable[[pd.DataFrame, Mapping[str, Any]], StackTestResult]


def freeze_production_assets(
    *,
    output_dir: str | Path,
    utility_runtime_manifest: str | Path,
    selector_manifest: str | Path,
    selector_lineage: Mapping[str, Any],
    cost_model_manifest: str | Path,
    ood_assets: Mapping[str, str | Path],
    code_version: str,
    protocol: PrequentialProtocol | None = None,
) -> Path:
    """Validate and atomically publish the immutable pre-test asset inventory."""

    active_protocol = protocol or PrequentialProtocol()
    target = Path(output_dir).resolve()
    if target.exists():
        raise FileExistsError(f"frozen production directory already exists: {target}")
    if not code_version.strip() or code_version == "unspecified":
        raise ValueError("an explicit code_version is required before freezing")
    runtime_path = Path(utility_runtime_manifest).resolve()
    selector_path = Path(selector_manifest).resolve()
    cost_path = Path(cost_model_manifest).resolve()
    for name, path in {
        "utility_runtime_manifest": runtime_path,
        "selector_manifest": selector_path,
        "cost_model_manifest": cost_path,
    }.items():
        if not path.is_file():
            raise ValueError(f"freeze asset is missing: {name}")
    if set(ood_assets) != _OOD_ASSET_NAMES:
        raise ValueError(
            "OOD assets must contain exactly vae_weights, train_scaler, and "
            "train_id_likelihood_reference"
        )

    runtime = load_utility_runtime(runtime_path)
    runtime_payload = _read_json(runtime_path, "utility runtime manifest")
    selector_payload = _read_json(selector_path, "selector manifest")
    if selector_payload.get("schema_version") != 1 or not isinstance(
        selector_payload.get("slots"), list
    ):
        raise ValueError("selector manifest does not implement the frozen slot contract")
    protocol_manifest = runtime_payload["protocol_manifest"]
    _validate_frozen_protocol(protocol_manifest, active_protocol, code_version)
    _validate_selector_lineage(selector_lineage, active_protocol, code_version)

    bundle_manifest_path = Path(runtime_payload["bundle_manifest"]["path"])
    bundle_payload = _read_json(bundle_manifest_path, "utility bundle manifest")
    bundle = UtilityModelBundle.load(bundle_manifest_path)
    eligibility_path = Path(runtime_payload["eligibility_manifest"]["path"])
    eligibility_payload = _read_json(eligibility_path, "eligibility manifest")
    eligibility = EligibilityReport.load(eligibility_path)
    evidence = eligibility.evidence_metadata
    if evidence.get("protocol_fingerprint") != active_protocol.fingerprint:
        raise ValueError("eligibility protocol fingerprint is stale or incompatible")
    if int(evidence.get("outer_fold_count", -1)) != 2:
        raise ValueError("eligibility must be based on exactly two outer Folds")
    if evidence.get("calibration_source") != "outer_unseen_predictions":
        raise ValueError("eligibility is not based on outer unseen evidence")
    if (
        bundle.lambda_cvar not in active_protocol.lambda_grid
        or bundle.lambda_mae not in active_protocol.lambda_grid
    ):
        raise ValueError("utility lambda values are outside the frozen grid")

    cost_payload = _read_json(cost_path, "cost model manifest")
    if cost_payload.get("cost_model_version") != active_protocol.cost_model_version:
        raise ValueError("cost model version is incompatible with the frozen protocol")

    assets: dict[str, dict[str, str]] = {
        "utility_runtime_manifest": _asset_entry(runtime_path),
        "selector_manifest": _asset_entry(selector_path),
        "utility_bundle_manifest": _asset_entry(bundle_manifest_path),
        "isotonic_and_platt_calibrator": _asset_entry(
            Path(runtime_payload["calibrator"]["path"])
        ),
        "eligibility_and_residual_distribution": _asset_entry(eligibility_path),
        "cost_model_manifest": _asset_entry(cost_path),
    }
    for head_name, metadata in sorted(bundle_payload["heads"].items()):
        assets[f"utility_head:{head_name}"] = _asset_entry(
            bundle_manifest_path.parent / metadata["path"]
        )
    for name, source in sorted(ood_assets.items()):
        path = Path(source).resolve()
        if not path.is_file():
            raise ValueError(f"freeze OOD asset is missing: {name}")
        assets[f"ood:{name}"] = _asset_entry(path)

    residual_distribution = {
        key: float(channel.residual_q05)
        for key, channel in sorted(eligibility.channels.items())
    }
    core = {
        "schema_version": 1,
        "status": "frozen_before_test",
        "gating_stacks": ["legacy", "tri_module"],
        "default_gating_stack": "legacy",
        "legacy_retained": True,
        "code_version": code_version,
        "protocol_fingerprint": active_protocol.fingerprint,
        "valid_range": [active_protocol.initial_start, active_protocol.fold_2_end],
        "outer_fold_count": 2,
        "final_calibration_source": "outer_unseen_predictions",
        "test_range": [str(TEST_START.date()), str(TEST_END.date())],
        "test_run_count_before_freeze": 0,
        "selector": {
            "asset": assets["selector_manifest"],
            "lineage": dict(selector_lineage),
        },
        "utility_runtime_artifact_id": runtime.artifact_id,
        "utility_bundle_artifact_id": bundle.artifact_id,
        "feature_schema": list(runtime.feature_schema),
        "lambda_cvar": bundle.lambda_cvar,
        "lambda_mae": bundle.lambda_mae,
        "calibrators": {
            "continuous_utility": "IsotonicRegression",
            "win_probability": "Platt scaling",
        },
        "residual_distribution": residual_distribution,
        "eligibility_artifact_id": eligibility_payload["artifact_id"],
        "cost_model": cost_payload,
        "ood_rules": {
            "enter_threshold": active_protocol.ood_enter_threshold,
            "recover_threshold": active_protocol.ood_recover_threshold,
            "horizon": active_protocol.horizon,
            "embargo_steps": active_protocol.embargo_steps,
        },
        "assets": assets,
        "declarations": {
            "test_is_evaluation_only": True,
            "test_must_not_tune_assets": True,
            "test_must_not_switch_default_stack": True,
            "test_must_not_delete_legacy": True,
        },
    }
    freeze_id = _freeze_identity(core)
    payload = {**core, "freeze_id": freeze_id}

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        manifest = staging / ArtifactNames.FROZEN_PRODUCTION_MANIFEST_JSON
        _write_json(manifest, payload)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target / ArtifactNames.FROZEN_PRODUCTION_MANIFEST_JSON


def run_one_shot_test_comparison(
    *,
    frozen_manifest: str | Path,
    test_frame: pd.DataFrame,
    legacy_evaluator: LegacyEvaluator,
    tri_module_evaluator: TriModuleEvaluator,
) -> Path:
    """Run the two real stacks once and atomically publish the advisory report."""

    manifest_path = Path(frozen_manifest).resolve()
    freeze = _load_and_validate_freeze(manifest_path)
    official_dir = manifest_path.parent / "official_test"
    if official_dir.exists():
        raise FileExistsError("the formal test has already been published for this freeze")
    data = _validated_test_frame(test_frame)
    input_identity = deterministic_frame_identity(data)
    one_shot_run_id = _canonical_hash(
        {"freeze_id": freeze["freeze_id"], "test_input_identity": input_identity}
    )
    asset_hashes_before = _verify_frozen_assets(freeze)
    lock_path = manifest_path.parent / ".official_test.lock"
    try:
        lock_descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise FileExistsError("the formal test is already running for this freeze") from error
    os.close(lock_descriptor)
    staging: Path | None = None
    try:
        staging = Path(
            tempfile.mkdtemp(prefix=".official_test.", dir=manifest_path.parent)
        )
        legacy = legacy_evaluator(data.copy(deep=True))
        tri_module = tri_module_evaluator(data.copy(deep=True), freeze)
        _validate_stack_result(legacy, "legacy", input_identity, freeze)
        _validate_stack_result(tri_module, "tri_module", input_identity, freeze)
        if legacy.cost_model_version != tri_module.cost_model_version:
            raise ValueError("legacy and tri-module cost-model versions differ")
        if legacy.audit_schema_version != tri_module.audit_schema_version:
            raise ValueError("legacy and tri-module audit schemas differ")
        asset_hashes_after = _verify_frozen_assets(freeze)
        if asset_hashes_after != asset_hashes_before:
            raise RuntimeError("a frozen production asset changed during formal test")

        report = {
            "schema_version": 1,
            "status": "completed_once",
            "one_shot_run_id": one_shot_run_id,
            "freeze_id": freeze["freeze_id"],
            "freeze_manifest_sha256": _file_hash(manifest_path),
            "test_input_identity": input_identity,
            "test_range": freeze["test_range"],
            "cost_model_version": legacy.cost_model_version,
            "audit_schema_version": legacy.audit_schema_version,
            "legacy": legacy.to_dict(),
            "tri_module": tri_module.to_dict(),
            "declarations": {
                "test_used_for_tuning": False,
                "automatic_default_switch": False,
                "automatic_legacy_deletion": False,
                "automatic_threshold_update": False,
                "second_test_scheduled": False,
                "legacy_retained": True,
                "default_gating_stack": "legacy",
                "legacy_deletion_requires_separate_user_decision": True,
                "comparison_is_advisory": True,
                "no_claim_of_eliminating_losing_trades": True,
            },
        }
        report_path = staging / ArtifactNames.FROZEN_TEST_COMPARISON_JSON
        _write_json(report_path, report)
        _write_json(
            staging / ArtifactNames.FROZEN_TEST_RUN_MARKER_JSON,
            {
                "one_shot_run_id": one_shot_run_id,
                "freeze_id": freeze["freeze_id"],
                "test_input_identity": input_identity,
                "rerun_allowed_under_same_freeze": False,
            },
        )
        os.replace(staging, official_dir)
    except BaseException:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        lock_path.unlink(missing_ok=True)
    return official_dir / ArtifactNames.FROZEN_TEST_COMPARISON_JSON


def deterministic_frame_identity(frame: pd.DataFrame) -> str:
    columns = sorted(str(column) for column in frame.columns)
    normalized = frame.loc[:, columns].copy()
    for column in columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[column]):
            normalized[column] = normalized[column].dt.strftime(
                "%Y-%m-%dT%H:%M:%S.%f"
            )
    sort_keys = [
        key for key in ("trading_day", "TradingDay", "contract", "step_index")
        if key in normalized
    ]
    if sort_keys:
        normalized = normalized.sort_values(sort_keys, kind="stable")
    normalized = normalized.reset_index(drop=True)
    digest = hashlib.sha256()
    digest.update(json.dumps(columns, separators=(",", ":")).encode("utf-8"))
    digest.update(
        json.dumps([str(dtype) for dtype in normalized.dtypes]).encode("utf-8")
    )
    digest.update(pd.util.hash_pandas_object(normalized, index=False).values.tobytes())
    return digest.hexdigest()


def _validate_frozen_protocol(
    manifest: Mapping[str, Any], protocol: PrequentialProtocol, code_version: str
) -> None:
    if manifest.get("protocol_fingerprint") != protocol.fingerprint:
        raise ValueError("utility runtime Fold protocol is incompatible")
    expected_protocol = json.loads(json.dumps(dict(protocol.__dict__)))
    if manifest.get("frozen_protocol") != expected_protocol:
        raise ValueError("utility runtime frozen protocol fields are incompatible")
    if int(manifest.get("outer_fold_count", -1)) != 2:
        raise ValueError("utility runtime must contain exactly two outer Folds")
    if manifest.get("final_calibration_source") != "outer_unseen_predictions":
        raise ValueError("final calibrator must use outer unseen predictions")
    if manifest.get("test_accessed") is not False:
        raise ValueError("utility runtime indicates that test was already accessed")
    if manifest.get("code_version") != code_version:
        raise ValueError("utility runtime code version is incompatible")


def _validate_selector_lineage(
    lineage: Mapping[str, Any], protocol: PrequentialProtocol, code_version: str
) -> None:
    expected = {
        "protocol_fingerprint": protocol.fingerprint,
        "valid_start": protocol.initial_start,
        "valid_end": protocol.fold_2_end,
        "build_scope": "final_full_valid",
        "code_version": code_version,
    }
    for key, value in expected.items():
        if lineage.get(key) != value:
            raise ValueError(f"selector lineage is incompatible: {key}")
    if not lineage.get("source_data_sha256"):
        raise ValueError("selector lineage requires source_data_sha256")


def _load_and_validate_freeze(path: Path) -> dict[str, Any]:
    payload = _read_json(path, "frozen production manifest")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported frozen production manifest schema")
    core = {key: value for key, value in payload.items() if key != "freeze_id"}
    if payload.get("freeze_id") != _freeze_identity(core):
        raise ValueError("frozen production manifest identity mismatch")
    if payload.get("status") != "frozen_before_test":
        raise ValueError("production assets were not frozen before test")
    if payload.get("test_range") != [str(TEST_START.date()), str(TEST_END.date())]:
        raise ValueError("formal test range differs from the preregistered range")
    _verify_frozen_assets(payload)
    return payload


def _validated_test_frame(frame: pd.DataFrame) -> pd.DataFrame:
    column = None
    for candidate in ("trading_day", "TradingDay"):
        if candidate in frame:
            column = candidate
            break
    if column is None:
        raise ValueError("formal test input requires official TradingDay")
    result = frame.copy(deep=True)
    days = pd.to_datetime(result[column]).dt.normalize()
    if result.empty or days.min() != TEST_START or days.max() != TEST_END:
        raise ValueError("formal test input must cover 2025-06-30 through 2026-02-05")
    if bool(((days < TEST_START) | (days > TEST_END)).any()):
        raise ValueError("formal test input contains rows outside the frozen range")
    result[column] = days
    return result


def _validate_stack_result(
    result: StackTestResult,
    expected_stack: str,
    input_identity: str,
    freeze: Mapping[str, Any],
) -> None:
    if not isinstance(result, StackTestResult):
        raise TypeError(f"{expected_stack} evaluator returned the wrong result contract")
    if result.stack_type != expected_stack:
        raise ValueError(f"{expected_stack} evaluator returned the wrong stack identity")
    if result.test_input_identity != input_identity:
        raise ValueError(f"{expected_stack} evaluator used a different test input")
    if result.cost_model_version != freeze["cost_model"]["cost_model_version"]:
        raise ValueError(f"{expected_stack} evaluator used an incompatible cost model")
    if result.audit_schema_version != AUDIT_SCHEMA_VERSION:
        raise ValueError(f"{expected_stack} evaluator used an incompatible audit schema")
    if set(result.metrics) != _REPORT_SECTIONS:
        raise ValueError(f"{expected_stack} comparison metrics are incomplete")
    _validate_audit_records(result, expected_stack, freeze)


def _validate_audit_records(
    result: StackTestResult, stack: str, freeze: Mapping[str, Any]
) -> None:
    prediction_ids: set[str] = set()
    decision_ids: set[str] = set()
    for record in result.prediction_records:
        missing = set(PREDICTION_RECORD_COLUMNS) - set(record)
        if missing:
            raise ValueError(f"{stack} prediction audit is missing fields: {sorted(missing)}")
        if record["gating_stack"] != stack:
            raise ValueError(f"{stack} prediction contains another gate-stack identity")
        if stack == "legacy" and any(
            record[field] is not None for field in _UTILITY_ONLY_FIELDS
        ):
            raise ValueError("legacy must keep utility-only audit fields null")
        if stack == "tri_module" and record["utility_artifact_id"] not in {
            None,
            freeze["utility_runtime_artifact_id"],
        }:
            raise ValueError("tri-module prediction references another utility artifact")
        prediction_ids.add(str(record["prediction_id"]))
        decision_ids.add(str(record["decision_id"]))

    snapshot_ids: set[str] = set()
    for record in result.account_records:
        missing = set(ACCOUNT_RECORD_COLUMNS) - set(record)
        if missing:
            raise ValueError(f"{stack} account audit is missing fields: {sorted(missing)}")
        snapshot_ids.add(str(record["snapshot_id"]))
    for record in result.trade_records:
        missing = set(TRADE_RECORD_COLUMNS) - set(record)
        if missing:
            raise ValueError(f"{stack} trade audit is missing fields: {sorted(missing)}")
        searchable = "|".join(
            str(record.get(key, "")).lower()
            for key in ("event_type", "execution_status", "decision_reason")
        )
        if "shadow" in searchable or "counterfactual" in searchable:
            raise ValueError("shadow/counterfactual fills cannot enter the real ledger")
        prediction_id = str(record.get("prediction_id") or "")
        if prediction_id and prediction_id not in prediction_ids:
            raise ValueError(f"{stack} trade references an unknown prediction")
        decision_id = str(record.get("decision_id") or "")
        if decision_id and decision_ids and decision_id not in decision_ids:
            raise ValueError(f"{stack} trade references an unknown decision")
        for key in ("pre_snapshot_id", "post_snapshot_id"):
            snapshot_id = str(record.get(key) or "")
            if snapshot_id and snapshot_id not in snapshot_ids:
                raise ValueError(f"{stack} trade references an unknown account snapshot")
        if (
            str(record.get("decision_reason", "")).upper() == "UTILITY_REJECTION"
            and "exchange"
            in str(record.get("execution_status", "")).lower()
        ):
            raise ValueError("utility rejection cannot be classified as exchange rejection")


def _verify_frozen_assets(freeze: Mapping[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, metadata in sorted(freeze.get("assets", {}).items()):
        path = Path(metadata["path"])
        if not path.is_file():
            raise ValueError(f"frozen asset is missing: {name}")
        observed[name] = _file_hash(path)
        if observed[name] != metadata["sha256"]:
            raise ValueError(f"frozen asset hash mismatch: {name}")
    if not observed:
        raise ValueError("frozen production manifest contains no assets")
    return observed


def _asset_entry(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"asset does not exist: {resolved}")
    return {"path": str(resolved), "sha256": _file_hash(resolved)}


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {name}: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _freeze_identity(payload: Mapping[str, Any]) -> str:
    """Stable content identity independent of absolute artifact locations."""

    normalized = json.loads(json.dumps(payload))

    def remove_locations(value: Any) -> None:
        if isinstance(value, dict):
            value.pop("path", None)
            value.pop("candidate_episode_source", None)
            for nested in value.values():
                remove_locations(nested)
        elif isinstance(value, list):
            for nested in value:
                remove_locations(nested)

    remove_locations(normalized)
    return _canonical_hash(normalized)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
