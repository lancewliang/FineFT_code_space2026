"""Versioned preparation of expert-routing calibration and frozen replay.

The routing manifest declares the Optuna search space. Runtime arguments discover
the existing train/valid/test inputs and model artifacts; their identities are
computed internally for cache and frozen-replay consistency.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from argparse import ArgumentParser, Namespace
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypeVar, overload

import pandas as pd

from RL.DiHFT.high_level.gating.expert_hysteresis import ExpertHysteresisGating

SCHEMA_VERSION = 2
EXPERIMENT_MANIFEST_SCHEMA_VERSION = 3
STRATEGY_VERSION = "expert_hysteresis.v1"
RAW_SCORING_PROTOCOL_VERSION = "raw_vae_quantiles.v2"
RAW_SCORING_SEED = 12345
RAW_SCORING_BATCH_SIZE = 512
SCORING_FIELDS = (
    "slope_window_length",
    "volatility_window_length",
    "slope_gamma",
    "volatility_gamma",
)
ROUTING_FIELDS = (
    "slope_rule_base_threshold",
    "volatility_rule_base_threshold",
    "hysteresis_exit_ratio",
    "expert_switch_lead",
    "expert_confirm_steps",
)
SEARCH_FIELDS = SCORING_FIELDS + ROUTING_FIELDS
POLICY_FIELDS = (
    "gating_strategy",
    "strategy_version",
    *ROUTING_FIELDS,
    "slope_hold_threshold",
    "volatility_hold_threshold",
)
SELECTION_FIELDS = (
    "study_name",
    "study_storage_identity",
    "trial_number",
    "trial_state",
    "objective_name",
    "objective_version",
    "objective_direction",
    "objective_value",
)
# Freeze execution settings too: a final replay must not silently change the account,
# action space, fees, risk priority, or the specialist/scoring architecture.
RUNTIME_FIELDS = (
    "dataset_name",
    "experiment_name",
    "base_path",
    "selection_manifest",
    "vae_path",
    "z_dim",
    "vae_hidden_dims",
    "loss_type",
    "hidden_nodes",
    "time_info_dim",
    "max_holding_number",
    "strategy_position_unit",
    "diagnostic_contract_unit",
    "position_choices",
    "leverage_choices",
    "initial_leverage",
    "initial_wallet_balance",
    "initial_margin",
    "initial_unrealized_pnL",
    "initial_position",
    "order_book_depth",
    "transcation_cost",
    "short_estimated_rate",
    "long_estimated_rate",
    "action_persistence",
    "stop_loss_return_threshold",
    "stop_loss_cooldown_steps",
    "circuit_breaker_consecutive_stops",
    "circuit_breaker_cooling_steps",
    "enable_non_main_contract_defense",
    "non_main_defense_mode",
    "enable_mdd_penalty",
    "target_mdd",
    "mdd_penalty_weight",
    "min_trades_guard",
    "early_stop",
    "candidate_slot_order",
    "calibration_warmup_steps",
)


_NamespaceT = TypeVar("_NamespaceT")


class FrozenRoutingArgumentParser(ArgumentParser):
    """Preserve explicit CLI destinations so frozen values cannot be overridden."""

    @overload
    def parse_known_args(
        self,
        args: Iterable[str] | None = None,
        namespace: None = None,
    ) -> tuple[Namespace, list[str]]: ...

    @overload
    def parse_known_args(
        self,
        args: Iterable[str] | None,
        namespace: _NamespaceT,
    ) -> tuple[_NamespaceT, list[str]]: ...

    @overload
    def parse_known_args(self, *, namespace: _NamespaceT) -> tuple[_NamespaceT, list[str]]: ...

    def parse_known_args(
        self,
        args: Iterable[str] | None = None,
        namespace: object | None = None,
    ) -> tuple[object, list[str]]:
        tokens = list(sys.argv[1:] if args is None else args)
        if namespace is None:
            parsed, unknown = super().parse_known_args(tokens)
        else:
            parsed, unknown = super().parse_known_args(tokens, namespace)
        destinations = {
            option: action.dest for action in self._actions for option in action.option_strings
        }
        provided = set()
        for token in tokens:
            if token.startswith("--"):
                option = token.split("=", 1)[0]
                if option in destinations:
                    provided.add(destinations[option])
                else:
                    provided.update(
                        destination
                        for name, destination in destinations.items()
                        if name.startswith(option)
                    )
        # argparse supports caller-owned namespace objects as well as Namespace.
        setattr(parsed, "_explicit_routing_options", provided)  # noqa: B010
        return parsed, unknown


def file_sha256(path: str | Path) -> str:
    """Hash an existing evidence file without loading model weights into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _raw_score_source_identity(
    *,
    artifacts: dict[str, Any],
    scoring_protocol: dict[str, Any],
) -> str:
    """Hash runtime inputs that can change raw VAE quantiles.

    The identity is computed from the files and runtime architecture; callers do
    not need to enumerate those inputs in the routing search-space manifest.
    """
    return _identity(
        {
            "strategy_version": STRATEGY_VERSION,
            "scoring_protocol": scoring_protocol,
            "artifacts": artifacts,
        }
    )


def _read(path: str | Path, *, schema_version: int = SCHEMA_VERSION) -> dict[str, Any]:
    # Any is confined to the external JSON boundary, whose fields are validated below.
    with Path(path).open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict) or value.get("schema_version") != schema_version:
        raise ValueError(f"document requires schema_version={schema_version}")
    return value


def _checked_file(descriptor: dict[str, Any], expected_path: Path | None = None) -> Path:
    if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}:
        raise ValueError("artifact evidence requires path and sha256")
    path = Path(descriptor["path"]).resolve()
    if expected_path is not None and path != expected_path.resolve():
        raise ValueError(f"artifact path does not match runtime input: {expected_path}")
    if not path.is_file() or file_sha256(path) != descriptor["sha256"]:
        raise ValueError(f"artifact hash mismatch or missing file: {path}")
    return path


def resolve_expert_parameters(args: Namespace, *, require_explicit: bool = True) -> Namespace:
    """Validate explicit two-axis configuration without legacy fallback parsing."""
    if args.gating_strategy != "expert_hysteresis":
        return args
    version = getattr(args, "strategy_version", None) or STRATEGY_VERSION
    if version != STRATEGY_VERSION:
        raise ValueError(f"unsupported expert strategy_version: {version}")
    for field in SCORING_FIELDS + (
        "slope_rule_base_threshold",
        "volatility_rule_base_threshold",
    ):
        if getattr(args, field, None) is None:
            if require_explicit:
                raise ValueError(f"expert_hysteresis requires explicit {field}")
            base_field = (
                "window_length"
                if field.endswith("window_length")
                else "gamma"
                if field.endswith("gamma")
                else "rule_base_threshold"
            )
            setattr(args, field, getattr(args, base_field))
    for field in SCORING_FIELDS[:2]:
        value = getattr(args, field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field} must be a positive integer")
    for field in SCORING_FIELDS[2:]:
        value = getattr(args, field)
        if not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{field} must be finite and in (0, 1]")
    gate = ExpertHysteresisGating(
        slope_threshold=args.slope_rule_base_threshold,
        volatility_threshold=args.volatility_rule_base_threshold,
        hysteresis_exit_ratio=args.hysteresis_exit_ratio,
        expert_switch_lead=args.expert_switch_lead,
        expert_confirm_steps=args.expert_confirm_steps,
    )
    args.strategy_version = version
    args.slope_hold_threshold = gate.slope_threshold * gate.hysteresis_exit_ratio
    args.volatility_hold_threshold = gate.volatility_threshold * gate.hysteresis_exit_ratio
    args.window_length = max(args.slope_window_length, args.volatility_window_length)
    args.gamma = args.slope_gamma
    args.rule_base_threshold = min(
        args.slope_rule_base_threshold,
        args.volatility_rule_base_threshold,
    )
    return args


def _validate_search_space(
    search_space: object,
    args: Namespace | None = None,
) -> dict[str, dict[str, Any]]:
    """Validate the declared nine-dimensional calibration space and trial values."""
    if not isinstance(search_space, dict) or set(search_space) != set(SEARCH_FIELDS):
        raise ValueError("experiment search_space must define all nine effective dimensions")
    validated: dict[str, dict[str, Any]] = {}
    integer_fields = {
        "slope_window_length",
        "volatility_window_length",
        "expert_confirm_steps",
    }
    for field in SEARCH_FIELDS:
        descriptor = search_space[field]
        if not isinstance(descriptor, dict):
            raise TypeError(f"search_space.{field} must be an object")
        expected_type = "int" if field in integer_fields else "float"
        if descriptor.get("type") != expected_type:
            raise ValueError(f"search_space.{field} requires type={expected_type}")
        if not {"type", "low", "high"}.issubset(descriptor):
            raise ValueError(f"search_space.{field} requires low and high")
        low, high = descriptor["low"], descriptor["high"]
        if isinstance(low, bool) or isinstance(high, bool):
            raise TypeError(f"search_space.{field} bounds must be numeric")
        if expected_type == "int":
            if not isinstance(low, int) or not isinstance(high, int):
                raise TypeError(f"search_space.{field} bounds must be integers")
        elif not all(isinstance(value, (int, float)) for value in (low, high)):
            raise TypeError(f"search_space.{field} bounds must be numeric")
        if not math.isfinite(float(low)) or not math.isfinite(float(high)) or low > high:
            raise ValueError(f"search_space.{field} bounds are invalid")
        if field in ("slope_window_length", "volatility_window_length") and low < 1:
            raise ValueError(f"search_space.{field} requires positive windows")
        if field == "expert_confirm_steps" and low < 1:
            raise ValueError("search_space.expert_confirm_steps requires positive integers")
        if field in (
            "slope_gamma",
            "volatility_gamma",
            "slope_rule_base_threshold",
            "volatility_rule_base_threshold",
        ) and not 0 < low <= high <= 1:
            raise ValueError(f"search_space.{field} bounds must lie in (0, 1]")
        if field == "hysteresis_exit_ratio" and not 0 < low <= high < 1:
            raise ValueError(
                "search_space.hysteresis_exit_ratio bounds must satisfy 0 < low <= high < 1"
            )
        if field == "expert_switch_lead" and low < 0:
            raise ValueError("search_space.expert_switch_lead must be non-negative")
        if args is not None:
            value = getattr(args, field)
            if value < low or value > high:
                raise ValueError(f"effective trial value lies outside search_space: {field}")
        validated[field] = dict(descriptor)
    return validated


def effective_trial_configuration(args: Namespace) -> dict[str, Any]:
    """Return the validated nine-dimensional configuration used by one Trial."""
    resolve_expert_parameters(args)
    return {
        **{field: getattr(args, field) for field in SEARCH_FIELDS},
        "slope_hold_threshold": args.slope_hold_threshold,
        "volatility_hold_threshold": args.volatility_hold_threshold,
    }


def validate_trial_configuration(args: Namespace) -> Namespace:
    """Validate a Trial against already prepared immutable experiment sources."""
    resolve_expert_parameters(args)
    search_space = getattr(args, "search_space", None)
    if search_space is None or not getattr(args, "frozen_experiment_identity", None):
        raise ValueError("Trial configuration requires prepared experiment sources")
    _validate_search_space(search_space, args)
    return args


def _contract_files(directory: Path) -> list[Path]:
    files = sorted(directory.glob("*.feather"))
    if not files:
        raise ValueError(f"experiment requires contract data: {directory}")
    return files


def _artifacts(args: Namespace) -> tuple[dict[str, Path], dict[str, list[Path]], list[str]]:
    dataset = Path(args.base_path) / args.dataset_name
    manifest_path = Path(args.selection_manifest)
    with manifest_path.open(encoding="utf-8") as source:
        selection = json.load(source)
    axes = selection["axes"]
    if not axes["slope"] or not axes["volatility"]:
        raise ValueError("empty Selection Manifest axes")
    paths = {
        "selection_manifest": manifest_path,
        "specialist_model": Path(selection["artifacts"]["model_assembly"]),
    }
    for field in (
        "rl_state_features",
        "vae_slope_state_features",
        "vae_volatility_state_features",
        "maintenance_margin_ratio_dict",
    ):
        paths[field] = dataset / f"{field}.npy"
    for axis in ("slope", "volatility"):
        for label in axes[axis]:
            root = Path(args.vae_path) / args.dataset_name / args.experiment_name / axis / label
            paths[f"{axis}.{label}.model"] = root / "model_latest.pth"
            paths[f"{axis}.{label}.reference"] = root / "id_logpx.npy"
    data = {
        "fit": _contract_files(dataset / "train"),
        "selection": _contract_files(Path(selection["valid_root"])),
        "calibration": _contract_files(dataset / "valid"),
        "final": _contract_files(dataset / "test"),
    }
    for role, files in data.items():
        for path in files:
            paths[f"data.{role}.{path.stem}"] = path
    columns = [f"{axis}.{label}" for axis in ("slope", "volatility") for label in axes[axis]]
    return paths, data, columns


def describe_experiment_inputs(args: Namespace) -> dict[str, Any]:
    """Inventory runtime inputs for internal identity and cache validation."""
    paths, _, columns = _artifacts(args)
    return {
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for name, path in paths.items()
        },
        "column_order": columns,
    }


def _runtime_scoring_protocol(args: Namespace, columns: list[str]) -> dict[str, Any]:
    """Return the fixed legacy scoring settings without requiring manifest input."""
    return {
        "protocol_version": RAW_SCORING_PROTOCOL_VERSION,
        "seed": RAW_SCORING_SEED,
        "batch_size": RAW_SCORING_BATCH_SIZE,
        "column_order": columns,
        "architecture": {
            field: getattr(args, field) for field in ("z_dim", "vae_hidden_dims", "loss_type")
        },
    }


def prepare_experiment(
    args: Namespace,
    phase: str = "calibration",
    *,
    validate_trial: bool = True,
) -> Namespace:
    """Validate routing search space and discover runtime inputs before rollout.

    The input manifest only declares the Optuna search space. Data files, model
    artifacts, and scoring settings are resolved from the normal runtime arguments.
    """
    if args.gating_strategy != "expert_hysteresis":
        return args
    if phase not in ("calibration", "final"):
        raise ValueError("experiment phase must be calibration or final")
    if validate_trial:
        resolve_expert_parameters(args)
    elif getattr(args, "strategy_version", None) not in (None, STRATEGY_VERSION):
        raise ValueError(f"unsupported expert strategy_version: {args.strategy_version}")
    manifest_path = getattr(args, "experiment_manifest", None)
    if not manifest_path:
        raise ValueError("expert_hysteresis requires an experiment_manifest search-space file")
    for field in ("base_path", "selection_manifest", "vae_path"):
        setattr(args, field, str(Path(getattr(args, field)).resolve()))
    manifest = _read(manifest_path, schema_version=EXPERIMENT_MANIFEST_SCHEMA_VERSION)
    unsupported_fields = set(manifest) - {
        "schema_version",
        "experiment_id",
        "strategy_version",
        "search_space",
    }
    if unsupported_fields:
        raise ValueError(
            "routing search-space manifest has unsupported fields: "
            + ", ".join(sorted(unsupported_fields))
        )
    if manifest.get("strategy_version") != STRATEGY_VERSION:
        raise ValueError("routing search-space strategy_version is incompatible")
    search_space = _validate_search_space(
        manifest.get("search_space"),
        args if validate_trial else None,
    )
    inputs = describe_experiment_inputs(args)
    columns = inputs["column_order"]
    scoring = _runtime_scoring_protocol(args, columns)
    source_identity = _raw_score_source_identity(
        artifacts=inputs["artifacts"],
        scoring_protocol=scoring,
    )
    stage = "valid" if phase == "calibration" else "test"
    if args.eval_stage != stage:
        raise ValueError(f"{phase} must use eval_stage={stage}")
    args.experiment_manifest = str(Path(manifest_path).resolve())
    args.experiment_id = manifest.get("experiment_id") or (
        f"{args.experiment_name}-{source_identity[:12]}"
    )
    args.raw_score_source_identity = source_identity
    args.frozen_experiment_identity = _identity(
        {
            "experiment_id": args.experiment_id,
            "strategy_version": STRATEGY_VERSION,
            "search_space": search_space,
            "raw_score_source_identity": source_identity,
        }
    )
    args.raw_scoring_protocol = scoring
    # Keep this alias until cache generation is moved out of vae_routing_util.
    args.scoring_protocol = args.raw_scoring_protocol
    args.search_space = search_space
    args.calibration_warmup_steps = max(
        search_space["slope_window_length"]["high"],
        search_space["volatility_window_length"]["high"],
    )
    with Path(args.selection_manifest).open(encoding="utf-8") as source:
        selection = json.load(source)
    args.candidate_slot_order = [slot["slot_id"] for slot in selection["slots"]]
    args.experiment_phase = phase
    if phase == "final" and not getattr(args, "frozen_config", None):
        raise ValueError("final expert evaluation requires a frozen selected configuration")
    return args


def _selected_trial(args: Namespace) -> dict[str, Any]:
    values = {field: getattr(args, field, None) for field in SELECTION_FIELDS}
    for field in (
        "study_name",
        "study_storage_identity",
        "objective_name",
        "objective_version",
    ):
        if not isinstance(values[field], str) or not values[field]:
            raise ValueError(f"selected Trial requires nonempty {field}")
    if (
        isinstance(values["trial_number"], bool)
        or not isinstance(values["trial_number"], int)
        or values["trial_number"] < 0
    ):
        raise ValueError("selected Trial requires a non-negative trial_number")
    if values["trial_state"] != "COMPLETE":
        raise ValueError("only a COMPLETE Trial can be frozen")
    if values["objective_direction"] not in ("maximize", "minimize"):
        raise ValueError("selected Trial objective_direction must be maximize or minimize")
    if (
        isinstance(values["objective_value"], bool)
        or not isinstance(values["objective_value"], (int, float))
        or not math.isfinite(float(values["objective_value"]))
    ):
        raise ValueError("selected Trial requires a finite objective_value")
    return values


def export_frozen_experiment(args: Namespace, path: str | Path) -> str:
    """Freeze one complete selected Trial for replay by steps 8--10."""
    prepare_experiment(args, "calibration")
    configuration = {
        field: getattr(args, field) for field in POLICY_FIELDS + SCORING_FIELDS + RUNTIME_FIELDS
    }
    document = {
        "schema_version": SCHEMA_VERSION,
        "document_type": "expert_hysteresis_frozen_config",
        "experiment_id": args.experiment_id,
        "experiment_manifest": args.experiment_manifest,
        "experiment_identity": args.frozen_experiment_identity,
        "selected_on": "calibration",
        "selection": _selected_trial(args),
        "search_space": args.search_space,
        "configuration": configuration,
    }
    document["config_hash"] = _identity(document)
    destination = Path(path)
    if destination.exists() and _read(destination) != document:
        raise ValueError("refusing to replace an already frozen selected configuration")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    args.frozen_config = str(destination.resolve())
    args.frozen_config_hash = document["config_hash"]
    args.frozen_selection = dict(document["selection"])
    return str(destination)


def load_frozen_experiment(
    args: Namespace,
    path: str | Path,
    *,
    phase: str = "final",
) -> Namespace:
    """Restore a selected policy for its declared calibration or final replay phase."""
    if phase not in ("calibration", "final"):
        raise ValueError(f"unsupported frozen evaluation phase: {phase}")
    document = _read(path)
    identity = document.pop("config_hash", None)
    if (
        identity != _identity(document)
        or document.get("document_type") != "expert_hysteresis_frozen_config"
        or document.get("selected_on") != "calibration"
    ):
        raise ValueError("frozen config hash mismatch or non-calibration selection")
    selection = document.get("selection")
    if not isinstance(selection, dict) or set(selection) != set(SELECTION_FIELDS):
        raise ValueError("frozen selected Trial identity is incomplete")
    selected_args = Namespace(**selection)
    _selected_trial(selected_args)
    configuration = document["configuration"]
    if set(configuration) != set(POLICY_FIELDS + SCORING_FIELDS + RUNTIME_FIELDS):
        raise ValueError("frozen configuration is incomplete")
    for field in getattr(args, "_explicit_routing_options", ()):
        if field in configuration and getattr(args, field) != configuration[field]:
            raise ValueError(f"cannot override frozen configuration: {field}")
    if configuration["gating_strategy"] != "expert_hysteresis":
        raise ValueError("frozen configuration does not select expert_hysteresis")
    if "experiment_manifest" in getattr(args, "_explicit_routing_options", ()) and (
        Path(args.experiment_manifest).resolve() != Path(document["experiment_manifest"]).resolve()
    ):
        raise ValueError("cannot override frozen experiment_manifest")
    for field, value in configuration.items():
        setattr(args, field, value)
    args.experiment_manifest = document["experiment_manifest"]
    args.frozen_config = str(Path(path).resolve())
    args.frozen_config_hash = identity
    # The frozen hash is the final replay/audit identity.  It must not be
    # replaced by a partial projection of runtime flags or output purpose.
    args.frozen_configuration_identity = identity
    args.frozen_selection = dict(selection)
    prepare_experiment(args, phase)
    if document["experiment_identity"] != args.frozen_experiment_identity:
        raise ValueError("frozen experiment source identity changed")
    if document.get("search_space") != args.search_space:
        raise ValueError("frozen search_space changed")
    return args


def _cache_document(args: Namespace, directory: str | Path) -> dict[str, Any]:
    import numpy as np

    root = Path(directory)
    files = _contract_files(Path(args.base_path) / args.dataset_name / args.eval_stage)
    columns = args.raw_scoring_protocol["column_order"]
    expected_names = {f"{path.stem}.npy" for path in files}
    if {path.name for path in root.glob("*.npy")} != expected_names:
        raise ValueError("quantile cache contract set differs from frozen dataset")
    descriptors = {}
    for data_path in files:
        path = root / f"{data_path.stem}.npy"
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        row_count = len(pd.read_feather(data_path, columns=["timestamp"]))
        if values.shape != (row_count, len(columns)) or values.dtype != np.float32:
            raise ValueError(f"quantile cache shape/dtype mismatch: {path}")
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"quantile cache scores must be finite and in [0, 1]: {path}")
        descriptors[path.name] = {"sha256": file_sha256(path), "shape": list(values.shape)}
    return {
        "schema_version": SCHEMA_VERSION,
        "document_type": "expert_hysteresis_raw_quantile_cache",
        "raw_score_source_identity": args.raw_score_source_identity,
        "raw_scoring_protocol": args.raw_scoring_protocol,
        "eval_stage": args.eval_stage,
        "contracts": descriptors,
    }


def write_quantile_cache_manifest(args: Namespace, directory: str | Path) -> str:
    """Bind freshly generated scores to frozen sources, never to route state."""
    document = _cache_document(args, directory)
    path = Path(directory) / "quantile_cache_manifest.json"
    if path.exists() and _read(path) != document:
        raise ValueError("refusing to relabel an incompatible quantile cache")
    path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return str(path)


def validate_quantile_cache(args: Namespace, directory: str | Path) -> str:
    """Check cache provenance, order, coverage, bounds, and content identity."""
    path = Path(directory) / "quantile_cache_manifest.json"
    if not path.is_file() or _read(path) != _cache_document(args, directory):
        raise ValueError("quantile cache provenance does not match frozen scoring sources")
    return str(directory)


def write_evaluation_receipt(args: Namespace, directory: str | Path) -> str:
    """Persist replay identities after checking that a selected policy did not change."""
    document = _read(args.frozen_config)
    identity = document.pop("config_hash", None)
    if identity != _identity(document):
        raise ValueError("frozen configuration changed during final evaluation")
    if any(getattr(args, field) != value for field, value in document["configuration"].items()):
        raise ValueError("effective configuration changed during final evaluation")
    phase = "calibration" if args.eval_stage == "valid" else "final"
    prepare_experiment(args, phase)
    if args.frozen_experiment_identity != document["experiment_identity"]:
        raise ValueError("experiment sources changed during final evaluation")
    cache = Path(args.precomputed_quantiles_dir) / "quantile_cache_manifest.json"
    validate_quantile_cache(args, cache.parent)
    target = Path(directory)
    protected = (
        target / "frozen_evaluation_receipt.json",
        target / "frozen_config.json",
        target / "experiment_manifest.json",
    )
    if any(path.exists() for path in protected):
        raise FileExistsError("refusing to overwrite frozen evaluation audit/config evidence")
    target.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "document_type": "expert_hysteresis_evaluation_receipt",
        "experiment_id": args.experiment_id,
        "experiment_identity": args.frozen_experiment_identity,
        "config_hash": identity,
        "selection": document["selection"],
        "eval_stage": args.eval_stage,
        "evaluation_purpose": getattr(args, "evaluation_purpose", "final"),
        "frozen_config": {"path": args.frozen_config, "sha256": file_sha256(args.frozen_config)},
        "score_cache_manifest": {"path": str(cache.resolve()), "sha256": file_sha256(cache)},
    }
    path = target / "frozen_evaluation_receipt.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    (target / "frozen_config.json").write_bytes(Path(args.frozen_config).read_bytes())
    (target / "experiment_manifest.json").write_bytes(Path(args.experiment_manifest).read_bytes())
    return str(path)
