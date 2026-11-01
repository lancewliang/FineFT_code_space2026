"""Shared, lightweight contract for the low-level trading observation.

The environment, replay buffers and model loaders all depend on this module.
It intentionally contains no environment or torch imports so that malformed
observations fail at the boundary instead of being silently padded or
truncated by a consumer.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


TRADING_INFO_KEYS = (
    "position_exposure",
    "single_holding_return_rate",
    "single_holding_max_drawdown",
    "current_holding_duration_norm",
    "unrealized_loss_to_notional_ratio",
    "last_long_exit_record_valid",
    "steps_since_last_long_exit_norm",
    "price_rise_since_last_long_exit_tanh",
    "last_short_exit_record_valid",
    "steps_since_last_short_exit_norm",
    "price_drop_since_last_short_exit_tanh",
)
TRADING_INFO_DIM = len(TRADING_INFO_KEYS)
TRADING_INFO_BASELINE_DIM = 4
TRADING_INFO_SCHEMA = "trading_info.v2"
TRADING_INFO_SCHEMA_VERSION = TRADING_INFO_SCHEMA
EXIT_MEMORY_SCALE_WINDOW = 18
EXIT_MEMORY_SCALE_OBSERVATIONS = EXIT_MEMORY_SCALE_WINDOW + 1
EXIT_MEMORY_MAX_AGE_STEPS = 256
EXIT_MEMORY_TANH_DIVISOR = 5.0
EXIT_PRICE_SCALE_FLOOR_DEFAULT = 5.0
CAUSAL_PRICE_SCALE_SCHEMA = "causal_price_scale.v1"
CAUSAL_PRICE_SCALE_COLUMN = "causal_mark_price_mean_abs_change_18"
CAUSAL_PRICE_SCALE_VALID_COLUMN = "causal_mark_price_scale_valid"
CAUSAL_PRICE_SCALE_ROW_COLUMN = "source_row_index"


class TradingObservationError(ValueError):
    """Raised when a trading observation or its metadata is invalid."""


def validate_exit_price_scale_floor(value: Any) -> float:
    """Return a finite, positive experiment scale floor."""
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise TradingObservationError(
            f"exit_price_scale_floor must be a finite positive number, got {value!r}"
        ) from exc
    if not np.isfinite(normalized) or normalized <= 0:
        raise TradingObservationError(
            f"exit_price_scale_floor must be a finite positive number, got {value!r}"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class TradingObservationConfig:
    """Versioned values that give a trading observation its meaning."""

    schema: str = TRADING_INFO_SCHEMA
    keys: tuple[str, ...] = TRADING_INFO_KEYS
    exit_price_scale_floor: float = EXIT_PRICE_SCALE_FLOOR_DEFAULT
    scale_window: int = EXIT_MEMORY_SCALE_WINDOW
    max_exit_age_steps: int = EXIT_MEMORY_MAX_AGE_STEPS
    tanh_divisor: float = EXIT_MEMORY_TANH_DIVISOR
    holding_duration_norm_steps: float = 180.0
    exit_price_scale_floor_source: str = "experiment_override"
    exit_price_scale_floor_note: str = (
        "5 yuan/ton is an experiment scale floor, not the exchange tick size"
    )
    price_source: str = "mark_price"
    price_unit: str = "yuan_per_tonne"

    def __post_init__(self) -> None:
        validate_exit_price_scale_floor(self.exit_price_scale_floor)
        if self.schema != TRADING_INFO_SCHEMA or tuple(self.keys) != TRADING_INFO_KEYS:
            raise TradingObservationError("trading observation schema or field order is invalid")
        if self.scale_window != EXIT_MEMORY_SCALE_WINDOW:
            raise TradingObservationError("scale_window must be 18")
        if self.max_exit_age_steps != EXIT_MEMORY_MAX_AGE_STEPS:
            raise TradingObservationError("max_exit_age_steps must be 256")
        if self.tanh_divisor != EXIT_MEMORY_TANH_DIVISOR:
            raise TradingObservationError("tanh_divisor must be 5")
        if not np.isfinite(self.holding_duration_norm_steps) or self.holding_duration_norm_steps <= 0:
            raise TradingObservationError("holding_duration_norm_steps must be finite and positive")

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "keys": list(self.keys),
            "dimension": len(self.keys),
            "exit_price_scale_floor": float(self.exit_price_scale_floor),
            "exit_price_scale_floor_source": self.exit_price_scale_floor_source,
            "exit_price_scale_floor_note": self.exit_price_scale_floor_note,
            "scale_window": self.scale_window,
            "scale_observations": EXIT_MEMORY_SCALE_OBSERVATIONS,
            "max_exit_age_steps": self.max_exit_age_steps,
            "tanh_divisor": float(self.tanh_divisor),
            "holding_duration_norm_steps": float(self.holding_duration_norm_steps),
            "price_source": self.price_source,
            "price_unit": self.price_unit,
        }


@dataclass(frozen=True, slots=True)
class ExitMemoryRecord:
    """The latest evidenced complete exit for one direction in an episode."""

    valid: bool = False
    observation_step: int = 0
    execution_price: float = 0.0
    price_scale: float = 0.0

    @classmethod
    def empty(cls) -> "ExitMemoryRecord":
        return cls()

    @classmethod
    def create(cls, observation_step: int, execution_price: float, price_scale: float) -> "ExitMemoryRecord":
        if not isinstance(observation_step, (int, np.integer)) or observation_step < 0:
            raise TradingObservationError("exit observation_step must be a non-negative integer")
        if not np.isfinite(execution_price) or execution_price <= 0:
            raise TradingObservationError("exit execution_price must be finite and positive")
        if not np.isfinite(price_scale) or price_scale <= 0:
            raise TradingObservationError("exit price_scale must be finite and positive")
        return cls(True, int(observation_step), float(execution_price), float(price_scale))


def calculate_causal_price_scale(
    reference_prices: Any,
    decision_step: int,
    exit_price_scale_floor: Any,
    *,
    scale_window: int = EXIT_MEMORY_SCALE_WINDOW,
) -> float:
    """Calculate ``max(floor, mean(abs(diff(P[t-18:t+1]))))`` causally."""
    floor = validate_exit_price_scale_floor(exit_price_scale_floor)
    prices = np.asarray(reference_prices, dtype=np.float64).reshape(-1)
    if scale_window != EXIT_MEMORY_SCALE_WINDOW:
        raise TradingObservationError("scale_window must be 18")
    if decision_step < scale_window or decision_step >= prices.size:
        raise TradingObservationError(
            f"decision_step {decision_step} requires {EXIT_MEMORY_SCALE_OBSERVATIONS} same-contract prices"
        )
    window = prices[decision_step - scale_window : decision_step + 1]
    if not np.isfinite(window).all() or (window <= 0).any():
        raise TradingObservationError("causal price scale requires finite positive mark_price values")
    return max(floor, float(np.abs(np.diff(window)).mean()))


def encode_exit_memory(
    record: ExitMemoryRecord,
    *,
    current_step: int,
    current_price: Any,
    direction: str,
) -> tuple[float, float, float]:
    """Encode one directional record as validity, age and signed price movement."""
    if current_step < 0:
        raise TradingObservationError("current_step must not be negative")
    price = float(current_price)
    if not np.isfinite(price) or price <= 0:
        raise TradingObservationError("current reference price must be finite and positive")
    if direction not in {"long", "short"}:
        raise TradingObservationError(f"unknown exit direction {direction!r}")
    if not record.valid:
        return 0.0, 0.0, 0.0
    if record.observation_step > current_step:
        raise TradingObservationError("exit record cannot be from the future")
    age = min(current_step - record.observation_step, EXIT_MEMORY_MAX_AGE_STEPS)
    normalized_move = (price - record.execution_price) / record.price_scale
    signed_move = normalized_move if direction == "long" else -normalized_move
    return 1.0, age / EXIT_MEMORY_MAX_AGE_STEPS, float(
        np.tanh(signed_move / EXIT_MEMORY_TANH_DIVISOR)
    )


def validate_trading_info(value: Any, *, name: str = "trading_info") -> np.ndarray:
    """Validate one ordered, finite float32 observation."""
    array = np.asarray(value)
    if array.ndim != 1 or array.shape[0] != TRADING_INFO_DIM:
        raise TradingObservationError(
            f"{name} must have schema {TRADING_INFO_SCHEMA} and width {TRADING_INFO_DIM}; got shape {array.shape}"
        )
    try:
        result = np.asarray(array, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise TradingObservationError(f"{name} must contain numeric values") from exc
    if not np.isfinite(result).all():
        raise TradingObservationError(f"{name} contains non-finite values")
    return result


def validate_trading_info_batch(value: Any, *, name: str = "trading_info") -> np.ndarray:
    """Validate a ``(batch, 11)`` finite float32 observation matrix."""
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != TRADING_INFO_DIM:
        raise TradingObservationError(
            f"{name} must have schema {TRADING_INFO_SCHEMA} and shape (B, {TRADING_INFO_DIM}); got shape {array.shape}"
        )
    try:
        result = np.asarray(array, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise TradingObservationError(f"{name} must contain numeric values") from exc
    if not np.isfinite(result).all():
        raise TradingObservationError(f"{name} contains non-finite values")
    return result


def validate_trading_metadata(metadata: Mapping[str, Any]) -> TradingObservationConfig:
    """Validate persisted metadata before loading a model or replay artifact."""
    if not isinstance(metadata, Mapping):
        raise TradingObservationError("trading observation metadata must be a mapping")
    expected = TradingObservationConfig()
    if metadata.get("schema") != expected.schema or tuple(metadata.get("keys", ())) != expected.keys:
        raise TradingObservationError("trading observation schema or field order does not match")
    if int(metadata.get("dimension", -1)) != TRADING_INFO_DIM:
        raise TradingObservationError("trading observation metadata dimension does not match")
    if int(metadata.get("scale_window", -1)) != EXIT_MEMORY_SCALE_WINDOW:
        raise TradingObservationError("scale_window does not match the active schema")
    if int(metadata.get("scale_observations", -1)) != EXIT_MEMORY_SCALE_OBSERVATIONS:
        raise TradingObservationError("scale_observations does not match the active schema")
    if int(metadata.get("max_exit_age_steps", -1)) != EXIT_MEMORY_MAX_AGE_STEPS:
        raise TradingObservationError("max_exit_age_steps does not match the active schema")
    if float(metadata.get("tanh_divisor", np.nan)) != EXIT_MEMORY_TANH_DIVISOR:
        raise TradingObservationError("tanh_divisor does not match the active schema")
    return TradingObservationConfig(
        exit_price_scale_floor=validate_exit_price_scale_floor(
            metadata.get("exit_price_scale_floor")
        ),
        holding_duration_norm_steps=float(metadata.get("holding_duration_norm_steps")),
        exit_price_scale_floor_source=str(metadata.get("exit_price_scale_floor_source", "")),
        exit_price_scale_floor_note=str(metadata.get("exit_price_scale_floor_note", "")),
        price_source=str(metadata.get("price_source", "")),
        price_unit=str(metadata.get("price_unit", "")),
    )


def validate_trading_state_dict(state_dict: Mapping[str, Any]) -> int:
    """Reject a bare checkpoint whose trading encoder is not eleven-dimensional."""
    widths = []
    for key, value in state_dict.items():
        if str(key).endswith("fc_trading.weight"):
            shape = tuple(getattr(value, "shape", ()))
            if len(shape) != 2:
                raise TradingObservationError(f"{key} is not a matrix")
            widths.append((str(key), int(shape[1])))
    if not widths:
        raise TradingObservationError("checkpoint has no fc_trading.weight")
    dimensions = {width for _, width in widths}
    if dimensions != {TRADING_INFO_DIM}:
        details = ", ".join(f"{key}={width}" for key, width in widths)
        raise TradingObservationError(
            f"checkpoint trading encoder width must be {TRADING_INFO_DIM}; found {details}"
        )
    return TRADING_INFO_DIM


def save_trading_observation_metadata(path: str | Path, config: TradingObservationConfig) -> None:
    """Write a validated metadata sidecar without replacing incompatible data."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if load_trading_observation_metadata(destination).to_metadata() != config.to_metadata():
            raise TradingObservationError(f"refusing incompatible metadata overwrite at {destination}")
        return
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(config.to_metadata(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def load_trading_observation_metadata(path: str | Path) -> TradingObservationConfig:
    source = Path(path)
    try:
        metadata = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TradingObservationError(f"cannot read trading observation metadata from {source}") from exc
    return validate_trading_metadata(metadata)


def prepare_trading_observation_artifact(
    artifact_directory: str | Path,
    config: TradingObservationConfig,
) -> Path:
    """Create or validate the metadata sidecar for a model/replay directory."""
    directory = Path(artifact_directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "trading_observation.json"
    save_trading_observation_metadata(destination, config)
    return destination


def require_trading_info_in_info(info: Mapping[str, Any], *, name: str = "info['trading_info']") -> np.ndarray:
    """Require an explicitly supplied observation; never silently fill zeros."""
    if "trading_info" not in info:
        raise TradingObservationError(f"missing required {name}; zero-vector fallback is not allowed")
    return validate_trading_info(info["trading_info"], name=name)


def add_exit_price_scale_floor_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--exit_price_scale_floor",
        type=validate_exit_price_scale_floor,
        required=True,
        help="Experiment exit-price scale floor in yuan/tonne; fu validation uses 5.",
    )


def adjust_causal_warmup_start(
    requested_start: int,
    partition_start: int,
    partition_length: int,
    *,
    scale_window: int = EXIT_MEMORY_SCALE_WINDOW,
) -> tuple[int, int]:
    """Return the first partition-local row with a complete causal window."""
    if requested_start < partition_start or partition_length < 0:
        raise TradingObservationError("invalid partition coordinates")
    if scale_window != EXIT_MEMORY_SCALE_WINDOW:
        raise TradingObservationError("scale_window must be 18")
    effective_start = max(requested_start, partition_start + scale_window)
    partition_end = partition_start + partition_length
    if effective_start > partition_end:
        raise TradingObservationError("partition has no decision row with a complete causal price warm-up")
    return effective_start, effective_start - requested_start
