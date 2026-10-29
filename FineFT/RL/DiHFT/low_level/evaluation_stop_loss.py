"""Stop-loss state for standalone low-level evaluation."""

from __future__ import annotations

import argparse
import math
from typing import Protocol

DEFAULT_THRESHOLD = 0.03


class Holding(Protocol):
    position: float
    unrealized_pnl: float
    current_markprice: float


def validate_threshold(value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("stop_loss_return_threshold must be finite and non-negative")
    return result


def add_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--stop_loss_return_threshold",
        type=validate_threshold,
        default=DEFAULT_THRESHOLD,
        help="止损比例，0 关闭；默认 0.03",
    )


class EvaluationStopLoss:
    """Latch a stop until the environment reports that the exit actually filled."""

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = validate_threshold(threshold)
        self.pending = False

    def before_action(self, env: Holding) -> bool:
        if self.threshold and env.position != 0:
            position, mark, pnl = (
                float(env.position),
                float(env.current_markprice),
                float(env.unrealized_pnl),
            )
            if not all(math.isfinite(x) for x in (position, mark, pnl)) or mark <= 0:
                raise ValueError(
                    "stop-loss requires finite holding state and positive mark price"
                )
            if -pnl / (abs(position) * mark) >= self.threshold:
                self.pending = True
        return self.pending

    def after_execution(self, position: float) -> None:
        if self.pending and float(position) == 0:
            self.pending = False


def executed_position(env: Holding) -> float:
    # Terminal paths may leave env.position at its pre-execution value.
    execution = getattr(env, "last_wallet_change", None)
    return float(execution.position if execution is not None else env.position)
