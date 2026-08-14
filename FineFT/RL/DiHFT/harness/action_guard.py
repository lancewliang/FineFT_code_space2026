from collections import deque
from dataclasses import dataclass

from env.env_class.futures_util import (
    map_action_to_position_leverage,
    map_position_leverage_to_action,
)


LABEL_ACTION_SEMANTIC_METADATA = {
    "limit_down": (-1, "limit"),
    "strong_down": (-1, "strong"),
    "weak_down": (-1, "weak"),
    "sideways": (0, "sideways"),
    "weak_up": (1, "weak"),
    "strong_up": (1, "strong"),
    "limit_up": (1, "limit"),
}
SUPPORTED_LABEL_ACTION_SEMANTICS = frozenset(LABEL_ACTION_SEMANTIC_METADATA)


def _label_direction(semantic: str) -> int:
    direction, _ = LABEL_ACTION_SEMANTIC_METADATA[semantic]
    return direction


def _is_opposed_position_change(
    current_position: float, target_position: float, semantic: str
) -> bool:
    label_direction = _label_direction(semantic)
    if label_direction == 0 or target_position == 0:
        return False
    is_opposed_target = target_position * label_direction < 0
    return is_opposed_target and target_position != current_position


@dataclass(frozen=True)
class GuardDecision:
    action: int
    reason: str
    opposed_action_count: int
    opposed_action_capacity: int


class RollingLabelActionGuard:
    def __init__(
        self,
        *,
        semantic: str,
        window_size: int,
        capacity: int,
        stop_loss_ratio: float,
        leverage_choices: list[int],
        position_list: list[float],
    ) -> None:
        self.semantic = semantic
        self.window_size = window_size
        self.capacity = capacity
        self.stop_loss_ratio = stop_loss_ratio
        self.leverage_choices = leverage_choices
        self.position_list = position_list
        self._final_action_history: deque[tuple[int, bool]] = deque(
            maxlen=window_size
        )

    def apply(
        self,
        proposed_action: int,
        *,
        current_action: int,
        current_position: float,
        current_mark_price: float,
        current_holding_opening_price: float,
    ) -> GuardDecision:
        proposed_position, _ = map_action_to_position_leverage(
            proposed_action, self.leverage_choices, self.position_list
        )
        proposed_is_opposed = _is_opposed_position_change(
            current_position, proposed_position, self.semantic
        )
        previous_step_count = self.window_size - 1
        recent_history = (
            list(self._final_action_history)[-previous_step_count:]
            if previous_step_count
            else []
        )
        exceeds_quota = (
            proposed_is_opposed
            and sum(is_opposed for _, is_opposed in recent_history) + 1
            > self.capacity
        )

        final_action = proposed_action
        reason = "allowed"
        if exceeds_quota:
            if self._should_stop_loss(
                current_position,
                current_mark_price,
                current_holding_opening_price,
            ):
                final_action = map_position_leverage_to_action(
                    0,
                    self.leverage_choices[0],
                    self.leverage_choices,
                    self.position_list,
                )
                reason = "stop_loss_close"
            else:
                final_action = current_action
                reason = "quota_hold"

        final_is_opposed = proposed_is_opposed if not exceeds_quota else False
        self._final_action_history.append((final_action, final_is_opposed))
        opposed_count = sum(
            is_opposed for _, is_opposed in recent_history
        ) + int(final_is_opposed)
        return GuardDecision(
            action=final_action,
            reason=reason,
            opposed_action_count=opposed_count,
            opposed_action_capacity=self.capacity,
        )

    def _should_stop_loss(
        self,
        current_position: float,
        current_mark_price: float,
        current_holding_opening_price: float,
    ) -> bool:
        if (
            current_position == 0
            or current_holding_opening_price <= 0
            or current_position * _label_direction(self.semantic) >= 0
        ):
            return False
        price_change = current_mark_price / current_holding_opening_price - 1
        if current_position > 0:
            return price_change <= -self.stop_loss_ratio
        return price_change >= self.stop_loss_ratio
