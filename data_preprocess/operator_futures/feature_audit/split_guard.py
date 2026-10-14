"""Evaluation window schedule generation, boundary maturity purification, and Test sealing."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional


@dataclass(frozen=True)
class EvaluationWindow:
    window_id: str
    start_time: datetime
    end_time: datetime
    partition: str  # "train", "valid", "test"
    is_historical_reference: bool = False
    is_partial: bool = False


def _add_months(dt: datetime, months: int) -> datetime:
    """Approximate or exact month addition."""
    year = dt.year + (dt.month + months - 1) // 12
    month = (dt.month + months - 1) % 12 + 1
    # Handle day overflow
    day = min(dt.day, 28)
    return datetime(year, month, day, dt.hour, dt.minute, dt.second)


def generate_evaluation_windows(
    start_date: datetime,
    train_end_date: datetime,
    valid_end_date: datetime,
    warmup_months: int = 6,
    window_months: int = 3,
) -> List[EvaluationWindow]:
    """Generates standardized evaluation windows:
    - Warmup reference window (initial N months)
    - Chronological evaluation windows in Train
    - Chronological evaluation windows in Valid (split at boundary)
    """
    windows: List[EvaluationWindow] = []

    # 1. Warmup reference
    warmup_end = _add_months(start_date, warmup_months)
    if warmup_end > train_end_date:
        warmup_end = train_end_date

    windows.append(EvaluationWindow(
        window_id="warmup_ref",
        start_time=start_date,
        end_time=warmup_end,
        partition="train",
        is_historical_reference=True,
        is_partial=False,
    ))

    # 2. Train evaluation windows
    curr_start = warmup_end
    w_idx = 1
    while curr_start < train_end_date:
        next_end = _add_months(curr_start, window_months)
        is_partial = False
        if next_end >= train_end_date:
            next_end = train_end_date
            is_partial = True

        windows.append(EvaluationWindow(
            window_id=f"train_w{w_idx:02d}",
            start_time=curr_start,
            end_time=next_end,
            partition="train",
            is_historical_reference=False,
            is_partial=is_partial,
        ))
        curr_start = next_end
        w_idx += 1

    # 3. Valid evaluation windows
    curr_start = train_end_date
    w_idx = 1
    while curr_start < valid_end_date:
        next_end = _add_months(curr_start, window_months)
        is_partial = False
        if next_end >= valid_end_date:
            next_end = valid_end_date
            is_partial = True

        windows.append(EvaluationWindow(
            window_id=f"valid_w{w_idx:02d}",
            start_time=curr_start,
            end_time=next_end,
            partition="valid",
            is_historical_reference=False,
            is_partial=is_partial,
        ))
        curr_start = next_end
        w_idx += 1

    return windows
