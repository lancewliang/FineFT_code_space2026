"""Calendar and trading session adapter for exact active trading time accumulation."""

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class SessionInterval:
    trading_day: str
    name: str
    start: datetime
    end: datetime

    @property
    def duration_minutes(self) -> float:
        return (self.end - self.start).total_seconds() / 60.0


class CalendarAdapter:
    """Manages exchange trading calendar and session timelines."""

    def __init__(self, intervals: List[SessionInterval], trading_days: List[str]):
        self.intervals = sorted(intervals, key=lambda x: x.start)
        self.trading_days = trading_days
        self._td_to_index = {td: idx for idx, td in enumerate(trading_days)}
        self._td_to_close: Dict[str, datetime] = {}
        for interval in self.intervals:
            self._td_to_close[interval.trading_day] = interval.end

        self._starts_np = np.array([np.datetime64(i.start, "us") for i in self.intervals])
        self._ends_np = np.array([np.datetime64(i.end, "us") for i in self.intervals])

    @classmethod
    def from_manifest(cls, manifest_path: str | Path) -> "CalendarAdapter":
        p = Path(manifest_path)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        intervals: List[SessionInterval] = []
        trading_days: List[str] = []

        for day_entry in data.get("trading_days", []):
            td = str(day_entry["trading_day"])
            trading_days.append(td)
            for sess in day_entry.get("sessions", []):
                s_dt = datetime.strptime(sess["start"], "%Y-%m-%d %H:%M:%S")
                e_dt = datetime.strptime(sess["end"], "%Y-%m-%d %H:%M:%S")
                intervals.append(SessionInterval(
                    trading_day=td,
                    name=sess.get("name", "session"),
                    start=s_dt,
                    end=e_dt
                ))

        return cls(intervals, sorted(list(set(trading_days))))

    def resolve_decision_time(self, dt: datetime) -> Tuple[datetime, Optional[SessionInterval]]:
        """If decision dt is inside a session, returns dt and session.
        If dt is outside/between sessions, advances conservatively to the next session start."""
        if not self.intervals:
            return dt, None
        t_dt = np.datetime64(dt, "us")
        idx = int(np.searchsorted(self._ends_np, t_dt, side="right"))
        if idx >= len(self.intervals):
            return dt, None
        interval = self.intervals[idx]
        if interval.start <= dt <= interval.end:
            return dt, interval
        elif dt < interval.start:
            return interval.start, interval
        return dt, None

    def get_trading_day(self, dt: datetime) -> Optional[str]:
        """Resolves the TradingDay for dt based on session intervals."""
        _, interval = self.resolve_decision_time(dt)
        if interval:
            return interval.trading_day
        return self.trading_days[-1] if self.trading_days else None

    def add_active_trading_minutes(self, start_dt: datetime, active_minutes: int) -> Tuple[Optional[datetime], Optional[str]]:
        """Accumulates exactly active_minutes within trading sessions."""
        if active_minutes <= 0:
            return start_dt, None

        current_dt, current_interval = self.resolve_decision_time(start_dt)
        if current_interval is None:
            return None, None

        remaining_minutes = float(active_minutes)
        idx = self.intervals.index(current_interval)

        while idx < len(self.intervals):
            interval = self.intervals[idx]
            eff_start = max(current_dt, interval.start)
            available_in_interval = (interval.end - eff_start).total_seconds() / 60.0

            if available_in_interval >= remaining_minutes:
                target_dt = eff_start + timedelta(minutes=remaining_minutes)
                return target_dt, interval.trading_day
            else:
                remaining_minutes -= available_in_interval
                idx += 1
                if idx < len(self.intervals):
                    current_dt = self.intervals[idx].start

        return None, None

    def get_trading_day_close(self, current_trading_day: str, offset: int = 0) -> Tuple[Optional[datetime], Optional[str]]:
        """Returns the day-session close datetime for current_trading_day + offset trading days."""
        if current_trading_day not in self._td_to_index:
            return None, None
        curr_idx = self._td_to_index[current_trading_day]
        target_idx = curr_idx + offset
        if target_idx < 0 or target_idx >= len(self.trading_days):
            return None, None
        target_td = self.trading_days[target_idx]
        return self._td_to_close.get(target_td), target_td
