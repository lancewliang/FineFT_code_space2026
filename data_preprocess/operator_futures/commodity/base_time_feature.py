from pathlib import Path
from datetime import datetime, date, timedelta
import math

import numpy as np
import polars as pl

from .config import TradingSession, get_commodity_config


BASE_TIME_FEATURE_COLUMNS: list[str] = [
    "trading_minute_progress",
    "morning_session",
    "afternoon_session",
    "night_session",
    "is_opening_30m",
    "is_closing_30m",
    "contract_month_sin",
    "contract_month_cos",
    "contract_life_remaining_ratio",
]


def _parse_date(val: str | date | datetime) -> date:
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, date):
        return val
    text = str(val).replace("-", "")
    return datetime.strptime(text, "%Y%m%d").date()


def _extract_contract_month(contract: str) -> int:
    digits = "".join(ch for ch in contract if ch.isdigit())
    if len(digits) < 2:
        raise ValueError(f"Cannot extract delivery month from contract code {contract!r}")
    month = int(digits[-2:])
    if not (1 <= month <= 12):
        raise ValueError(f"Invalid contract delivery month {month} in contract {contract!r}")
    return month


def _match_session(dt: datetime, sessions: tuple[TradingSession, ...]) -> tuple[TradingSession, datetime, datetime] | None:
    t = dt.time()
    d = dt.date()
    for s in sessions:
        if s.start <= s.end:
            if s.start <= t <= s.end:
                start_dt = datetime.combine(d, s.start)
                end_dt = datetime.combine(d, s.end)
                return s, start_dt, end_dt
        else:
            if t >= s.start:
                start_dt = datetime.combine(d, s.start)
                end_dt = datetime.combine(d + timedelta(days=1), s.end)
                return s, start_dt, end_dt
            elif t <= s.end:
                start_dt = datetime.combine(d - timedelta(days=1), s.start)
                end_dt = datetime.combine(d, s.end)
                return s, start_dt, end_dt
    return None


def generate_base_time_features(
    base_df: pl.DataFrame,
    symbol: str,
    contract: str,
    trading_day: str | date | datetime,
    last_trading_day: str | date | datetime,
    total_trading_day_count: int,
) -> pl.DataFrame:
    if "timestamp" not in base_df.columns:
        raise ValueError("base_df must contain 'timestamp' column")

    if total_trading_day_count <= 0:
        raise ValueError("total_trading_day_count must be positive")

    config = get_commodity_config(symbol)
    month = _extract_contract_month(contract)
    theta = 2.0 * math.pi * month / 12.0
    sin_val = math.sin(theta)
    cos_val = math.cos(theta)

    curr_date = _parse_date(trading_day)
    last_date = _parse_date(last_trading_day)

    if curr_date > last_date:
        remaining_days = 1
    else:
        remaining_days = int(np.busday_count(curr_date, last_date)) + 1
    remaining_ratio = max(remaining_days, 1) / float(total_trading_day_count)

    timestamps = base_df["timestamp"].to_list()
    progress_list = []
    morning_list = []
    afternoon_list = []
    night_list = []
    opening_30m_list = []
    closing_30m_list = []

    for ts in timestamps:
        if isinstance(ts, (int, float)):
            # convert ms / us / ns or unix sec if integer
            if ts > 1e16:
                dt = datetime.fromtimestamp(ts / 1e9)
            elif ts > 1e13:
                dt = datetime.fromtimestamp(ts / 1e6)
            elif ts > 1e10:
                dt = datetime.fromtimestamp(ts / 1e3)
            else:
                dt = datetime.fromtimestamp(ts)
        elif isinstance(ts, str):
            dt = datetime.fromisoformat(ts)
        else:
            dt = ts

        match = _match_session(dt, config.trading_sessions)
        if match is None:
            # Fallback for boundary timestamps: pick closest session
            best_s = None
            best_start_dt = None
            best_end_dt = None
            best_diff = float("inf")
            d = dt.date()
            for s in config.trading_sessions:
                s_dt = datetime.combine(d, s.start)
                e_dt = datetime.combine(d, s.end)
                diff = min(abs((dt - s_dt).total_seconds()), abs((dt - e_dt).total_seconds()))
                if diff < best_diff:
                    best_diff = diff
                    best_s, best_start_dt, best_end_dt = s, s_dt, e_dt
            s, start_dt, end_dt = best_s, best_start_dt, best_end_dt
        else:
            s, start_dt, end_dt = match

        duration_sec = (end_dt - start_dt).total_seconds()
        elapsed_sec = (dt - start_dt).total_seconds()

        if duration_sec <= 0:
            progress = 0.0
        else:
            progress = min(max(elapsed_sec / duration_sec, 0.0), 1.0)

        progress_list.append(float(progress))

        # Session one-hot
        if 6 <= s.start.hour < 12:
            morning_list.append(1.0)
            afternoon_list.append(0.0)
            night_list.append(0.0)
        elif 12 <= s.start.hour < 18:
            morning_list.append(0.0)
            afternoon_list.append(1.0)
            night_list.append(0.0)
        else:
            morning_list.append(0.0)
            afternoon_list.append(0.0)
            night_list.append(1.0)

        # 30m flags
        opening_30m_list.append(1.0 if elapsed_sec <= 1800 else 0.0)
        closing_30m_list.append(1.0 if (duration_sec - elapsed_sec) <= 1800 else 0.0)

    n = len(timestamps)
    result = pl.DataFrame({
        "timestamp": base_df["timestamp"],
        "trading_minute_progress": progress_list,
        "morning_session": morning_list,
        "afternoon_session": afternoon_list,
        "night_session": night_list,
        "is_opening_30m": opening_30m_list,
        "is_closing_30m": closing_30m_list,
        "contract_month_sin": [float(sin_val)] * n,
        "contract_month_cos": [float(cos_val)] * n,
        "contract_life_remaining_ratio": [float(remaining_ratio)] * n,
    })
    return result


def generate_and_write_base_time_feature(
    base_feature_path: pl.DataFrame | Path,
    output_root: Path,
    symbol: str,
    contract: str,
    target_freq: str,
    date: str,
    last_trading_day: str,
    total_trading_day_count: int,
) -> Path:
    if isinstance(base_feature_path, (str, Path)):
        path = Path(base_feature_path)
        if not path.exists():
            raise FileNotFoundError(f"BASE_FEATURE path does not exist: {path}")
        base_df = pl.read_ipc(path)
    else:
        base_df = base_feature_path

    res = generate_base_time_features(
        base_df=base_df,
        symbol=symbol,
        contract=contract,
        trading_day=date,
        last_trading_day=last_trading_day,
        total_trading_day_count=total_trading_day_count,
    )
    out_dir = output_root / "BASE_TIME_FEATURE" / symbol / contract / target_freq
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date}.feather"
    res.write_ipc(out_path)
    return out_path
