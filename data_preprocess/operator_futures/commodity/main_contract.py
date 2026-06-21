from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .config import get_commodity_config


def normalize_timestamp(row: pd.Series) -> pd.Timestamp:
    action_day = str(row["ActionDay"])
    update_time = str(row["UpdateTime"])
    return pd.to_datetime(f"{action_day} {update_time}", format="%Y%m%d %H:%M:%S.%f")


def calculate_contract_volume(df: pd.DataFrame) -> float:
    if "Volume" not in df.columns or df.empty:
        return 0.0

    volume = pd.to_numeric(df["Volume"], errors="coerce")
    if volume.dropna().empty:
        return 0.0
    return float(volume.max() - volume.min())


def iter_contract_files(
    raw_root: Path, commodity_name: str, year: str
) -> Iterable[Path]:
    year_dir = raw_root / commodity_name / year
    if not year_dir.exists():
        raise FileNotFoundError(
            f"Commodity raw year directory does not exist: {year_dir}"
        )
    return iter(sorted(year_dir.glob("*/*/*.csv")))


def _eligible_contracts(
    frames: Dict[str, pd.DataFrame], symbol: str
) -> Dict[str, pd.DataFrame]:
    config = get_commodity_config(symbol)
    eligible: Dict[str, pd.DataFrame] = {}
    for contract, frame in frames.items():
        normalized = contract.lower()
        if not normalized.startswith(config.symbol):
            continue

        month_text = normalized[-2:]
        if not month_text.isdigit():
            continue

        if int(month_text) in config.main_contract_months:
            eligible[contract] = frame
    return eligible


def _largest_volume_contract(frames: Dict[str, pd.DataFrame]) -> Optional[str]:
    volumes = {
        contract: calculate_contract_volume(frame) for contract, frame in frames.items()
    }
    positive = {contract: volume for contract, volume in volumes.items() if volume > 0}
    if not positive:
        return None
    return max(positive, key=positive.get)


def select_main_contract_for_day(
    previous_day_frames: Dict[str, pd.DataFrame],
    current_day_frames: Dict[str, pd.DataFrame],
    symbol: str,
) -> Tuple[str, str]:
    previous_eligible = _eligible_contracts(previous_day_frames, symbol)
    current_eligible = _eligible_contracts(current_day_frames, symbol)

    previous_choice = _largest_volume_contract(previous_eligible)
    if (
        previous_choice in current_eligible
        and calculate_contract_volume(current_eligible[previous_choice]) > 0
    ):
        return previous_choice, "previous_trading_day_volume"

    fallback = _largest_volume_contract(current_eligible)
    if fallback is None:
        raise ValueError(f"No tradable eligible contract found for symbol {symbol!r}")
    return fallback, "current_trading_day_fallback"


def stitch_main_contract_frames(
    selected_frames: Iterable[Tuple[str, str, pd.DataFrame, Path]]
) -> pd.DataFrame:
    output: List[pd.DataFrame] = []
    for trading_day, contract, frame, source_file in selected_frames:
        copied = frame.copy()
        copied["timestamp"] = copied.apply(normalize_timestamp, axis=1)
        copied["main_contract"] = contract
        copied["source_contract"] = copied.get("InstrumentID", contract)
        copied["source_file"] = str(source_file)
        copied["main_contract_trading_day"] = trading_day
        output.append(copied)

    if not output:
        raise ValueError("No selected main-contract frames to stitch")

    stitched = pd.concat(output, ignore_index=True)
    return stitched.sort_values("timestamp").reset_index(drop=True)


def load_contract_frames_by_trading_day(
    raw_root: Path, commodity_name: str, year: str
) -> Dict[str, Dict[str, Tuple[pd.DataFrame, Path]]]:
    days: Dict[str, Dict[str, Tuple[pd.DataFrame, Path]]] = {}
    for file_path in iter_contract_files(raw_root, commodity_name, year):
        frame = pd.read_csv(file_path)
        if frame.empty:
            continue
        required = {"InstrumentID", "TradingDay", "ActionDay", "UpdateTime"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{file_path} missing required columns: {sorted(missing)}")

        trading_days = frame["TradingDay"].astype(str).unique()
        if len(trading_days) != 1:
            raise ValueError(
                f"{file_path} contains multiple TradingDay values: {trading_days}"
            )
        contract = str(frame["InstrumentID"].iloc[0])
        trading_day = str(trading_days[0])
        days.setdefault(trading_day, {})[contract] = (frame, file_path)
    return days


def build_main_contract_continuous_frame(
    raw_root: Path, commodity_name: str, year: str, symbol: str
) -> pd.DataFrame:
    days = load_contract_frames_by_trading_day(raw_root, commodity_name, year)
    selected = []
    previous_frames: Dict[str, pd.DataFrame] = {}
    for trading_day in sorted(days):
        current_items = days[trading_day]
        current_frames = {
            contract: frame for contract, (frame, _) in current_items.items()
        }
        contract, reason = select_main_contract_for_day(
            previous_frames, current_frames, symbol
        )
        frame, source_file = current_items[contract]
        copied = frame.copy()
        copied["main_contract_selection_reason"] = reason
        selected.append((trading_day, contract, copied, source_file))
        previous_frames = current_frames
    return stitch_main_contract_frames(selected)
