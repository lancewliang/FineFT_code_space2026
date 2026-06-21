import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import polars as pl

from .config import get_commodity_config


logger = logging.getLogger(__name__)


def normalize_timestamp(row) -> datetime:
    action_day = str(row["ActionDay"])
    update_time = str(row["UpdateTime"])
    return datetime.strptime(f"{action_day} {update_time}", "%Y%m%d %H:%M:%S.%f")


def with_normalized_timestamp(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        (
            pl.col("ActionDay").cast(pl.Utf8)
            + pl.lit(" ")
            + pl.col("UpdateTime").cast(pl.Utf8)
        )
        .str.strptime(
            pl.Datetime("us"),
            format="%Y%m%d %H:%M:%S%.f",
            strict=True,
        )
        .alias("timestamp")
    )


def calculate_contract_volume(df: pl.DataFrame) -> float:
    if "Volume" not in df.columns or df.height == 0:
        return 0.0

    volume = df.select(
        pl.col("Volume").cast(pl.Float64, strict=False).alias("Volume")
    )["Volume"].drop_nulls()
    if len(volume) == 0:
        return 0.0
    return float(volume.max() - volume.min())


def _parse_date(value: str):
    return datetime.strptime(value, "%Y-%m-%d").date()


def infer_years_for_date_range(start_date: str, end_date: str) -> List[str]:
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if end <= start:
        raise ValueError(
            f"end_date must be greater than start_date for left-open range: "
            f"{start_date} -> {end_date}"
        )

    last_included = end - timedelta(days=1)
    return [str(year) for year in range(start.year, last_included.year + 1)]


def iter_contract_files(
    raw_root: Path, commodity_name: str, year: str
) -> Iterable[Path]:
    year_dir = raw_root / commodity_name / year
    if not year_dir.exists():
        raise FileNotFoundError(
            f"Commodity raw year directory does not exist: {year_dir}"
        )
    files = set(year_dir.glob("*.csv"))
    files.update(year_dir.glob("*/*/*.csv"))
    return iter(sorted(files))


def _eligible_contracts(
    frames: Dict[str, pl.DataFrame], symbol: str
) -> Dict[str, pl.DataFrame]:
    config = get_commodity_config(symbol)
    eligible: Dict[str, pl.DataFrame] = {}
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


def _largest_volume_contract(frames: Dict[str, pl.DataFrame]) -> Optional[str]:
    volumes = {
        contract: calculate_contract_volume(frame) for contract, frame in frames.items()
    }
    positive = {contract: volume for contract, volume in volumes.items() if volume > 0}
    if not positive:
        return None
    return max(positive, key=positive.get)


def _log_selected_main_contract_file(
    trading_day: str,
    contract: str,
    reason: str,
    source_file: Path,
    previous_day_frames: Dict[str, pl.DataFrame],
    current_day_frames: Dict[str, pl.DataFrame],
) -> None:
    previous_volume = (
        calculate_contract_volume(previous_day_frames[contract])
        if contract in previous_day_frames
        else None
    )
    current_volume = (
        calculate_contract_volume(current_day_frames[contract])
        if contract in current_day_frames
        else None
    )
    logger.info(
        "Selected commodity main-contract file: trading_day=%s contract=%s "
        "reason=%s previous_day_volume=%s current_day_volume=%s "
        "file_name=%s source_file=%s",
        trading_day,
        contract,
        reason,
        previous_volume,
        current_volume,
        source_file.name,
        source_file,
    )


def select_main_contract_for_day(
    previous_day_frames: Dict[str, pl.DataFrame],
    current_day_frames: Dict[str, pl.DataFrame],
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
    selected_frames: Iterable[Tuple[str, str, pl.DataFrame, Path]]
) -> pl.DataFrame:
    output: List[pl.DataFrame] = []
    for trading_day, contract, frame, source_file in selected_frames:
        source_contract = (
            pl.col("InstrumentID").cast(pl.Utf8)
            if "InstrumentID" in frame.columns
            else pl.lit(contract)
        )
        copied = with_normalized_timestamp(frame).with_columns(
            pl.lit(contract).alias("main_contract"),
            source_contract.alias("source_contract"),
            pl.lit(str(source_file)).alias("source_file"),
            pl.lit(trading_day).alias("main_contract_trading_day"),
        )
        output.append(copied)

    if not output:
        raise ValueError("No selected main-contract frames to stitch")

    return pl.concat(output, how="vertical").sort("timestamp")


def load_contract_frames_by_trading_day(
    raw_root: Path, commodity_name: str, year: str
) -> Dict[str, Dict[str, Tuple[pl.DataFrame, Path]]]:
    days: Dict[str, Dict[str, Tuple[pl.DataFrame, Path]]] = {}
    file_paths = list(iter_contract_files(raw_root, commodity_name, year))
    logger.info(
        "Loading commodity raw files: commodity=%s year=%s files=%d",
        commodity_name,
        year,
        len(file_paths),
    )
    for file_path in file_paths:
        frame = pl.read_csv(file_path)
        if frame.height == 0:
            logger.debug(
                "Skipping empty commodity raw file: file_name=%s source_file=%s",
                file_path.name,
                file_path,
            )
            continue
        required = {"InstrumentID", "TradingDay", "ActionDay", "UpdateTime"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{file_path} missing required columns: {sorted(missing)}")

        trading_days = frame["TradingDay"].cast(pl.Utf8).unique().to_list()
        if len(trading_days) != 1:
            raise ValueError(
                f"{file_path} contains multiple TradingDay values: {trading_days}"
            )
        contract = str(frame.item(0, "InstrumentID"))
        trading_day = str(trading_days[0])
        days.setdefault(trading_day, {})[contract] = (frame, file_path)
        logger.debug(
            "Loaded commodity contract file: trading_day=%s contract=%s "
            "rows=%d file_name=%s source_file=%s",
            trading_day,
            contract,
            frame.height,
            file_path.name,
            file_path,
        )
    contract_count = sum(len(contracts) for contracts in days.values())
    logger.info(
        "Loaded commodity raw files: commodity=%s year=%s trading_days=%d contracts=%d",
        commodity_name,
        year,
        len(days),
        contract_count,
    )
    return days


def load_contract_frames_by_trading_day_for_years(
    raw_root: Path, commodity_name: str, years: Sequence[str]
) -> Dict[str, Dict[str, Tuple[pl.DataFrame, Path]]]:
    days: Dict[str, Dict[str, Tuple[pl.DataFrame, Path]]] = {}
    for year in years:
        year_days = load_contract_frames_by_trading_day(
            raw_root, commodity_name, str(year)
        )
        for trading_day, contracts in year_days.items():
            if trading_day in days:
                overlap = sorted(set(days[trading_day]).intersection(contracts))
                if overlap:
                    raise ValueError(
                        f"Duplicate contract data for TradingDay {trading_day}: "
                        f"{overlap}"
                    )
            days.setdefault(trading_day, {}).update(contracts)
    logger.info(
        "Loaded commodity raw date range candidates: commodity=%s years=%s trading_days=%d",
        commodity_name,
        ",".join(str(year) for year in years),
        len(days),
    )
    return days


def _trading_day_in_range(trading_day: str, start_date: str, end_date: str) -> bool:
    trading_ts = datetime.strptime(trading_day, "%Y%m%d").date()
    return _parse_date(start_date) <= trading_ts < _parse_date(end_date)


def build_main_contract_continuous_frame(
    raw_root: Path, commodity_name: str, year: str, symbol: str
) -> pl.DataFrame:
    logger.info(
        "Building commodity main-contract series: symbol=%s commodity=%s year=%s",
        symbol,
        commodity_name,
        year,
    )
    days = load_contract_frames_by_trading_day(raw_root, commodity_name, year)
    selected = []
    previous_frames: Dict[str, pl.DataFrame] = {}
    for trading_day in sorted(days):
        current_items = days[trading_day]
        current_frames = {
            contract: frame for contract, (frame, _) in current_items.items()
        }
        contract, reason = select_main_contract_for_day(
            previous_frames, current_frames, symbol
        )
        frame, source_file = current_items[contract]
        _log_selected_main_contract_file(
            trading_day,
            contract,
            reason,
            source_file,
            previous_frames,
            current_frames,
        )
        copied = frame.with_columns(
            pl.lit(reason).alias("main_contract_selection_reason")
        )
        selected.append((trading_day, contract, copied, source_file))
        previous_frames = current_frames
    stitched = stitch_main_contract_frames(selected)
    logger.info(
        "Built commodity main-contract series: symbol=%s selected_days=%d rows=%d",
        symbol,
        len(selected),
        stitched.height,
    )
    return stitched


def build_main_contract_continuous_frame_for_date_range(
    raw_root: Path,
    commodity_name: str,
    start_date: str,
    end_date: str,
    symbol: str,
) -> pl.DataFrame:
    years = infer_years_for_date_range(start_date, end_date)
    logger.info(
        "Building commodity main-contract series: symbol=%s commodity=%s start_date=%s end_date=%s years=%s",
        symbol,
        commodity_name,
        start_date,
        end_date,
        ",".join(years),
    )
    days = load_contract_frames_by_trading_day_for_years(
        raw_root, commodity_name, years
    )
    selected = []
    previous_frames: Dict[str, pl.DataFrame] = {}
    for trading_day in sorted(days):
        current_items = days[trading_day]
        current_frames = {
            contract: frame for contract, (frame, _) in current_items.items()
        }
        if not _trading_day_in_range(trading_day, start_date, end_date):
            previous_frames = current_frames
            continue

        contract, reason = select_main_contract_for_day(
            previous_frames, current_frames, symbol
        )
        frame, source_file = current_items[contract]
        _log_selected_main_contract_file(
            trading_day,
            contract,
            reason,
            source_file,
            previous_frames,
            current_frames,
        )
        copied = frame.with_columns(
            pl.lit(reason).alias("main_contract_selection_reason")
        )
        selected.append((trading_day, contract, copied, source_file))
        previous_frames = current_frames
    stitched = stitch_main_contract_frames(selected)
    logger.info(
        "Built commodity main-contract series: symbol=%s selected_days=%d rows=%d",
        symbol,
        len(selected),
        stitched.height,
    )
    return stitched
