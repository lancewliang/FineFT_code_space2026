import logging
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import polars as pl

from .config import get_commodity_config


logger = logging.getLogger(__name__)


MAIN_CONTRACT_SELECTION_RULE = (
    "monthly_top_2_or_10_days_above_configured_daily_volume_threshold"
)


@dataclass(frozen=True)
class MainContractSummaryTradingDay:
    trading_day: str
    date: str
    source_file: str
    daily_volume: float

    @classmethod
    def from_dict(cls, payload: dict) -> "MainContractSummaryTradingDay":
        if not isinstance(payload, dict):
            raise ValueError(f"summary trading day must be an object: {payload}")
        for field in ("trading_day", "date", "source_file", "daily_volume"):
            if field not in payload:
                raise ValueError(f"summary trading day missing {field}: {payload}")
        daily_volume = payload["daily_volume"]
        if not isinstance(daily_volume, (int, float)):
            raise ValueError("summary trading day daily_volume must be numeric")
        return cls(
            trading_day=str(payload["trading_day"]),
            date=str(payload["date"]),
            source_file=str(payload["source_file"]),
            daily_volume=float(daily_volume),
        )

    def to_dict(self) -> dict:
        return {
            "trading_day": self.trading_day,
            "date": self.date,
            "source_file": self.source_file,
            "daily_volume": self.daily_volume,
        }


@dataclass(frozen=True)
class MainContractSummaryContract:
    contract: str
    selected_months: List[str]
    trading_days: List[MainContractSummaryTradingDay]

    @classmethod
    def from_dict(cls, payload: dict) -> "MainContractSummaryContract":
        if not isinstance(payload, dict):
            raise ValueError(f"summary contract must be an object: {payload}")
        for field in ("contract", "trading_day_count", "trading_days"):
            if field not in payload:
                raise ValueError(f"summary contract missing {field}: {payload}")

        trading_days_payload = payload["trading_days"]
        if not isinstance(trading_days_payload, list):
            raise ValueError(f"summary contract missing trading_days: {payload}")
        if payload["trading_day_count"] != len(trading_days_payload):
            raise ValueError(
                f"trading_day_count mismatch for contract {payload['contract']}"
            )

        return cls(
            contract=str(payload["contract"]),
            selected_months=[str(month) for month in payload.get("selected_months", [])],
            trading_days=[
                MainContractSummaryTradingDay.from_dict(item)
                for item in trading_days_payload
            ],
        )

    def to_dict(self) -> dict:
        ordered_days = sorted(self.trading_days, key=lambda item: item.trading_day)
        return {
            "contract": self.contract,
            "start_trading_day": ordered_days[0].trading_day,
            "end_trading_day": ordered_days[-1].trading_day,
            "trading_day_count": len(ordered_days),
            "selected_months": sorted(self.selected_months),
            "trading_days": [item.to_dict() for item in ordered_days],
        }


@dataclass(frozen=True)
class MainContractSummary:
    symbol: str
    commodity_name: str
    start_date: str
    end_date: str
    contracts: List[MainContractSummaryContract]
    selection_rule: str = MAIN_CONTRACT_SELECTION_RULE

    @classmethod
    def from_dict(cls, payload: dict) -> "MainContractSummary":
        if not isinstance(payload, dict):
            raise ValueError("main contract summary must be a JSON object")
        contracts = payload.get("contracts")
        if not isinstance(contracts, list) or not contracts:
            raise ValueError("main contract summary must contain non-empty contracts")

        return cls(
            symbol=str(payload["symbol"]),
            commodity_name=str(payload["commodity_name"]),
            start_date=str(payload["start_date"]),
            end_date=str(payload["end_date"]),
            selection_rule=str(
                payload.get(
                    "selection_rule",
                    MAIN_CONTRACT_SELECTION_RULE,
                )
            ),
            contracts=[
                MainContractSummaryContract.from_dict(item) for item in contracts
            ],
        )

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "commodity_name": self.commodity_name,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "selection_rule": self.selection_rule,
            "contracts": [item.to_dict() for item in self.contracts],
        }


def load_main_contract_summary(path: Path) -> MainContractSummary:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"main contract summary does not exist: {path}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid main contract summary JSON: {path}") from exc
    return MainContractSummary.from_dict(payload)


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


def load_contract_files_by_trading_day_for_years(
    raw_root: Path, commodity_name: str, years: Sequence[str]
) -> Dict[str, Dict[str, Path]]:
    days: Dict[str, Dict[str, Path]] = {}
    for year in years:
        file_paths = list(iter_contract_files(raw_root, commodity_name, str(year)))
        logger.info(
            "Loading commodity raw file paths: commodity=%s year=%s files=%d",
            commodity_name,
            year,
            len(file_paths),
        )
        for file_path in file_paths:
            frame = pl.read_csv(file_path, n_rows=1)
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
                raise ValueError(
                    f"{file_path} missing required columns: {sorted(missing)}"
                )

            trading_days = frame["TradingDay"].cast(pl.Utf8).unique().to_list()
            if len(trading_days) != 1:
                raise ValueError(
                    f"{file_path} contains multiple TradingDay values: {trading_days}"
                )
            contract = str(frame.item(0, "InstrumentID"))
            trading_day = str(trading_days[0])
            existing = days.setdefault(trading_day, {}).get(contract)
            if existing is not None:
                raise ValueError(
                    f"Duplicate contract data for TradingDay {trading_day} "
                    f"contract {contract}: {existing} and {file_path}"
                )
            days.setdefault(trading_day, {})[contract] = file_path
            logger.debug(
                "Loaded commodity contract file path: trading_day=%s contract=%s "
                "file_name=%s source_file=%s",
                trading_day,
                contract,
                file_path.name,
                file_path,
            )
    contract_count = sum(len(contracts) for contracts in days.values())
    logger.info(
        "Loaded commodity raw file paths: commodity=%s years=%s trading_days=%d contracts=%d",
        commodity_name,
        ",".join(str(year) for year in years),
        len(days),
        contract_count,
    )
    return days


def _trading_day_in_range(trading_day: str, start_date: str, end_date: str) -> bool:
    trading_ts = datetime.strptime(trading_day, "%Y%m%d").date()
    return _parse_date(start_date) <= trading_ts < _parse_date(end_date)


def _format_trading_day_file_date(trading_day: str) -> str:
    return datetime.strptime(trading_day, "%Y%m%d").date().isoformat()


def _clip_contract_trading_days(
    contract: str,
    trading_days: List[MainContractSummaryTradingDay],
    selected_months: List[str],
) -> List[MainContractSummaryTradingDay]:
    ordered_days = sorted(trading_days, key=lambda item: item.trading_day)
    if not ordered_days:
        raise ValueError(f"No retained trading days for contract {contract}: empty")
    if not selected_months:
        raise ValueError(
            f"No retained trading days for contract {contract}: missing selected months"
        )
    if len(ordered_days) <= 10:
        raise ValueError(
            f"No retained trading days for contract {contract}: fewer than 11 raw trading days"
        )

    window_start = _parse_date(f"{min(selected_months)}-01")
    end_cutoff = ordered_days[-11].trading_day
    retained_days = [
        day
        for day in ordered_days
        if _parse_date(day.date) >= window_start and day.trading_day <= end_cutoff
    ]
    if not retained_days:
        raise ValueError(
            f"No retained trading days for contract {contract} after window clipping"
        )
    return retained_days


def build_main_contract_summary_model_for_date_range(
    raw_root: Path,
    commodity_name: str,
    start_date: str,
    end_date: str,
    symbol: str,
) -> MainContractSummary:
    years = infer_years_for_date_range(start_date, end_date)
    logger.info(
        "Building commodity main-contract summary: symbol=%s commodity=%s start_date=%s end_date=%s years=%s",
        symbol,
        commodity_name,
        start_date,
        end_date,
        ",".join(years),
    )
    days = load_contract_files_by_trading_day_for_years(
        raw_root, commodity_name, years
    )

    monthly_volumes: Dict[str, Dict[str, float]] = {}
    monthly_high_volume_days: Dict[str, Dict[str, int]] = {}
    contract_days: Dict[str, List[MainContractSummaryTradingDay]] = {}
    selected_months_by_contract: Dict[str, set] = {}
    high_volume_threshold = get_commodity_config(
        symbol
    ).main_contract_daily_volume_threshold

    for trading_day in sorted(days):
        if not _trading_day_in_range(trading_day, start_date, end_date):
            continue

        frames = {}
        source_files = days[trading_day]
        for contract, file_path in source_files.items():
            frame = pl.read_csv(file_path)
            missing = {"Volume"}.difference(frame.columns)
            if missing:
                raise ValueError(
                    f"{file_path} missing required columns: {sorted(missing)}"
                )
            frames[contract] = frame

        eligible = _eligible_contracts(frames, symbol)
        month = _format_trading_day_file_date(trading_day)[:7]
        for contract, frame in eligible.items():
            daily_volume = calculate_contract_volume(frame)
            monthly_volumes.setdefault(month, {}).setdefault(contract, 0.0)
            monthly_volumes[month][contract] += daily_volume
            if (
                high_volume_threshold is not None
                and daily_volume > high_volume_threshold
            ):
                monthly_high_volume_days.setdefault(month, {}).setdefault(contract, 0)
                monthly_high_volume_days[month][contract] += 1
            contract_days.setdefault(contract, []).append(
                MainContractSummaryTradingDay(
                    trading_day=trading_day,
                    date=_format_trading_day_file_date(trading_day),
                    source_file=str(source_files[contract]),
                    daily_volume=daily_volume,
                )
            )

    for month in sorted(set(monthly_volumes) | set(monthly_high_volume_days)):
        volumes = monthly_volumes.get(month, {})
        positive = {
            contract: volume for contract, volume in volumes.items() if volume > 0
        }
        top_contracts = sorted(
            positive,
            key=lambda contract: (-positive[contract], contract),
        )[:2]
        for contract in top_contracts:
            selected_months_by_contract.setdefault(contract, set()).add(month)
        for contract, count in monthly_high_volume_days.get(month, {}).items():
            if count >= 10:
                selected_months_by_contract.setdefault(contract, set()).add(month)

    if not selected_months_by_contract:
        raise ValueError(f"No monthly top-2 contracts found for symbol {symbol!r}")

    contracts = []
    for contract in sorted(selected_months_by_contract):
        selected_months = sorted(selected_months_by_contract[contract])
        actual_trading_days = _clip_contract_trading_days(
            contract,
            contract_days[contract],
            selected_months,
        )
        contracts.append(
            MainContractSummaryContract(
                contract=contract,
                selected_months=selected_months,
                trading_days=actual_trading_days,
            )
        )

    return MainContractSummary(
        symbol=symbol,
        commodity_name=commodity_name,
        start_date=start_date,
        end_date=end_date,
        contracts=contracts,
    )


def build_main_contract_summary_for_date_range(
    raw_root: Path,
    commodity_name: str,
    start_date: str,
    end_date: str,
    symbol: str,
) -> dict:
    return build_main_contract_summary_model_for_date_range(
        raw_root=raw_root,
        commodity_name=commodity_name,
        start_date=start_date,
        end_date=end_date,
        symbol=symbol,
    ).to_dict()


def write_main_contract_summary_for_date_range(
    raw_root: Path,
    commodity_name: str,
    output_dir: Path,
    start_date: str,
    end_date: str,
    symbol: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    for legacy_path in sorted(output_dir.glob("????-??-??.csv")):
        try:
            datetime.strptime(legacy_path.stem, "%Y-%m-%d")
        except ValueError:
            continue
        legacy_path.unlink()
        logger.info(
            "Removed legacy commodity main-contract daily file: output=%s",
            legacy_path,
        )
    summary = build_main_contract_summary_for_date_range(
        raw_root=raw_root,
        commodity_name=commodity_name,
        start_date=start_date,
        end_date=end_date,
        symbol=symbol,
    )
    path = output_dir / "main_contract_summary.json"
    if path.exists():
        logger.info("Overwriting commodity main-contract summary: output=%s", path)
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Wrote commodity main-contract summary: output=%s contracts=%d",
        path,
        len(summary["contracts"]),
    )
    return path
