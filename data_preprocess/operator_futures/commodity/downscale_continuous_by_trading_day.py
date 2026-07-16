import argparse
from datetime import datetime
from dataclasses import dataclass
import logging
from pathlib import Path
import time

import polars as pl

from operator_futures.data_quality import DataQualityValidator

from .downscale import (
    ASK_PRICE_COLUMNS,
    ASK_VOLUME_COLUMNS,
    BID_PRICE_COLUMNS,
    BID_VOLUME_COLUMNS,
    create_second_level_snapshots,
    downscale_base_features,
    downscale_derivative_reference,
    downscale_orderbook,
    downscale_quote_features,
)
from .main_contract import MainContractSummary, load_main_contract_summary


logger = logging.getLogger(__name__)


SECOND_LEVEL_DOWNSCALE_REQUIRED_COLUMNS = (
    "timestamp",
    "LastPrice",
    "LowPrice",
    "HighPrice",
    "LowerLimitPrice",
    "UpperLimitPrice",
    "Volume",
    "Turnover",
    *BID_PRICE_COLUMNS,
    *ASK_PRICE_COLUMNS,
    *BID_VOLUME_COLUMNS,
    *ASK_VOLUME_COLUMNS,
)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _trading_day_output_name(trading_day: str) -> str:
    trading_day = str(trading_day)
    if len(trading_day) == 8 and trading_day.isdigit():
        return datetime.strptime(trading_day, "%Y%m%d").date().isoformat()
    return trading_day


@dataclass(frozen=True)
class SummaryTradingDaySource:
    contract: str
    date: str
    source_file: Path


def iter_summary_trading_days(
    summary: MainContractSummary, contract_filter: str | None = None
):
    matched = False
    for contract in summary.contracts:
        contract_name = contract.contract
        if contract_filter is not None and contract_name != contract_filter:
            continue
        matched = True
        for day in contract.trading_days:
            source_file = Path(day.source_file)
            if not source_file.exists():
                raise FileNotFoundError(f"source_file does not exist: {source_file}")
            yield SummaryTradingDaySource(
                contract=contract_name,
                date=day.date,
                source_file=source_file,
            )
    if contract_filter is not None and not matched:
        raise ValueError(f"contract {contract_filter!r} not found in summary")


def _write_downscaled_day(
    day_frame: pl.DataFrame,
    output_root: Path,
    target_freq: str,
    symbol: str,
    contract: str,
    depth: int,
) -> str:
    trading_days = (
        day_frame.select(pl.col("TradingDay").cast(pl.Utf8).unique().sort())
        .to_series()
        .to_list()
    )
    if len(trading_days) != 1:
        raise ValueError(
            f"Daily continuous file must contain one TradingDay: {trading_days}"
        )
    trading_day = trading_days[0]
    second = create_second_level_snapshots(day_frame)
    DataQualityValidator.validate_no_illegal_values(
        second,
        stage="second_level_snapshots",
        contract=contract,
        trading_day=trading_day,
        columns=SECOND_LEVEL_DOWNSCALE_REQUIRED_COLUMNS,
    )
    outputs = {
        "DOWNSCALE_DERTIC": downscale_derivative_reference(
            second, target_freq, symbol
        ),
        "DOWNSCALE_ORDERBOOK_25": downscale_orderbook(
            second, target_freq, depth=depth
        ),
        "BASE_FEATURE": downscale_base_features(second, target_freq, symbol),
        "COMMODITY_QUOTE_FEATURE": downscale_quote_features(second, target_freq),
    }
    output_name = _trading_day_output_name(trading_day)
    for folder, frame in outputs.items():
        DataQualityValidator.validate_no_illegal_values(
            frame,
            stage="feature_output",
            feature_name=folder,
            contract=contract,
            trading_day=trading_day,
        )
    for folder, frame in outputs.items():
        path = output_root / folder / symbol / contract / target_freq
        path.mkdir(parents=True, exist_ok=True)
        frame.write_ipc(path / f"{output_name}.feather")
        frame.write_csv(path / f"{output_name}.csv")
    return trading_day


def downscale_continuous_by_trading_day(
    summary_path: Path,
    output_root: Path,
    target_freq: str,
    symbol: str,
    depth: int = 5,
    contract: str | None = None,
) -> None:
    started_at = time.monotonic()
    logger.info(
        "Starting commodity summary downscale: summary=%s output_root=%s target_freq=%s symbol=%s contract=%s depth=%d",
        summary_path,
        output_root,
        target_freq,
        symbol,
        contract,
        depth,
    )
    summary = load_main_contract_summary(summary_path)
    processed = []
    for item in iter_summary_trading_days(summary, contract):
        raw = pl.read_csv(item.source_file)
        logger.info(
            "Downscaling commodity contract source file: contract=%s date=%s input=%s rows=%d",
            item.contract,
            item.date,
            item.source_file,
            raw.height,
        )
        trading_day = _write_downscaled_day(
            raw,
            output_root,
            target_freq,
            symbol,
            item.contract,
            depth,
        )
        processed.append((item.contract, trading_day))

    logger.info(
        "Finished commodity summary downscale: contract_days=%d elapsed_seconds=%.2f",
        len(processed),
        time.monotonic() - started_at,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Downscale commodity main-contract source files from summary"
    )
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--target_freq", default="5min")
    parser.add_argument("--symbol", default="fu")
    parser.add_argument("--contract")
    parser.add_argument("--depth", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    downscale_continuous_by_trading_day(
        summary_path=Path(args.summary),
        output_root=Path(args.output_root),
        target_freq=args.target_freq,
        symbol=args.symbol,
        depth=args.depth,
        contract=args.contract,
    )


if __name__ == "__main__":
    main()
