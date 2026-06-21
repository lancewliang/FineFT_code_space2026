from dataclasses import dataclass
from datetime import time
from typing import Dict, Tuple


@dataclass(frozen=True)
class TradingSession:
    start: time
    end: time

    def __post_init__(self) -> None:
        if self.start >= self.end:
            raise ValueError("trading session start must be before end")


@dataclass(frozen=True)
class CommodityConfig:
    symbol: str
    display_name: str
    dataset_name: str
    orderbook_depth: int
    funding_enabled: bool
    buy_fee_rate: float
    sell_fee_rate: float
    main_contract_months: Tuple[int, ...]
    contract_unit: float
    use_contract_multiplier: bool
    trading_sessions: Tuple[TradingSession, ...]

    def __post_init__(self) -> None:
        if self.contract_unit <= 0:
            raise ValueError("contract_unit must be positive")
        if not self.trading_sessions:
            raise ValueError("trading_sessions must not be empty")


COMMODITY_CONFIGS: Dict[str, CommodityConfig] = {
    "fu": CommodityConfig(
        symbol="fu",
        display_name="燃料油",
        dataset_name="fu",
        orderbook_depth=5,
        funding_enabled=False,
        buy_fee_rate=0.0001,
        sell_fee_rate=0.0003,
        main_contract_months=tuple(range(1, 13)),
        contract_unit=10,
        use_contract_multiplier=False,
        trading_sessions=(
            TradingSession(time(9, 0), time(10, 15)),
            TradingSession(time(10, 30), time(11, 30)),
            TradingSession(time(13, 30), time(15, 0)),
            TradingSession(time(21, 0), time(23, 0)),
        ),
    )
}


def get_commodity_config(symbol: str) -> CommodityConfig:
    normalized = symbol.lower()
    if normalized not in COMMODITY_CONFIGS:
        supported = ", ".join(sorted(COMMODITY_CONFIGS))
        raise ValueError(
            f"Unsupported commodity symbol {symbol!r}; supported: {supported}"
        )
    return COMMODITY_CONFIGS[normalized]
