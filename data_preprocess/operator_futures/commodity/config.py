from dataclasses import dataclass
from typing import Dict, Tuple


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

    def __post_init__(self) -> None:
        if self.contract_unit <= 0:
            raise ValueError("contract_unit must be positive")


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
