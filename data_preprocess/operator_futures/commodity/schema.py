from typing import Dict, List


DERIVATIVE_REFERENCE_COLUMNS = [
    "symbol",
    "funding_timestamp",
    "funding_rate",
    "index_price",
    "mark_price",
]
PRICE_LIMIT_COLUMNS = [
    "LowerLimitPrice",
    "UpperLimitPrice",
]


def resample_kwargs() -> Dict[str, str]:
    return {"closed": "right", "label": "right"}


def build_orderbook_columns(depth: int) -> List[str]:
    if depth < 1:
        raise ValueError("orderbook depth must be positive")

    columns: List[str] = []
    for side in ("ask", "bid"):
        for level in range(1, depth + 1):
            columns.append(f"{side}{level}_price")
            columns.append(f"{side}{level}_size")
    return columns


def get_reward_execution_columns(depth: int) -> List[str]:
    return [
        "timestamp",
        "contract",
        *build_orderbook_columns(depth),
        *PRICE_LIMIT_COLUMNS,
        *DERIVATIVE_REFERENCE_COLUMNS,
    ]
