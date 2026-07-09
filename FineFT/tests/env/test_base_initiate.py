import pandas as pd

from env.env_initiate.base_initiate import initiate_base_env


def _df(depth):
    rows = []
    for t, price in enumerate([100.0, 101.0]):
        timestamp = pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=t)
        row = {
            "timestamp": timestamp,
            "mark_price": price,
            "funding_rate": 0.0,
            "funding_timestamp": timestamp + pd.Timedelta(hours=8),
            "feature_a": float(t),
        }
        for level in range(1, depth + 1):
            row[f"ask{level}_price"] = price + level
            row[f"ask{level}_size"] = 10.0
            row[f"bid{level}_price"] = price - level
            row[f"bid{level}_size"] = 10.0
        rows.append(row)
    return pd.DataFrame(rows)


def test_base_env_uses_configured_order_book_depth():
    env = initiate_base_env(
        _df(depth=5),
        ["feature_a"],
        max_holding_number=1,
        position_choices=3,
        order_book_depth=5,
    )

    _, info = env.reset()

    assert len(info["ask_qyts"]) == 5
    assert len(info["bid_qyts"]) == 5
