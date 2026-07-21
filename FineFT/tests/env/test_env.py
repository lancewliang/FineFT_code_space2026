import sys
from pathlib import Path

FINEFT_ROOT = Path(__file__).resolve().parents[2]
if str(FINEFT_ROOT) not in sys.path:
    sys.path.insert(0, str(FINEFT_ROOT))

import numpy as np
import pandas as pd

from env.env_initiate.base_initiate import initiate_base_env


def _sample_data(rows=50):
    timestamps = pd.date_range("2026-01-01", periods=rows, freq="min")
    data = {
        "timestamp": timestamps,
        "funding_timestamp": timestamps + pd.Timedelta(hours=8),
        "funding_rate": np.zeros(rows),
        "mark_price": np.linspace(100.0, 101.0, rows),
        "feature_a": np.linspace(0.0, 1.0, rows),
    }
    for level in range(1, 26):
        data[f"ask{level}_price"] = data["mark_price"] + level * 0.01
        data[f"ask{level}_size"] = np.full(rows, 10.0)
        data[f"bid{level}_price"] = data["mark_price"] - level * 0.01
        data[f"bid{level}_size"] = np.full(rows, 10.0)
    return pd.DataFrame(data), ["feature_a"]


def main():
    df, features = _sample_data()
    env = initiate_base_env(df, features)
    state, info = env.reset()
    for _ in range(30):
        action = info["avaiable_action_list"][0]
        print(action)

        state, reward, done, info = env.step(action)
        print(info)
    for _ in range(10):
        action = 4
        state, reward, done, info = env.step(action)
        print(info)
    print("final balance", env.wallet_balance + env.unrealized_pnl)


if __name__ == "__main__":
    main()
