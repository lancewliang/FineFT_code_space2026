"""Check prefix invariance of the production delivery-defense code path.

Stops at environment construction, after production computes its terminal mask.
No simulated order is submitted and no result artifact is written.
"""
from __future__ import annotations

import logging
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "FineFT"))
from RL.DiHFT.high_level.vae_routing_util import vae_risk_aware_routing


class EndProbe(Exception):
    """Stop after the real terminal mask has been constructed."""


def production_mask(frame: pd.DataFrame) -> np.ndarray:
    router = object.__new__(vae_risk_aware_routing)
    settings = {
        "vae_slope_indicators": [], "vae_volatility_indicators": [],
        "precomputed_quantiles_dir": None, "enable_delivery_month_defense": True,
        "delivery_defense_days": 5, "tech_indicator_list": [], "max_holding_number": 1,
        "position_choices": 3, "leverage_choices": [5], "long_estimated_rate": 0,
        "short_estimated_rate": 0, "transcation_cost": .0003,
        "maintenance_margin_ratio_dict": {}, "early_stop": 0,
        "initial_state": (6000, 0, 0, 0, 5), "order_book_depth": 5,
    }
    for key, value in settings.items():
        setattr(router, key, value)
    with patch("RL.DiHFT.high_level.vae_routing_util.initiate_base_env", side_effect=EndProbe):
        try:
            router.run_single_valid_df(frame, "/unused", contract_name="fu2509")
        except EndProbe:
            pass
    mask = router.is_terminal_defense_step
    return np.zeros(len(frame), dtype=bool) if mask is None else mask.copy()


def main() -> int:
    logging.getLogger().setLevel(logging.ERROR)
    market = pd.read_feather(ROOT / "dataset/10min/fu/valid/fu2509.feather", columns=["timestamp"])
    dates = sorted(market.timestamp.dt.date.unique())
    prefix = market[market.timestamp.dt.date < dates[-2]].copy()
    full_mask = production_mask(market)[:len(prefix)]
    prefix_mask = production_mask(prefix)
    changed = np.flatnonzero(full_mask != prefix_mask)
    print(f"TERMINAL PREFIX {'FAIL' if len(changed) else 'PASS'}: {len(changed)} historical masks change when two future dates are appended")
    if len(changed):
        print(f"First changed decision: {prefix.timestamp.iloc[changed[0]]}; prefix={prefix_mask[changed[0]]}, extended={full_mask[changed[0]]}")
    return int(len(changed) > 0)


if __name__ == "__main__":
    raise SystemExit(main())
