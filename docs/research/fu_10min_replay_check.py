"""Replay a captured portfolio path through the production metric function.

This checks realized performance, not causal effects of a policy modification.
Run in the finetf environment from the repository root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "FineFT"))
from analysis.calculate_metric.calculate_metric import calculate_metric


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-target", action="store_true")
    parser.add_argument("--check-audit", action="store_true")
    parser.add_argument("--result-dir", type=Path, default=ROOT / "result/DiHFT/final_result/fu/10min_parallel/main_sub")
    args = parser.parse_args()
    frame = pd.read_csv(args.result_dir / "portfolio_action_timeline.csv")
    saved = json.loads((args.result_dir / "summary_metrics.json").read_text())["portfolio"]
    capital = saved["initial_capital"]
    rewards = frame["step_net_pnl"].to_numpy()
    metrics = calculate_metric(capital, np.r_[0.0, rewards], freq=12)
    equity = capital + np.r_[0.0, rewards.cumsum()]
    np.testing.assert_allclose(equity[1:], frame["portfolio_equity"], atol=1e-7)
    np.testing.assert_allclose([metrics[0], metrics[2]], [saved["tr"], saved["mdd"]], atol=1e-10)
    print(f"REPLAY PASS: net={rewards.sum():.4f}, TR={metrics[0]:.8%}, MDD={metrics[2]:.8%}, steps={len(rewards)}")
    if args.check_audit:
        total, mismatches = 0, 0
        for file in sorted((args.result_dir / "contracts").glob("*/audit_trades.json")):
            for trade in json.loads(file.read_text()):
                if trade["executed_quantity"] <= 0:
                    continue
                total += 1
                expected = abs(trade["fill_price"] * trade["executed_quantity"]) * 0.0003
                mismatches += not np.isclose(expected, trade["commission"], atol=1e-7, rtol=1e-7)
        print(f"AUDIT {'FAIL' if mismatches else 'PASS'}: fee/fill mismatch {mismatches}/{total} unit-size fills")
        return int(mismatches > 0)
    if args.check_target:
        passed = metrics[0] >= 0.50 and metrics[2] <= 0.05
        print(f"TARGET {'PASS' if passed else 'FAIL'}: TR>=50%, MDD<=5%")
        return 0 if passed else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
