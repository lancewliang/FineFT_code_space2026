"""Replay saved actions without model inference and reconcile every equity step.

Run from the repository root after ``conda activate finetf``. This writes evidence
only; it never overwrites the saved strategy histories or runs a new test trial.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "FineFT"))
from env.env_initiate.base_initiate import initiate_base_env


def replay_contract(contract: str, output_dir: Path) -> dict:
    market_path = REPO_ROOT / "dataset/10min/fu/valid" / f"{contract}.feather"
    history_dir = REPO_ROOT / "result/DiHFT/final_result/fu/10min_parallel/contracts" / contract
    market = pd.read_feather(market_path)
    actions = np.load(history_dir / "micro_action_history.npy")
    saved_equity = np.load(history_dir / "total_asset_history.npy")
    saved_rewards = np.load(history_dir / "reward_history.npy")
    if len(market) != len(actions) + 1 or len(saved_equity) != len(actions) + 1:
        raise ValueError(f"Unexpected history alignment for {contract}")
    if not np.allclose(np.diff(saved_equity), saved_rewards, rtol=0, atol=1e-8):
        raise ValueError(f"Saved reward/equity mismatch for {contract}")
    environment = initiate_base_env(
        market, feature_list=["mark_price"], max_holding_number=1,
        position_choices=3, leverage_choice=[5], long_estimated_rate=0,
        short_estimated_rate=0, commission_rate=0.0003,
        initial_state=(float(saved_equity[0]), 0, 0, 0, 5), order_book_depth=5,
    )
    environment.reset()
    records = []
    for index, action in enumerate(actions):
        old_position = float(environment.position)
        previous_equity = float(environment.wallet_balance + environment.unrealized_pnl)
        previous_day = environment.day
        reference = float(market.mark_price.iloc[previous_day])
        requested_position = {0: -1.0, 1: 0.0, 2: 1.0}[int(action)]
        _, _, done, _ = environment.step(int(action))
        executed_position = float(environment.position)
        next_reference = float(market.mark_price.iloc[environment.day])
        equity = float(environment.wallet_balance + environment.unrealized_pnl)
        net_change = equity - previous_equity
        gross_change = executed_position * (next_reference - reference)
        actual_fee = float(environment.commission_fee_step)
        residual = gross_change - net_change - actual_fee
        records.append({
            "step_index": index,
            "reference_timestamp": str(market.timestamp.iloc[previous_day]),
            "next_reference_timestamp": str(market.timestamp.iloc[environment.day]),
            "requested_action": int(action), "requested_position": requested_position,
            "previous_executed_position": old_position,
            "executed_position": executed_position,
            "requested_executed_mismatch": requested_position != executed_position,
            "executed_quantity_change": executed_position - old_position,
            "reference_price": reference, "next_reference_price": next_reference,
            "bid1_price": float(market.bid1_price.iloc[previous_day]),
            "ask1_price": float(market.ask1_price.iloc[previous_day]),
            "bid1_size": float(market.bid1_size.iloc[previous_day]),
            "ask1_size": float(market.ask1_size.iloc[previous_day]),
            "actual_commission_fee": actual_fee,
            "saved_net_equity_change": float(saved_rewards[index]),
            "replayed_net_equity_change": net_change,
            "actual_position_reference_gross_change": gross_change,
            "gross_minus_net_residual": gross_change - net_change,
            "gross_minus_net_minus_fee_residual": residual,
            "environment_reported_slippage": float(environment.slippage_step),
            "saved_equity_after_step": float(saved_equity[index + 1]),
            "replayed_equity_after_step": equity,
            "limit_down_single_sided_ratio": float(market.limit_down_single_sided_ratio.iloc[previous_day]),
            "limit_up_single_sided_ratio": float(market.limit_up_single_sided_ratio.iloc[previous_day]),
            "done": bool(done),
        })
    frame = pd.DataFrame(records)
    replayed_equity = np.asarray(environment.margine_balance_history)
    max_error = float(np.max(np.abs(saved_equity - replayed_equity)))
    if max_error > 1e-8:
        raise ValueError(f"Replay disagrees with saved equity for {contract}: {max_error}")
    evidence_rows = frame.loc[
        (frame.executed_quantity_change != 0)
        | frame.requested_executed_mismatch
        | (frame.gross_minus_net_minus_fee_residual.abs() > 1e-8)
    ]
    evidence_rows.to_csv(output_dir / f"{contract}_execution_steps.csv", index=False)
    target_positions = np.asarray([{0: -1, 1: 0, 2: 1}[int(action)] for action in actions])
    prices = market.mark_price.to_numpy()
    mismatches = frame.loc[frame.requested_executed_mismatch]
    negatives = frame.loc[frame.gross_minus_net_minus_fee_residual < -1e-8]
    return {
        "contract": contract, "steps": len(actions),
        "csv_evidence_steps": len(evidence_rows),
        "csv_scope": "All execution changes, requested/executed mismatches, and nonzero execution residuals; every equity step is replayed and reconciled before filtering.",
        "source_market": str(market_path.relative_to(REPO_ROOT)),
        "source_history": str(history_dir.relative_to(REPO_ROOT)),
        "saved_equity_matches_replay": True, "max_equity_absolute_error": max_error,
        "legacy_truncated_requested_position_gross_pnl": float(np.dot(target_positions[:-1], np.diff(prices[:len(actions)]))),
        "complete_requested_position_gross_pnl": float(np.dot(target_positions, np.diff(prices))),
        "complete_executed_position_gross_pnl": float(frame.actual_position_reference_gross_change.sum()),
        "saved_net_pnl": float(saved_rewards.sum()),
        "actual_execution_commission": float(frame.actual_commission_fee.sum()),
        "gross_minus_net_minus_fee_residual": float(frame.gross_minus_net_minus_fee_residual.sum()),
        "environment_reported_slippage": float(frame.environment_reported_slippage.sum()),
        "requested_executed_mismatches": mismatches.to_dict(orient="records"),
        "negative_execution_residual_steps": int(len(negatives)),
        "largest_negative_execution_residual_step": negatives.sort_values("gross_minus_net_minus_fee_residual").iloc[0].to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contracts", nargs="+", default=["fu2508", "fu2510"])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [replay_contract(contract, arguments.output_dir) for contract in arguments.contracts]
    result = {
        "purpose": "Audit saved validation execution; no model fitting or independent test execution.",
        "replay_parameters": {"position_choices": 3, "max_holding_number": 1, "leverage": 5, "commission_rate": 0.0003, "order_book_depth": 5, "long_estimated_rate": 0, "short_estimated_rate": 0},
        "limits": [
            "The audited existing feathers were generated under the original preprocessing convention (HEAD f5a0461 before this audit): mark_price with first() and orderbook with last(); raw within-bar timestamps are absent from these feathers. Changing the source now does not regenerate historical data.",
            "Residual reconciliation does not establish a live execution advantage or same-instant slippage.",
            "Existing open_long_position reports open_value minus initial margin as slippage, omitting leverage; environment_reported_slippage is recorded as evidence, not accepted as correct.",
        ],
        "contracts": summaries,
    }
    (arguments.output_dir / "execution_replay_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps([{k: row[k] for k in ["contract", "steps", "max_equity_absolute_error", "complete_executed_position_gross_pnl", "saved_net_pnl", "actual_execution_commission", "gross_minus_net_minus_fee_residual"]} for row in summaries], indent=2))


if __name__ == "__main__":
    main()
