"""Descriptive attribution of the frozen valid run; no hindsight trading claims."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "result/DiHFT/final_result/fu/10min_parallel/main_sub"
OUT = ROOT / "docs/research/fu_10min_followup_evidence"


def analyze_selection_and_features(step: pd.DataFrame, episode: pd.DataFrame) -> dict:
    """Summarize handoffs and re-score saved selection evidence without retraining."""
    handoffs = []
    for row in episode.itertuples():
        frame = step[(step.contract == row.contract) & step.step.between(row.entry_step, row.exit_step)]
        handoffs.append({"contract": row.contract, "entry_step": row.entry_step, "distinct_slots": frame.loc[frame.slot.lt(9), "slot"].nunique(), "slot_changes": int(frame.slot.diff().fillna(0).ne(0).sum()), "entry_slot": row.entry_slot, "pnl": row.pnl})
    pd.DataFrame(handoffs).to_csv(OUT / "episode_handoffs.csv", index=False)
    ranks = pd.read_csv(ROOT / "analysis_result/DiHFT/low_level/fu/10min_parallel/two_dimensional_selection/two_dimensional_candidate_rankings.csv")
    outputs = []
    for z in (0.2, 1.0, 1.28, 1.645):
        table = ranks.copy()
        table["robust_pair_score"] = np.minimum(table.mean_return_per_step_volatility_run-z*table.standard_error_volatility_run, table.mean_return_per_step_slope_run-z*table.standard_error_slope_run)
        table["robust_eligible"] = table.eligible & table.robust_pair_score.gt(0)
        for axis in ("slope", "volatility"):
            table["robust_eligible"] &= (table[f"mean_return_per_step_{axis}"]-z*table[f"standard_error_{axis}"]).gt(0)
        for slot, group in table.groupby("slot_id"):
            valid = group[group.robust_eligible].sort_values("robust_pair_score", ascending=False)
            outputs.append({"z": z, "slot_id": slot, "eligible": len(valid), "candidate": valid.iloc[0].candidate_id if len(valid) else None, "score": valid.iloc[0].robust_pair_score if len(valid) else None})
    pd.DataFrame(outputs).to_csv(OUT / "selector_lcb_sensitivity.csv", index=False)
    feature_root = ROOT / "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/10min/fu/train"
    manifest = json.loads((feature_root / "feature_selection_manifest.json").read_text())
    drift = pd.read_csv(feature_root / "distribution_audit_metrics.csv")
    evidence = []
    for stream in ("rl_stream", "vae_slope_stream", "vae_volatility_stream"):
        names = manifest[stream]["selected_features"]
        values = drift[drift.feature.isin(names)].forward_psi
        evidence.append({"stream": stream, "features": len(names), "features_with_psi": len(values), "forward_psi_p90": values.quantile(.9), "forward_psi_max": values.max()})
    pd.DataFrame(evidence).to_csv(OUT / "saved_feature_drift_evidence.csv", index=False)
    wide = step[step.role.ge(.5)].pivot_table(index="timestamp", columns="role", values="price_change", aggfunc="first")
    timeline = pd.read_csv(RESULT / "portfolio_action_timeline.csv")
    both = timeline[timeline.main_position.ne(0) & timeline.sub_position.ne(0)]
    return {"multiple_slot_episodes": sum(x["distinct_slots"] > 1 for x in handoffs), "mean_distinct_slots": float(np.mean([x["distinct_slots"] for x in handoffs])), "both_held_steps": len(both), "both_held_same_direction_ratio": float((both.main_position*both.sub_position > 0).mean()), "main_sub_price_change_corr": float(wide.corr().loc[.5, 1.0])}


def grouped(frame: pd.DataFrame, field: str) -> pd.DataFrame:
    return frame.groupby(field, dropna=False, observed=True).agg(
        episodes=("pnl", "size"), net_pnl=("pnl", "sum"),
        win_rate=("pnl", lambda s: (s > 0).mean()),
        mean_pnl=("pnl", "mean"), median_bars=("bars", "median"),
        hard_stops=("exit_reason", lambda s: (s == "HARD_STOP_LOSS").sum()),
        mfe_sum=("mfe_net_path", "sum"),
    )


def main() -> None:
    OUT.mkdir(exist_ok=True)
    steps, episodes, predictions, contract_info = [], [], [], []
    for directory in sorted((RESULT / "contracts").iterdir()):
        name = directory.name
        market = pd.read_feather(ROOT / "dataset/10min/fu/valid" / f"{name}.feather")
        n = len(market) - 1
        pos = np.load(directory / "executed_position_history.npy")
        reward = np.load(directory / "reward_history.npy")
        fee = np.load(directory / "commission_fee_history.npy")
        price = market.mark_price.to_numpy()
        trades = pd.DataFrame(json.loads((directory / "audit_trades.json").read_text()))
        pred = pd.DataFrame(json.loads((directory / "audit_predictions.json").read_text()))
        macro = np.load(directory / "macro_action_history.npy")
        macro = np.r_[np.full(n-len(macro), 9), macro]
        trade_by_step = trades.drop_duplicates("step_index").set_index("step_index")
        reasons = trade_by_step.decision_reason.reindex(range(n)).to_numpy()
        signed_dp = pos * np.diff(price)
        frame = pd.DataFrame({"contract": name, "step": np.arange(n), "timestamp": market.timestamp.iloc[:-1], "role": market.prev_day_contract_role_tier.iloc[:-1], "position": pos, "slot": macro, "reason": reasons, "net_pnl": reward, "gross_pnl": signed_dp, "fee": fee, "mark_price": price[:-1], "price_change": np.diff(price)})
        # Mirror current production terminal defense for attribution only.
        dates = sorted(market.timestamp.dt.date.unique())
        frame["terminal"] = market.timestamp.iloc[:-1].dt.date.isin(dates[-5:])
        frame["eligible"] = (frame.role >= 0.5) & ~frame.terminal & (frame.step >= 147)
        steps.append(frame)
        eq = np.r_[6000.0, 6000+reward.cumsum()]
        np.testing.assert_allclose(eq, np.load(directory / "total_asset_history.npy"), atol=1e-7)
        if len(pred):
            evidence = pred.routing_evidence.map(json.loads)
            for axis in ("slope", "volatility"):
                weights = np.asarray([x[axis] for x in evidence])
                ordered = np.sort(weights, axis=1)
                pred[f"{axis}_max"] = ordered[:, -1]
                pred[f"{axis}_margin"] = (ordered[:, -1]-ordered[:, -2])/weights.sum(axis=1)
            predictions.append(pred)
            pred_by_step = pred.set_index("step_index")
        else:
            pred_by_step = pd.DataFrame()
        active = None
        previous = 0.0
        for i, p in enumerate(pos):
            if active is not None and (p == 0 or p * previous < 0):
                start = active
                pnl_path = np.r_[0.0, reward[start:i+1].cumsum()]
                entry_record = trade_by_step.loc[start]
                exit_record = trade_by_step.loc[i]
                direction = int(np.sign(previous))
                # Audit fill_price uses a post-step mark plus execution slippage.
                # Unit-size, constant-fee-rate run: fee/rate recovers notional.
                fill = float(fee[start] / 0.0003)
                # Past-only style vote, as in ADR 0007, five prices preceding entry.
                momentum = price[start-1]-price[start-5] if start >= 5 else 0.0
                style = "trend_following" if momentum * direction > 0 else ("mean_reversion" if momentum * direction < 0 else "unclassified")
                row = {"contract": name, "entry_step": start, "exit_step": i, "entry_time": market.timestamp.iloc[start], "exit_time": market.timestamp.iloc[i], "direction": direction, "entry_slot": int(macro[start]), "exit_slot": int(macro[i]), "entry_role": float(frame.role.iloc[start]), "exit_role": float(frame.role.iloc[i]), "exit_reason": exit_record.decision_reason, "pnl": float(reward[start:i+1].sum()), "bars": i-start, "entry_fill": fill, "exit_fill": float(fee[i]/0.0003), "reported_entry_fill": float(entry_record.fill_price), "reported_exit_fill": float(exit_record.fill_price), "mfe_net_path": float(pnl_path.max()), "mae_net_path": float(pnl_path.min()), "giveback_net_path": float(pnl_path.max()-pnl_path[-1]), "style": style, "entry_momentum5": momentum}
                row["gross_fill_pnl"] = direction*(row["exit_fill"]-fill)
                row["fees"] = float(fee[start:i+1].sum())
                row["mfe_mark"] = max(0.0, float(np.max(direction*(price[start:i+1]-fill))))
                row["mae_mark"] = min(0.0, float(np.min(direction*(price[start:i+1]-fill))))
                if start in pred_by_step.index:
                    pr = pred_by_step.loc[start]
                    for col in ("slope_max", "volatility_max", "slope_margin", "volatility_margin", "action_value_margin", "action_value_short", "action_value_flat", "action_value_long"):
                        row[col] = float(pr[col])
                    row["prediction_slot"] = int(pr.routing_slot)
                    assert row["entry_slot"] == row["prediction_slot"]
                # Actual next 6/12/24 marks are ex-post opportunity diagnostics.
                for horizon in (6, 12, 24):
                    end = min(i+horizon, n)
                    row[f"continuation_{horizon}_gross"] = float(direction*(price[end]-price[i]))
                episodes.append(row)
                active = None
            if p != 0 and (previous == 0 or previous*p < 0):
                active = i
            previous = p
        assert active is None, f"Unclosed episode: {name}"
        contract_info.append({"contract": name, "start": str(market.timestamp.iloc[0]), "end": str(market.timestamp.iloc[-1]), "net_pnl": reward.sum(), "warmup_bars": int((reasons=="WARM_UP").sum())})
    step = pd.concat(steps, ignore_index=True)
    episode = pd.DataFrame(episodes)
    pred = pd.concat(predictions, ignore_index=True)
    np.testing.assert_allclose(episode.pnl.sum(), step.net_pnl.sum(), atol=1e-7)
    np.testing.assert_allclose(episode.gross_fill_pnl-episode.fees, episode.pnl, atol=1e-6)
    episode.to_csv(OUT / "episodes.csv", index=False)
    step.to_csv(OUT / "steps.csv", index=False)
    pred.to_csv(OUT / "predictions.csv", index=False)
    for field in ("entry_slot", "direction", "entry_role", "exit_reason", "style", "contract"):
        grouped(episode, field).to_csv(OUT / f"episodes_by_{field}.csv")
    step.groupby("slot").agg(steps=("net_pnl", "size"), net_pnl=("net_pnl", "sum"), gross_pnl=("gross_pnl", "sum"), fee=("fee", "sum")).to_csv(OUT / "step_slot_attribution.csv")
    step.groupby(["role", "position"]).agg(steps=("net_pnl", "size"), net_pnl=("net_pnl", "sum")).to_csv(OUT / "role_direction_attribution.csv")
    summary = {"episodes": len(episode), "wins": int((episode.pnl>0).sum()), "losses": int((episode.pnl<0).sum()), "win_rate": float((episode.pnl>0).mean()), "profit_factor": float(episode.loc[episode.pnl>0, "pnl"].sum()/abs(episode.loc[episode.pnl<0, "pnl"].sum())), "sum_winners": float(episode.loc[episode.pnl>0, "pnl"].sum()), "sum_losers": float(episode.loc[episode.pnl<0, "pnl"].sum()), "median_bars": float(episode.bars.median()), "mean_bars": float(episode.bars.mean()), "mfe_sum_hindsight_only": float(episode.mfe_net_path.sum()), "giveback_sum_hindsight_only": float(episode.giveback_net_path.sum()), "losers_previously_positive": int(((episode.pnl<0)&(episode.mfe_net_path>0)).sum()), "losers_mfe_ge20": int(((episode.pnl<0)&(episode.mfe_net_path>=20)).sum()), "contracts": contract_info}
    summary["handoffs_and_exposure"] = analyze_selection_and_features(step, episode)
    timeline = step.groupby("timestamp").net_pnl.sum().sort_index()
    eq = 12000+timeline.cumsum()
    drawdown = 1-eq/eq.cummax()
    trough = drawdown.idxmax()
    peak = eq.loc[:trough].idxmax()
    summary["maxdd"] = {"peak_time": str(peak), "trough_time": str(trough), "peak_equity": float(eq.loc[peak]), "trough_equity": float(eq.loc[trough]), "cash_loss": float(eq.loc[trough]-eq.loc[peak])}
    dd_window = step[(step.timestamp>peak)&(step.timestamp<=trough)]
    dd_window.groupby(["contract", "position"]).net_pnl.sum().to_csv(OUT / "drawdown_attribution.csv")
    step["month"] = step.timestamp.dt.to_period("M").astype(str)
    step.groupby("month").net_pnl.sum().to_csv(OUT / "monthly_pnl.csv")
    # Conditional association on realized entries; not a causal filter backtest.
    for col in ("slope_max", "volatility_max", "slope_margin", "volatility_margin", "action_value_margin"):
        episode[f"{col}_quartile"] = pd.qcut(episode[col], 4, duplicates="drop")
        grouped(episode, f"{col}_quartile").to_csv(OUT / f"entry_quality_{col}.csv")
    eligible = step[step.eligible]
    summary["gate"] = {"eligible_steps": len(eligible), "passed_steps": int((eligible.slot<9).sum()), "pass_rate": float((eligible.slot<9).mean()), "reason_counts": step.reason.value_counts().to_dict(), "position_counts_eligible": eligible.position.value_counts().to_dict(), "prediction_margin_quantiles": pred[["slope_margin", "volatility_margin", "slope_max", "volatility_max"]].quantile([0.1,0.5,0.9]).to_dict()}
    # Frozen-path PnL scaling is sensitivity only; no state-changing replay.
    sensitivity = []
    for name, mask in [("main_step_only", step.role==1), ("sub_step_only", step.role==.5), ("long_step_only", step.position>0), ("short_step_only", step.position<0)]:
        pnls=step.net_pnl.where(mask, 0).groupby(step.timestamp).sum().sort_index()
        values=np.r_[12000,12000+pnls.cumsum()]
        sensitivity.append({"mask": name,"net_pnl": pnls.sum(),"mdd": float(max(1-values/np.maximum.accumulate(values)))})
    pd.DataFrame(sensitivity).to_csv(OUT / "frozen_path_masks_not_backtests.csv", index=False)
    (OUT / "attribution_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    for col in ("entry_slot", "direction", "entry_role", "exit_reason", "style"):
        print(col, grouped(episode, col).to_string(), sep="\n")


if __name__ == "__main__":
    main()
