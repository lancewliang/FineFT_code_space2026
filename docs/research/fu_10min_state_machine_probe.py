"""Isolated, one-factor-at-a-time valid replays using the actual router and env.

No production model/config/result is changed. These are development experiments,
not independent out-of-sample tests. Baseline rewards are compared step by step.
"""
from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "FineFT"))
import numpy as np
import pandas as pd
import torch
from RL.DiHFT.high_level.vae_routing_final_result_macro_action import parse_and_prepare_args
from RL.DiHFT.high_level.vae_routing_util import vae_risk_aware_routing
from RL.DiHFT.high_level.gating.base import GatingDecision

OUT = ROOT / "docs/research/fu_10min_followup_evidence"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    logging.getLogger().setLevel(logging.ERROR)
    args = parse_and_prepare_args([
        "--base_path", "dataset/10min", "--dataset_name", "fu",
        "--experiment_name", "10min_parallel", "--eval_stage", "valid",
        "--output_sub_dir", "followup_diagnostic_probe",
        "--max_holding_number", "1", "--initial_wallet_balance", "6000",
        "--position_choices", "3", "--order_book_depth", "5",
        "--transcation_cost", "0.0003", "--short_estimated_rate", "0",
        "--long_estimated_rate", "0", "--enable_non_main_contract_defense",
        "--non_main_defense_mode", "graceful_hold_exit",
        "--enable_delivery_month_defense", "--delivery_defense_days", "5",
        "--action_persistence", "6", "--stop_loss_return_threshold", "0.03",
        "--stop_loss_cooldown_steps", "12", "--circuit_breaker_consecutive_stops", "3",
        "--circuit_breaker_cooling_steps", "24", "--no-save_artifacts",
    ])
    router = vae_risk_aware_routing(args)
    original_agent_act = router.agent_act
    original_get_action = router.get_action
    original_network_state = copy.deepcopy(router.low_level_network.state_dict())
    original_slots = copy.deepcopy(router.selection_manifest.slots)
    files = [(c, pd.read_feather(p)) for c, p in router.find_contract_files()]
    (OUT / "resolved_parameters.json").write_text(json.dumps(vars(args), indent=2))
    cases = [
        ("baseline", {}),
        ("slot6_flat", {"flat_slot": 6}),
        ("slot8_flat", {"flat_slot": 8}),
        ("main_only", {"main_only": True}),
        ("stop_2pct", {"stop_loss_return_threshold": 0.02}),
        ("stop_disabled", {"stop_loss_return_threshold": 0.0}),
        ("persistence_1", {"action_persistence": 1}),
        ("persistence_3", {"action_persistence": 3}),
        ("threshold_055", {"slope_rule_base_threshold": 0.55, "volatility_rule_base_threshold": 0.55}),
        ("exit_ratio_1", {"hysteresis_exit_ratio": 1.0}),
        ("hier_margin_002", {"gating_strategy": "hierarchical", "slope_margin_threshold": 0.02, "volatility_margin_threshold": 0.02}),
        ("hier_margin_012", {"gating_strategy": "hierarchical", "slope_margin_threshold": 0.12, "volatility_margin_threshold": 0.12}),
        ("slot4_flat", {"flat_slot": 4}),
        ("delivery_disabled", {"enable_delivery_month_defense": False}),
        ("slope_threshold_055", {"slope_rule_base_threshold": 0.55}),
        ("vol_threshold_055", {"volatility_rule_base_threshold": 0.55}),
        ("vol_gamma_0925", {"volatility_gamma": 0.925}),
        ("trailing_1pct_half", {"trailing": True}),
        ("benchmark_long", {"benchmark": "long", "slope_rule_base_threshold": 0.0, "volatility_rule_base_threshold": 0.0}),
        ("benchmark_mom48", {"benchmark": 48, "slope_rule_base_threshold": 0.0, "volatility_rule_base_threshold": 0.0}),
        ("benchmark_mom12", {"benchmark": 12, "slope_rule_base_threshold": 0.0, "volatility_rule_base_threshold": 0.0}),
        ("slot4_second_candidate", {"replacement": [4, 50, 4]}),
    ]
    if len(sys.argv) > 1:
        cases = [c for c in cases if c[0] in sys.argv[1:]]
    else:
        cases = cases[:12]
    summary_path = OUT / ("state_machine_probes_extra.csv" if len(sys.argv)>1 else "state_machine_probes.csv")
    summary = []
    if len(sys.argv)>1 and summary_path.exists():
        requested = {name for name, _ in cases}
        previous = pd.read_csv(summary_path)
        summary = previous.loc[~previous["case"].isin(requested)].to_dict("records")
    for name, changes in cases:
        start = time.monotonic()
        cfg = copy.deepcopy(args)
        for key, value in changes.items():
            if key not in ("flat_slot", "main_only", "benchmark", "trailing", "replacement"):
                setattr(cfg, key, value)
        router.reconfigure_routing(cfg)
        if changes.get("main_only"):
            gate = router.gating_strategy

            class MainOnlyAdmission:
                def decide(self, volatility_weights, slope_weights, current_position=0.0):
                    role = router.df.prev_day_contract_role_tier.iloc[int(router.step_idx)]
                    if role < 1.0 and current_position == 0.0:
                        return GatingDecision(is_defensive=True, volatility_index=-1, slope_index=-1, reject_reason="main_only_admission", metrics={})
                    return gate.decide(volatility_weights, slope_weights, current_position)

            router.gating_strategy = MainOnlyAdmission()
        router.enable_delivery_month_defense = cfg.enable_delivery_month_defense
        router.agent_act = original_agent_act
        router.get_action = original_get_action
        router.low_level_network.load_state_dict(original_network_state)
        if "replacement" in changes:
            slot, epoch, bin_index = changes["replacement"]
            state = torch.load(ROOT / f"result/DiHFT/low_level/fu/10min_parallel/weights_advantage_pretrain/epoch_{epoch}/trained_model.pkl", map_location="cpu", weights_only=False)
            prefix = f"qnet_list.{bin_index}."
            sub_state = {k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
            router.low_level_network.qnet_list[slot].load_state_dict(sub_state)
        if "benchmark" in changes:
            def benchmark_action(s, info):
                lookback = changes["benchmark"]
                if lookback == "long":
                    return 2 if info["avaliable_action"][2] else 1
                idx = int(router.step_idx)
                prices = router.df.mark_price.to_numpy()
                signal = np.sign(prices[idx]-prices[max(0, idx-int(lookback))])
                action = int(signal+1)
                return action if info["avaliable_action"][action] else 1
            router.agent_act = benchmark_action
        if changes.get("trailing"):
            def trailing_action(info, s, current_position, current_leverage, current_unrealized_pnl=0.0, current_markprice=0.0):
                if current_position == 0 or current_position*router._probe_previous <= 0:
                    router._probe_peak = 0.0
                router._probe_previous = current_position
                router._probe_peak = max(router._probe_peak, current_unrealized_pnl)
                if current_position != 0 and router._probe_peak >= .01*abs(current_position)*current_markprice and current_unrealized_pnl <= .5*router._probe_peak:
                    return router._apply_defensive_action(info, current_position, current_leverage)
                return original_get_action(info, s, current_position, current_leverage, current_unrealized_pnl, current_markprice)
            router.get_action = trailing_action
        for dest, source in zip(router.selection_manifest.slots, original_slots):
            dest.clear()
            dest.update(copy.deepcopy(source))
        if "flat_slot" in changes:
            router.selection_manifest.slots[changes["flat_slot"]]["kind"] = "empty_model"
        rows, parts, histories = [], [], []
        for contract, market in files:
            frame = market.copy()
            router.reset_routing_state()
            router._probe_peak = 0.0
            router._probe_previous = 0.0
            result = router.run_single_valid_df(frame, str(OUT / name / contract), contract_name=contract)
            reward = np.asarray(result["step_net_pnls"])
            rows.append({"case": name, "contract": contract, "net_pnl": float(reward.sum()), "trades": result["trades"], "hard_stops": router.hard_stop_loss_count})
            parts.append(pd.Series(reward, index=market.timestamp.iloc[:len(reward)]))
            if name == "baseline":
                histories.append((contract, reward))
        timeline = pd.concat(parts, axis=1).fillna(0).sort_index().sum(axis=1)
        eq = np.r_[12000.0, 12000 + timeline.cumsum().to_numpy()]
        mdd = float(np.max(1 - eq / np.maximum.accumulate(eq)))
        item = {"case": name, "net_pnl": float(timeline.sum()), "tr": float(timeline.sum()/12000), "mdd": mdd, "trades": sum(x["trades"] for x in rows), "hard_stops": sum(x["hard_stops"] for x in rows), "seconds": time.monotonic()-start}
        if name == "baseline":
            for contract, reward in histories:
                saved = np.load(ROOT / "result/DiHFT/final_result/fu/10min_parallel/main_sub/contracts" / contract / "reward_history.npy")
                np.testing.assert_allclose(reward, saved, atol=1e-7)
            item["exact_reward_replay"] = True
        pd.DataFrame(rows).to_csv(OUT / f"probe_{name}_contracts.csv", index=False)
        pd.DataFrame({"timestamp": timeline.index, "net_pnl": timeline.values}).to_csv(OUT / f"probe_{name}_timeline.csv", index=False)
        summary.append(item)
        pd.DataFrame(summary).to_csv(summary_path, index=False)
        print(json.dumps(item), flush=True)


if __name__ == "__main__":
    main()
