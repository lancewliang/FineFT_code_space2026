import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


FINEFT_ROOT = Path(__file__).resolve().parents[2]
if str(FINEFT_ROOT) not in sys.path:
    sys.path.insert(0, str(FINEFT_ROOT))

from analysis.pick_agent.FineFT_single_agent_with_different_position import picker


def _picker(tmp_path):
    save_dir = tmp_path / "fu" / "10min_nstep6_costw5"
    save_dir.mkdir(parents=True)
    return picker(
        Namespace(
            base_path="dataset/10min",
            dataset_name="fu",
            num_label=5,
            epoch_num=1,
            initial_position=1,
            save_path=str(tmp_path),
            model_save_path="result/DiHFT/potential_model",
            std_preference=0.1,
            experiment_name="10min_nstep6_costw5",
            position_choices=3,
            hidden_nodes=128,
        )
    )


def _cross_contract_record(label, initial_action, bin_index, rewards, lengths):
    contracts = [f"fu25{i:02d}" for i in range(len(rewards))]
    n = len(rewards)
    return {
        "label": label,
        "initial_action": initial_action,
        "bin_index": bin_index,
        "contract": contracts,
        "df_path": [
            f"{contract}/{label}/df_{index}.feather"
            for index, contract in enumerate(contracts)
        ],
        "reward_sum": rewards,
        "df_length": lengths,
        "turnover": [0.0 for _ in rewards],
        "mean_position": [0.0] * n,
        "mean_abs_position": [0.0] * n,
        "long_step_ratio": [0.0] * n,
        "short_step_ratio": [0.0] * n,
        "flat_step_ratio": [1.0] * n,
        "long_reward_sum": [0.0] * n,
        "short_reward_sum": [0.0] * n,
        "flat_reward_sum": [0.0] * n,
        "net_position_exposure": [0.0] * n,
        "limit_up_step_ratio": [0.0] * n,
        "limit_down_step_ratio": [0.0] * n,
        "limit_up_long_reward_sum": [0.0] * n,
        "limit_down_short_reward_sum": [0.0] * n,
        "limit_up_reverse_short_ratio": [0.0] * n,
        "limit_down_reverse_long_ratio": [0.0] * n,
    }


def test_transform_single_epoch_result_uses_sample_equal_cross_contract_rewards(
    tmp_path,
):
    p = _picker(tmp_path)
    result = [
        _cross_contract_record(
            "label_0",
            initial_action=0,
            bin_index=1,
            rewards=[10.0, 6.0],
            lengths=[5, 3],
        )
    ]

    transformed = p.transform_single_epoch_result(result, "epoch_1")

    assert transformed[0]["normalized_reward"].tolist() == [2.0, 2.0]
    assert transformed[0]["trans_reward_mean"] == 2.0
    assert transformed[0]["trans_reward_std"] == 0.0


def test_transform_single_epoch_result_rejects_legacy_contract_label_schema(
    tmp_path,
):
    p = _picker(tmp_path)
    legacy = [
        {
            "label": "fu2409/label_0",
            "initial_action": 0,
            "bin_index": 1,
            "df_path": ["df_0.feather"],
            "reward_sum": [1.0],
            "df_length": [1],
            "turnover": [0.0],
        }
    ]

    with pytest.raises(ValueError, match="rerun test_agent_index.py"):
        p.transform_single_epoch_result(legacy, "epoch_1")


def test_picker_rejects_label_set_mismatch(tmp_path):
    p = _picker(tmp_path)
    result_all = pd.DataFrame(
        [
            {
                "label": "label_0",
                "bin_index": 1,
                "epoch_path": "epoch_1",
                "trans_reward_mean": 0.2,
            }
        ]
    )

    with pytest.raises(ValueError, match="label_1"):
        p.pick_best_agent_regarding_dynamics_bin_index_path(result_all)


def _make_candidate_row(label, bin_index, epoch_path, initial_action, score):
    if label == "label_0":
        net_exp, l_ratio, s_ratio, l_rew, s_rew, lim_u_rew, lim_d_rew = -0.5, 0.0, 0.6, 0.0, 5.0, 0.0, 5.0
    elif label in ["label_1"]:
        net_exp, l_ratio, s_ratio, l_rew, s_rew, lim_u_rew, lim_d_rew = -0.5, 0.0, 0.6, 0.0, 5.0, 0.0, 0.0
    elif label in ["label_2"]:
        net_exp, l_ratio, s_ratio, l_rew, s_rew, lim_u_rew, lim_d_rew = 0.0, 0.1, 0.1, 0.5, 0.5, 0.0, 0.0
    elif label in ["label_3"]:
        net_exp, l_ratio, s_ratio, l_rew, s_rew, lim_u_rew, lim_d_rew = 0.5, 0.6, 0.0, 5.0, 0.0, 0.0, 0.0
    else:  # label_4
        net_exp, l_ratio, s_ratio, l_rew, s_rew, lim_u_rew, lim_d_rew = 0.5, 0.6, 0.0, 5.0, 0.0, 5.0, 0.0

    return {
        "label": label,
        "bin_index": bin_index,
        "epoch_path": epoch_path,
        "initial_action": initial_action,
        "trans_reward_mean": score,
        "candidate_mean_exposure": net_exp,
        "candidate_long_ratio": l_ratio,
        "candidate_short_ratio": s_ratio,
        "candidate_long_reward_mean": l_rew,
        "candidate_short_reward_mean": s_rew,
        "candidate_limit_up_long_reward_mean": lim_u_rew,
        "candidate_limit_down_short_reward_mean": lim_d_rew,
        "candidate_limit_up_reverse_short_ratio": 0.0,
        "candidate_limit_down_reverse_long_ratio": 0.0,
    }


def test_final_selection_keeps_current_result_all_initial_action_mean(tmp_path):
    p = _picker(tmp_path)
    p.label_semantics_path = _make_label_semantics_file(tmp_path)
    rows = []
    for initial_action, score in [(0, 1.0), (1, 3.0)]:
        rows.append(_make_candidate_row("label_0", 0, "epoch_1", initial_action, score))
    for initial_action, score in [(0, 1.9), (1, 1.9)]:
        rows.append(_make_candidate_row("label_0", 1, "epoch_2", initial_action, score))
    for label_index in range(1, 5):
        rows.append(_make_candidate_row(f"label_{label_index}", label_index, f"epoch_{label_index}", 0, 0.1))

    best = p.pick_best_agent_regarding_dynamics_bin_index_path(pd.DataFrame(rows))

    label_0 = best[best["label"] == "label_0"].iloc[0]
    assert label_0["bin_index"] == 0
    assert label_0["epoch_path"] == "epoch_1"
    assert label_0["reward_max"] == 2.0


def test_write_selection_manifest_records_label_choices(tmp_path):
    p = _picker(tmp_path)
    best = pd.DataFrame(
        [
            {
                "label": f"label_{index}",
                "epoch_path": f"epoch_{index}",
                "bin_index": index,
                "reward_max": float(index),
                "source_rows": index + 1,
            }
            for index in range(5)
        ]
    )

    manifest_path = p.write_selection_manifest(best)

    manifest = json.loads(Path(manifest_path).read_text())
    assert manifest["dataset_name"] == "fu"
    assert manifest["experiment_name"] == "10min_nstep6_costw5"
    assert manifest["selection_method"] == "sample_equal_current_picker_logic"
    assert manifest["labels"][0]["label"] == "label_0"
    assert manifest["labels"][0]["model_path"] == "epoch_0/trained_model.pkl"


def test_pick_best_agent_fails_when_label_has_only_nan_rewards(tmp_path):
    p = _picker(tmp_path)
    p.label_semantics_path = _make_label_semantics_file(tmp_path)
    rows = [
        _make_candidate_row(f"label_{label_index}", label_index, "epoch_1", 0, 0.2)
        for label_index in range(4)
    ]
    rows.append(
        _make_candidate_row("label_4", 2, "epoch_1", 0, np.nan)
    )

    with pytest.raises(ValueError, match="label_4"):
        p.pick_best_agent_regarding_dynamics_bin_index_path(
            pd.DataFrame(rows)
        )

from analysis.pick_agent.DiHFT_high_level_heurstic import Picker as DiHFT_Picker

def test_dihft_picker_multi_contract_analysis(tmp_path):
    # Setup mock epoch structure with two contracts
    epoch_dir = tmp_path / "result" / "DiHFT" / "high_level" / "fu" / "30min_multi" / "vae_risk_aware_routing" / "param_1"
    c1_dir = epoch_dir / "contracts" / "fu2409"
    c2_dir = epoch_dir / "contracts" / "fu2411"
    c1_dir.mkdir(parents=True)
    c2_dir.mkdir(parents=True)

    for cdir, r_val in [(c1_dir, 10.0), (c2_dir, 20.0)]:
        np.save(cdir / "initial_margin_history.npy", np.array([100.0, 100.0]))
        np.save(cdir / "maintain_marigine_history.npy", np.array([50.0, 50.0]))
        np.save(cdir / "new_position_required_money_history.npy", np.array([0.0, 0.0]))
        np.save(cdir / "micro_action_history.npy", np.array([1, 1]))
        np.save(cdir / "reward_history.npy", np.array([r_val, r_val]))
        np.save(cdir / "total_asset_history.npy", np.array([1000.0, 1020.0]))
        np.save(cdir / "unrealized_pnl_history.npy", np.array([0.0, 0.0]))
        np.save(cdir / "wallet_balance_history.npy", np.array([1000.0, 1020.0]))

    args = Namespace(
        dataset_name="fu",
        experiment_name="30min_multi",
        save_path=str(tmp_path / "analysis_result"),
        early_stop=0,
    )
    picker = DiHFT_Picker(args)
    res = picker.analysis_single_epoch(str(epoch_dir))
    assert res["num_contracts"] == 2
    assert res["tr"] > 0


def test_dihft_picker_find_valid_contract_files_uses_base_path(tmp_path):
    custom_base = tmp_path / "custom_base"
    valid_dir = custom_base / "fu" / "valid"
    valid_dir.mkdir(parents=True)
    feather_file = valid_dir / "c1.feather"
    feather_file.write_text("dummy")

    args = Namespace(
        base_path=str(custom_base),
        dataset_name="fu",
        experiment_name="30min_multi",
        save_path=str(tmp_path / "analysis_result"),
        early_stop=0,
    )
    picker = DiHFT_Picker(args)
    files = picker._find_valid_contract_files()
    assert len(files) == 1
    assert files[0] == ("c1", str(feather_file))


def _semantic_cross_contract_record(
    label,
    initial_action,
    bin_index,
    rewards,
    lengths,
    net_position_exposure=0.0,
    long_step_ratio=0.0,
    short_step_ratio=0.0,
    flat_step_ratio=1.0,
    long_reward_sum=0.0,
    short_reward_sum=0.0,
    limit_up_step_ratio=0.0,
    limit_down_step_ratio=0.0,
    limit_up_long_reward_sum=0.0,
    limit_down_short_reward_sum=0.0,
    limit_up_reverse_short_ratio=0.0,
    limit_down_reverse_long_ratio=0.0,
):
    contracts = [f"fu25{i:02d}" for i in range(len(rewards))]
    n = len(rewards)
    return {
        "label": label,
        "initial_action": initial_action,
        "bin_index": bin_index,
        "contract": contracts,
        "df_path": [f"{c}/{label}/df_0.feather" for c in contracts],
        "reward_sum": rewards,
        "df_length": lengths,
        "turnover": [0.0] * n,
        "mean_position": [net_position_exposure] * n,
        "mean_abs_position": [abs(net_position_exposure)] * n,
        "long_step_ratio": [long_step_ratio] * n,
        "short_step_ratio": [short_step_ratio] * n,
        "flat_step_ratio": [flat_step_ratio] * n,
        "long_reward_sum": [long_reward_sum] * n,
        "short_reward_sum": [short_reward_sum] * n,
        "flat_reward_sum": [0.0] * n,
        "net_position_exposure": [net_position_exposure] * n,
        "limit_up_step_ratio": [limit_up_step_ratio] * n,
        "limit_down_step_ratio": [limit_down_step_ratio] * n,
        "limit_up_long_reward_sum": [limit_up_long_reward_sum] * n,
        "limit_down_short_reward_sum": [limit_down_short_reward_sum] * n,
        "limit_up_reverse_short_ratio": [limit_up_reverse_short_ratio] * n,
        "limit_down_reverse_long_ratio": [limit_down_reverse_long_ratio] * n,
    }


def _make_label_semantics_file(tmp_path, labels=None, labeling_method="slope"):
    if labels is None:
        labels = [
            {
                "label": "label_0",
                "direction": "strong_down",
                "direction_sign": -1,
                "strength": 2,
                "description": "跌停",
                "limit_state": "limit_down",
                "limit_state_sign": -1,
            },
            {
                "label": "label_1",
                "direction": "down",
                "direction_sign": -1,
                "strength": 1,
                "description": "下跌",
                "limit_state": "none",
                "limit_state_sign": 0,
            },
            {
                "label": "label_2",
                "direction": "sideways",
                "direction_sign": 0,
                "strength": 0,
                "description": "震荡",
                "limit_state": "none",
                "limit_state_sign": 0,
            },
            {
                "label": "label_3",
                "direction": "up",
                "direction_sign": 1,
                "strength": 1,
                "description": "上涨",
                "limit_state": "none",
                "limit_state_sign": 0,
            },
            {
                "label": "label_4",
                "direction": "strong_up",
                "direction_sign": 1,
                "strength": 2,
                "description": "涨停",
                "limit_state": "limit_up",
                "limit_state_sign": 1,
            },
        ]
    sem_path = tmp_path / "label_semantics.json"
    data = {
        "dataset_name": "fu",
        "labeling_method": labeling_method,
        "dynamic_number": 3,
        "label_number": 5,
        "labels": labels,
    }
    sem_path.write_text(json.dumps(data, indent=2))
    return str(sem_path)


def test_picker_rejects_edge_label_price_limit_convention_violation(tmp_path):
    # label_0 is missing limit_down semantics
    bad_labels = [
        {
            "label": "label_0",
            "direction": "down",
            "direction_sign": -1,
            "strength": 1,
            "description": "普通下跌",
            "limit_state": "none",
            "limit_state_sign": 0,
        },
        {
            "label": "label_1",
            "direction": "down",
            "direction_sign": -1,
            "strength": 1,
            "description": "下跌",
            "limit_state": "none",
            "limit_state_sign": 0,
        },
        {
            "label": "label_2",
            "direction": "sideways",
            "direction_sign": 0,
            "strength": 0,
            "description": "震荡",
            "limit_state": "none",
            "limit_state_sign": 0,
        },
        {
            "label": "label_3",
            "direction": "up",
            "direction_sign": 1,
            "strength": 1,
            "description": "上涨",
            "limit_state": "none",
            "limit_state_sign": 0,
        },
        {
            "label": "label_4",
            "direction": "strong_up",
            "direction_sign": 1,
            "strength": 2,
            "description": "涨停",
            "limit_state": "limit_up",
            "limit_state_sign": 1,
        },
    ]
    sem_path = _make_label_semantics_file(tmp_path, labels=bad_labels)
    p = _picker(tmp_path)
    p.label_semantics_path = sem_path
    with pytest.raises(ValueError, match="label_0"):
        p.load_label_semantics()


def test_picker_rejects_dtw_without_explicit_semantics(tmp_path):
    p = _picker(tmp_path)
    p.labeling_method = "DTW"
    p.label_semantics_path = None
    with pytest.raises(ValueError, match="DTW"):
        p.load_label_semantics()


def test_bullish_label_selects_long_profitable_candidate(tmp_path):
    p = _picker(tmp_path)
    p.label_semantics_path = _make_label_semantics_file(tmp_path)
    # For label_3 (up, direction_sign=1):
    # Cand A: higher reward (10), but bearish (net_position_exposure = -0.5, long_step_ratio = 0)
    # Cand B: lower reward (5), bullish (net_position_exposure = 0.5, long_step_ratio = 0.6, long_reward_sum = 5.0)
    recs = []
    # label_0..2, 4 dummy recs that pass
    for lbl, net_exp, l_ratio, s_ratio, l_rew, s_rew, lim_u_rew, lim_d_rew in [
        ("label_0", -0.5, 0.0, 0.6, 0.0, 5.0, 0.0, 5.0),
        ("label_1", -0.5, 0.0, 0.6, 0.0, 5.0, 0.0, 0.0),
        ("label_2", 0.0, 0.1, 0.1, 0.5, 0.5, 0.0, 0.0),
        ("label_4", 0.5, 0.6, 0.0, 5.0, 0.0, 5.0, 0.0),
    ]:
        recs.append(
            _semantic_cross_contract_record(
                lbl, 0, 0, [5.0], [10],
                net_position_exposure=net_exp,
                long_step_ratio=l_ratio,
                short_step_ratio=s_ratio,
                long_reward_sum=l_rew,
                short_reward_sum=s_rew,
                limit_up_long_reward_sum=lim_u_rew,
                limit_down_short_reward_sum=lim_d_rew,
            )
        )
    # label_3 candidate A (bearish, high reward)
    recs.append(
        _semantic_cross_contract_record(
            "label_3", 0, 0, [10.0], [10],
            net_position_exposure=-0.5, long_step_ratio=0.0, short_step_ratio=0.8,
            long_reward_sum=0.0, short_reward_sum=10.0
        )
    )
    # label_3 candidate B (bullish, lower reward)
    recs.append(
        _semantic_cross_contract_record(
            "label_3", 0, 1, [5.0], [10],
            net_position_exposure=0.5, long_step_ratio=0.6, short_step_ratio=0.0,
            long_reward_sum=5.0, short_reward_sum=0.0
        )
    )

    df_all = p.transform_single_epoch_result_all(recs, "epoch_1")
    best = p.pick_best_agent_regarding_dynamics_bin_index_path(df_all)

    l3 = best[best["label"] == "label_3"].iloc[0]
    assert l3["bin_index"] == 1  # candidate B selected


def test_bearish_label_selects_short_profitable_candidate(tmp_path):
    p = _picker(tmp_path)
    p.label_semantics_path = _make_label_semantics_file(tmp_path)
    # For label_1 (down, direction_sign=-1):
    # Cand A: higher reward (10), but bullish (net_position_exposure = 0.5, short_step_ratio = 0)
    # Cand B: lower reward (5), bearish (net_position_exposure = -0.5, short_step_ratio = 0.6, short_reward_sum = 5.0)
    recs = []
    for lbl, net_exp, l_ratio, s_ratio, l_rew, s_rew, lim_u_rew, lim_d_rew in [
        ("label_0", -0.5, 0.0, 0.6, 0.0, 5.0, 0.0, 5.0),
        ("label_2", 0.0, 0.1, 0.1, 0.5, 0.5, 0.0, 0.0),
        ("label_3", 0.5, 0.6, 0.0, 5.0, 0.0, 0.0, 0.0),
        ("label_4", 0.5, 0.6, 0.0, 5.0, 0.0, 5.0, 0.0),
    ]:
        recs.append(
            _semantic_cross_contract_record(
                lbl, 0, 0, [5.0], [10],
                net_position_exposure=net_exp,
                long_step_ratio=l_ratio,
                short_step_ratio=s_ratio,
                long_reward_sum=l_rew,
                short_reward_sum=s_rew,
                limit_up_long_reward_sum=lim_u_rew,
                limit_down_short_reward_sum=lim_d_rew,
            )
        )
    # label_1 Cand A (bullish, high reward)
    recs.append(
        _semantic_cross_contract_record(
            "label_1", 0, 0, [10.0], [10],
            net_position_exposure=0.5, long_step_ratio=0.8, short_step_ratio=0.0,
            long_reward_sum=10.0, short_reward_sum=0.0
        )
    )
    # label_1 Cand B (bearish, lower reward)
    recs.append(
        _semantic_cross_contract_record(
            "label_1", 0, 1, [5.0], [10],
            net_position_exposure=-0.5, long_step_ratio=0.0, short_step_ratio=0.6,
            long_reward_sum=0.0, short_reward_sum=5.0
        )
    )

    df_all = p.transform_single_epoch_result_all(recs, "epoch_1")
    best = p.pick_best_agent_regarding_dynamics_bin_index_path(df_all)

    l1 = best[best["label"] == "label_1"].iloc[0]
    assert l1["bin_index"] == 1


def test_sideways_label_rejects_strong_directional_exposure(tmp_path):
    p = _picker(tmp_path)
    p.label_semantics_path = _make_label_semantics_file(tmp_path)
    # For label_2 (sideways, direction_sign=0):
    # Cand A: higher reward (10), but strong exposure (net_position_exposure = 0.5)
    # Cand B: lower reward (5), controlled exposure (net_position_exposure = 0.05)
    recs = []
    for lbl, net_exp, l_ratio, s_ratio, l_rew, s_rew, lim_u_rew, lim_d_rew in [
        ("label_0", -0.5, 0.0, 0.6, 0.0, 5.0, 0.0, 5.0),
        ("label_1", -0.5, 0.0, 0.6, 0.0, 5.0, 0.0, 0.0),
        ("label_3", 0.5, 0.6, 0.0, 5.0, 0.0, 0.0, 0.0),
        ("label_4", 0.5, 0.6, 0.0, 5.0, 0.0, 5.0, 0.0),
    ]:
        recs.append(
            _semantic_cross_contract_record(
                lbl, 0, 0, [5.0], [10],
                net_position_exposure=net_exp,
                long_step_ratio=l_ratio,
                short_step_ratio=s_ratio,
                long_reward_sum=l_rew,
                short_reward_sum=s_rew,
                limit_up_long_reward_sum=lim_u_rew,
                limit_down_short_reward_sum=lim_d_rew,
            )
        )
    # label_2 Cand A (strong exposure, high reward)
    recs.append(
        _semantic_cross_contract_record(
            "label_2", 0, 0, [10.0], [10],
            net_position_exposure=0.5, long_step_ratio=0.8, short_step_ratio=0.0
        )
    )
    # label_2 Cand B (controlled exposure, lower reward)
    recs.append(
        _semantic_cross_contract_record(
            "label_2", 0, 1, [5.0], [10],
            net_position_exposure=0.05, long_step_ratio=0.1, short_step_ratio=0.1
        )
    )

    df_all = p.transform_single_epoch_result_all(recs, "epoch_1")
    best = p.pick_best_agent_regarding_dynamics_bin_index_path(df_all)

    l2 = best[best["label"] == "label_2"].iloc[0]
    assert l2["bin_index"] == 1


def test_limit_up_label_requires_limit_state_long_profitability(tmp_path):
    p = _picker(tmp_path)
    p.label_semantics_path = _make_label_semantics_file(tmp_path)
    # For label_4 (limit_up):
    # Cand A: high reward, high reverse short ratio (0.5 > 0.2)
    # Cand B: lower reward, long profitable in limit up (limit_up_long_reward_sum = 3.0, limit_up_reverse_short_ratio = 0.05)
    recs = []
    for lbl, net_exp, l_ratio, s_ratio, l_rew, s_rew, lim_u_rew, lim_d_rew in [
        ("label_0", -0.5, 0.0, 0.6, 0.0, 5.0, 0.0, 5.0),
        ("label_1", -0.5, 0.0, 0.6, 0.0, 5.0, 0.0, 0.0),
        ("label_2", 0.0, 0.1, 0.1, 0.5, 0.5, 0.0, 0.0),
        ("label_3", 0.5, 0.6, 0.0, 5.0, 0.0, 0.0, 0.0),
    ]:
        recs.append(
            _semantic_cross_contract_record(
                lbl, 0, 0, [5.0], [10],
                net_position_exposure=net_exp,
                long_step_ratio=l_ratio,
                short_step_ratio=s_ratio,
                long_reward_sum=l_rew,
                short_reward_sum=s_rew,
                limit_up_long_reward_sum=lim_u_rew,
                limit_down_short_reward_sum=lim_d_rew,
            )
        )
    # label_4 Cand A
    recs.append(
        _semantic_cross_contract_record(
            "label_4", 0, 0, [10.0], [10],
            net_position_exposure=0.5, long_step_ratio=0.5, short_step_ratio=0.4,
            long_reward_sum=10.0, limit_up_step_ratio=0.5, limit_up_long_reward_sum=1.0,
            limit_up_reverse_short_ratio=0.5
        )
    )
    # label_4 Cand B
    recs.append(
        _semantic_cross_contract_record(
            "label_4", 0, 1, [5.0], [10],
            net_position_exposure=0.5, long_step_ratio=0.6, short_step_ratio=0.0,
            long_reward_sum=5.0, limit_up_step_ratio=0.5, limit_up_long_reward_sum=3.0,
            limit_up_reverse_short_ratio=0.05
        )
    )

    df_all = p.transform_single_epoch_result_all(recs, "epoch_1")
    best = p.pick_best_agent_regarding_dynamics_bin_index_path(df_all)

    l4 = best[best["label"] == "label_4"].iloc[0]
    assert l4["bin_index"] == 1


def test_limit_down_label_requires_limit_state_short_profitability(tmp_path):
    p = _picker(tmp_path)
    p.label_semantics_path = _make_label_semantics_file(tmp_path)
    # For label_0 (limit_down):
    # Cand A: high reward, high reverse long ratio (0.5 > 0.2)
    # Cand B: lower reward, short profitable in limit down (limit_down_short_reward_sum = 3.0, limit_down_reverse_long_ratio = 0.05)
    recs = []
    for lbl, net_exp, l_ratio, s_ratio, l_rew, s_rew, lim_u_rew, lim_d_rew in [
        ("label_1", -0.5, 0.0, 0.6, 0.0, 5.0, 0.0, 0.0),
        ("label_2", 0.0, 0.1, 0.1, 0.5, 0.5, 0.0, 0.0),
        ("label_3", 0.5, 0.6, 0.0, 5.0, 0.0, 0.0, 0.0),
        ("label_4", 0.5, 0.6, 0.0, 5.0, 0.0, 5.0, 0.0),
    ]:
        recs.append(
            _semantic_cross_contract_record(
                lbl, 0, 0, [5.0], [10],
                net_position_exposure=net_exp,
                long_step_ratio=l_ratio,
                short_step_ratio=s_ratio,
                long_reward_sum=l_rew,
                short_reward_sum=s_rew,
                limit_up_long_reward_sum=lim_u_rew,
                limit_down_short_reward_sum=lim_d_rew,
            )
        )
    # label_0 Cand A
    recs.append(
        _semantic_cross_contract_record(
            "label_0", 0, 0, [10.0], [10],
            net_position_exposure=-0.5, long_step_ratio=0.4, short_step_ratio=0.5,
            short_reward_sum=10.0, limit_down_step_ratio=0.5, limit_down_short_reward_sum=1.0,
            limit_down_reverse_long_ratio=0.5
        )
    )
    # label_0 Cand B
    recs.append(
        _semantic_cross_contract_record(
            "label_0", 0, 1, [5.0], [10],
            net_position_exposure=-0.5, long_step_ratio=0.0, short_step_ratio=0.6,
            short_reward_sum=5.0, limit_down_step_ratio=0.5, limit_down_short_reward_sum=3.0,
            limit_down_reverse_long_ratio=0.05
        )
    )

    df_all = p.transform_single_epoch_result_all(recs, "epoch_1")
    best = p.pick_best_agent_regarding_dynamics_bin_index_path(df_all)

    l0 = best[best["label"] == "label_0"].iloc[0]
    assert l0["bin_index"] == 1


def test_picker_falls_back_when_no_candidate_matches_strict_semantics(tmp_path):
    p = _picker(tmp_path)
    p.label_semantics_path = _make_label_semantics_file(tmp_path)
    # All candidates for label_3 are bearish, but label 0, 1, 2, 4 pass
    recs = [
        _semantic_cross_contract_record("label_0", 0, 0, [5.0], [10], net_position_exposure=-0.5, short_step_ratio=0.6, short_reward_sum=5.0, limit_down_short_reward_sum=5.0),
        _semantic_cross_contract_record("label_1", 0, 0, [5.0], [10], net_position_exposure=-0.5, short_step_ratio=0.6, short_reward_sum=5.0),
        _semantic_cross_contract_record("label_2", 0, 0, [5.0], [10], net_position_exposure=0.0, long_step_ratio=0.1, short_step_ratio=0.1),
        _semantic_cross_contract_record("label_3", 0, 0, [5.0], [10], net_position_exposure=-0.5, long_step_ratio=0.0, short_step_ratio=0.6, short_reward_sum=5.0),
        _semantic_cross_contract_record("label_4", 0, 0, [5.0], [10], net_position_exposure=0.5, long_step_ratio=0.6, long_reward_sum=5.0, limit_up_long_reward_sum=5.0),
    ]

    df_all = p.transform_single_epoch_result_all(recs, "epoch_1")
    best = p.pick_best_agent_regarding_dynamics_bin_index_path(df_all)
    l3 = best[best["label"] == "label_3"].iloc[0]
    assert l3["behavior_summary"]["selection_note"] == "positive reward model fallback (soft directional semantics)"


def test_picker_rejects_stale_result_lacking_behavior_metrics(tmp_path):
    p = _picker(tmp_path)
    p.label_semantics_path = _make_label_semantics_file(tmp_path)
    stale_rec = [
        {
            "label": "label_0",
            "initial_action": 0,
            "bin_index": 0,
            "contract": ["fu2500"],
            "df_path": ["fu2500/label_0/df_0.feather"],
            "reward_sum": [5.0],
            "df_length": [10],
            "turnover": [0.0],
        }
    ]
    with pytest.raises(ValueError, match="rerun test_agent_index.py"):
        p.transform_single_epoch_result(stale_rec, "epoch_1")
