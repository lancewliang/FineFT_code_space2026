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


def test_final_selection_keeps_current_result_all_initial_action_mean(tmp_path):
    p = _picker(tmp_path)
    rows = []
    for initial_action, score in [(0, 1.0), (1, 3.0)]:
        rows.append(
            {
                "label": "label_0",
                "bin_index": 0,
                "epoch_path": "epoch_1",
                "initial_action": initial_action,
                "trans_reward_mean": score,
            }
        )
    for initial_action, score in [(0, 1.9), (1, 1.9)]:
        rows.append(
            {
                "label": "label_0",
                "bin_index": 1,
                "epoch_path": "epoch_2",
                "initial_action": initial_action,
                "trans_reward_mean": score,
            }
        )
    for label_index in range(1, 5):
        rows.append(
            {
                "label": f"label_{label_index}",
                "bin_index": label_index,
                "epoch_path": f"epoch_{label_index}",
                "initial_action": 0,
                "trans_reward_mean": 0.1,
            }
        )

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
    rows = [
        {
            "label": f"label_{label_index}",
            "bin_index": label_index,
            "epoch_path": "epoch_1",
            "trans_reward_mean": 0.2,
        }
        for label_index in range(4)
    ]
    rows.append(
        {
            "label": "label_4",
            "bin_index": 2,
            "epoch_path": "epoch_1",
            "trans_reward_mean": np.nan,
        }
    )

    with pytest.raises(ValueError, match="label_4"):
        _picker(tmp_path).pick_best_agent_regarding_dynamics_bin_index_path(
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
