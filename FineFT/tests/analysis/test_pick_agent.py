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
