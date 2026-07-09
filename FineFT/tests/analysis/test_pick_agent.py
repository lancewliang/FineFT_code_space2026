import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd


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


def test_pick_best_agent_skips_label_with_only_nan_rewards(tmp_path):
    result_all = pd.DataFrame(
        [
            {
                "label": "label_0",
                "bin_index": 1,
                "epoch_path": "epoch_1",
                "trans_reward_mean": 0.2,
            },
            {
                "label": "label_4",
                "bin_index": 2,
                "epoch_path": "epoch_1",
                "trans_reward_mean": np.nan,
            },
        ]
    )

    best_agent_info = _picker(tmp_path).pick_best_agent_regarding_dynamics_bin_index_path(
        result_all
    )

    assert best_agent_info["label"].tolist() == ["label_0"]
    assert best_agent_info["bin_index"].tolist() == [1]
    assert best_agent_info["reward_max"].tolist() == [0.2]
