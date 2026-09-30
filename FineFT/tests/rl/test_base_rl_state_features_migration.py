import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch import nn

FINEFT_ROOT = Path(__file__).resolve().parents[2]
if str(FINEFT_ROOT) not in sys.path:
    sys.path.insert(0, str(FINEFT_ROOT))

from common.artifacts import ArtifactNames


BASE_RL_FILES = [
    "cdqn_test.py",
    "cdqn_train.py",
    "dqn_test.py",
    "dqn_train.py",
    "dra_test.py",
    "dra_train.py",
    "ensemble_dqn_test.py",
    "ensemble_dqn_train.py",
    "ncqrdqn_test.py",
    "ncqrdqn_train.py",
    "ppo_test.py",
    "ppo_train.py",
    "rule_imbalance_volume_util.py",
    "rule_macd_util.py",
    "sunrise_dqn_test.py",
    "sunrise_dqn_train.py",
    "winnow_test.py",
]

TARGET_DIRS = [
    FINEFT_ROOT / "RL" / "base",
    FINEFT_ROOT / "env",
    FINEFT_ROOT / "common",
]


def test_target_directories_have_no_bare_state_features_npy():
    bare_state_features_pattern = re.compile(r'(?<![a-zA-Z0-9_])state_features\.npy')
    violations = []

    for target_dir in TARGET_DIRS:
        for ext in ("*.py", "*.sh"):
            for file_path in target_dir.rglob(ext):
                content = file_path.read_text(encoding="utf-8")
                matches = bare_state_features_pattern.findall(content)
                if matches:
                    violations.append(str(file_path))

    assert not violations, f"Legacy state_features.npy found in: {violations}"


def test_base_rl_scripts_use_artifact_names():
    base_dir = FINEFT_ROOT / "RL" / "base"
    for filename in BASE_RL_FILES:
        file_path = base_dir / filename
        assert file_path.exists(), f"Expected file {filename} does not exist"
        content = file_path.read_text(encoding="utf-8")
        assert (
            "ArtifactNames.RL_STATE_FEATURES_NPY" in content
        ), f"{filename} does not reference ArtifactNames.RL_STATE_FEATURES_NPY"


def test_env_initiate_scripts_use_artifact_names():
    for rel_path in [
        "env/env_initiate/simple_initiate.py",
        "env/env_initiate/agg_initiate.py",
        "env/performance_test/agg_env.py",
    ]:
        file_path = FINEFT_ROOT / rel_path
        content = file_path.read_text(encoding="utf-8")
        assert (
            "ArtifactNames.RL_STATE_FEATURES_NPY" in content
        ), f"{rel_path} does not reference ArtifactNames.RL_STATE_FEATURES_NPY"


def test_dqn_trainer_loads_rl_features_and_fails_fast_on_legacy_only(tmp_path):
    from RL.base.dqn_train import DQN

    dataset_name = "mock_symbol"
    ds_dir = tmp_path / dataset_name
    train_dir = ds_dir / "train"
    train_dir.mkdir(parents=True)
    (train_dir / "df_0.feather").touch()

    margin_path = ds_dir / ArtifactNames.MAINTENANCE_MARGIN_RATIO_DICT_NPY
    np.save(margin_path, {"50000": [0.004, 0]})

    legacy_features_path = ds_dir / "state_features.npy"
    np.save(legacy_features_path, np.array(["legacy_f1", "legacy_f2"]))

    args = MagicMock()
    args.base_path = str(tmp_path)
    args.dataset_name = dataset_name
    args.result_path = str(tmp_path / "result")
    args.buffer_size = 1000
    args.n_step = 1
    args.gamma = 0.99
    args.tau = 0.005
    args.batch_size = 32
    args.update_times = 1
    args.epsilon_init = 1.0
    args.epsilon_min = 0.1
    args.epsilon_step = 1000
    args.lr_init = 1e-4
    args.lr_min = 1e-5
    args.lr_step = 1000
    args.num_sample = 1
    args.hidden_nodes = 64
    args.target_freq = 100
    args.device = "cpu"
    args.max_holding_number = 8
    args.position_choices = 9
    args.leverage_choices = [5]
    args.long_estimated_rate = 0.0005
    args.short_estimated_rate = 0.0
    args.transcation_cost = 0.0002
    args.early_stop = 0
    args.initial_wallet_balance = 100000.0
    args.initial_margin = 0.0
    args.initial_unrealized_pnL = 0.0
    args.initial_position = 0.0
    args.initial_leverage = 5

    # Fails fast with FileNotFoundError when rl_state_features.npy is missing,
    # even though legacy state_features.npy exists on disk.
    with pytest.raises(FileNotFoundError):
        DQN(args)

    # When rl_state_features.npy is provided, it loads successfully.
    rl_features_path = ds_dir / ArtifactNames.RL_STATE_FEATURES_NPY
    np.save(rl_features_path, np.array(["rl_f1", "rl_f2", "rl_f3"]))

    trainer = DQN(args)

    assert list(trainer.tech_indicator_list) == ["rl_f1", "rl_f2", "rl_f3"]

@pytest.fixture(autouse=True)
def restore_pythonhashseed():
    import os
    orig = os.environ.get("PYTHONHASHSEED")
    yield
    if orig is not None:
        os.environ["PYTHONHASHSEED"] = orig
    else:
        os.environ.pop("PYTHONHASHSEED", None)
