from __future__ import annotations

import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

FINEFT_ROOT = Path(__file__).resolve().parents[2]
if str(FINEFT_ROOT) not in sys.path:
    sys.path.insert(0, str(FINEFT_ROOT))

from common.artifacts import ArtifactNames

TARGET_DIRS = [
    FINEFT_ROOT / "RL" / "DiHFT",
    FINEFT_ROOT / "RL" / "EarnHFT",
    FINEFT_ROOT / "RL" / "SL",
]

MIGRATED_RL_FILES = [
    # EarnHFT
    FINEFT_ROOT / "RL" / "EarnHFT" / "low_level" / "ddqn_pes_risk_aware.py",
    FINEFT_ROOT / "RL" / "EarnHFT" / "low_level" / "test_ddqn.py",
    FINEFT_ROOT / "RL" / "EarnHFT" / "high_level" / "dqn.py",
    FINEFT_ROOT / "RL" / "EarnHFT" / "high_level" / "test_dqn.py",
    # DiHFT low_level
    FINEFT_ROOT / "RL" / "DiHFT" / "low_level" / "test_agent_average.py",
    FINEFT_ROOT / "RL" / "DiHFT" / "low_level" / "trace_udpate_index.py",
    # DiHFT high_level
    FINEFT_ROOT / "RL" / "DiHFT" / "high_level" / "train_high_level.py",
    FINEFT_ROOT / "RL" / "DiHFT" / "high_level" / "train_high_level_seq.py",
    FINEFT_ROOT / "RL" / "DiHFT" / "high_level" / "test_high_level.py",
    FINEFT_ROOT / "RL" / "DiHFT" / "high_level" / "test_single_agent.py",
    # DiHFT ablation
    FINEFT_ROOT / "RL" / "DiHFT" / "ablation" / "safe_routing.py",
    FINEFT_ROOT / "RL" / "DiHFT" / "ablation" / "converge_steps_sun" / "EarnHFT_PES.py",
    FINEFT_ROOT / "RL" / "DiHFT" / "ablation" / "converge_steps_sun" / "EarnHFT_PES_test.py",
    FINEFT_ROOT / "RL" / "DiHFT" / "ablation" / "converge_steps_sun" / "EarnHFT_random.py",
    FINEFT_ROOT / "RL" / "DiHFT" / "ablation" / "converge_steps_sun" / "EarnHFT_random_test.py",
    FINEFT_ROOT / "RL" / "DiHFT" / "ablation" / "converge_steps_sun" / "FineFT.py",
    FINEFT_ROOT / "RL" / "DiHFT" / "ablation" / "converge_steps_sun" / "FineFT_test.py",
    FINEFT_ROOT / "RL" / "DiHFT" / "ablation" / "converge_steps_sun" / "FineFT_without_pretrain.py",
    FINEFT_ROOT / "RL" / "DiHFT" / "ablation" / "converge_steps_sun" / "FineFT_wo_pretrain_test.py",
    # SL
    FINEFT_ROOT / "RL" / "SL" / "test_adaboost.py",
]


def test_target_directories_have_no_bare_state_features_npy():
    bare_state_features_pattern = re.compile(r"(?<![a-zA-Z0-9_])state_features\.npy")
    violations: list[str] = []

    for target_dir in TARGET_DIRS:
        for ext in ("*.py", "*.sh"):
            for file_path in target_dir.rglob(ext):
                content = file_path.read_text(encoding="utf-8")
                matches = bare_state_features_pattern.findall(content)
                if matches:
                    violations.append(str(file_path))

    assert not violations, f"Legacy state_features.npy found in: {violations}"


def test_all_migrated_files_use_rl_artifact_name():
    for file_path in MIGRATED_RL_FILES:
        assert file_path.exists(), f"File does not exist: {file_path}"
        content = file_path.read_text(encoding="utf-8")
        assert (
            "ArtifactNames.RL_STATE_FEATURES_NPY" in content
        ), f"{file_path.name} does not reference ArtifactNames.RL_STATE_FEATURES_NPY"


def test_earnhft_high_level_uses_high_level_artifact_name():
    for file_name in ["dqn.py", "test_dqn.py"]:
        file_path = FINEFT_ROOT / "RL" / "EarnHFT" / "high_level" / file_name
        content = file_path.read_text(encoding="utf-8")
        assert (
            "ArtifactNames.HIGH_LEVEL_STATE_FEATURES_NPY" in content
        ), f"{file_name} does not reference ArtifactNames.HIGH_LEVEL_STATE_FEATURES_NPY"


def test_safe_routing_uses_both_rl_and_vae_artifact_names():
    file_path = FINEFT_ROOT / "RL" / "DiHFT" / "ablation" / "safe_routing.py"
    content = file_path.read_text(encoding="utf-8")
    assert "ArtifactNames.RL_STATE_FEATURES_NPY" in content
    assert "ArtifactNames.VAE_STATE_FEATURES_NPY" in content


def test_earnhft_ddqn_fails_fast_when_rl_features_missing_and_legacy_present(tmp_path):
    from RL.EarnHFT.low_level.ddqn_pes_risk_aware import DQN as ddqn_pes_risk_aware

    dataset_name = "test_ds"
    ds_dir = tmp_path / dataset_name
    train_dir = ds_dir / "train"
    train_dir.mkdir(parents=True)
    (train_dir / "df_0.feather").touch()

    margin_path = ds_dir / ArtifactNames.MAINTENANCE_MARGIN_RATIO_DICT_NPY
    np.save(margin_path, {"50000": [0.004, 0]})

    # Legacy state_features.npy is present
    legacy_features_path = ds_dir / "state_features.npy"
    np.save(legacy_features_path, np.array(["f1", "f2"]))

    args = MagicMock()
    args.base_path = str(tmp_path)
    args.dataset_name = dataset_name
    args.result_path = str(tmp_path / "result")
    args.beta = 0.5
    args.type = "even"
    args.device = "cpu"
    args.gpu_index = 0
    args.buffer_size = 100
    args.n_step = 1
    args.gamma = 0.99
    args.tau = 0.005
    args.batch_size = 16
    args.update_times = 1
    args.epsilon_init = 1.0
    args.epsilon_min = 0.1
    args.epsilon_step = 100
    args.lr_init = 1e-4
    args.lr_min = 1e-5
    args.lr_step = 100
    args.hidden_nodes = 32
    args.target_freq = 50
    args.max_holding_number = 4
    args.position_choices = 5
    args.leverage_choices = [5]
    args.long_estimated_rate = 0.0005
    args.short_estimated_rate = 0.0
    args.transcation_cost = 0.0002
    args.num_sample = 1
    args.early_stop = 0
    args.initial_wallet_balance = 100000.0
    args.initial_margin = 0.0
    args.initial_unrealized_pnL = 0.0
    args.initial_position = 0.0
    args.initial_leverage = 5

    # Should fail fast with FileNotFoundError because rl_state_features.npy is missing
    with pytest.raises(FileNotFoundError):
        ddqn_pes_risk_aware(args)

    # When rl_state_features.npy is provided, initialization proceeds past feature loading
    np.save(ds_dir / ArtifactNames.RL_STATE_FEATURES_NPY, np.array(["f1", "f2", "f3"]))
    agent = ddqn_pes_risk_aware(args)
    assert list(agent.tech_indicator_list) == ["f1", "f2", "f3"]


def test_safe_routing_fails_fast_on_missing_rl_or_vae_features(tmp_path):
    from RL.DiHFT.ablation.safe_routing import vae_risk_aware_routing

    dataset_name = "test_symbol"
    ds_dir = tmp_path / dataset_name
    ds_dir.mkdir(parents=True)

    margin_path = ds_dir / ArtifactNames.MAINTENANCE_MARGIN_RATIO_DICT_NPY
    np.save(margin_path, {"50000": [0.004, 0]})

    legacy_features_path = ds_dir / "state_features.npy"
    np.save(legacy_features_path, np.array(["legacy_1", "legacy_2"]))

    args = MagicMock()
    args.base_path = str(tmp_path)
    args.dataset_name = dataset_name
    args.device = "cpu"
    args.gpu_index = 0
    args.max_holding_number = 4
    args.position_choices = 5
    args.leverage_choices = [5]
    args.long_estimated_rate = 0.0005
    args.short_estimated_rate = 0.0
    args.transcation_cost = 0.0002
    args.early_stop = 0
    args.initial_wallet_balance = 100000.0
    args.initial_state = 0
    args.initial_position = 0
    args.initial_leverage = 5
    args.window_length = 5
    args.low_level_path = str(tmp_path / "low_level")
    args.hidden_nodes = 32
    args.time_info_dim = 1
    args.N = 2
    args.action_type = "continuous"
    args.rank_path = str(tmp_path / "rank")
    args.TCN_channels = [16, 16]
    args.high_level_seq_length = 4
    args.seed = 42
    args.gamma = 0.99
    args.vae_path = str(tmp_path / "vae")
    args.vae_hidden_dims = [32, 16]
    args.z_dim = 4
    args.loss_type = "NLL"

    # 1. Missing rl_state_features.npy fails fast
    with pytest.raises(FileNotFoundError):
        vae_risk_aware_routing(args)

    # 2. Add rl_state_features.npy, but missing vae_state_features.npy fails fast
    np.save(ds_dir / ArtifactNames.RL_STATE_FEATURES_NPY, np.array(["rl_f1", "rl_f2"]))
    with pytest.raises(FileNotFoundError):
        vae_risk_aware_routing(args)

    # 3. Add vae_state_features.npy
    np.save(ds_dir / ArtifactNames.VAE_STATE_FEATURES_NPY, np.array(["vae_f1"]))
    
    args.label_number = 1
    # Mock network loading so __init__ completes
    with patch("torch.load", return_value={}):
        with patch("RL.DiHFT.ablation.safe_routing.MLP_VAE.load_state_dict"):
            with patch("RL.DiHFT.ablation.safe_routing.ensemble_Qnet.load_state_dict"):
                label_dir = tmp_path / "vae" / dataset_name / "label_0"
                label_dir.mkdir(parents=True)
                (label_dir / "model_latest.pth").touch()
                np.save(label_dir / "id_logpx.npy", np.array([0.1, 0.2]))
                
                agent = vae_risk_aware_routing(args)
                assert list(agent.tech_indicator_list) == ["rl_f1", "rl_f2"]
                assert list(agent.vae_indicator_list) == ["vae_f1"]
