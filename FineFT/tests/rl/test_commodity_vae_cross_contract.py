import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


FINEFT_ROOT = Path(__file__).resolve().parents[2]
VAE_ROOT = FINEFT_ROOT / "RL" / "DiHFT" / "VAE"
if str(FINEFT_ROOT) not in sys.path:
    sys.path.insert(0, str(FINEFT_ROOT))
if str(VAE_ROOT) not in sys.path:
    sys.path.insert(0, str(VAE_ROOT))

from RL.DiHFT.VAE import main as vae_main
from RL.DiHFT.VAE import merge_vae_train
from RL.DiHFT.VAE import process as vae_process
from RL.DiHFT.VAE import summary as vae_summary
from RL.DiHFT.VAE.manifests import (
    ContractDatasetLoader,
    ContractLogpxResult,
    LabelArraySource,
    LabelSummary,
    LabelTrainingManifest,
    RoutingSummary,
    TestContractSource,
    TrainBaselineLogpx,
)


def _dataset_root(tmp_path):
    return tmp_path / "dataset" / "10min"


def _vae_dir(tmp_path):
    return _dataset_root(tmp_path) / "fu" / "VAE_data"


def _save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(data, dtype=float))


def test_materialize_label_training_data_merges_contract_arrays_and_writes_manifest(tmp_path):
    vae_dir = _vae_dir(tmp_path)
    _save(vae_dir / "fu2505" / "label_0.npy", [[1.0, 2.0], [3.0, 4.0]])
    _save(vae_dir / "fu2509" / "label_0.npy", [[5.0, 6.0]])
    (vae_dir / "fu2510").mkdir(parents=True)
    (vae_dir / "test").mkdir()

    result = merge_vae_train.materialize_label_training_data(
        data_base_path=str(_dataset_root(tmp_path)),
        dataset_name="fu",
        label_index=0,
    )

    merged = np.load(vae_dir / "train" / "label_0.npy")
    np.testing.assert_array_equal(
        merged,
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    )
    assert isinstance(result, LabelTrainingManifest)
    assert result.merged_path == str(vae_dir / "train" / "label_0.npy")
    assert result.total_samples == 3
    assert result.feature_dim == 2
    assert [item.contract for item in result.included_contracts] == [
        "fu2505",
        "fu2509",
    ]
    assert result.included_contracts[0].sample_count == 2
    assert result.included_contracts[1].sample_count == 1
    assert result.missing_contracts == ["fu2510"]

    manifest = json.loads((vae_dir / "train" / "label_0_manifest.json").read_text())
    assert manifest == result.to_dict()
    assert manifest["dataset_name"] == "fu"
    assert manifest["label"] == "label_0"
    assert manifest["included_contracts"][0]["source_file"].endswith(
        "fu2505/label_0.npy"
    )


def test_materialize_label_training_data_fails_when_no_contract_has_label(tmp_path):
    vae_dir = _vae_dir(tmp_path)
    (vae_dir / "fu2505").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="label_4"):
        merge_vae_train.materialize_label_training_data(
            data_base_path=str(_dataset_root(tmp_path)),
            dataset_name="fu",
            label_index=4,
        )

    assert not (vae_dir / "train" / "label_4.npy").exists()


def test_materialize_label_training_data_rejects_non_2d_arrays(tmp_path):
    vae_dir = _vae_dir(tmp_path)
    _save(vae_dir / "fu2505" / "label_0.npy", [1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="two-dimensional"):
        merge_vae_train.materialize_label_training_data(
            data_base_path=str(_dataset_root(tmp_path)),
            dataset_name="fu",
            label_index=0,
        )


def test_materialize_label_training_data_rejects_feature_dim_mismatch(tmp_path):
    vae_dir = _vae_dir(tmp_path)
    _save(vae_dir / "fu2505" / "label_0.npy", [[1.0, 2.0]])
    _save(vae_dir / "fu2509" / "label_0.npy", [[3.0, 4.0, 5.0]])

    with pytest.raises(ValueError, match="feature dimension"):
        merge_vae_train.materialize_label_training_data(
            data_base_path=str(_dataset_root(tmp_path)),
            dataset_name="fu",
            label_index=0,
        )


def test_discover_label_sources_reads_contract_label_arrays_as_objects(tmp_path):
    vae_dir = _vae_dir(tmp_path)
    _save(vae_dir / "fu2505" / "label_0.npy", [[1.0, 2.0]])
    _save(vae_dir / "fu2509" / "label_0.npy", [[3.0, 4.0]])
    (vae_dir / "fu2510").mkdir(parents=True)

    sources, missing_contracts = merge_vae_train.discover_label_sources(
        data_base_path=str(_dataset_root(tmp_path)),
        dataset_name="fu",
        label_index=0,
    )

    assert all(isinstance(source, LabelArraySource) for source in sources)
    assert [source.contract for source in sources] == ["fu2505", "fu2509"]
    assert sources[0].source_file.endswith("fu2505/label_0.npy")
    assert missing_contracts == ["fu2510"]


def test_discover_test_sources_reads_contract_test_arrays(tmp_path):
    vae_dir = _vae_dir(tmp_path)
    _save(vae_dir / "test" / "test_fu2508.npy", [[1.0, 2.0]])
    _save(vae_dir / "test" / "test_fu2509.npy", [[3.0, 4.0]])

    sources = vae_main.discover_test_sources(
        data_base_path=str(_dataset_root(tmp_path)),
        dataset_name="fu",
    )

    assert all(isinstance(source, TestContractSource) for source in sources)
    assert [source.contract for source in sources] == ["fu2508", "fu2509"]
    assert sources[0].source_file.endswith("test_fu2508.npy")


def test_discover_test_sources_fails_when_no_test_arrays(tmp_path):
    (_vae_dir(tmp_path) / "test").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="no test_.*\\.npy"):
        vae_main.discover_test_sources(
            data_base_path=str(_dataset_root(tmp_path)),
            dataset_name="fu",
        )


def test_prepare_contract_dataset_loader_list_wraps_sources_as_objects(tmp_path):
    vae_dir = _vae_dir(tmp_path)
    source_file = vae_dir / "test" / "test_fu2508.npy"
    _save(source_file, [[1.0, 2.0], [3.0, 4.0]])
    sources = [TestContractSource(contract="fu2508", source_file=str(source_file))]

    loaders = vae_process.prepare_contract_dataset_loader_list(
        sources,
        expected_feature_dim=2,
    )

    assert len(loaders) == 1
    assert isinstance(loaders[0], ContractDatasetLoader)
    assert loaders[0].contract == "fu2508"
    assert loaders[0].source_file == str(source_file)
    assert len(loaders[0].loader.dataset) == 2


def test_prepare_contract_dataset_loader_list_rejects_feature_dim_mismatch(tmp_path):
    vae_dir = _vae_dir(tmp_path)
    source_file = vae_dir / "test" / "test_fu2508.npy"
    _save(source_file, [[1.0, 2.0, 3.0]])
    sources = [TestContractSource(contract="fu2508", source_file=str(source_file))]

    with pytest.raises(ValueError, match="feature dimension"):
        vae_process.prepare_contract_dataset_loader_list(
            sources,
            expected_feature_dim=2,
        )


def test_write_contract_logpx_outputs_writes_per_contract_and_aggregate_files(tmp_path):
    save_path = tmp_path / "result" / "DiHFT" / "vae_results" / "fu" / "label_0"
    results = [
        ContractLogpxResult(
            contract="fu2508",
            source_file="dataset/10min/fu/VAE_data/test/test_fu2508.npy",
            logpx=np.array([-1.0, -2.0]),
        ),
        ContractLogpxResult(
            contract="fu2509",
            source_file="dataset/10min/fu/VAE_data/test/test_fu2509.npy",
            logpx=np.array([-3.0]),
        ),
    ]

    summary = vae_summary.write_contract_logpx_outputs(
        results,
        save_path=str(save_path),
        dataset_name="fu",
        label="label_0",
    )

    np.testing.assert_array_equal(
        np.load(save_path / "ood_logpx_fu2508.npy"),
        np.array([-1.0, -2.0]),
    )
    np.testing.assert_array_equal(
        np.load(save_path / "ood_logpx_all.npy"),
        np.array([-1.0, -2.0, -3.0]),
    )
    per_contract_csv = pd.read_csv(save_path / "ood_logpx_fu2508.csv")
    assert per_contract_csv.to_dict("records") == [
        {
            "contract": "fu2508",
            "source_file": "dataset/10min/fu/VAE_data/test/test_fu2508.npy",
            "row_index": 0,
            "logpx": -1.0,
        },
        {
            "contract": "fu2508",
            "source_file": "dataset/10min/fu/VAE_data/test/test_fu2508.npy",
            "row_index": 1,
            "logpx": -2.0,
        },
    ]
    all_csv = pd.read_csv(save_path / "ood_logpx_all.csv")
    assert all_csv["contract"].tolist() == ["fu2508", "fu2508", "fu2509"]
    assert isinstance(summary, LabelSummary)
    summary_file = json.loads((save_path / "summary.json").read_text())
    assert summary_file == summary.to_dict()
    assert summary.dataset_name == "fu"
    assert summary.label == "label_0"
    assert summary.test.contracts["fu2508"].summary.stats.samples == 2
    assert summary.test.all.stats.samples == 3
    assert "roc_auc" not in json.dumps(summary_file).lower()


def test_write_contract_logpx_outputs_includes_enhanced_summary_metrics(tmp_path):
    save_path = tmp_path / "result" / "DiHFT" / "vae_results" / "fu" / "label_0"
    train_baseline = TrainBaselineLogpx(
        source_file="dataset/10min/fu/VAE_data/train/label_0.npy",
        input_samples=4,
        analyzed_samples=4,
        logpx=np.array([-10.0, -8.0, -6.0, -4.0]),
    )
    results = [
        ContractLogpxResult(
            contract="fu2508",
            source_file="dataset/10min/fu/VAE_data/test/test_fu2508.npy",
            input_samples=3,
            logpx=np.array([-9.0, -7.0]),
        ),
        ContractLogpxResult(
            contract="fu2509",
            source_file="dataset/10min/fu/VAE_data/test/test_fu2509.npy",
            input_samples=2,
            logpx=np.array([-5.0, -3.0]),
        ),
    ]

    summary = vae_summary.write_contract_logpx_outputs(
        results,
        save_path=str(save_path),
        dataset_name="fu",
        label="label_0",
        train_baseline=train_baseline,
    )

    assert isinstance(summary, LabelSummary)
    summary_file = json.loads((save_path / "summary.json").read_text())
    assert summary_file == summary.to_dict()
    assert summary.train_baseline is not None
    assert summary.train_baseline.source_file.endswith("label_0.npy")
    assert summary.train_baseline.summary.integrity.input_samples == 4
    assert summary.train_baseline.summary.integrity.analyzed_samples == 4
    assert summary.train_baseline.summary.integrity.sample_mismatch is False
    assert set(summary.train_baseline.summary.stats.quantiles) == {
        "q01",
        "q05",
        "q25",
        "q50",
        "q75",
        "q95",
        "q99",
    }
    fu2508 = summary.test.contracts["fu2508"]
    assert fu2508.summary.integrity.input_samples == 3
    assert fu2508.summary.integrity.analyzed_samples == 2
    assert fu2508.summary.integrity.sample_mismatch is True
    assert fu2508.summary.stats.samples == 2
    assert set(fu2508.summary.stats.quantiles) == set(
        summary.train_baseline.summary.stats.quantiles
    )
    assert fu2508.summary.acceptance is not None
    assert set(fu2508.summary.acceptance.to_dict()) == {
        "ge_train_q01_pct",
        "ge_train_q05_pct",
        "ge_train_q50_pct",
    }
    assert summary.test.all.integrity.analyzed_samples == 4
    assert "roc_auc" not in json.dumps(summary_file).lower()
    assert "accuracy" not in json.dumps(summary_file).lower()


def test_parser_accepts_explicit_train_and_analyze_only_flags():
    train_args = vae_main.parser.parse_args(
        [
            "--dataset_name",
            "fu",
            "--data_base_path",
            "dataset/10min",
            "--label_index",
            "0",
            "--train",
        ]
    )
    analyze_args = vae_main.parser.parse_args(
        [
            "--dataset_name",
            "fu",
            "--data_base_path",
            "dataset/10min",
            "--label_index",
            "0",
            "--analyze-only",
        ]
    )

    assert train_args.train is True
    assert train_args.analyze_only is False
    assert analyze_args.train is False
    assert analyze_args.analyze_only is True


def test_parser_rejects_routing_summary_workflow_flag():
    with pytest.raises(SystemExit):
        vae_main.parser.parse_args(
            [
                "--dataset_name",
                "fu",
                "--data_base_path",
                "dataset/10min",
                "--routing-summary",
            ]
        )


def test_write_routing_summary_compares_labels_by_contract(tmp_path):
    result_root = tmp_path / "result" / "DiHFT" / "vae_results" / "fu"
    for label, values in {
        "label_0": {
            "fu2508": [-1.0, -5.0, -2.0],
            "fu2509": [-3.0, -2.0],
        },
        "label_1": {
            "fu2508": [-2.0, -4.0, -3.0],
            "fu2509": [-2.5, -5.0],
        },
        "label_2": {
            "fu2508": [-4.0, -3.0],
            "fu2509": [-1.0, -4.0],
        },
    }.items():
        label_dir = result_root / label
        label_dir.mkdir(parents=True)
        for contract, logpx in values.items():
            np.save(label_dir / f"ood_logpx_{contract}.npy", np.asarray(logpx))

    summary = vae_summary.write_routing_summary(
        result_root=str(result_root),
        dataset_name="fu",
        labels=["label_0", "label_1", "label_2"],
        low_margin_threshold=1.0,
    )

    assert isinstance(summary, RoutingSummary)
    summary_file = json.loads((result_root / "routing_summary.json").read_text())
    assert summary_file == summary.to_dict()
    assert summary.dataset_name == "fu"
    assert summary.labels == ["label_0", "label_1", "label_2"]
    assert summary.score_type == "raw_logpx"
    assert summary.contracts["fu2508"].winner.samples == 2
    assert summary.contracts["fu2508"].input_samples_by_label["label_0"] == 3
    assert summary.contracts["fu2508"].sample_mismatch is True
    assert summary.contracts["fu2508"].winner.winner_counts == {
        "label_0": 1,
        "label_1": 0,
        "label_2": 1,
    }
    assert summary.all.winner_counts == {
        "label_0": 2,
        "label_1": 0,
        "label_2": 2,
    }
    assert (result_root / "routing_summary.json").exists()


def test_main_writes_routing_summary_after_analysis_when_all_labels_ready(tmp_path):
    vae_dir = _vae_dir(tmp_path)
    _save(vae_dir / "test" / "test_fu2508.npy", [[1.0, 2.0]])
    _save(vae_dir / "test" / "test_fu2509.npy", [[3.0, 4.0]])
    result_root = tmp_path / "result" / "DiHFT" / "vae_results" / "fu"
    for label, values in {
        "label_0": {
            "fu2508": [-1.0],
            "fu2509": [-3.0],
        },
        "label_1": {
            "fu2508": [-2.0],
            "fu2509": [-2.5],
        },
    }.items():
        label_dir = result_root / label
        label_dir.mkdir(parents=True)
        for contract, logpx in values.items():
            np.save(label_dir / f"ood_logpx_{contract}.npy", np.asarray(logpx))
    args = vae_main.parser.parse_args(
        [
            "--dataset_name",
            "fu",
            "--data_base_path",
            str(_dataset_root(tmp_path)),
            "--base_model_path",
            str(tmp_path / "result" / "DiHFT"),
            "--total_label_number",
            "2",
            "--analyze-only",
        ]
    )

    summary = vae_summary.maybe_write_routing_summary_after_analysis(args)

    assert isinstance(summary, RoutingSummary)
    assert summary.dataset_name == "fu"
    assert summary.all.winner_counts == {"label_0": 1, "label_1": 1}
    summary_file = json.loads((result_root / "routing_summary.json").read_text())
    assert summary_file == summary.to_dict()
    assert (result_root / "routing_summary.json").exists()


def test_fu_vae_shell_limits_parallel_training_jobs(tmp_path):
    script_path = (
        FINEFT_ROOT
        / "script"
        / "train"
        / "DiHFT"
        / "low_level"
        / "VAE_util_fu.sh"
    )
    assert "--routing-summary" not in script_path.read_text(encoding="utf-8")
    bin_dir = tmp_path / "bin"
    conda_base = tmp_path / "conda"
    state_dir = tmp_path / "state"
    bin_dir.mkdir()
    (conda_base / "etc" / "profile.d").mkdir(parents=True)
    state_dir.mkdir()
    (conda_base / "etc" / "profile.d" / "conda.sh").write_text(
        "conda() { :; }\n",
        encoding="utf-8",
    )
    (bin_dir / "conda").write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" == \"info\" && \"$2\" == \"--base\" ]]; then\n"
        f"  echo \"{conda_base}\"\n"
        "else\n"
        "  exit 0\n"
        "fi\n",
        encoding="utf-8",
    )
    (bin_dir / "python").write_text(
        "#!/usr/bin/env bash\n"
        "state_dir=${FAKE_VAE_STATE_DIR:?}\n"
        "lock_file=\"${state_dir}/lock\"\n"
        "current_file=\"${state_dir}/current\"\n"
        "max_file=\"${state_dir}/max\"\n"
        "mkdir -p \"${state_dir}\"\n"
        "touch \"${current_file}\" \"${max_file}\" \"${lock_file}\"\n"
        "inc=$(\n"
        "  {\n"
        "    flock -x 9\n"
        "    current=$(cat \"${current_file}\")\n"
        "    current=${current:-0}\n"
        "    current=$((current + 1))\n"
        "    echo \"${current}\" >\"${current_file}\"\n"
        "    max=$(cat \"${max_file}\")\n"
        "    max=${max:-0}\n"
        "    if ((current > max)); then echo \"${current}\" >\"${max_file}\"; fi\n"
        "    echo \"${current}\"\n"
        "  } 9>\"${lock_file}\"\n"
        ")\n"
        "if ((inc > 2)); then\n"
        "  echo \"parallel limit exceeded: ${inc}\" >&2\n"
        "  exit 42\n"
        "fi\n"
        "sleep 0.1\n"
        "{\n"
        "  flock -x 9\n"
        "  current=$(cat \"${current_file}\")\n"
        "  current=${current:-1}\n"
        "  echo $((current - 1)) >\"${current_file}\"\n"
        "} 9>\"${lock_file}\"\n",
        encoding="utf-8",
    )
    os.chmod(bin_dir / "conda", 0o755)
    os.chmod(bin_dir / "python", 0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["ROOTPATH"] = str(FINEFT_ROOT.parent)
    env["LABEL_COUNT"] = "5"
    env["MAX_PARALLEL_JOBS"] = "2"
    env["FAKE_VAE_STATE_DIR"] = str(state_dir)

    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=FINEFT_ROOT.parent,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (state_dir / "max").read_text().strip() == "2"


def test_fu_vae_shell_rejects_invalid_parallel_job_limit(tmp_path):
    script_path = (
        FINEFT_ROOT
        / "script"
        / "train"
        / "DiHFT"
        / "low_level"
        / "VAE_util_fu.sh"
    )
    env = os.environ.copy()
    env["ROOTPATH"] = str(FINEFT_ROOT.parent)
    env["MAX_PARALLEL_JOBS"] = "0"

    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=FINEFT_ROOT.parent,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=3,
    )

    assert result.returncode != 0
    assert "MAX_PARALLEL_JOBS must be a positive integer" in result.stderr


def test_fu_vae_shell_returns_failure_when_any_training_job_fails(tmp_path):
    script_path = (
        FINEFT_ROOT
        / "script"
        / "train"
        / "DiHFT"
        / "low_level"
        / "VAE_util_fu.sh"
    )
    bin_dir = tmp_path / "bin"
    conda_base = tmp_path / "conda"
    bin_dir.mkdir()
    (conda_base / "etc" / "profile.d").mkdir(parents=True)
    (conda_base / "etc" / "profile.d" / "conda.sh").write_text(
        "conda() { :; }\n",
        encoding="utf-8",
    )
    (bin_dir / "conda").write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" == \"info\" && \"$2\" == \"--base\" ]]; then\n"
        f"  echo \"{conda_base}\"\n"
        "else\n"
        "  exit 0\n"
        "fi\n",
        encoding="utf-8",
    )
    (bin_dir / "python").write_text(
        "#!/usr/bin/env bash\n"
        "label_index=\"\"\n"
        "while (($# > 0)); do\n"
        "  if [[ \"$1\" == \"--label_index\" ]]; then\n"
        "    label_index=\"$2\"\n"
        "    break\n"
        "  fi\n"
        "  shift\n"
        "done\n"
        "if [[ \"${label_index}\" == \"1\" ]]; then exit 7; fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    os.chmod(bin_dir / "conda", 0o755)
    os.chmod(bin_dir / "python", 0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["ROOTPATH"] = str(FINEFT_ROOT.parent)
    env["LABEL_COUNT"] = "3"
    env["MAX_PARALLEL_JOBS"] = "2"

    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=FINEFT_ROOT.parent,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "finished with failures" in result.stdout
