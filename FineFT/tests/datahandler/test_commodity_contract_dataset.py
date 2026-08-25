import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from datahandler.commodity_contract_dataset import (
    build_dataset_manifest,
    load_dataset_split_manifest,
    run_dataset_generation,
    write_stage_datasets,
    write_train_slices,
)
from datahandler.manifests import (
    DatasetManifest,
    DatasetSkippedContract,
    DatasetSplitManifest,
)


def _dataset_split_manifest(symbol="fu", target_freq="10min"):
    return {
        "symbol": symbol,
        "target_freq": target_freq,
        "sets": {
            "train": {
                "range": ["2026-01-01", "2026-01-06"],
                "contracts": [
                    {
                        "contract": "fu2508",
                        "trading_days": ["2026-01-01", "2026-01-02"],
                        "output_row_count": 4,
                    },
                    {
                        "contract": "fu2509",
                        "trading_days": ["2026-01-03"],
                        "output_row_count": 2,
                    },
                ],
                "skipped_contracts": [],
            },
            "valid": {
                "range": ["2026-01-06", "2026-01-09"],
                "contracts": [
                    {
                        "contract": "fu2508",
                        "trading_days": ["2026-01-06"],
                        "output_row_count": 2,
                    }
                ],
                "skipped_contracts": [
                    {"contract": "fu2509", "reason": "no trading days in valid range"}
                ],
            },
            "test": {
                "range": ["2026-01-09", "2026-01-11"],
                "contracts": [
                    {
                        "contract": "fu2509",
                        "trading_days": ["2026-01-09"],
                        "output_row_count": 2,
                    }
                ],
                "skipped_contracts": [
                    {"contract": "fu2508", "reason": "no trading days in test range"}
                ],
            },
        },
    }


def _write_dataset_split_manifest(path, manifest=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = manifest or _dataset_split_manifest()
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def _write_scale_save_file(root, stage, contract, rows=2):
    output = root / "SCALE_SAVE" / "fu" / "10min" / stage / f"{contract}.feather"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="10min"),
            "symbol": [contract] * rows,
            "feature_a": list(range(rows)),
            "bid1_price": list(range(100, 100 + rows)),
            "mark_price": list(range(200, 200 + rows)),
        }
    ).to_feather(output)
    return output


def _dataset_manifest_from_dict(manifest):
    payload = {
        "symbol": manifest.get("symbol", "fu"),
        "target_freq": manifest.get("target_freq", "10min"),
        "dataset_split_manifest_path": manifest.get(
            "dataset_split_manifest_path",
            "dataset_split_manifest.json",
        ),
        "state_features_source_path": manifest.get(
            "state_features_source_path",
            "",
        ),
        "state_features_path": manifest.get("state_features_path", ""),
        "sets": manifest.get("sets", {}),
    }
    return DatasetManifest.from_dict(payload)


def test_load_dataset_split_manifest_validates_symbol_and_target_freq(tmp_path):
    manifest_path = _write_dataset_split_manifest(tmp_path / "dataset_split_manifest.json")

    manifest = load_dataset_split_manifest(
        manifest_path,
        symbol="fu",
        target_freq="10min",
    )

    assert isinstance(manifest, DatasetSplitManifest)
    assert manifest.symbol == "fu"
    assert manifest.target_freq == "10min"
    assert [item.contract for item in manifest.sets["train"].contracts] == [
        "fu2508",
        "fu2509",
    ]


def test_load_dataset_split_manifest_fails_on_symbol_mismatch(tmp_path):
    manifest_path = _write_dataset_split_manifest(tmp_path / "dataset_split_manifest.json")

    with pytest.raises(ValueError, match="symbol"):
        load_dataset_split_manifest(
            manifest_path,
            symbol="al",
            target_freq="10min",
        )


def test_load_dataset_split_manifest_fails_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="dataset split manifest"):
        load_dataset_split_manifest(
            tmp_path / "dataset_split_manifest.json",
            symbol="fu",
            target_freq="10min",
        )


def test_load_dataset_split_manifest_fails_on_target_freq_mismatch(tmp_path):
    manifest_path = _write_dataset_split_manifest(
        tmp_path / "dataset_split_manifest.json",
        manifest=_dataset_split_manifest(target_freq="5min"),
    )

    with pytest.raises(ValueError, match="target_freq"):
        load_dataset_split_manifest(
            manifest_path,
            symbol="fu",
            target_freq="10min",
        )


def test_load_dataset_split_manifest_fails_when_stage_contracts_missing(tmp_path):
    split_manifest = _dataset_split_manifest()
    del split_manifest["sets"]["train"]["contracts"]
    manifest_path = _write_dataset_split_manifest(
        tmp_path / "dataset_split_manifest.json",
        manifest=split_manifest,
    )

    with pytest.raises(ValueError, match="sets.train.contracts"):
        load_dataset_split_manifest(
            manifest_path,
            symbol="fu",
            target_freq="10min",
        )


def test_build_dataset_manifest_uses_split_manifest_and_stage_scale_save_paths(tmp_path):
    split_manifest_path = _write_dataset_split_manifest(
        tmp_path / "dataset_split_manifest.json",
        manifest=_dataset_split_manifest(),
    )
    split_manifest = load_dataset_split_manifest(
        split_manifest_path,
        symbol="fu",
        target_freq="10min",
    )

    manifest = build_dataset_manifest(
        split_manifest=split_manifest,
        dataset_split_manifest_path=split_manifest_path,
        input_root=tmp_path / "SCALE_SAVE",
        state_features_path=tmp_path / "FEATURE_SELECTION" / "state_features.npy",
        output_root=tmp_path / "dataset" / "10min",
        symbol="fu",
        target_freq="10min",
        chunk_length=2,
        early_stop=1,
    )

    assert isinstance(manifest, DatasetManifest)
    assert manifest.dataset_split_manifest_path.endswith("dataset_split_manifest.json")
    assert manifest.state_features_path.endswith("dataset/10min/fu/state_features.npy")
    train_contracts = {
        item.contract: item for item in manifest.sets["train"].contracts
    }
    assert train_contracts["fu2508"].input_path.endswith(
        "SCALE_SAVE/fu/10min/train/fu2508.feather"
    )
    assert train_contracts["fu2508"].output_path.endswith(
        "dataset/10min/fu/train/fu2508.feather"
    )
    assert train_contracts["fu2508"].slice_outputs[0].path.endswith(
        "dataset/10min/fu/train/slice/df_0.feather"
    )
    split_skipped_contract = split_manifest.sets["valid"].skipped_contracts[0]
    assert isinstance(split_skipped_contract, DatasetSkippedContract)
    assert split_skipped_contract.contract == "fu2509"
    assert split_skipped_contract.reason == "no trading days in valid range"
    skipped_contract = manifest.sets["valid"].skipped_contracts[0]
    assert isinstance(skipped_contract, DatasetSkippedContract)
    assert skipped_contract.contract == "fu2509"
    assert skipped_contract.reason == "no trading days in valid range"
    assert manifest.to_dict()["sets"]["valid"]["skipped_contracts"] == [
        {"contract": "fu2509", "reason": "no trading days in valid range"}
    ]


def test_write_stage_datasets_copies_stage_files_and_state_features(tmp_path):
    train_file = _write_scale_save_file(tmp_path, "train", "fu2508", rows=3)
    valid_file = _write_scale_save_file(tmp_path, "valid", "fu2508", rows=2)
    test_file = _write_scale_save_file(tmp_path, "test", "fu2509", rows=2)
    state_features = tmp_path / "FEATURE_SELECTION" / "10min" / "fu" / "train" / "state_features.npy"
    state_features.parent.mkdir(parents=True)
    np.save(state_features, np.array(["feature_a"]))
    dataset_root = tmp_path / "dataset" / "10min" / "fu"
    manifest = _dataset_manifest_from_dict({
        "symbol": "fu",
        "target_freq": "10min",
        "state_features_source_path": str(state_features),
        "state_features_path": str(dataset_root / "state_features.npy"),
        "sets": {
            "train": {
                "contracts": [
                    {
                        "contract": "fu2508",
                        "input_path": str(train_file),
                        "output_path": str(dataset_root / "train" / "fu2508.feather"),
                    }
                ],
                "skipped_contracts": [],
            },
            "valid": {
                "contracts": [
                    {
                        "contract": "fu2508",
                        "input_path": str(valid_file),
                        "output_path": str(dataset_root / "valid" / "fu2508.feather"),
                    }
                ],
                "skipped_contracts": [],
            },
            "test": {
                "contracts": [
                    {
                        "contract": "fu2509",
                        "input_path": str(test_file),
                        "output_path": str(dataset_root / "test" / "fu2509.feather"),
                    }
                ],
                "skipped_contracts": [],
            },
        },
    })

    write_stage_datasets(manifest)

    assert (dataset_root / "train" / "fu2508.feather").exists()
    assert pd.read_feather(dataset_root / "train" / "fu2508.feather")[
        "feature_a"
    ].tolist() == [0, 1, 2]
    assert np.load(dataset_root / "state_features.npy", allow_pickle=True).tolist() == [
        "feature_a"
    ]
    assert manifest.sets["train"].contracts[0].output_row_count == 3
    assert manifest.sets["valid"].contracts_total_count == 2
    assert manifest.sets["test"].contracts_total_count == 2
    assert not (dataset_root / "train.feather").exists()


def test_write_stage_datasets_fails_when_state_features_missing(tmp_path):
    train_file = _write_scale_save_file(tmp_path, "train", "fu2508", rows=2)
    dataset_root = tmp_path / "dataset" / "10min" / "fu"
    manifest = _dataset_manifest_from_dict({
        "state_features_source_path": str(tmp_path / "missing" / "state_features.npy"),
        "state_features_path": str(dataset_root / "state_features.npy"),
        "sets": {
            "train": {
                "contracts": [
                    {
                        "contract": "fu2508",
                        "input_path": str(train_file),
                        "output_path": str(dataset_root / "train" / "fu2508.feather"),
                    }
                ],
                "skipped_contracts": [],
            }
        },
    })

    with pytest.raises(FileNotFoundError, match="state_features"):
        write_stage_datasets(manifest)


def test_write_stage_datasets_fails_when_scale_save_file_missing(tmp_path):
    state_features = tmp_path / "FEATURE_SELECTION" / "state_features.npy"
    state_features.parent.mkdir(parents=True)
    np.save(state_features, np.array(["feature_a"]))
    dataset_root = tmp_path / "dataset" / "10min" / "fu"
    manifest = _dataset_manifest_from_dict({
        "state_features_source_path": str(state_features),
        "state_features_path": str(dataset_root / "state_features.npy"),
        "sets": {
            "train": {
                "contracts": [
                    {
                        "contract": "fu2508",
                        "input_path": str(tmp_path / "SCALE_SAVE" / "fu" / "10min" / "train" / "fu2508.feather"),
                        "output_path": str(dataset_root / "train" / "fu2508.feather"),
                    }
                ],
                "skipped_contracts": [],
            }
        },
    })

    with pytest.raises(FileNotFoundError, match="fu2508"):
        write_stage_datasets(manifest)


def test_write_stage_datasets_fails_when_state_features_empty(tmp_path):
    train_file = _write_scale_save_file(tmp_path, "train", "fu2508", rows=2)
    state_features = tmp_path / "FEATURE_SELECTION" / "state_features.npy"
    state_features.parent.mkdir(parents=True)
    np.save(state_features, np.array([]))
    dataset_root = tmp_path / "dataset" / "10min" / "fu"
    manifest = _dataset_manifest_from_dict({
        "state_features_source_path": str(state_features),
        "state_features_path": str(dataset_root / "state_features.npy"),
        "sets": {
            "train": {
                "contracts": [
                    {
                        "contract": "fu2508",
                        "input_path": str(train_file),
                        "output_path": str(dataset_root / "train" / "fu2508.feather"),
                    }
                ],
                "skipped_contracts": [],
            }
        },
    })

    with pytest.raises(ValueError, match="state feature"):
        write_stage_datasets(manifest)


def test_write_stage_datasets_fails_when_copied_stage_data_empty(tmp_path):
    train_file = _write_scale_save_file(tmp_path, "train", "fu2508", rows=0)
    state_features = tmp_path / "FEATURE_SELECTION" / "state_features.npy"
    state_features.parent.mkdir(parents=True)
    np.save(state_features, np.array(["feature_a"]))
    dataset_root = tmp_path / "dataset" / "10min" / "fu"
    manifest = _dataset_manifest_from_dict({
        "state_features_source_path": str(state_features),
        "state_features_path": str(dataset_root / "state_features.npy"),
        "sets": {
            "train": {
                "contracts": [
                    {
                        "contract": "fu2508",
                        "input_path": str(train_file),
                        "output_path": str(dataset_root / "train" / "fu2508.feather"),
                    }
                ],
                "skipped_contracts": [],
            }
        },
    })

    with pytest.raises(ValueError, match="empty stage dataset"):
        write_stage_datasets(manifest)


def test_write_train_slices_uses_contiguous_indices_and_single_contract_files(tmp_path):
    train_dir = tmp_path / "dataset" / "10min" / "fu" / "train"
    train_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="D"),
            "symbol": ["fu2508"] * 5,
            "feature_a": range(5),
        }
    ).to_feather(train_dir / "fu2508.feather")
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="D"),
            "symbol": ["fu2509"] * 3,
            "feature_a": range(10, 13),
        }
    ).to_feather(train_dir / "fu2509.feather")
    manifest = _dataset_manifest_from_dict({
        "sets": {
            "train": {
                "contracts": [
                    {
                        "contract": "fu2508",
                        "input_path": str(train_dir / "fu2508.feather"),
                        "output_path": str(train_dir / "fu2508.feather"),
                        "slice_outputs": [
                            {
                                "index": 0,
                                "path": str(train_dir / "slice" / "df_0.feather"),
                                "row_start": 0,
                                "row_end": 3,
                            },
                            {
                                "index": 1,
                                "path": str(train_dir / "slice" / "df_1.feather"),
                                "row_start": 2,
                                "row_end": 5,
                            },
                        ],
                    },
                    {
                        "contract": "fu2509",
                        "input_path": str(train_dir / "fu2509.feather"),
                        "output_path": str(train_dir / "fu2509.feather"),
                        "slice_outputs": [
                            {
                                "index": 2,
                                "path": str(train_dir / "slice" / "df_2.feather"),
                                "row_start": 0,
                                "row_end": 3,
                            }
                        ],
                    },
                ]
            }
        }
    })

    write_train_slices(manifest)

    slice_paths = sorted((train_dir / "slice").glob("df_*.feather"))
    assert [path.name for path in slice_paths] == [
        "df_0.feather",
        "df_1.feather",
        "df_2.feather",
    ]
    assert pd.read_feather(slice_paths[0])["symbol"].unique().tolist() == ["fu2508"]
    assert pd.read_feather(slice_paths[2])["symbol"].unique().tolist() == ["fu2509"]


def test_commodity_data_handler_scripts_use_contract_dataset_tool():
    root = Path(__file__).resolve().parents[3]
    for script_name, symbol in [
        (f"commodity_data_handler_{frequency}_{symbol}.sh", symbol)
        for frequency in ("1min", "5min", "10min", "30min")
        for symbol in ("fu", "al")
    ]:
        text = (root / "FineFT" / "script" / "data" / script_name).read_text()
        assert "commodity_contract_dataset.py" in text
        assert f"--symbol {symbol}" in text or '--symbol "${SYMBOL}"' in text
        assert "--dataset_split_manifest_path" in text
        assert "SPLIT-TRAIN-VALID-TEST" in text
        assert "dataset_split_manifest.json" in text
        assert "--input_root" in text
        assert "PREPROCESS_DATASET/commodity-futures/SCALE_SAVE" in text
        assert "--state_features_path" in text
        assert "FEATURE_SELECTION" in text
        assert "train/state_features.npy" in text
        assert "valid_cross_contract_label_calibration.py" in text
        assert '--valid_dir "dataset/${TARGET_FREQ}/${SYMBOL}/valid"' in text
        assert "--data_path" not in text
        assert "--summary_path" not in text
        assert "--feature_union_path" not in text
        assert "preprocess_data.py --trading_pair" not in text
        assert "valid_cross_contract_label_calibration.py --data_path dataset/" not in text


def test_run_dataset_generation_writes_manifest_stage_files_and_train_slices(tmp_path):
    manifest_path = _write_dataset_split_manifest(tmp_path / "dataset_split_manifest.json")
    _write_scale_save_file(tmp_path, "train", "fu2508", rows=4)
    _write_scale_save_file(tmp_path, "train", "fu2509", rows=2)
    _write_scale_save_file(tmp_path, "valid", "fu2508", rows=2)
    _write_scale_save_file(tmp_path, "test", "fu2509", rows=2)
    state_features = tmp_path / "FEATURE_SELECTION" / "10min" / "fu" / "train" / "state_features.npy"
    state_features.parent.mkdir(parents=True)
    np.save(state_features, np.array(["feature_a"]))

    returned_manifest = run_dataset_generation(
        dataset_split_manifest_path=manifest_path,
        input_root=tmp_path / "SCALE_SAVE",
        state_features_path=state_features,
        output_root=tmp_path / "dataset" / "10min",
        symbol="fu",
        target_freq="10min",
        chunk_length=4,
        early_stop=1,
    )

    dataset_root = tmp_path / "dataset" / "10min" / "fu"
    assert isinstance(returned_manifest, DatasetManifest)
    assert (dataset_root / "dataset_manifest.json").exists()
    assert (dataset_root / "train" / "fu2508.feather").exists()
    assert (dataset_root / "train" / "fu2509.feather").exists()
    assert (dataset_root / "valid" / "fu2508.feather").exists()
    assert (dataset_root / "test" / "fu2509.feather").exists()
    manifest = json.loads((dataset_root / "dataset_manifest.json").read_text())
    assert manifest == returned_manifest.to_dict()
    expected_counts = {"train": 6, "valid": 2, "test": 2}
    for set_name, expected_count in expected_counts.items():
        assert manifest["sets"][set_name]["contracts_total_count"] == expected_count
        for contract in manifest["sets"][set_name]["contracts"]:
            output_path = Path(contract["output_path"])
            assert contract["output_row_count"] == len(pd.read_feather(output_path))
    assert sorted(path.name for path in (dataset_root / "train" / "slice").glob("df_*.feather")) == [
        "df_0.feather",
        "df_1.feather",
    ]
    slice_paths = sorted((dataset_root / "train" / "slice").glob("df_*.feather"))
    slice_row_counts = [len(pd.read_feather(path)) for path in slice_paths]
    assert slice_row_counts == [4, 2]
    train_slices = [
        item
        for contract in manifest["sets"]["train"]["contracts"]
        for item in contract["slice_outputs"]
    ]
    assert [item["output_row_count"] for item in train_slices] == slice_row_counts
    assert not (dataset_root / "valid" / "label_0").exists()
