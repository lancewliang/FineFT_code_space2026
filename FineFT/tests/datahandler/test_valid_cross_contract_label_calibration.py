import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def _write_contract(valid_dir, contract, base_price, rows=96, price_scale=1.0):
    increments = np.resize(
        np.array([1.0] * 8 + [-1.0] * 8 + [2.0] * 8 + [-2.0] * 8),
        rows,
    )
    prices = price_scale * (base_price + np.cumsum(increments))
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="10min"),
            "symbol": [contract] * rows,
            "bid1_price": prices,
            "state_feature": np.arange(rows, dtype=float),
        }
    ).to_feather(valid_dir / f"{contract}.feather")


def test_build_valid_dataset_publishes_one_shared_threshold_set(tmp_path):
    from datahandler import valid_cross_contract_label_calibration as calibration

    valid_dir = tmp_path / "valid"
    valid_dir.mkdir()
    _write_contract(valid_dir, "fu2501", 100.0)
    _write_contract(valid_dir, "fu2505", 1000.0)

    calibration.build_valid_dataset(
        valid_dir,
        dynamic_number=3,
        timestamp="timestamp",
        min_length_limit=4,
        merging_threshold=-1,
    )

    manifest = json.loads(
        (valid_dir / "slope" / "slice_manifest.json").read_text()
    )
    assert manifest["calibration"]["fit_scope"] == "valid_all_contracts"
    assert manifest["calibration"]["final_segment_count"] > 0
    assert manifest["calibration"]["shared_thresholds"] == manifest["calibration"][
        "thresholds"
    ]
    assert sorted(manifest["contracts"]) == ["fu2501", "fu2505"]
    assert sorted(manifest["labels"]) == ["label_0", "label_1", "label_2"]
    assert sorted(
        path.name for path in (valid_dir / "slope" / "fu2501").iterdir()
    ) == ["label_0", "label_1", "label_2"]
    for contract in ("fu2501", "fu2505"):
        contract_manifest = manifest["contracts"][contract]
        assert contract_manifest["input_row_count"] == 96
        assert contract_manifest["total_row_count"] == 96
    assert all(
        "contract" in file_info
        for label_info in manifest["labels"].values()
        for file_info in label_info["files"]
    )


def test_build_valid_dataset_keeps_slope_and_volatility_outputs_separate(tmp_path):
    from datahandler import valid_cross_contract_label_calibration as calibration

    valid_dir = tmp_path / "valid"
    valid_dir.mkdir()
    _write_contract(valid_dir, "fu2501", 100.0)
    _write_contract(valid_dir, "fu2505", 1000.0)

    calibration.build_valid_dataset(
        valid_dir,
        dynamic_number=3,
        timestamp="timestamp",
        min_length_limit=4,
        merging_threshold=-1,
    )
    slope_manifest_path = valid_dir / "slope" / "slice_manifest.json"
    slope_manifest_before = slope_manifest_path.read_bytes()

    calibration.build_valid_dataset(
        valid_dir,
        dynamic_number=3,
        labeling_method="volatility",
        timestamp="timestamp",
        min_length_limit=4,
        merging_threshold=-1,
    )

    volatility_manifest = json.loads(
        (valid_dir / "volatility" / "slice_manifest.json").read_text()
    )
    assert slope_manifest_path.read_bytes() == slope_manifest_before
    assert volatility_manifest["valid_path"] == str(valid_dir / "volatility")
    assert volatility_manifest["calibration"]["labeling_method"] == "volatility"
    assert volatility_manifest["calibration"]["threshold_method"] == (
        "global_segment_quantile"
    )
    assert volatility_manifest["calibration"]["score_method"] == (
        "segment_log_return_volatility"
    )
    assert volatility_manifest["calibration"]["volatility_return_type"] == "log"
    assert volatility_manifest["calibration"]["volatility_annualized"] is False
    assert volatility_manifest["calibration"]["volatility_ddof"] == 0
    assert volatility_manifest["calibration"]["volatility_unit"] == "percent"
    assert volatility_manifest["calibration"]["volatility_statistics"]["min"] >= 0
    assert sorted(volatility_manifest["contracts"]["fu2501"]["labels"]) == [
        "label_0",
        "label_1",
        "label_2",
    ]
    assert all(
        path.name.startswith("label_")
        for path in (valid_dir / "volatility" / "fu2501").iterdir()
    )


def test_build_valid_dataset_publishes_global_segment_quantile_thresholds(
    tmp_path,
):
    from datahandler import valid_cross_contract_label_calibration as calibration

    valid_dir = tmp_path / "valid"
    valid_dir.mkdir()
    _write_contract(valid_dir, "fu2501", 100.0)
    _write_contract(valid_dir, "fu2505", 1000.0)

    calibration.build_valid_dataset(
        valid_dir,
        dynamic_number=4,
        threshold_method="global_segment_quantile",
        timestamp="timestamp",
        min_length_limit=4,
        merging_threshold=-1,
    )

    manifest = json.loads(
        (valid_dir / "slope" / "slice_manifest.json").read_text()
    )
    np.testing.assert_allclose(
        manifest["calibration"]["shared_thresholds"],
        [-0.381047258209, -0.000075363309, 0.392236945801],
    )
    assert manifest["calibration"]["threshold_method"] == (
        "global_segment_quantile"
    )
    assert manifest["calibration"]["threshold_weighting"] == "equal_segment"
    assert manifest["calibration"]["threshold_quantiles"] == [0.25, 0.5, 0.75]
    assert manifest["calibration"]["per_label_segment_count"] == {
        "label_0": 6,
        "label_1": 6,
        "label_2": 6,
        "label_3": 6,
    }
    assert manifest["calibration"]["per_label_row_count"] == {
        "label_0": 48,
        "label_1": 48,
        "label_2": 48,
        "label_3": 48,
    }
    assert manifest["calibration"]["per_label_contract_coverage"] == {
        "label_0": 1,
        "label_1": 1,
        "label_2": 1,
        "label_3": 1,
    }
    assert sorted(manifest["labels"]) == [
        "label_0",
        "label_1",
        "label_2",
        "label_3",
    ]


def test_cli_accepts_global_segment_quantile_threshold_method(tmp_path):
    from datahandler import valid_cross_contract_label_calibration as calibration

    valid_dir = tmp_path / "valid"
    valid_dir.mkdir()
    _write_contract(valid_dir, "fu2501", 100.0)
    _write_contract(valid_dir, "fu2505", 1000.0)

    subprocess.run(
        [
            sys.executable,
            calibration.__file__,
            "--valid_dir",
            str(valid_dir),
            "--dynamic_number",
            "4",
            "--threshold_method",
            "global_segment_quantile",
            "--timestamp",
            "timestamp",
            "--min_length_limit",
            "4",
            "--merging_threshold",
            "-1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = json.loads(
        (valid_dir / "slope" / "slice_manifest.json").read_text()
    )
    assert manifest["calibration"]["dynamic_number"] == 4
    assert manifest["calibration"]["threshold_method"] == (
        "global_segment_quantile"
    )


def test_cli_builds_volatility_labels_with_scale_invariant_scores(tmp_path):
    from datahandler import valid_cross_contract_label_calibration as calibration

    first_dir = tmp_path / "first" / "valid"
    second_dir = tmp_path / "second" / "valid"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)
    _write_contract(first_dir, "fu2501", 100.0)
    _write_contract(second_dir, "fu2501", 100.0, price_scale=10.0)

    for valid_dir in (first_dir, second_dir):
        subprocess.run(
            [
                sys.executable,
                calibration.__file__,
                "--valid_dir",
                str(valid_dir),
                "--dynamic_number",
                "3",
                "--labeling_method",
                "volatility",
                "--timestamp",
                "timestamp",
                "--min_length_limit",
                "4",
                "--merging_threshold",
                "-1",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    first_manifest = json.loads(
        (first_dir / "volatility" / "slice_manifest.json").read_text()
    )
    second_manifest = json.loads(
        (second_dir / "volatility" / "slice_manifest.json").read_text()
    )
    np.testing.assert_allclose(
        first_manifest["calibration"]["shared_thresholds"],
        second_manifest["calibration"]["shared_thresholds"],
    )
    np.testing.assert_allclose(
        list(first_manifest["calibration"]["volatility_statistics"].values()),
        list(second_manifest["calibration"]["volatility_statistics"].values()),
    )
    assert first_manifest["calibration"]["threshold_quantiles"] == [1 / 3, 2 / 3]


def test_build_valid_dataset_preserves_previous_generation_on_input_failure(tmp_path):
    from datahandler import valid_cross_contract_label_calibration as calibration

    valid_dir = tmp_path / "valid"
    valid_dir.mkdir()
    _write_contract(valid_dir, "fu2501", 100.0)
    calibration.build_valid_dataset(
        valid_dir,
        dynamic_number=3,
        timestamp="timestamp",
        min_length_limit=4,
        merging_threshold=-1,
    )
    previous_manifest = (valid_dir / "slope" / "slice_manifest.json").read_bytes()
    previous_output = (
        valid_dir / "slope" / "fu2501" / "label_0" / "df_0.feather"
    ).read_bytes()

    pd.DataFrame({"symbol": ["fu2505"], "timestamp": [0]}).to_feather(
        valid_dir / "fu2505.feather"
    )
    try:
        calibration.build_valid_dataset(valid_dir, timestamp="timestamp")
    except ValueError as exc:
        assert "missing required columns" in str(exc)
    else:
        raise AssertionError("malformed input should fail the complete build")

    assert (
        valid_dir / "slope" / "slice_manifest.json"
    ).read_bytes() == previous_manifest
    assert (
        valid_dir / "slope" / "fu2501" / "label_0" / "df_0.feather"
    ).read_bytes() == previous_output


def test_build_valid_dataset_records_skipped_contract_and_all_empty_labels(tmp_path):
    from datahandler import valid_cross_contract_label_calibration as calibration

    valid_dir = tmp_path / "valid"
    valid_dir.mkdir()
    _write_contract(valid_dir, "fu2501", 100.0)
    _write_contract(valid_dir, "fu2505", 200.0, rows=2)

    calibration.build_valid_dataset(
        valid_dir,
        dynamic_number=3,
        timestamp="timestamp",
        min_length_limit=4,
        merging_threshold=-1,
    )

    manifest = json.loads(
        (valid_dir / "slope" / "slice_manifest.json").read_text()
    )
    assert sorted(manifest["contracts"]) == ["fu2501"]
    assert manifest["skipped_contracts"]["fu2505"]["input_row_count"] == 2
    assert sorted(manifest["contracts"]["fu2501"]["labels"]) == [
        "label_0",
        "label_1",
        "label_2",
    ]
    assert manifest["contracts"]["fu2501"]["labels"]["label_2"][
        "file_count"
    ] == 0
    assert not list(
        (valid_dir / "slope" / "fu2501" / "label_2").glob("*.feather")
    )


def test_build_valid_dataset_preserves_labels_under_price_unit_scaling(tmp_path):
    from datahandler import valid_cross_contract_label_calibration as calibration

    first_dir = tmp_path / "first" / "valid"
    second_dir = tmp_path / "second" / "valid"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)
    _write_contract(first_dir, "fu2501", 100.0)
    _write_contract(second_dir, "fu2501", 100.0, price_scale=10.0)

    first_manifest = calibration.build_valid_dataset(
        first_dir,
        dynamic_number=3,
        timestamp="timestamp",
        min_length_limit=4,
        merging_threshold=-1,
    )
    second_manifest = calibration.build_valid_dataset(
        second_dir,
        dynamic_number=3,
        timestamp="timestamp",
        min_length_limit=4,
        merging_threshold=-1,
    )

    def labels_by_timestamp(valid_dir):
        result = {}
        for label_dir in sorted((valid_dir / "slope" / "fu2501").glob("label_*")):
            label = int(label_dir.name.split("_")[1])
            for path in label_dir.glob("*.feather"):
                for timestamp in pd.read_feather(path)["timestamp"]:
                    result[timestamp] = label
        return result

    np.testing.assert_allclose(
        first_manifest.calibration["shared_thresholds"],
        second_manifest.calibration["shared_thresholds"],
    )
    assert labels_by_timestamp(first_dir) == labels_by_timestamp(second_dir)


def test_build_valid_dataset_rejects_unsupported_final_labeling_before_publish(
    tmp_path,
):
    from datahandler import valid_cross_contract_label_calibration as calibration

    valid_dir = tmp_path / "valid"
    valid_dir.mkdir()
    (valid_dir / "slice_manifest.json").write_text("previous", encoding="utf-8")

    try:
        calibration.build_valid_dataset(valid_dir, labeling_method="quantile")
    except ValueError as exc:
        assert "supports final slope or volatility labeling only" in str(exc)
    else:
        raise AssertionError("quantile final labeling should be rejected")
    assert (valid_dir / "slice_manifest.json").read_text() == "previous"


def test_build_valid_dataset_rolls_back_when_publication_fails(tmp_path, monkeypatch):
    from datahandler import valid_cross_contract_label_calibration as calibration

    valid_dir = tmp_path / "valid"
    valid_dir.mkdir()
    _write_contract(valid_dir, "fu2501", 100.0)
    _write_contract(valid_dir, "fu2505", 1000.0)
    calibration.build_valid_dataset(
        valid_dir,
        dynamic_number=3,
        timestamp="timestamp",
        min_length_limit=4,
        merging_threshold=-1,
    )
    previous_manifest = (valid_dir / "slope" / "slice_manifest.json").read_bytes()
    previous_contracts = sorted(path.name for path in valid_dir.iterdir())

    original_move = calibration.shutil.move

    def fail_contract_publication(source, destination):
        if Path(source).name == "fu2501" and ".valid-cross-contract-staging-" in str(
            source
        ):
            raise OSError("injected publication failure")
        return original_move(source, destination)

    monkeypatch.setattr(calibration.shutil, "move", fail_contract_publication)
    with pytest.raises(OSError, match="injected publication failure"):
        calibration.build_valid_dataset(
            valid_dir,
            dynamic_number=3,
            timestamp="timestamp",
            min_length_limit=4,
            merging_threshold=-1,
        )

    assert (
        valid_dir / "slope" / "slice_manifest.json"
    ).read_bytes() == previous_manifest
    assert sorted(path.name for path in valid_dir.iterdir()) == previous_contracts
