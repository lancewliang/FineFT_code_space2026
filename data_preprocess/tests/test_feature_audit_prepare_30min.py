import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import polars as pl
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_preprocess.operator_futures.feature_audit.prepare_30min_inputs import prepare_30min_data


def test_prepare_30min_inputs_isolation(tmp_path):
    """Verifies 30min data generation does not touch summary and produces valid split files."""
    work_dir = tmp_path / "prep_work"
    work_dir.mkdir()

    # 1. Mock 10min input
    split_10min_dir = work_dir / "10min" / "fu"
    train_dir = split_10min_dir / "train"
    train_dir.mkdir(parents=True)

    timestamps = [
        "2024-01-02 09:10:00", "2024-01-02 09:20:00", "2024-01-02 09:30:00",
        "2024-01-02 09:40:00", "2024-01-02 09:50:00", "2024-01-02 10:00:00",
        "2024-01-02 10:10:00", "2024-01-02 10:30:00", "2024-01-02 11:00:00",
    ]
    df = pl.DataFrame({
        "contract": ["fu2405"] * len(timestamps),
        "timestamp": timestamps,
        "trading_day": ["20240102"] * len(timestamps),
        "feat_val": [1.0] * len(timestamps),
        "LastPrice": [3000.0] * len(timestamps),
    })
    df.write_ipc(train_dir / "fu2405.feather")
    df.write_ipc(split_10min_dir / "train.feather")

    manifest_10min = {
        "symbol": "fu",
        "target_freq": "10min",
        "sets": {"train": ["fu2405"], "valid": [], "test": []}
    }
    (split_10min_dir / "dataset_split_manifest.json").write_text(json.dumps(manifest_10min))

    # Mock summary file
    summary_file = work_dir / "main_contract_summary.json"
    summary_file.write_text(json.dumps({"dummy": "value"}))
    sum_hash_before = hashlib.sha256(summary_file.read_bytes()).hexdigest()

    # Output 30min dir
    split_30min_dir = work_dir / "30min" / "fu"

    # Run prepare_30min_data
    prepare_30min_data(
        source_10min_dir=split_10min_dir,
        target_30min_dir=split_30min_dir,
        summary_file=summary_file,
    )

    # 1. Summary file untouched
    sum_hash_after = hashlib.sha256(summary_file.read_bytes()).hexdigest()
    assert sum_hash_before == sum_hash_after

    # 2. 30min files generated
    assert (split_30min_dir / "train.feather").exists()
    assert (split_30min_dir / "train" / "fu2405.feather").exists()
    assert (split_30min_dir / "dataset_split_manifest.json").exists()

    # 3. Check downsampled 30min rows
    df_30 = pl.read_ipc(split_30min_dir / "train.feather")
    # Should only retain timestamps aligned to 30min boundaries (09:30, 10:00, 10:30, 11:00)
    assert len(df_30) < len(df)
    assert "09:30:00" in str(df_30["timestamp"].to_list()[0])
