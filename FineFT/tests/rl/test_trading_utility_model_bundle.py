from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from RL.DiHFT.trading_utility.model_bundle import UtilityModelBundle
from RL.DiHFT.trading_utility.model_training import (
    UtilityModelTrainer,
    UtilityTrainingData,
)
from RL.DiHFT.trading_utility.train import train_from_candidate_csv


def _training_data() -> UtilityTrainingData:
    rows = []
    net = []
    downside = []
    mae = []
    win = []
    contracts = []
    slots = []
    directions = []
    for index in range(80):
        x = (index - 40) / 10.0
        slot = index % 2
        direction = -1 if index % 3 == 0 else 1
        outcome = 12.0 * x + 2.0 * slot - 1.5 * direction
        rows.append([x, float(slot), float(direction)])
        net.append(outcome)
        downside.append(max(0.0, -outcome))
        mae.append(max(0.0, -0.5 * outcome) + 1.0)
        win.append(outcome > 0.0)
        contracts.append(f"fu{index % 4}")
        slots.append(slot)
        directions.append(direction)
    return UtilityTrainingData(
        features=np.asarray(rows, dtype=float),
        feature_names=("momentum", "slot_id", "direction"),
        net_pnl_bps=np.asarray(net),
        downside_loss_bps=np.asarray(downside),
        mae_bps=np.asarray(mae),
        wins=np.asarray(win),
        contracts=tuple(contracts),
        slots=np.asarray(slots),
        directions=np.asarray(directions),
    )


def test_four_head_bundle_roundtrips_through_explicit_manifest(tmp_path: Path) -> None:
    bundle = UtilityModelTrainer.fit(
        _training_data(),
        lambda_cvar=0.5,
        lambda_mae=1.0,
        random_seed=17,
    )
    sample = {"momentum": 1.5, "slot_id": 1.0, "direction": 1.0}

    before = bundle.predict_one(sample)
    manifest_path = bundle.save(tmp_path)
    restored = UtilityModelBundle.load(manifest_path)
    after = restored.predict_one(sample)

    assert np.isfinite(before.mean_net_pnl_bps)
    assert before.cvar_loss_bps >= 0.0
    assert before.mae_loss_bps >= 0.0
    assert 0.0 <= before.win_probability <= 1.0
    assert before.raw_utility_bps == pytest.approx(
        before.mean_net_pnl_bps
        - 0.5 * before.cvar_loss_bps
        - before.mae_loss_bps
    )
    assert after == pytest.approx(before)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["feature_schema"] == ["momentum", "slot_id", "direction"]
    assert set(manifest["heads"]) == {
        "mean_net_pnl",
        "lower_tail",
        "mae",
        "win_probability",
    }
    assert manifest["preferred_backend"] == "catboost"
    assert manifest["artifact_id"]


def test_bundle_loader_rejects_tampered_head(tmp_path: Path) -> None:
    manifest_path = UtilityModelTrainer.fit(_training_data()).save(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mean_asset = tmp_path / manifest["heads"]["mean_net_pnl"]["path"]
    mean_asset.write_bytes(mean_asset.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="hash mismatch"):
        UtilityModelBundle.load(manifest_path)


def test_independent_training_entry_reads_counterfactual_csv(tmp_path: Path) -> None:
    data = _training_data()
    frame = pd.DataFrame(
        {
            "feature_values": [json.dumps(row.tolist()) for row in data.features],
            "net_pnl_bps": data.net_pnl_bps,
            "downside_loss_bps": data.downside_loss_bps,
            "mae_bps": data.mae_bps,
            "win": data.wins,
            "contract": data.contracts,
            "slot_id": data.slots,
            "direction": data.directions,
        }
    )
    source = tmp_path / "candidate_episodes.csv"
    frame.to_csv(source, index=False)

    manifest_path = train_from_candidate_csv(
        episodes_csv=source,
        output_dir=tmp_path / "bundle",
        feature_names=data.feature_names,
        backend="sklearn",
    )

    assert manifest_path.name == "utility_bundle_manifest.json"
    assert UtilityModelBundle.load(manifest_path).feature_names == data.feature_names
