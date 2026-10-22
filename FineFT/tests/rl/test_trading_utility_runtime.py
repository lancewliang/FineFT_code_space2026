from __future__ import annotations

import json
import hashlib

import numpy as np
import pytest

from RL.DiHFT.trading_utility.eligibility import (
    ChannelEligibility,
    EligibilityPolicy,
    EligibilityReport,
)
from RL.DiHFT.trading_utility.model_bundle import UtilityModelBundle
from RL.DiHFT.trading_utility.model_training import (
    UtilityModelTrainer,
    UtilityTrainingData,
)
from RL.DiHFT.trading_utility.runtime_artifacts import publish_runtime_manifest
from RL.DiHFT.high_level.trading_utility.runtime import (
    load_utility_runtime,
)


class _IdentityCalibrator:
    def predict_utility(self, values):
        return np.asarray(values, dtype=float)

    def predict_win_probability(self, values):
        return np.asarray(values, dtype=float)


def _runtime_manifest(tmp_path):
    values = np.linspace(-2.0, 2.0, 24)
    data = UtilityTrainingData(
        features=values.reshape(-1, 1),
        feature_names=("signal",),
        net_pnl_bps=values * 2.0,
        downside_loss_bps=np.maximum(0.0, -values * 2.0),
        mae_bps=np.abs(values),
        wins=values > 0.0,
        contracts=tuple(f"fu{index % 3}" for index in range(len(values))),
        slots=np.zeros(len(values), dtype=int),
        directions=np.ones(len(values), dtype=int),
    )
    bundle = UtilityModelTrainer.fit(data, backend="sklearn", random_seed=7)
    bundle_manifest = bundle.save(tmp_path / "bundle")
    channel = ChannelEligibility(
        slot_id=0,
        direction=1,
        eligible=False,
        failure_reasons=("insufficient_episodes",),
        episode_count=24,
        contract_count=3,
        trading_day_block_count=8,
        quintile_counts=(5, 5, 5, 5, 4),
        win_rates=(0.1, 0.2, 0.3, 0.4, 0.5),
        payoff_ratios=(1.0, 1.1, 1.2, 1.3, 1.4),
        win_rate_spearman=1.0,
        payoff_ratio_spearman=1.0,
        mean_net_pnl_bps=0.0,
        mean_net_pnl_lcb95_bps=-1.0,
        raw_p_value=1.0,
        fdr_q_value=1.0,
        win_rate_trend_lcb95=-1.0,
        payoff_ratio_trend_lcb95=-1.0,
        residual_q05=-2.0,
    )
    eligibility_manifest = EligibilityReport(
        channels={"0:1": channel}, policy=EligibilityPolicy(bootstrap_iterations=20)
    ).save(tmp_path / "eligibility")
    frozen_protocol = {
        "horizon": 3,
        "ood_enter_threshold": 0.005,
        "ood_recover_threshold": 0.01,
    }
    protocol = {
        "protocol_fingerprint": hashlib.sha256(
            json.dumps(frozen_protocol, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "frozen_protocol": frozen_protocol,
        "outer_fold_count": 2,
    }
    return publish_runtime_manifest(
        output_dir=tmp_path / "runtime",
        bundle_manifest=bundle_manifest,
        calibrator=_IdentityCalibrator(),
        eligibility_manifest=eligibility_manifest,
        protocol_manifest=protocol,
    )


def test_runtime_manifest_loads_only_explicit_verified_assets(tmp_path) -> None:
    manifest = _runtime_manifest(tmp_path)

    runtime = load_utility_runtime(manifest)

    assert runtime.protocol_fingerprint
    assert runtime.feature_schema == ("signal",)
    assert runtime.horizon == 3
    assert runtime.artifact_id


def test_runtime_manifest_rejects_asset_damage_and_protocol_mismatch(tmp_path) -> None:
    manifest = _runtime_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["protocol_fingerprint"] = "changed-after-freeze"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="protocol fingerprint mismatch"):
        load_utility_runtime(manifest)
