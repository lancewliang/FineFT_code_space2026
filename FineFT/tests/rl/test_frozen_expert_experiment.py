"""Formal preparation round-trips complete policy and independently sourced scores."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from RL.DiHFT.high_level import frozen_experiment as frozen
from RL.DiHFT.high_level import vae_routing_final_result_macro_action as final
from RL.DiHFT.high_level import vae_routing_optuna as tuning
from RL.DiHFT.high_level import vae_routing_util as vru
from RL.DiHFT.VAE.vae import MLP_VAE
from tests.rl.test_expert_hysteresis_routing import make_router
from tests.rl.test_vae_routing_action_persistence import _sample_manifest_payload


def descriptor(path):
    return {"path": str(path.resolve()), "sha256": frozen.file_sha256(path)}


@pytest.fixture
def experiment(tmp_path):
    dataset = tmp_path / "dataset" / "fu"
    selection_root = tmp_path / "selection_data"
    frames = {}
    for role, directory, date in (
        ("fit", dataset / "train", "2020-01-02"),
        ("selection", selection_root, "2020-01-03"),
        ("calibration", dataset / "valid", "2020-02-02"),
        ("final", dataset / "test", "2020-03-02"),
    ):
        directory.mkdir(parents=True)
        path = directory / "fu2005.feather"
        pd.DataFrame(
            {
                "timestamp": pd.date_range(date, periods=4, freq="10min", tz="UTC"),
                "trading_day": [date] * 4,
                "feature": [0.1, 0.2, 0.3, 0.4],
            }
        ).to_feather(path)
        frames[role] = path
    for field in (
        "rl_state_features",
        "vae_slope_state_features",
        "vae_volatility_state_features",
        "maintenance_margin_ratio_dict",
    ):
        np.save(dataset / f"{field}.npy", np.array(["feature"]))
    model = tmp_path / "specialists.pth"
    model.write_bytes(b"controlled specialist identity")
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            _sample_manifest_payload(
                schema_version=1,
                axes={axis: [f"label_{i}" for i in range(3)] for axis in ("slope", "volatility")},
                slot_count=9,
                slot_index_formula="volatility_index * num_labels + slope_index",
                slots=[{"slot_id": i, "kind": "model"} for i in range(9)],
                valid_root=str(selection_root),
                artifacts={"model_assembly": str(model)},
            )
        )
    )
    for axis in ("slope", "volatility"):
        for label in range(3):
            root = tmp_path / "vae" / "fu" / "fixture" / axis / f"label_{label}"
            root.mkdir(parents=True)
            model = MLP_VAE(INPUT_DIM=1, Z_DIM=2, hidden_dims=[8, 4], loss_func="NLL")
            torch.save(model.state_dict(), root / "model_latest.pth")
            np.save(root / "id_logpx.npy", np.array([-2.0, -1.0, 0.0]))
    args = vru.parser.parse_args(
        [
            "--base_path",
            str(tmp_path / "dataset"),
            "--dataset_name",
            "fu",
            "--experiment_name",
            "fixture",
            "--selection_manifest",
            str(selection),
            "--vae_path",
            str(tmp_path / "vae"),
            "--gating_strategy",
            "expert_hysteresis",
            "--slope_window_length",
            "2",
            "--volatility_window_length",
            "3",
            "--slope_gamma",
            "0.9",
            "--volatility_gamma",
            "0.8",
            "--slope_rule_base_threshold",
            "0.6",
            "--volatility_rule_base_threshold",
            "0.8",
            "--hysteresis_exit_ratio",
            "0.5",
            "--expert_switch_lead",
            "0.15",
            "--expert_confirm_steps",
            "2",
            "--eval_stage",
            "valid",
            "--z_dim",
            "2",
        ]
    )
    args.vae_hidden_dims = [8, 4]
    evidence = tmp_path / "split_evidence.json"
    evidence.write_text(json.dumps({"split_version": "fixture.corrected.v1"}))
    periods = {}
    for role, start, end in (
        ("fit", "2020-01-01", "2020-01-10"),
        ("selection", "2020-01-01", "2020-01-11"),
        ("calibration", "2020-02-01", "2020-02-10"),
        ("final", "2020-03-01", "2020-03-10"),
    ):
        periods[role] = {
            "start": start + "T00:00:00Z",
            "end": end + "T00:00:00Z",
            "source_evidence": descriptor(evidence),
        }
    inputs = frozen.describe_experiment_inputs(args)
    document = {
        "schema_version": 1,
        "experiment_id": "controlled-fixture.v1",
        "strategy_version": "expert_hysteresis.v1",
        "split_version": "fixture.corrected.v1",
        "trading_day_atomic": True,
        "models_retrained_for_split": True,
        "periods": periods,
        "scoring": {
            "protocol_version": "seeded_vae_quantiles.v1",
            "seed": 12345,
            "batch_size": 8,
            "column_order": inputs["column_order"],
            "architecture": {
                key: getattr(args, key) for key in ("z_dim", "vae_hidden_dims", "loss_type")
            },
            **{key: getattr(args, key) for key in frozen.SCORING_FIELDS},
        },
        "artifacts": inputs["artifacts"],
    }
    manifest = tmp_path / "experiment.json"
    manifest.write_text(json.dumps(document))
    args.experiment_manifest = str(manifest)
    return args, document, frames


def save_manifest(args, document):
    Path(args.experiment_manifest).write_text(json.dumps(document))


def test_public_tuning_preparation_and_final_snapshot_roundtrip(experiment, tmp_path, monkeypatch):
    args, _, _ = experiment
    search = tuning.parser_all.parse_args(
        [
            "--dataset_name",
            "fu",
            "--experiment_name",
            "fixture",
            "--selection_manifest",
            args.selection_manifest,
            "--gating_strategy",
            "expert_hysteresis",
            "--hysteresis_exit_ratio",
            "0.5",
        ]
    )
    prepared = tuning.prepare_base_args(args, search)
    assert prepared.slope_hold_threshold == pytest.approx(0.3)
    assert prepared.volatility_hold_threshold == pytest.approx(0.4)
    assert prepared.strategy_version == "expert_hysteresis.v1"
    selected = frozen.export_frozen_experiment(prepared, tmp_path / "selected.json")
    # Auto-discovered legacy files must never override a frozen selected policy.
    monkeypatch.chdir(tmp_path)
    legacy = Path("result/DiHFT/final_result/fu/fixture/high_level_agent_para.txt")
    legacy.parent.mkdir(parents=True)
    legacy.write_text("strat_hierarchical_ws_999_wv_999\n")
    restored = final.parse_and_prepare_args(["--frozen_config", selected])
    for key in frozen.POLICY_FIELDS + frozen.SCORING_FIELDS + frozen.RUNTIME_FIELDS:
        assert getattr(restored, key) == getattr(prepared, key), key
    assert restored.eval_stage == "test"
    assert restored.frozen_experiment_identity == prepared.frozen_experiment_identity
    assert restored.frozen_configuration_identity == prepared.frozen_configuration_identity
    assert restored.para_file is None
    base_reloaded = vru.resolve_routing_parameters(
        vru.parser.parse_args(["--frozen_config", selected, "--eval_stage", "test"])
    )
    assert base_reloaded.gating_strategy == "expert_hysteresis"
    assert base_reloaded.expert_confirm_steps == prepared.expert_confirm_steps


@pytest.mark.parametrize(
    "flags",
    [
        ["--expert_confirm_steps", "9"],
        ["--slope_gamma", "0.6"],
        ["--gating_strategy", "absolute"],
        ["--initial_wallet_balance", "1"],
        ["--transcation_cost", "0.1"],
        ["--strategy_version", "future.v2"],
        ["--expert_confirm", "9"],
        ["--slope_gamma=0.6"],
    ],
)
def test_final_rejects_frozen_override(experiment, tmp_path, flags):
    args, _, _ = experiment
    selected = frozen.export_frozen_experiment(args, tmp_path / "selected.json")
    with pytest.raises(ValueError, match="override frozen"):
        final.parse_and_prepare_args(["--frozen_config", selected, *flags])
    with pytest.raises(ValueError, match="override frozen"):
        vru.resolve_routing_parameters(
            vru.parser.parse_args(
                [
                    "--frozen_config",
                    selected,
                    "--eval_stage",
                    "test",
                    *flags,
                ]
            )
        )


def test_final_requires_complete_frozen_config_and_rejects_legacy_mix(experiment, tmp_path):
    args, _, _ = experiment
    with pytest.raises(ValueError, match="requires frozen_config"):
        final.parse_and_prepare_args(["--gating_strategy", "expert_hysteresis"])
    selected = frozen.export_frozen_experiment(args, tmp_path / "selected.json")
    with pytest.raises(ValueError, match="legacy"):
        final.parse_and_prepare_args(["--frozen_config", selected, "--optuna_csv", "old.csv"])
    document = json.loads(Path(selected).read_text())
    document["configuration"].pop("expert_confirm_steps")
    Path(selected).write_text(json.dumps(document))
    with pytest.raises(ValueError, match="identity"):
        final.parse_and_prepare_args(["--frozen_config", selected])


@pytest.mark.parametrize(
    "change,match",
    [
        ("overlap_selection", "overlaps independent"),
        ("overlap_final", "overlaps final"),
        ("atomic", "TradingDay-atomic"),
        ("retrained", "fitted for"),
        ("version", "strategy_version"),
        ("score", "fixed scoring"),
        ("model", "hash mismatch"),
        ("missing", "identity set"),
        ("day", "TradingDay crosses"),
        ("outside", "outside declared"),
        ("absent_day", "evidence missing"),
    ],
)
def test_formal_prepare_rejects_missing_or_contaminated_evidence(experiment, change, match):
    args, document, frames = experiment
    if change == "overlap_selection":
        document["periods"]["selection"]["end"] = "2020-02-03T00:00:00Z"
    elif change == "overlap_final":
        document["periods"]["final"]["start"] = "2020-02-05T00:00:00Z"
    elif change == "atomic":
        document["trading_day_atomic"] = False
    elif change == "retrained":
        document["models_retrained_for_split"] = False
    elif change == "version":
        document["strategy_version"] = "expert_hysteresis.v2"
    elif change == "score":
        args.slope_gamma = 0.7
    elif change == "model":
        Path(document["artifacts"]["slope.label_0.model"]["path"]).write_bytes(b"changed")
    elif change == "missing":
        document["artifacts"].pop("slope.label_0.reference")
    else:
        path = frames["calibration"]
        frame = pd.read_feather(path)
        if change == "day":
            frame["trading_day"] = "2020-01-02"
        elif change == "outside":
            frame["timestamp"] = pd.Timestamp("2020-01-02", tz="UTC")
        else:
            frame = frame.drop(columns=["trading_day"])
        frame.to_feather(path)
        document["artifacts"]["data.calibration.fu2005"] = descriptor(path)
    save_manifest(args, document)
    with pytest.raises(ValueError, match=match):
        frozen.prepare_experiment(args)


def test_missing_manifest_and_legacy_parameters_do_not_claim_independence(experiment):
    args, _, _ = experiment
    args.experiment_manifest = None
    with pytest.raises(ValueError, match="experiment_manifest"):
        frozen.prepare_experiment(args)
    args.para_file = "legacy.txt"
    with pytest.raises(ValueError, match="legacy"):
        vru.resolve_routing_parameters(args)


def test_selected_snapshot_detects_source_mutation_and_cannot_be_replaced(experiment, tmp_path):
    args, document, _ = experiment
    selected = frozen.export_frozen_experiment(args, tmp_path / "selected.json")
    args.expert_confirm_steps = 3
    with pytest.raises(ValueError, match="already frozen"):
        frozen.export_frozen_experiment(args, selected)
    document["experiment_id"] = "another-experiment"
    save_manifest(args, document)
    with pytest.raises(ValueError, match="source identity changed"):
        final.parse_and_prepare_args(["--frozen_config", selected])


def test_invalid_policy_fails_at_public_resolver(experiment):
    args, _, _ = experiment
    for field, value in (
        ("expert_confirm_steps", 0),
        ("hysteresis_exit_ratio", 1.0),
        ("expert_switch_lead", float("nan")),
        ("slope_rule_base_threshold", 0),
    ):
        invalid = copy.deepcopy(args)
        setattr(invalid, field, value)
        with pytest.raises(ValueError):
            vru.resolve_routing_parameters(invalid)


def test_quantile_cache_requires_matching_provenance_and_has_no_route_state(experiment, tmp_path):
    args, _, _ = experiment
    frozen.prepare_experiment(args)
    directory = tmp_path / "cache"
    directory.mkdir()
    values = np.array([[0.1, 0.2, 0.6, 0.1, 0.8, 0.2]] * 4, dtype=np.float32)
    np.save(directory / "fu2005.npy", values)
    with pytest.raises(ValueError, match="provenance"):
        frozen.validate_quantile_cache(args, directory)
    frozen.write_quantile_cache_manifest(args, directory)
    assert frozen.validate_quantile_cache(args, directory) == str(directory)
    metadata = json.loads((directory / "quantile_cache_manifest.json").read_text())
    assert set(metadata) == {
        "schema_version",
        "experiment_identity",
        "scoring",
        "eval_stage",
        "contracts",
    }
    scores = np.load(directory / "fu2005.npy", mmap_mode="r")
    assert not scores.flags.writeable
    changed = values.copy()
    changed[0, 0] = 0.3
    np.save(directory / "fu2005.npy", changed)
    with pytest.raises(ValueError, match="provenance"):
        frozen.validate_quantile_cache(args, directory)


def test_frozen_identical_scores_via_cache_or_protocol_have_identical_public_routes(
    tmp_path, monkeypatch
):
    values = np.array([[0.1, 0.2, 0.6, 0.1, 0.8, 0.2]] * 4, dtype=np.float32)
    path = tmp_path / "scores.npy"
    np.save(path, values)
    cached = np.load(path, mmap_mode="r")
    first, second = make_router(), make_router()
    for router in (first, second):
        del router.calculate_axis_window_result
        router.gamma = 0.9
        router.window_length = 2
        router.axis_gammas = {"slope": 0.9, "volatility": 0.9}
        router.reset_routing_state()
    first.current_contract_quantiles = cached
    second.current_contract_quantiles = None
    second.vae_models = {axis: list(range(3)) for axis in ("slope", "volatility")}
    second.in_ds_logpx = {
        axis: [np.array([0.0]) for _ in range(3)] for axis in ("slope", "volatility")
    }
    monkeypatch.setattr(vru, "analyze_single_sample", lambda *args: (None, 0.0))
    second.device = "cpu"
    sequence = iter(values.flatten())
    second.find_quantile = lambda value, reference: next(sequence)
    # get_quantiles traverses slope labels then volatility labels, exactly the cache order.
    for step in range(2):
        first.step_idx = second.step_idx = step
        first.get_quantiles(np.array([0.0]), np.array([0.0]))
        second.get_quantiles(np.array([0.0]), np.array([0.0]))
    outputs = []
    for router in (first, second):
        outputs.append(
            [
                router.get_action(
                    {"previous_action": 1, "avaliable_action": [1, 1, 1], "trading_info": [0.0]},
                    np.array([step]),
                    0.0,
                    1,
                )
                for step in range(2)
            ]
        )
    assert outputs[0] == outputs[1] == [1, 2]
    assert first.audit_predictions == second.audit_predictions
    # Interleaving a third run starts a separate K-step streak on the same read-only cache.
    third = make_router()
    del third.calculate_axis_window_result
    third.gamma = 0.9
    third.window_length = 2
    third.axis_gammas = {"slope": 0.9, "volatility": 0.9}
    third.reset_routing_state()
    third.current_contract_quantiles = cached
    third.step_idx = 0
    for step in range(2):
        third.step_idx = step
        third.get_quantiles(np.array([0.0]), np.array([0.0]))
    assert (
        third.get_action(
            {"previous_action": 1, "avaliable_action": [1, 1, 1], "trading_info": [0.0]},
            np.array([0.0]),
            0.0,
            1,
        )
        == 1
    )
    assert third.audit_predictions == []


def test_public_final_replay_preserves_snapshot_and_writes_receipt(
    experiment, tmp_path, monkeypatch
):
    from types import SimpleNamespace

    args, _, _ = experiment
    selected = frozen.export_frozen_experiment(args, tmp_path / "selected.json")
    output = tmp_path / "final-output"
    cache = tmp_path / "final-cache"
    observed = []

    def controlled_scores(final_args):
        cache.mkdir()
        np.save(cache / "fu2005.npy", np.full((4, 6), 0.7, dtype=np.float32))
        frozen.write_quantile_cache_manifest(final_args, cache)
        return str(cache)

    class ControlledReplay:
        def __init__(self, final_args):
            self.args = final_args
            self.test_path = str(output)

        def test(self):
            observed.append({key: getattr(self.args, key) for key in frozen.POLICY_FIELDS})
            return 0.123

    monkeypatch.setattr(final, "ensure_precomputed_vae_quantiles", controlled_scores)
    monkeypatch.setattr(final, "vae_risk_aware_routing", ControlledReplay)
    for module, name in (
        ("analysis.diagnostics.trading_diagnostics", "TradingDiagnosticsCalculator"),
        (
            "analysis.diagnostics.synchronized_portfolio_diagnostics",
            "SynchronizedPortfolioCalculator",
        ),
    ):
        monkeypatch.setattr(f"{module}.{name}", lambda **kwargs: SimpleNamespace(run=lambda: None))
    monkeypatch.setattr("sys.argv", ["replay", "--frozen_config", selected])
    assert final.main() == 0.123
    assert observed == [{key: getattr(args, key) for key in frozen.POLICY_FIELDS}]
    receipt = json.loads((output / "frozen_evaluation_receipt.json").read_text())
    assert receipt["configuration_identity"] == args.frozen_configuration_identity
    assert receipt["experiment_identity"] == args.frozen_experiment_identity
    assert receipt["eval_stage"] == "test"
    assert (output / "frozen_config.json").read_bytes() == Path(selected).read_bytes()
    restored = final.parse_and_prepare_args(["--frozen_config", selected])
    restored.precomputed_quantiles_dir = str(cache)
    restored.expert_confirm_steps += 1
    with pytest.raises(ValueError, match="effective configuration changed"):
        frozen.write_evaluation_receipt(restored, output)


def test_frozen_scoring_protocol_generates_reproducible_compatible_cache(
    experiment, tmp_path, monkeypatch
):
    args, _, _ = experiment
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    args.precomputed_quantiles_dir = str(tmp_path / "first-scores")
    first = vru.ensure_precomputed_vae_quantiles(args)
    other = copy.deepcopy(args)
    other.precomputed_quantiles_dir = str(tmp_path / "second-scores")
    second = vru.ensure_precomputed_vae_quantiles(other)
    np.testing.assert_array_equal(
        np.load(Path(first) / "fu2005.npy"), np.load(Path(second) / "fu2005.npy")
    )
    assert json.loads((Path(first) / "quantile_cache_manifest.json").read_text()) == json.loads(
        (Path(second) / "quantile_cache_manifest.json").read_text()
    )
    assert vru.ensure_precomputed_vae_quantiles(args) == first
    incompatible = json.loads((Path(first) / "quantile_cache_manifest.json").read_text())
    incompatible["scoring"]["seed"] += 1
    (Path(first) / "quantile_cache_manifest.json").write_text(json.dumps(incompatible))
    with pytest.raises(ValueError, match="provenance"):
        vru.ensure_precomputed_vae_quantiles(args)


def test_negative_boolean_override_cannot_change_frozen_risk_controls(experiment, tmp_path):
    args, _, _ = experiment
    args.enable_mdd_penalty = True
    selected = frozen.export_frozen_experiment(args, tmp_path / "selected.json")
    with pytest.raises(ValueError, match="override frozen"):
        final.parse_and_prepare_args(["--frozen_config", selected, "--no-enable_mdd_penalty"])
